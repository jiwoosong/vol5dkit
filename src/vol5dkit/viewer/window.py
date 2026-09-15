"""Qt controls, viewer state and one bounded data-preparation worker."""

from __future__ import annotations

from dataclasses import replace
import math
import sys
import threading

if any(name in sys.modules for name in ("PyQt5", "PyQt6", "PySide2")):
    raise RuntimeError("vol5dkit uses PySide6; start a process without another Qt binding")

from PySide6 import QtCore, QtWidgets

from ._data import Request, make_source, prepare_frame, relative_index
from .canvas import PLANE_AXES, VolumeCanvas


_application = None
_viewers = set()
_loop_running = False


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Create and update viewers on the Python main thread")


def _app():
    global _application
    _main_thread()
    application = QtWidgets.QApplication.instance()
    if application is None:
        _application = QtWidgets.QApplication([])
        application = _application
        application.setApplicationName("vol5dkit")
    if not isinstance(application, QtWidgets.QApplication):
        raise RuntimeError("The existing Qt application does not support widgets")
    return application


def _limits(clim):
    if clim is None:
        return None
    if len(clim) != 2 or not all(math.isfinite(v) for v in clim) or clim[0] > clim[1]:
        raise ValueError("clim must contain finite (low, high) values with low <= high")
    return tuple(clim)


class _Worker(QtCore.QObject):
    finished = QtCore.Signal(int, object, str)

    @QtCore.Slot(object)
    def prepare(self, request):
        try:
            frame = prepare_frame(request)
        except Exception as exc:
            # Do not send tracebacks retaining temporary tensor storage to Qt.
            self.finished.emit(request.revision, None, f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(request.revision, frame, "")


class Viewer(QtWidgets.QMainWindow):
    """A/B volume viewer. All public methods run on the main thread.

    ``frame`` is the last completely displayed snapshot; ``last_error`` is a
    preparation error string, or None. Registration shares detached storage:
    do not write into it concurrently with this viewer's reads.
    """

    _prepare = QtCore.Signal(object)

    def __init__(self, a, b=None, *, rgb=False, interpolation="nearest", clim=None):
        _app()
        if interpolation not in ("nearest", "linear"):
            raise ValueError("interpolation must be 'nearest' or 'linear'")
        sources = {"A": make_source(a, rgb)}
        if b is not None:
            sources["B"] = make_source(b, rgb)
        limits = _limits(clim)
        super().__init__()
        # Closing this window must not implicitly stop a host application's loop.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle("vol5dkit")
        self.resize(1200, 940)
        self.sources = sources
        self.active = "A"
        self.positions = (0.0, 0.5, 0.5, 0.5)
        self.channels = {name: 0 for name in sources}
        self.clim = limits
        self.interpolation = interpolation
        self.frame = None
        self.last_error = None
        self._revision = 0
        self._range_after = 0
        self._busy = False
        self._pending = None
        self._closed = False
        self._syncing = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)
        self.slot_box = QtWidgets.QComboBox()
        self.slot_box.addItems(["A", "B"])
        toolbar.addWidget(QtWidgets.QLabel("Input"))
        toolbar.addWidget(self.slot_box)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["Scalar", "RGB"])
        toolbar.addWidget(self.mode_box)
        self.channel_box = QtWidgets.QSpinBox()
        self.channel_box.setPrefix("C ")
        toolbar.addWidget(self.channel_box)
        self.interpolation_box = QtWidgets.QComboBox()
        self.interpolation_box.addItems(["nearest", "linear"])
        self.interpolation_box.setCurrentText(interpolation)
        toolbar.addWidget(self.interpolation_box)
        self.colormap_box = QtWidgets.QComboBox()
        self.colormap_box.addItems(["grays", "viridis", "hot", "coolwarm", "fire"])
        toolbar.addWidget(self.colormap_box)
        self.view3d_box = QtWidgets.QComboBox()
        self.view3d_box.addItems(["slices", "volume", "hidden"])
        toolbar.addWidget(QtWidgets.QLabel("3D"))
        toolbar.addWidget(self.view3d_box)
        self.stride_box = QtWidgets.QSpinBox()
        self.stride_box.setRange(1, 4096)
        self.stride_box.setPrefix("Preview stride ")
        self.stride_box.setToolTip("Explicit 3D volume preview stride; native planes stay unchanged")
        toolbar.addWidget(self.stride_box)
        self.opacity_box = QtWidgets.QDoubleSpinBox()
        self.opacity_box.setRange(0.0, 1.0)
        self.opacity_box.setSingleStep(0.05)
        self.opacity_box.setValue(0.15)
        self.opacity_box.setPrefix("Opacity ")
        toolbar.addWidget(self.opacity_box)
        fit = QtWidgets.QPushButton("Fit")
        toolbar.addWidget(fit)
        toolbar.addStretch()

        limits_row = QtWidgets.QHBoxLayout()
        layout.addLayout(limits_row)
        limits_row.addWidget(QtWidgets.QLabel("Shared scalar range"))
        self.low_edit = QtWidgets.QLineEdit()
        self.high_edit = QtWidgets.QLineEdit()
        self.low_edit.setPlaceholderText("low")
        self.high_edit.setPlaceholderText("high")
        limits_row.addWidget(self.low_edit)
        limits_row.addWidget(self.high_edit)
        apply_range = QtWidgets.QPushButton("Apply")
        reset_range = QtWidgets.QPushButton("Range from current input")
        limits_row.addWidget(apply_range)
        limits_row.addWidget(reset_range)
        self.crosshair_box = QtWidgets.QCheckBox("Crosshair")
        self.crosshair_box.setChecked(True)
        self.crosshair_box.setToolTip("Show the shared SHW point in all three native planes")
        limits_row.addWidget(self.crosshair_box)
        limits_row.addStretch()

        axes = QtWidgets.QGridLayout()
        layout.addLayout(axes)
        self.axis_sliders = {}
        self.axis_boxes = {}
        self.axis_labels = {}
        for row, name in enumerate(("T", "S", "H", "W")):
            axes.addWidget(QtWidgets.QLabel(name), row, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            box = QtWidgets.QSpinBox()
            label = QtWidgets.QLabel()
            axes.addWidget(slider, row, 1)
            axes.addWidget(box, row, 2)
            axes.addWidget(label, row, 3)
            self.axis_sliders[name], self.axis_boxes[name], self.axis_labels[name] = slider, box, label
            slider.valueChanged.connect(lambda index, n=name: self._move_axis(n, index))
            box.valueChanged.connect(lambda index, n=name: self._move_axis(n, index))
        axes.setColumnStretch(1, 1)

        self.canvas = VolumeCanvas(on_hover=self._hover, on_pick=self._pick, on_scroll=self._scroll)
        self.canvas.widget.setToolTip(
            "Left click/drag: move the shared SHW crosshair\n"
            "Wheel: move through slices; Ctrl+wheel: zoom\n"
            "Shift+left drag: pan; right drag: zoom"
        )
        layout.addWidget(self.canvas.widget, 1)
        self.status = QtWidgets.QLabel("Preparing native slices…")
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.slot_box.currentTextChanged.connect(self._select_slot)
        self.mode_box.currentTextChanged.connect(self._change_mode)
        self.channel_box.valueChanged.connect(self._change_channel)
        self.interpolation_box.currentTextChanged.connect(self._change_interpolation)
        self.colormap_box.currentTextChanged.connect(self.canvas.set_colormap)
        self.view3d_box.currentTextChanged.connect(self._change_3d)
        self.stride_box.valueChanged.connect(lambda _: self._request())
        self.opacity_box.valueChanged.connect(self.canvas.set_opacity)
        self.crosshair_box.toggled.connect(self.canvas.set_crosshair_visible)
        fit.clicked.connect(self.canvas.reset_view)
        apply_range.clicked.connect(self._apply_range)
        reset_range.clicked.connect(self._reset_range)
        self.low_edit.returnPressed.connect(self._apply_range)
        self.high_edit.returnPressed.connect(self._apply_range)

        from PySide6.QtGui import QShortcut, QKeySequence
        for key, name in (("A", "A"), ("B", "B")):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda n=name: self._select_slot(n))
        toggle = QShortcut(QKeySequence("Tab"), self)
        toggle.activated.connect(lambda: self._select_slot("B" if self.active == "A" else "A"))

        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._prepare.connect(self._worker.prepare)
        self._worker.finished.connect(self._received)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        _viewers.add(self)
        self._sync_controls()
        self._request()

    def _sync_controls(self):
        self._syncing = True
        try:
            source = self.sources[self.active]
            shape = source.volume.shape
            self.slot_box.model().item(1).setEnabled("B" in self.sources)
            self.slot_box.setCurrentText(self.active)
            self.mode_box.setCurrentText("RGB" if source.rgb else "Scalar")
            self.mode_box.model().item(1).setEnabled(shape[1] == 3)
            self.channel_box.setRange(0, shape[1] - 1)
            self.channels[self.active] = min(self.channels[self.active], shape[1] - 1)
            self.channel_box.setValue(self.channels[self.active])
            self.channel_box.setEnabled(not source.rgb and shape[1] > 1)
            self.colormap_box.setEnabled(not source.rgb)
            self.view3d_box.model().item(1).setEnabled(not source.rgb)
            if source.rgb and self.view3d_box.currentText() == "volume":
                self.view3d_box.setCurrentText("slices")
            self.stride_box.setEnabled(self.view3d_box.currentText() == "volume")
            self.opacity_box.setEnabled(self.view3d_box.currentText() == "volume")
            for n, length, position in zip(("T", "S", "H", "W"), (shape[0], *shape[2:]), self.positions):
                index = relative_index(position, length)
                for control in (self.axis_sliders[n], self.axis_boxes[n]):
                    control.setRange(0, length - 1)
                    control.setValue(index)
                    control.setEnabled(length > 1)
                text = f"/ {length - 1}"
                if n == "T":
                    text += f"   coordinate {source.volume.times[index]:g}"
                self.axis_labels[n].setText(text)
            if self.clim is not None:
                self.low_edit.setText(str(self.clim[0]))
                self.high_edit.setText(str(self.clim[1]))
        finally:
            self._syncing = False

    def _request(self):
        if self._syncing or self._closed:
            return
        self._revision += 1
        self.last_error = None
        request = Request(
            self._revision, self.active, self.sources[self.active], self.positions,
            channel=self.channels[self.active], clim=self.clim,
            volume_3d=self.view3d_box.currentText() == "volume",
            preview_stride=self.stride_box.value(),
        )
        self.status.setText(f"Preparing {self.active}…")
        if self._busy:
            self._pending = request
        else:
            self._busy = True
            self._prepare.emit(request)

    @QtCore.Slot(int, object, str)
    def _received(self, revision, frame, error):
        self._busy = False
        if self._closed:
            return
        # Establish the first scalar range even if that initial frame was
        # superseded by a slice/A-B request. Explicit range resets invalidate it.
        if frame is not None and not frame.source.rgb and self.clim is None and revision >= self._range_after:
            self.clim = frame.clim
            self._sync_controls()
        if revision == self._revision:
            if error:
                self.last_error = error
                self.status.setText(error)
            else:
                try:
                    self.canvas.set_frame(
                        frame, interpolation=self.interpolation,
                        mode_3d=self.view3d_box.currentText(), opacity=self.opacity_box.value(),
                    )
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.status.setText(self.last_error)
                else:
                    self.frame = frame
                    self.setWindowTitle(f"vol5dkit · {frame.slot} · {tuple(frame.source.volume.shape)}")
                    self.status.setText(
                        f"{frame.slot}  T,C,S,H,W = {frame.indices}  |  "
                        f"{frame.source.volume.dtype} on {frame.source.volume.device}  |  "
                        "Click/drag: crosshair · Wheel: slice · Ctrl+wheel: zoom"
                    )
        if self._pending is not None:
            request, self._pending = self._pending, None
            self._busy = True
            self._prepare.emit(replace(request, clim=self.clim))

    def _select_slot(self, name):
        if self._syncing or name not in self.sources or name == self.active:
            return
        self.active = name
        self._sync_controls()
        self._request()

    def _move_axis(self, name, index):
        if self._syncing:
            return
        axis = ("T", "S", "H", "W").index(name)
        shape = self.sources[self.active].volume.shape
        length = (shape[0], *shape[2:])[axis]
        positions = list(self.positions)
        positions[axis] = index / (length - 1) if length > 1 else positions[axis]
        self.positions = tuple(positions)
        self._sync_controls()
        self._request()

    def _change_channel(self, index):
        if not self._syncing:
            self.channels[self.active] = index
            self._request()

    def _change_mode(self, mode):
        if self._syncing:
            return
        try:
            source = make_source(self.sources[self.active].volume, mode == "RGB")
        except (TypeError, ValueError) as exc:
            self.last_error = str(exc)
            self.status.setText(self.last_error)
            self._sync_controls()
            return
        # Changing display mode must retain the original producer dependency.
        source = replace(source, ready=self.sources[self.active].ready)
        self.sources[self.active] = source
        self._sync_controls()
        self._request()

    def _change_interpolation(self, value):
        self.interpolation = value
        self.canvas.set_interpolation(value)

    def _change_3d(self, value):
        if not self._syncing:
            self._sync_controls()
            self._request()

    def _apply_range(self):
        try:
            # Keep integer endpoints exact for int64/uint64 narrow windows.
            values = []
            for editor in (self.low_edit, self.high_edit):
                value = editor.text().strip()
                try:
                    values.append(int(value))
                except ValueError:
                    values.append(float(value))
            self.clim = _limits(values)
        except (TypeError, ValueError, OverflowError) as exc:
            self.last_error = f"Invalid range: {exc}"
            self.status.setText(self.last_error)
            return
        self._request()

    def _reset_range(self):
        self.clim = None
        self._range_after = self._revision + 1
        self._request()

    def _hover(self, name, row, col):
        frame = self.frame
        if frame is None:
            return
        plane = next(p for p in frame.planes if p.name == name)
        row, col = math.floor(row + 0.5), math.floor(col + 0.5)
        if not (0 <= row < plane.raw.shape[0] and 0 <= col < plane.raw.shape[1]):
            return
        r_axis, c_axis, _ = PLANE_AXES[name]
        shw = list(frame.indices[2:])
        shw[r_axis], shw[c_axis] = row, col
        world = frame.source.volume.index_to_world(shw)
        value = plane.raw[row, col].tolist()
        self.status.setText(
            f"{frame.slot} · {name} · SHW {tuple(shw)} · value {value} · "
            f"XYZ ({world[0]:g}, {world[1]:g}, {world[2]:g})"
        )

    def _pick(self, name, row, col):
        if self.frame is None or self.frame.source is not self.sources[self.active]:
            return
        shape = self.sources[self.active].volume.shape[2:]
        positions = list(self.positions)
        for axis, value in zip(PLANE_AXES[name][:2], (row, col)):
            index = min(max(math.floor(value + 0.5), 0), shape[axis] - 1)
            if shape[axis] > 1:
                positions[axis + 1] = index / (shape[axis] - 1)
        if tuple(positions) == self.positions:
            return
        self.positions = tuple(positions)
        self._sync_controls()
        self._request()

    def _scroll(self, name, steps):
        if self.frame is None or self.frame.source is not self.sources[self.active]:
            return
        axis = PLANE_AXES[name][2]
        length = self.sources[self.active].volume.shape[axis + 2]
        # Accumulate against the requested point, even while an older frame is
        # being prepared. Fast wheel input must not lose steps.
        current = relative_index(self.positions[axis + 1], length)
        index = min(max(current + steps, 0), length - 1)
        if index != current:
            self._move_axis("SHW"[axis], index)

    def update(self, slot=None, volume=None, *, rgb=None):
        """Replace A/B with a detached alias, preserving navigation and range.

        Call within the producing CUDA stream or establish the calling stream's
        dependency first. With no arguments this retains QWidget.update().
        """
        if slot is None and volume is None:
            return super().update()
        _main_thread()
        if self._closed:
            raise RuntimeError("Cannot update a closed viewer")
        if slot not in ("A", "B"):
            raise ValueError("slot must be 'A' or 'B'")
        previous = self.sources.get(slot)
        if rgb is None:
            rgb = previous.rgb if previous is not None else False
        source = make_source(volume, rgb)
        self.sources[slot] = source
        self.channels.setdefault(slot, 0)
        self._sync_controls()
        if slot == self.active:
            self._request()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._pending = None
            self._thread.quit()
            self._thread.wait()
            self.canvas.close()
            self.frame = None
            self.sources.clear()
            _viewers.discard(self)
            if _loop_running and not _viewers:
                _app().quit()
        event.accept()


def view(a, b=None, *, rgb=False, interpolation="nearest", clim=None, block=True):
    """Open a viewer and optionally run the desktop event loop."""
    viewer = Viewer(a, b, rgb=rgb, interpolation=interpolation, clim=clim)
    del a, b
    viewer.show()
    if block:
        run()
    return viewer


def run():
    """Run this library's event loop until its last viewer closes."""
    global _loop_running
    application = _app()
    if _loop_running or QtCore.QThread.currentThread().loopLevel() > 0:
        raise RuntimeError("A Qt event loop is already running; use view(..., block=False)")
    if not _viewers:
        return 0
    _loop_running = True
    try:
        return application.exec()
    finally:
        _loop_running = False
