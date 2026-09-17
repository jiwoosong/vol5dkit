"""Qt controls, viewer state and one bounded data-preparation worker."""

from __future__ import annotations

from dataclasses import replace
import math
import sys
import threading

if any(name in sys.modules for name in ("PyQt5", "PyQt6", "PySide2")):
    raise RuntimeError("vol5dkit uses PySide6; start a process without another Qt binding")

from PySide6 import QtCore, QtWidgets

from .._view import Display, _validate_window
from ._data import Request, make_source, prepare_frame, relative_index
from .canvas import PLANE_AXES, VolumeCanvas, available_colormaps


class _Worker(QtCore.QObject):
    finished = QtCore.Signal(int, object, str)

    def __init__(self):
        super().__init__()
        self._volume_key = None
        self._volume_buffer = None

    def clear_volume(self):
        self._volume_key = self._volume_buffer = None

    @QtCore.Slot(object)
    def prepare(self, request):
        try:
            key = None
            if request.volume_3d and not request.source.rgb:
                key = (
                    request.input_index, request.source.volume,
                    relative_index(request.positions[0], request.source.volume.shape[0]),
                    request.channel, request.window,
                )
            if key is None or key != self._volume_key:
                self.clear_volume()
            frame = prepare_frame(request, volume_buffer=self._volume_buffer)
            if frame.volume_buffer is not None:
                self._volume_key = (*key[:-1], frame.window)
                self._volume_buffer = frame.volume_buffer
        except Exception as exc:
            # Do not send tracebacks retaining temporary tensor storage to Qt.
            self.finished.emit(request.revision, None, f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(request.revision, frame, "")


class _ColormapBox(QtWidgets.QComboBox):
    """Load optional providers only when the picker is first opened."""

    opening = QtCore.Signal()

    def showPopup(self):
        self.opening.emit()
        super().showPopup()


class Viewer(QtWidgets.QMainWindow):
    """The child process's native-grid snapshot window.

    ``frame`` is the last completely displayed snapshot; ``last_error`` is a
    preparation/display error string, or None. Inputs are immutable CPU snapshots
    loaded by the child entry point, which also owns the QApplication.
    """

    _prepare = QtCore.Signal(object)

    def __init__(self, inputs, *, window=None, interpolation="nearest"):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Create viewers on the Python main thread")
        if not isinstance(QtWidgets.QApplication.instance(), QtWidgets.QApplication):
            raise RuntimeError("Create a QApplication before the snapshot viewer")
        if interpolation not in ("nearest", "linear"):
            raise ValueError("interpolation must be 'nearest' or 'linear'")
        if not inputs:
            raise ValueError("at least one input is required")
        sources, names, windows, colormaps = {}, {}, {}, {}
        for input_index, display in enumerate(inputs):
            if not isinstance(display, Display):
                raise TypeError("viewer inputs must be normalized Display objects")
            sources[input_index] = make_source(display.data, display.rgb)
            names[input_index] = display.name or f"Volume {input_index + 1}"
            windows[input_index] = _validate_window(display.window)
            colormaps[input_index] = display.cmap
        limits = _validate_window(window)
        super().__init__()
        self.setWindowTitle("vol5dkit")
        self.resize(1200, 840)
        self.sources = sources
        self.names = names
        self.windows = windows
        self.colormaps = colormaps
        self.active = 0
        self.positions = (0.0, 0.5, 0.5, 0.5)
        self.channels = {name: 0 for name in sources}
        self.shared_window = limits
        self.share_window = limits is not None
        self._shared_range_source = None
        self.interpolation = interpolation
        self.frame = None
        self.last_error = None
        self._revision = 0
        self._range_generation = 0
        self._inflight = None
        self._inflight_generation = 0
        self._pending_generation = 0
        self._colormaps_loaded = False
        self._busy = False
        self._pending = None
        self._closed = False
        self._syncing = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = self.addToolBar("Viewer")
        toolbar.setMovable(False)
        self.input_box = QtWidgets.QComboBox()
        for input_index, name in self.names.items():
            self.input_box.addItem(name, input_index)
        self.input_box.setMinimumContentsLength(14)
        self.input_box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        toolbar.addWidget(QtWidgets.QLabel("Input"))
        toolbar.addWidget(self.input_box)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["Scalar", "RGB"])
        toolbar.addWidget(self.mode_box)
        self.channel_box = QtWidgets.QSpinBox()
        self.channel_box.setPrefix("C ")
        toolbar.addWidget(self.channel_box)
        fit = QtWidgets.QPushButton("Fit")
        toolbar.addWidget(fit)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.controls_dock = QtWidgets.QDockWidget("Controls", self)
        self.controls_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.controls_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        panel = QtWidgets.QWidget()
        settings = QtWidgets.QVBoxLayout(panel)
        settings.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        settings.setContentsMargins(8, 8, 8, 8)
        self.controls_scroll.setWidget(panel)
        self.controls_dock.setWidget(self.controls_scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)
        toolbar.addAction(self.controls_dock.toggleViewAction())

        navigation = QtWidgets.QGroupBox("Navigation")
        settings.addWidget(navigation)
        axes = QtWidgets.QGridLayout(navigation)
        self.axis_sliders = {}
        self.axis_boxes = {}
        self.axis_labels = {}
        for axis, name in enumerate(("T", "S", "H", "W")):
            row = axis * 2
            axes.addWidget(QtWidgets.QLabel(name), row, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            box = QtWidgets.QSpinBox()
            label = QtWidgets.QLabel()
            label.setWordWrap(True)
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            axes.addWidget(label, row, 1)
            axes.addWidget(box, row, 2)
            axes.addWidget(slider, row + 1, 0, 1, 3)
            self.axis_sliders[name], self.axis_boxes[name], self.axis_labels[name] = slider, box, label
            slider.valueChanged.connect(lambda index, n=name: self._move_axis(n, index))
            box.valueChanged.connect(lambda index, n=name: self._move_axis(n, index))
        axes.setColumnStretch(1, 1)

        display = QtWidgets.QGroupBox("Display")
        settings.addWidget(display)
        display_layout = QtWidgets.QFormLayout(display)
        self.interpolation_box = QtWidgets.QComboBox()
        self.interpolation_box.addItems(["nearest", "linear"])
        self.interpolation_box.setCurrentText(interpolation)
        display_layout.addRow("Interpolation", self.interpolation_box)
        self.colormap_box = _ColormapBox()
        self.colormap_box.setEditable(True)
        self.colormap_box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.colormap_box.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.colormap_box.completer().setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.colormap_box.completer().setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        for value, label in available_colormaps():
            self.colormap_box.addItem(label, value)
        display_layout.addRow("Colormap", self.colormap_box)
        self.scalar_range = QtWidgets.QWidget()
        range_layout = QtWidgets.QGridLayout(self.scalar_range)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.share_box = QtWidgets.QCheckBox("Share window")
        self.share_box.setToolTip("Share the current input's window; uncheck to restore each input's own window.")
        range_layout.addWidget(self.share_box, 0, 0, 1, 2)
        self.low_edit = QtWidgets.QLineEdit()
        self.high_edit = QtWidgets.QLineEdit()
        self.low_edit.setPlaceholderText("low")
        self.high_edit.setPlaceholderText("high")
        range_layout.addWidget(self.low_edit, 1, 0)
        range_layout.addWidget(self.high_edit, 1, 1)
        apply_range = QtWidgets.QPushButton("Apply")
        reset_range = QtWidgets.QPushButton("Auto")
        reset_range.setToolTip("Use finite min/max over this input's complete TCSHW tensor.")
        self.auto_range_button = reset_range
        range_layout.addWidget(apply_range, 2, 0)
        range_layout.addWidget(reset_range, 2, 1)
        display_layout.addRow(self.scalar_range)
        self.crosshair_box = QtWidgets.QCheckBox("Crosshair")
        self.crosshair_box.setChecked(True)
        self.crosshair_box.setToolTip("Show the shared SHW point in all three native planes")
        display_layout.addRow(self.crosshair_box)

        view3d = QtWidgets.QGroupBox("3D")
        settings.addWidget(view3d)
        view3d_layout = QtWidgets.QFormLayout(view3d)
        self.view3d_box = QtWidgets.QComboBox()
        for label, mode in (("Planes", "slices"), ("Volume", "volume"), ("Off", "hidden")):
            self.view3d_box.addItem(label, mode)
        view3d_layout.addRow("Mode", self.view3d_box)
        self.volume_controls = QtWidgets.QWidget()
        volume_layout = QtWidgets.QFormLayout(self.volume_controls)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        self.volume_shape_label = QtWidgets.QLabel()
        self.volume_shape_label.setWordWrap(True)
        volume_layout.addRow(self.volume_shape_label)
        self.opacity_box = QtWidgets.QDoubleSpinBox()
        self.opacity_box.setRange(0.0, 1.0)
        self.opacity_box.setSingleStep(0.05)
        self.opacity_box.setValue(0.15)
        self.opacity_box.setToolTip(
            "Per-sample alpha = normalized scalar value × this setting.\n"
            "Samples accumulate along the viewing ray; this is not the final\n"
            "image's transparency percentage. The scalar window also affects\n"
            "the result. Every voxel is used."
        )
        volume_layout.addRow("3D opacity", self.opacity_box)
        view3d_layout.addRow(self.volume_controls)
        settings.addStretch()

        self.canvas = VolumeCanvas(on_hover=self._hover, on_pick=self._pick, on_scroll=self._scroll)
        self.canvas.widget.setToolTip(
            "Left click/drag: move the shared SHW crosshair\n"
            "Wheel: move through slices; Ctrl+wheel: zoom\n"
            "Shift+left drag: pan; right drag: zoom"
        )
        layout.addWidget(self.canvas.widget, 1)
        self.canvas.widget.setMinimumSize(1, 1)
        self.status = QtWidgets.QLabel("Preparing native slices…")
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.status.setFixedHeight(2 * self.status.fontMetrics().lineSpacing() + 4)
        self.statusBar().addWidget(self.status, 1)
        self.resizeDocks([self.controls_dock], [300], QtCore.Qt.Orientation.Horizontal)

        self.input_box.currentIndexChanged.connect(self._select_input)
        self.mode_box.currentTextChanged.connect(self._change_mode)
        self.channel_box.valueChanged.connect(self._change_channel)
        self.interpolation_box.currentTextChanged.connect(self._change_interpolation)
        self.colormap_box.activated.connect(self._choose_colormap)
        self.colormap_box.lineEdit().returnPressed.connect(self._choose_colormap)
        self.colormap_box.opening.connect(self._load_colormaps)
        self.view3d_box.currentTextChanged.connect(self._change_3d)
        self.share_box.toggled.connect(self._change_share)
        self.opacity_box.valueChanged.connect(self.canvas.set_opacity)
        self.crosshair_box.toggled.connect(self.canvas.set_crosshair_visible)
        fit.clicked.connect(self.canvas.reset_view)
        apply_range.clicked.connect(self._apply_range)
        reset_range.clicked.connect(self._reset_range)
        self.low_edit.returnPressed.connect(self._apply_range)
        self.high_edit.returnPressed.connect(self._apply_range)

        from PySide6.QtGui import QShortcut, QKeySequence
        for key, name in (("A", 0), ("B", 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda n=name: self._select_input(n))
        toggle = QShortcut(QKeySequence("Tab"), self)
        toggle.activated.connect(self._next_input)

        # Invalid startup palettes fail before the worker is started.
        try:
            name = self.colormaps[self.active]
            self._prepared_palette = (self.canvas.palette if name == self.canvas.colormap
                                      else self.canvas.prepare_colormap(name))
            self.canvas.set_colormap(self._prepared_palette)
        except Exception:
            self.canvas.close()
            raise

        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._prepare.connect(self._worker.prepare)
        self._worker.finished.connect(self._received)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        self._sync_controls()
        self._request()

    @property
    def current_window(self):
        return self.shared_window if self.share_window else self.windows[self.active]

    def _window_for(self, input_index):
        return self.shared_window if self.share_window else self.windows[input_index]

    def _set_window(self, value):
        if self.share_window:
            self.shared_window = value
        else:
            self.windows[self.active] = value

    def _sync_colormap(self):
        value = self.colormaps[self.active]
        # Preserve the familiar unprefixed VisPy API while displaying its source.
        index = self.colormap_box.findData(value)
        if index < 0 and ":" not in value:
            index = self.colormap_box.findData("vispy:" + value)
        if index < 0:
            self.colormap_box.addItem(value, value)
            index = self.colormap_box.count() - 1
        with QtCore.QSignalBlocker(self.colormap_box):
            self.colormap_box.setCurrentIndex(index)

    def _sync_controls(self):
        self._syncing = True
        try:
            source = self.sources[self.active]
            shape = source.volume.shape
            self.input_box.setCurrentIndex(self.active)
            self.mode_box.setCurrentText("RGB" if source.rgb else "Scalar")
            self.mode_box.model().item(1).setEnabled(shape[1] == 3)
            self.channel_box.setRange(0, shape[1] - 1)
            self.channels[self.active] = min(self.channels[self.active], shape[1] - 1)
            self.channel_box.setValue(self.channels[self.active])
            self.channel_box.setEnabled(not source.rgb and shape[1] > 1)
            self.colormap_box.setEnabled(not source.rgb)
            self.scalar_range.setEnabled(not source.rgb)
            self.share_box.setChecked(self.share_window)
            self._sync_colormap()
            self.view3d_box.model().item(1).setEnabled(not source.rgb)
            if source.rgb and self.view3d_box.currentData() == "volume":
                self.view3d_box.setCurrentIndex(0)
            volume_mode = self.view3d_box.currentData() == "volume"
            self.volume_controls.setVisible(volume_mode)
            self.opacity_box.setEnabled(volume_mode)
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
            if self.current_window is not None:
                self.low_edit.setText(str(self.current_window[0]))
                self.high_edit.setText(str(self.current_window[1]))
            else:
                self.low_edit.clear()
                self.high_edit.clear()
        finally:
            self._syncing = False

    def _request(self):
        if self._syncing or self._closed:
            return
        self._revision += 1
        self.last_error = None
        request = Request(
            self._revision, self.active, self.sources[self.active], self.positions,
            channel=self.channels[self.active], window=self.current_window,
            volume_3d=self.view3d_box.currentData() == "volume",
            range_source=self._shared_range_source if self.share_window and self.current_window is None else None,
        )
        self.status.setText(f"Preparing {self.names[self.active]}…")
        if request.volume_3d:
            self.volume_shape_label.setText("Preparing volume…")
        if self._busy:
            self._pending = request
            self._pending_generation = self._range_generation
        else:
            self._busy = True
            self._inflight = request
            self._inflight_generation = self._range_generation
            self._prepare.emit(request)

    @QtCore.Slot(int, object, str)
    def _received(self, revision, frame, error):
        self._busy = False
        request, self._inflight = self._inflight, None
        if self._closed:
            return
        # A superseded navigation request may still fill its own immutable
        # snapshot's range cache. Manual edits/Auto/share changes invalidate it.
        if (frame is not None and request is not None and request.window is None
                and not frame.source.rgb
                and self._inflight_generation == self._range_generation
                and self.sources.get(frame.input_index) is frame.source):
            if self.share_window:
                if (self.shared_window is None
                        and request.range_source is self._shared_range_source):
                    self.shared_window = frame.window
            elif self.windows[frame.input_index] is None:
                self.windows[frame.input_index] = frame.window
            self._sync_controls()
        if revision == self._revision:
            if error:
                self.last_error = error
                self.status.setText(error)
            else:
                previous_palette = self.canvas.palette
                try:
                    # Commit input palette and pixels in the same GUI callback.
                    # Until then the last complete frame keeps its own palette.
                    self.canvas.set_frame(
                        frame, interpolation=self.interpolation,
                        mode_3d=self.view3d_box.currentData(), opacity=self.opacity_box.value(),
                        palette=self._prepared_palette,
                    )
                except Exception as exc:
                    if self.canvas.palette is not previous_palette:
                        try:
                            self.canvas.set_colormap(previous_palette)
                        except Exception:
                            # A renderer/provider failure may also prevent
                            # rollback. Keep the original display error visible.
                            pass
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.status.setText(self.last_error)
                else:
                    self.frame = frame
                    if frame.volume_buffer is not None:
                        shape = " × ".join(str(n) for n in frame.volume_buffer.shape)
                        self.volume_shape_label.setText(f"Volume SHW: {shape}")
                    self.setWindowTitle(f"vol5dkit · {self.names[frame.input_index]} · {tuple(frame.source.volume.shape)}")
                    self.status.setText(
                        f"{self.names[frame.input_index]}  T,C,S,H,W = {frame.indices}  |  "
                        f"{frame.source.volume.dtype} on {frame.source.volume.device}  |  "
                        "Click/drag: crosshair · Wheel: slice · Ctrl+wheel: zoom"
                    )
        if self._pending is not None:
            request, self._pending = self._pending, None
            self._busy = True
            # Reuse a range just computed for this input instead of scanning it
            # again for the pending slice. The queue still contains one request.
            self._inflight = replace(request, window=self._window_for(request.input_index))
            self._inflight_generation = self._pending_generation
            self._prepare.emit(self._inflight)

    def _select_input(self, input_index):
        if self._syncing or input_index not in self.sources or input_index == self.active:
            return
        try:
            palette = self.canvas.prepare_colormap(self.colormaps[input_index])
        except Exception as exc:
            self.last_error = f"Colormap: {type(exc).__name__}: {exc}"
            self.status.setText(self.last_error)
            self._sync_controls()
            return
        self.active = input_index
        self._prepared_palette = palette
        self._sync_controls()
        self._request()

    def _next_input(self):
        self._select_input((self.active + 1) % len(self.sources))

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
        self.sources[self.active] = source
        self._sync_controls()
        self._request()

    def _change_interpolation(self, value):
        self.interpolation = value
        self.canvas.set_interpolation(value)

    def _load_colormaps(self):
        if self._colormaps_loaded:
            return
        try:
            choices = available_colormaps(extended=True)
        except Exception as exc:
            self.last_error = f"Colormaps: {type(exc).__name__}: {exc}"
            self.status.setText(self.last_error)
            return
        with QtCore.QSignalBlocker(self.colormap_box):
            self.colormap_box.clear()
            for value, label in choices:
                self.colormap_box.addItem(label, value)
        self._colormaps_loaded = True
        self._sync_colormap()
        self.last_error = None
        self.status.setText("Colormaps loaded")

    def _choose_colormap(self, *_):
        text = self.colormap_box.currentText()
        index = self.colormap_box.findText(text)
        value = self.colormap_box.itemData(index) if index >= 0 else text.strip()
        self._change_colormap(value)

    def _change_colormap(self, value):
        try:
            palette = self.canvas.prepare_colormap(value)
            if self.frame is not None and self.frame.source is self.sources[self.active]:
                self.canvas.set_colormap(palette)
        except Exception as exc:
            self.last_error = f"Colormap: {type(exc).__name__}: {exc}"
            self.status.setText(self.last_error)
            self._sync_colormap()
        else:
            self.colormaps[self.active] = value
            self._prepared_palette = palette
            self._sync_colormap()
            self.last_error = None
            self.status.setText(f"Colormap: {value}")

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
            limits = _validate_window(values)
        except (TypeError, ValueError, OverflowError) as exc:
            self.last_error = f"Invalid range: {exc}"
            self.status.setText(self.last_error)
            return
        self._range_generation += 1
        self._set_window(limits)
        self._request()

    def _reset_range(self):
        self._range_generation += 1
        self._set_window(None)
        if self.share_window:
            self._shared_range_source = self.sources[self.active]
        self._sync_controls()
        self._request()

    def _change_share(self, checked):
        if self._syncing:
            return
        self._range_generation += 1
        if checked:
            self.shared_window = self.windows[self.active]
            self._shared_range_source = self.sources[self.active] if self.shared_window is None else None
        else:
            self._shared_range_source = None
        self.share_window = checked
        self._sync_controls()
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
            f"{self.names[frame.input_index]} · {name} · SHW {tuple(shw)} · value {value} · "
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

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._pending = None
            self._thread.quit()
            self._thread.wait()
            self._inflight = None
            self._worker.clear_volume()
            self._shared_range_source = None
            self.canvas.close()
            self.frame = None
            self.sources.clear()
        event.accept()
