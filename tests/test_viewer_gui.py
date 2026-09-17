"""Real Qt/VisPy integration; opt in with VOL5DKIT_TEST_GUI=1."""

import gc
import os
from pathlib import Path
import threading
import time
import weakref

import numpy as np
import pytest
import torch

pytest.importorskip("vispy")
pytest.importorskip("PySide6")
if os.environ.get("VOL5DKIT_TEST_GUI") != "1":
    pytest.skip("set VOL5DKIT_TEST_GUI=1 for desktop GUI integration", allow_module_level=True)

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from vol5dkit import Volume, Display
from vol5dkit._view import _normalize_inputs
from vol5dkit.viewer import window
from vol5dkit.viewer._data import relative_index

pytestmark = pytest.mark.gui


def _open(*inputs, **kwargs):
    viewer = window.Viewer(_normalize_inputs(inputs), **kwargs)
    viewer.show()
    return viewer


def _wait(application, predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for Qt/data preparation")


def _settled(application, viewer):
    _wait(application, lambda: not viewer._busy and viewer._pending is None)
    assert viewer.last_error is None, viewer.last_error
    assert viewer.frame is not None
    assert viewer.frame.revision == viewer._revision
    assert viewer.canvas._frame is viewer.frame


def _pixel_position(viewer, name, row, col):
    viewer.canvas.render()  # Resolve the current view/camera transforms.
    point = viewer.canvas._images[name].get_transform(map_from="visual", map_to="canvas").map((col + 0.5, row + 0.5))
    x, y = point[:2] / point[3]
    return QtCore.QPoint(round(x), round(y))


def _wheel(viewer, position, delta, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
    widget = viewer.canvas.widget
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position), QtCore.QPointF(widget.mapToGlobal(position)),
        QtCore.QPoint(), QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton, modifiers,
        QtCore.Qt.ScrollPhase.NoScrollPhase, False,
    )
    QtWidgets.QApplication.sendEvent(widget, event)


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def close_windows(application):
    yield
    for widget in application.topLevelWidgets():
        if isinstance(widget, window.Viewer):
            widget.close()
            widget.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()


@pytest.fixture
def viewer(application):
    a = Volume(torch.arange(2 * 5 * 7 * 9, dtype=torch.float32).reshape(2, 1, 5, 7, 9), spacing=(2, 1, 0.5), times=(0, 0.25))
    viewer = _open(
        Display(a, name='Original'),
        Display(Volume(torch.full((1, 1, 1, 4, 6), 100.0)), name='Prediction', cmap='fire'),
        Display(Volume(torch.full((1, 3, 4, 6, 8), 0.5)), name='Colors', rgb=True),
    )
    _settled(application, viewer)
    yield viewer
    viewer.close()
    application.processEvents()


def test_startup_native_frame_range_hover_and_full_window_capture(application, viewer):
    assert viewer.frame.indices == (0, 0, 2, 3, 4)
    assert viewer.current_window == (0, 629)
    assert viewer.interpolation == "nearest"
    assert viewer.canvas.render().shape[-1] == 4
    viewer._hover("HW", 1, 2)
    assert "value 137.0" in viewer.status.text()
    assert "SHW (2, 1, 2)" in viewer.status.text()
    assert "XYZ (1, 1, 4)" in viewer.status.text()
    assert "coordinate 0" in viewer.axis_labels["T"].text()
    viewer.update()  # QWidget's no-argument paint request remains available.
    application.processEvents()
    artifact = Path(__file__).resolve().parents[1] / ".test-artifacts" / "viewer.png"
    artifact.parent.mkdir(exist_ok=True)
    assert viewer.grab().save(str(artifact))


def test_ab_relative_position_no_drift_including_singleton(application, viewer):
    viewer.axis_sliders["S"].setValue(1)
    viewer.axis_boxes["H"].setValue(5)
    _settled(application, viewer)
    positions = viewer.positions
    for _ in range(4):
        viewer.input_box.setCurrentIndex(1)
        _settled(application, viewer)
        assert viewer.positions == positions
        assert viewer.frame.indices == (0, 0, 0, relative_index(positions[2], 4), relative_index(positions[3], 6))
        assert not viewer.axis_boxes["S"].isEnabled()
        viewer.input_box.setCurrentIndex(0)
        _settled(application, viewer)
        assert viewer.positions == positions
        assert viewer.frame.indices == (0, 0, 1, 5, 4)


def test_inputs_restore_channel_mode_window_colormap_and_interpolation(application, viewer):
    viewer.interpolation_box.setCurrentText("linear")
    viewer._change_colormap("hot")
    viewer.low_edit.setText("10")
    viewer.high_edit.setText("200")
    viewer._apply_range()
    _settled(application, viewer)
    viewer._next_input()
    _settled(application, viewer)
    assert viewer.frame.input_index == 1
    assert viewer.current_window == (100, 100)
    assert viewer.canvas.colormap == "fire"
    viewer._next_input()
    _settled(application, viewer)
    assert viewer.frame.input_index == 2 and viewer.frame.source.rgb
    assert not viewer.channel_box.isEnabled()
    assert viewer.canvas._images["HW"]._texture.internalformat == "rgb32f"
    viewer.mode_box.setCurrentText("Scalar")
    viewer.channel_box.setValue(2)
    _settled(application, viewer)
    assert viewer.current_window == (0.5, 0.5)
    viewer._next_input()
    _settled(application, viewer)
    assert viewer.frame.input_index == 0 and viewer.frame.window == (10, 200)
    assert viewer.canvas.colormap == "hot"
    assert viewer.canvas._images["HW"].interpolation == "linear"
    viewer._select_input(2)
    _settled(application, viewer)
    assert viewer.frame.indices[1] == 2 and not viewer.frame.source.rgb
    assert [viewer.input_box.itemText(i) for i in range(3)] == ["Original", "Prediction", "Colors"]


def test_shared_window_restores_individual_values_and_auto_is_current_input(application):
    a = Volume(torch.tensor([0., 3., -10., 50.]).reshape(2, 2, 1, 1, 1))
    b = Volume(torch.tensor([20., 30.]).reshape(1, 1, 1, 1, 2))
    viewer = _open(Display(a, window=(-1, 1)), b)
    _settled(application, viewer)
    assert viewer.current_window == (-1, 1) and not viewer.share_box.isChecked()
    viewer.share_box.setChecked(True)
    _settled(application, viewer)
    viewer._select_input(1)
    _settled(application, viewer)
    assert viewer.frame.window == (-1, 1) and viewer.windows[1] is None
    viewer._reset_range()
    _settled(application, viewer)
    assert viewer.frame.window == (20, 30) and viewer.windows[1] is None
    viewer.share_box.setChecked(False)
    _settled(application, viewer)
    assert viewer.current_window == (20, 30)
    viewer._select_input(0)
    _settled(application, viewer)
    assert viewer.current_window == (-1, 1)
    viewer._reset_range()
    _settled(application, viewer)
    assert viewer.current_window == (-10, 50)  # All T and C, not only the selected frame.


def test_explicit_common_window_overrides_but_preserves_individual_options(application):
    a = Volume(torch.zeros(1, 1, 2, 2, 2))
    viewer = _open(Display(a, window=(10, 20)), Display(a, window=(-4, 4)), window=(1, 3))
    _settled(application, viewer)
    assert viewer.share_box.isChecked() and viewer.current_window == (1, 3)
    viewer._select_input(1)
    _settled(application, viewer)
    assert viewer.frame.window == (1, 3)
    viewer.share_box.setChecked(False)
    _settled(application, viewer)
    assert viewer.frame.window == (-4, 4)
    viewer._select_input(0)
    _settled(application, viewer)
    assert viewer.frame.window == (10, 20)


def test_full_volume_and_pick(application, viewer):
    assert not hasattr(viewer, "stride_box")
    original = tuple(plane.raw.clone() for plane in viewer.frame.planes)
    viewer.view3d_box.setCurrentText("Volume")
    _settled(application, viewer)
    assert viewer.frame.volume_buffer.shape == (5, 7, 9)
    assert all(torch.equal(raw, plane.raw) for raw, plane in zip(original, viewer.frame.planes))
    assert viewer.canvas.render().shape[-1] == 4
    assert viewer.canvas._volume_visual.visible
    viewer.opacity_box.setValue(0.3)
    assert viewer.canvas._opacity == 0.3
    viewer._pick("HW", 0, 8)
    _settled(application, viewer)
    assert viewer.frame.indices[3:] == (0, 8)
    viewer.view3d_box.setCurrentText("Planes")
    _settled(application, viewer)
    assert viewer.frame.volume_buffer is None and viewer.canvas._volume_visual is None


def test_colormap_failure_rolls_back_selection_and_reports_error(application, viewer, monkeypatch):
    frame = viewer.frame
    previous = viewer.canvas._images["HW"].cmap
    make_colormap = viewer.canvas.prepare_colormap

    def fail_hot(name):
        if name == "hot":
            raise ValueError("test sampling failure")
        return make_colormap(name)

    monkeypatch.setattr(viewer.canvas, "prepare_colormap", fail_hot)
    viewer._change_colormap("hot")
    assert viewer.colormap_box.currentData() == "vispy:grays"
    assert viewer.canvas.colormap == "grays"
    assert viewer.canvas._images["HW"].cmap is previous
    assert "test sampling failure" in viewer.status.text()
    assert "test sampling failure" in viewer.last_error
    assert viewer.frame is frame
    viewer._change_colormap("fire")
    assert viewer.last_error is None
    assert viewer.canvas.colormap == "fire"
    assert viewer.canvas.render().shape[-1] == 4


def test_scalar_rgb_and_3d_controls_show_only_relevant_settings(application, viewer):
    assert not viewer.volume_controls.isVisible()
    assert viewer.scalar_range.isEnabled() and viewer.colormap_box.isEnabled()
    assert viewer.opacity_box.value() == 0.15
    viewer.view3d_box.setCurrentText("Volume")
    _settled(application, viewer)
    assert viewer.volume_controls.isVisible() and viewer.opacity_box.isEnabled()
    assert viewer.volume_shape_label.text() == "Volume SHW: 5 × 7 × 9"
    viewer._select_input(2)
    _settled(application, viewer)
    assert viewer.view3d_box.currentText() == "Planes"
    assert not viewer.volume_controls.isVisible()
    assert not viewer.view3d_box.model().item(1).isEnabled()
    assert not viewer.colormap_box.isEnabled() and not viewer.scalar_range.isEnabled()
    assert not viewer.low_edit.isEnabled() and not viewer.high_edit.isEnabled()
    viewer.mode_box.setCurrentText("Scalar")
    _settled(application, viewer)
    assert viewer.scalar_range.isEnabled() and viewer.colormap_box.isEnabled()
    viewer.view3d_box.setCurrentText("Off")
    _settled(application, viewer)
    assert not viewer.volume_controls.isVisible()
    assert not any(image.visible for image in viewer.canvas._cutplanes.values())


def test_controls_resize_collapse_and_remain_accessible_in_small_window(application, viewer):
    viewer.view3d_box.setCurrentText("Volume")
    _settled(application, viewer)
    artifacts = Path(__file__).resolve().parents[1] / ".test-artifacts"
    artifacts.mkdir(exist_ok=True)
    viewer.resize(1200, 840)
    application.processEvents()
    assert viewer.grab().save(str(artifacts / "viewer-controls.png"))
    viewer.resizeDocks([viewer.controls_dock], [340], QtCore.Qt.Orientation.Horizontal)
    application.processEvents()
    wide = viewer.controls_dock.width()
    viewer.resizeDocks([viewer.controls_dock], [300], QtCore.Qt.Orientation.Horizontal)
    application.processEvents()
    assert viewer.controls_dock.width() < wide
    point, revision = viewer.positions, viewer._revision
    toggle = viewer.controls_dock.toggleViewAction()
    toggle.trigger()
    application.processEvents()
    assert not viewer.controls_dock.isVisible()
    toggle.trigger()
    application.processEvents()
    assert viewer.controls_dock.isVisible()
    assert viewer.positions == point and viewer._revision == revision

    viewer.resize(800, 600)
    application.processEvents()
    assert viewer.width() == 800 and viewer.height() == 600
    scroll = viewer.controls_scroll
    assert scroll.verticalScrollBar().maximum() > 0
    for control in (*viewer.axis_sliders.values(), *viewer.axis_boxes.values(),
                    viewer.colormap_box, viewer.low_edit, viewer.high_edit,
                    viewer.crosshair_box, viewer.view3d_box, viewer.opacity_box):
        # QAbstractSpinBox exposes an input-method cursor rect; Qt's
        # ensureWidgetVisible may reveal that rect without the outer frame.
        center = control.mapTo(scroll.widget(), control.rect().center())
        scroll.ensureVisible(center.x(), center.y(), control.width() // 2 + 8, control.height() // 2 + 8)
        application.processEvents()
        rectangle = QtCore.QRect(control.mapTo(scroll.viewport(), QtCore.QPoint()), control.size())
        assert scroll.viewport().rect().contains(rectangle), control
    assert viewer.grab().save(str(artifacts / "viewer-small.png"))


def test_native_click_moves_one_shared_point_in_all_planes(application, viewer):
    for name, row, col, axes in (("HW", 1, 7, (1, 2)), ("SW", 4, 2, (0, 2)), ("SH", 1, 5, (0, 1))):
        expected = list(viewer.frame.indices[2:])
        expected[axes[0]], expected[axes[1]] = row, col
        position = _pixel_position(viewer, name, row, col)
        QtTest.QTest.mouseClick(viewer.canvas.widget, QtCore.Qt.MouseButton.LeftButton, pos=position)
        _settled(application, viewer)
        assert viewer.frame.indices == (0, 0, *expected)
        assert [viewer.axis_boxes[n].value() for n in "SHW"] == expected
        tensor = viewer.sources[0].volume.tensor[0, 0]
        s, h, w = expected
        for plane, expected_raw in zip(viewer.frame.planes, (tensor[s], tensor[:, h], tensor[:, :, w])):
            assert torch.equal(plane.raw, expected_raw)
        assert all(line.visible for line in viewer.canvas._crosshairs.values())


def test_drag_moves_crosshair_and_shift_drag_only_pans(application, viewer):
    widget = viewer.canvas.widget
    start = _pixel_position(viewer, "HW", 1, 1)
    end = _pixel_position(viewer, "HW", 5, 7)
    camera = viewer.canvas._views["HW"].camera
    before = (camera.rect.left, camera.rect.bottom, camera.rect.width, camera.rect.height)
    QtTest.QTest.mousePress(widget, QtCore.Qt.MouseButton.LeftButton, pos=start)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end), QtCore.QPointF(widget.mapToGlobal(end)),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MouseButton.LeftButton, pos=end)
    _settled(application, viewer)
    assert viewer.frame.indices == (0, 0, 2, 5, 7)
    np.testing.assert_allclose((camera.rect.left, camera.rect.bottom, camera.rect.width, camera.rect.height), before)

    point = viewer.positions
    revision = viewer._revision
    QtTest.QTest.mousePress(widget, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier, start)
    pan_end = start + QtCore.QPoint(20, 15)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove, QtCore.QPointF(pan_end), QtCore.QPointF(widget.mapToGlobal(pan_end)),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier, pan_end)
    assert viewer.positions == point and viewer._revision == revision
    assert (camera.rect.left, camera.rect.bottom) != before[:2]
    np.testing.assert_allclose((camera.rect.width, camera.rect.height), before[2:])


def test_native_wheel_navigates_each_axis_and_control_wheel_only_zooms(application, viewer):
    for name, axis in (("HW", 0), ("SW", 1), ("SH", 2)):
        before = list(viewer.frame.indices[2:])
        position = _pixel_position(viewer, name, 1, 1)
        camera = viewer.canvas._views[name].camera
        old_width = camera.rect.width
        _wheel(viewer, position, 120)
        _settled(application, viewer)
        before[axis] += 1
        assert viewer.frame.indices[2:] == tuple(before)
        assert camera.rect.width == old_width
        point, revision = viewer.positions, viewer._revision
        _wheel(viewer, position, 120, QtCore.Qt.KeyboardModifier.ControlModifier)
        application.processEvents()
        assert viewer.positions == point and viewer._revision == revision
        assert camera.rect.width < old_width


def test_wheel_accumulates_pending_steps_and_stops_at_boundaries(application, viewer):
    position = _pixel_position(viewer, "HW", 1, 1)
    _wheel(viewer, position, -60)
    assert viewer.axis_boxes["S"].value() == 2
    _wheel(viewer, position, -60)
    _wheel(viewer, position, -120)
    assert viewer.axis_boxes["S"].value() == 0
    revision = viewer._revision
    _wheel(viewer, position, -120)
    assert viewer._revision == revision
    _settled(application, viewer)
    assert viewer.frame.indices[2] == 0
    _wheel(viewer, position, 120 * 20)
    _settled(application, viewer)
    assert viewer.frame.indices[2] == 4
    revision = viewer._revision
    _wheel(viewer, position, 120)
    assert viewer._revision == revision
    viewer._select_input(1)
    _settled(application, viewer)
    point, revision = viewer.positions, viewer._revision
    _wheel(viewer, _pixel_position(viewer, "HW", 1, 1), 120)
    assert viewer.positions == point and viewer._revision == revision


def test_crosshair_can_be_hidden_without_changing_display_data(application, viewer):
    frame, revision = viewer.frame, viewer._revision
    viewer.crosshair_box.setChecked(False)
    assert all(not line.visible for line in viewer.canvas._crosshairs.values())
    viewer.canvas.render()
    viewer.crosshair_box.setChecked(True)
    assert all(line.visible for line in viewer.canvas._crosshairs.values())
    assert viewer.frame is frame and viewer._revision == revision


def test_navigation_ignores_old_input_while_a_switch_is_pending(application, viewer):
    viewer._select_input(1)
    point, revision = viewer.positions, viewer._revision
    viewer._pick("HW", 0, 0)
    viewer._scroll("SW", 1)
    assert viewer.positions == point and viewer._revision == revision
    _settled(application, viewer)
    viewer._pick("HW", 0, 0)
    _settled(application, viewer)
    assert viewer.frame.indices[3:] == (0, 0)


def test_latest_revision_applies_pixels_labels_and_values_together(application, viewer, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame
    applied = []
    set_frame = viewer.canvas.set_frame

    def slow_prepare(request, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(request, **kwargs)

    def record(frame, **kwargs):
        applied.append(frame.revision)
        set_frame(frame, **kwargs)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    monkeypatch.setattr(viewer.canvas, "set_frame", record)
    viewer.axis_sliders["S"].setValue(0)
    _wait(application, entered.is_set)
    old_revision = viewer._revision
    for index in (1, 2, 3, 4):
        viewer.axis_sliders["S"].setValue(index)
    latest_revision = viewer._revision
    assert viewer._pending.revision == latest_revision
    release.set()
    _settled(application, viewer)
    assert applied == [latest_revision]
    assert old_revision not in applied
    assert viewer.frame.indices[2] == 4
    assert viewer.canvas._labels["HW"].text == "HW · S=4"
    assert viewer.frame.planes[0].raw[0, 0].item() == 252
    assert np.array_equal(viewer.canvas._images["HW"]._data, viewer.frame.planes[0].image)


def test_preparation_error_keeps_last_complete_display(application):
    a = Volume(torch.zeros(1, 1, 3, 3, 3))
    b = Volume(torch.full((1, 3, 3, 3, 3), 2.0))
    viewer = _open(a, Display(b, rgb=True))
    _settled(application, viewer)
    frame = viewer.frame
    viewer._select_input(1)
    _wait(application, lambda: not viewer._busy and viewer._pending is None)
    assert "[0, 1]" in viewer.last_error
    assert viewer.frame is frame and viewer.canvas._frame is frame
    viewer.mode_box.setCurrentText("Scalar")
    _settled(application, viewer)
    assert not viewer.frame.source.rgb


def test_snapshot_references_released_on_close_and_reopen(application):
    volume = Volume(torch.ones(1, 1, 3, 4, 5))
    viewer = _open(volume)
    _settled(application, viewer)
    assert viewer.sources[0].volume is volume
    retained = weakref.ref(volume.tensor)
    del volume
    viewer.close()
    application.processEvents()
    gc.collect()
    assert not viewer._thread.isRunning()
    assert retained() is None
    assert viewer.frame is None and not viewer.sources
    assert viewer._worker._volume_buffer is None
    another = _open(Volume(torch.zeros(1, 1, 3, 4, 5)))
    _settled(application, another)
    assert QtWidgets.QApplication.instance() is application
    another.close()


def test_close_during_preparation_releases_queued_result(application, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame

    def slow_prepare(request, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(request, **kwargs)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    viewer = _open(Volume(torch.ones(1, 1, 3, 4, 5)))
    _wait(application, entered.is_set)
    retained = weakref.ref(viewer.sources[0].volume.tensor)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    viewer.close()
    timer.join()
    gc.collect()
    assert not viewer._thread.isRunning()
    # No later event-loop iteration should be necessary to drop the data.
    assert retained() is None


def test_failed_input_colormap_switch_preserves_active_display(application, viewer, monkeypatch):
    frame, point, revision = viewer.frame, viewer.positions, viewer._revision
    original = viewer.canvas.prepare_colormap

    def failing(name):
        if name == "fire":
            raise RuntimeError("provider unavailable")
        return original(name)

    monkeypatch.setattr(viewer.canvas, "prepare_colormap", failing)
    viewer.input_box.setCurrentIndex(1)
    assert viewer.active == 0 and viewer.input_box.currentIndex() == 0
    assert viewer.positions == point and viewer._revision == revision
    assert viewer.frame is frame and viewer.canvas._frame is frame
    assert viewer.colormaps[0] == "grays"
    assert "provider unavailable" in viewer.last_error


def test_picker_loads_optional_sources_lazily_and_preserves_list_on_failure(application, viewer, monkeypatch):
    previous = [viewer.colormap_box.itemData(i) for i in range(viewer.colormap_box.count())]
    assert not viewer._colormaps_loaded

    def failing(*, extended=False):
        assert extended
        raise ImportError("broken optional package")

    monkeypatch.setattr(window, "available_colormaps", failing)
    viewer._load_colormaps()
    assert not viewer._colormaps_loaded
    assert [viewer.colormap_box.itemData(i) for i in range(viewer.colormap_box.count())] == previous
    assert "broken optional package" in viewer.last_error
    monkeypatch.setattr(window, "available_colormaps", lambda **_: [("vispy:grays", "grays · VisPy"), ("mpl:hot", "hot · Matplotlib")])
    viewer._load_colormaps()
    assert viewer._colormaps_loaded and viewer.colormap_box.isEditable()
    assert viewer.colormap_box.findData("mpl:hot") >= 0
    assert viewer.colormap_box.currentData() == "vispy:grays"


@pytest.mark.parametrize("transition", ["manual", "switch", "share", "auto_again"])
def test_stale_auto_range_cannot_replace_new_range_or_another_inputs_range(application, viewer, monkeypatch, transition):
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame

    def slow_prepare(request, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(request, **kwargs)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    viewer._reset_range()
    _wait(application, entered.is_set)
    try:
        if transition == "manual":
            viewer.low_edit.setText("7")
            viewer.high_edit.setText("9")
            viewer._apply_range()
        elif transition == "switch":
            viewer._select_input(1)
        elif transition == "share":
            viewer.share_box.setChecked(True)
            viewer._select_input(1)
        else:
            viewer._reset_range()
        release.set()
        _settled(application, viewer)
    finally:
        release.set()
    if transition == "manual":
        assert viewer.current_window == (7, 9) and viewer.frame.window == (7, 9)
    elif transition in ("switch", "share"):
        assert viewer.frame.input_index == 1
        if transition == "share":
            assert viewer.current_window == (0, 629) and viewer.frame.window == (0, 629)
            assert viewer.windows[1] is None  # Shared Auto never overwrites an individual window.
        else:
            assert viewer.current_window == (100, 100) and viewer.frame.window == (100, 100)
    else:
        assert viewer.current_window == (0, 629)
    assert viewer.canvas._frame is viewer.frame


def test_large_integer_window_edit_stays_exact(application):
    low = 2**63 + 123
    volume = Volume(torch.tensor([low, low + 4], dtype=torch.uint64).reshape(1, 1, 1, 1, 2))
    viewer = _open(volume)
    _settled(application, viewer)
    assert viewer.current_window == (low, low + 4)
    assert viewer.low_edit.text() == str(low)
    viewer.low_edit.setText(str(low + 1))
    viewer.high_edit.setText(str(low + 3))
    viewer._apply_range()
    _settled(application, viewer)
    assert viewer.current_window == (low + 1, low + 3)
    assert all(type(value) is int for value in viewer.current_window)


@pytest.mark.parametrize("auto_pending", [False, True])
def test_shared_auto_retains_initiating_input_when_navigation_replaces_request(application, viewer, monkeypatch, auto_pending):
    viewer.share_box.setChecked(True)
    _settled(application, viewer)
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame

    def slow_prepare(request, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(request, **kwargs)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    if auto_pending:
        viewer.axis_sliders["S"].setValue(0)
        _wait(application, entered.is_set)
        viewer._reset_range()  # Auto A is pending, then replaced by navigation to B.
    else:
        viewer._reset_range()
        _wait(application, entered.is_set)  # Auto A is already running.
    try:
        viewer._select_input(1)
        assert viewer._pending.input_index == 1
        release.set()
        _settled(application, viewer)
    finally:
        release.set()
    assert viewer.frame.input_index == 1 and viewer.frame.window == (0, 629)
    assert viewer.shared_window == (0, 629)
    assert viewer.windows[1] is None


def test_input_palette_changes_only_with_accepted_pixels(application, viewer, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame
    old_frame = viewer.frame

    def slow_prepare(request, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(request, **kwargs)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    viewer._select_input(1)
    _wait(application, entered.is_set)
    try:
        assert viewer.active == 1 and viewer.frame is old_frame
        assert viewer.canvas.colormap == "grays"
        viewer._change_colormap("hot")  # Editing a pending input must also wait.
        assert viewer.colormaps[1] == "hot" and viewer.canvas.colormap == "grays"
        assert viewer.canvas._frame is old_frame
        release.set()
        _settled(application, viewer)
    finally:
        release.set()
    assert viewer.canvas.colormap == "hot" and viewer.frame.input_index == 1


def test_native_shortcuts_cycle_all_inputs_and_select_first_two(application, viewer):
    viewer.activateWindow()
    viewer.canvas.widget.setFocus()
    application.processEvents()
    for expected in (1, 2, 0):
        QtTest.QTest.keyClick(viewer.canvas.widget, QtCore.Qt.Key.Key_Tab)
        _settled(application, viewer)
        assert viewer.active == expected
    QtTest.QTest.keyClick(viewer.canvas.widget, QtCore.Qt.Key.Key_B)
    _settled(application, viewer)
    assert viewer.active == 1
    QtTest.QTest.keyClick(viewer.canvas.widget, QtCore.Qt.Key.Key_A)
    _settled(application, viewer)
    assert viewer.active == 0


def test_frame_upload_failure_preserves_previous_palette_and_error(application, viewer, monkeypatch):
    frame, palette = viewer.frame, viewer.canvas.palette

    def fail_frame(*args, **kwargs):
        raise RuntimeError("frame upload failed")

    monkeypatch.setattr(viewer.canvas, "set_frame", fail_frame)
    viewer._select_input(1)
    _wait(application, lambda: not viewer._busy and viewer._pending is None)
    assert viewer.frame is frame and viewer.canvas._frame is frame
    assert viewer.last_error == "RuntimeError: frame upload failed"
    assert viewer.canvas.palette is palette


def test_volume_buffer_reused_for_spatial_moves_and_invalidated_by_sampling_state(application, monkeypatch):
    volume = Volume(torch.arange(2 * 2 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 2, 4, 5, 6))
    viewer = _open(volume, Volume(volume.tensor + 1), window=(0, 500))
    viewer.view3d_box.setCurrentText("Volume")
    _settled(application, viewer)
    buffer = viewer.frame.volume_buffer
    visual = viewer.canvas._volume_visual
    assert visual._last_data is buffer
    uploads = []
    set_data = visual.set_data

    def record(data, **kwargs):
        uploads.append(data)
        return set_data(data, **kwargs)

    monkeypatch.setattr(visual, "set_data", record)
    viewer.opacity_box.setValue(0.3)
    palette = viewer.canvas.palette
    for axis in ("S", "H", "W"):
        viewer.axis_boxes[axis].setValue(0)
        _settled(application, viewer)
        assert viewer.frame.volume_buffer is buffer
        assert viewer._worker._volume_buffer is buffer
        assert viewer.canvas.palette is palette
    assert not uploads

    for change in (
        lambda: viewer.axis_boxes["T"].setValue(1),
        lambda: viewer.channel_box.setValue(1),
        lambda: (viewer.low_edit.setText("10"), viewer._apply_range()),
        lambda: viewer._select_input(1),
    ):
        change()
        _settled(application, viewer)
        assert viewer.frame.volume_buffer is not buffer
        buffer = viewer.frame.volume_buffer
        assert viewer._worker._volume_buffer is buffer
    assert len(uploads) == 4
    viewer.view3d_box.setCurrentText("Off")
    _settled(application, viewer)
    assert viewer._worker._volume_key is None
    assert viewer._worker._volume_buffer is None
    assert viewer.frame.volume_buffer is None
    assert viewer.canvas._volume_visual is None
