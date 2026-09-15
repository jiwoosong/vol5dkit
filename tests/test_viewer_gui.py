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

from vol5dkit import Volume, view
from vol5dkit.viewer import window
from vol5dkit.viewer._data import relative_index

pytestmark = pytest.mark.gui


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


@pytest.fixture
def application():
    application = window._app()
    yield application
    for viewer in tuple(window._viewers):
        viewer.close()
    application.processEvents()


@pytest.fixture
def viewer(application):
    a = Volume(torch.arange(2 * 5 * 7 * 9, dtype=torch.float32).reshape(2, 1, 5, 7, 9), spacing=(2, 1, 0.5), times=(0, 0.25))
    viewer = view(a, block=False)
    _settled(application, viewer)
    yield viewer
    viewer.close()
    application.processEvents()


def test_startup_native_frame_range_hover_and_full_window_capture(application, viewer):
    assert viewer.frame.indices == (0, 0, 2, 3, 4)
    assert viewer.clim == (0, 314)
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
    viewer.update("B", Volume(torch.zeros(1, 1, 1, 4, 6)))
    for _ in range(4):
        viewer.slot_box.setCurrentText("B")
        _settled(application, viewer)
        assert viewer.positions == positions
        assert viewer.frame.indices == (0, 0, 0, relative_index(positions[2], 4), relative_index(positions[3], 6))
        assert not viewer.axis_boxes["S"].isEnabled()
        viewer.slot_box.setCurrentText("A")
        _settled(application, viewer)
        assert viewer.positions == positions
        assert viewer.frame.indices == (0, 0, 1, 5, 4)


def test_registration_rgb_and_scalar_range_and_interpolation_persist(application, viewer):
    limits = viewer.clim
    viewer.interpolation_box.setCurrentText("linear")
    viewer.update("A", Volume(torch.ones(2, 1, 5, 7, 9) * 999))
    _settled(application, viewer)
    assert viewer.frame.clim == limits
    assert viewer.canvas._images["HW"].interpolation == "linear"
    viewer.update("B", Volume(torch.rand(1, 3, 4, 6, 8)), rgb=True)
    assert viewer.slot_box.model().item(1).isEnabled()
    viewer.slot_box.setCurrentText("B")
    _settled(application, viewer)
    assert viewer.frame.source.rgb
    assert viewer.mode_box.currentText() == "RGB"
    assert not viewer.channel_box.isEnabled()
    assert viewer.canvas._images["HW"]._texture.internalformat == "rgb32f"
    viewer.update("B", Volume(torch.ones(1, 3, 4, 6, 8) * 0.5))
    _settled(application, viewer)
    assert viewer.frame.source.rgb
    viewer.mode_box.setCurrentText("Scalar")
    _settled(application, viewer)
    assert not viewer.frame.source.rgb
    assert viewer.frame.clim == limits
    assert viewer.canvas._images["HW"].interpolation == "linear"


def test_preview_volume_and_pick(application, viewer):
    viewer.stride_box.setValue(2)
    viewer.view3d_box.setCurrentText("volume")
    _settled(application, viewer)
    assert viewer.frame.volume.shape == (3, 4, 5)
    assert viewer.canvas.render().shape[-1] == 4
    assert viewer.canvas._volume_visual.visible
    viewer.opacity_box.setValue(0.3)
    assert viewer.canvas._opacity == 0.3
    viewer._pick("HW", 0, 8)
    _settled(application, viewer)
    assert viewer.frame.indices[3:] == (0, 8)
    viewer.view3d_box.setCurrentText("slices")
    _settled(application, viewer)
    assert viewer.frame.volume is None
    assert viewer.canvas._volume_visual is None


def test_native_click_moves_one_shared_point_in_all_planes(application, viewer):
    for name, row, col, axes in (("HW", 1, 7, (1, 2)), ("SW", 4, 2, (0, 2)), ("SH", 1, 5, (0, 1))):
        expected = list(viewer.frame.indices[2:])
        expected[axes[0]], expected[axes[1]] = row, col
        position = _pixel_position(viewer, name, row, col)
        QtTest.QTest.mouseClick(viewer.canvas.widget, QtCore.Qt.MouseButton.LeftButton, pos=position)
        _settled(application, viewer)
        assert viewer.frame.indices == (0, 0, *expected)
        assert [viewer.axis_boxes[n].value() for n in "SHW"] == expected
        tensor = viewer.sources["A"].volume.tensor[0, 0]
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
    viewer.update("A", Volume(torch.zeros(1, 1, 1, 7, 9)))
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


def test_navigation_ignores_old_source_while_a_replacement_is_pending(application, viewer):
    viewer.update("A", Volume(torch.zeros(1, 1, 3, 4, 5)))
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

    def slow_prepare(request):
        entered.set()
        assert release.wait(5)
        return prepare(request)

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


def test_preparation_error_keeps_last_complete_display(application, viewer):
    frame = viewer.frame
    viewer.update("A", Volume(torch.full((1, 3, 3, 3, 3), 2.0)), rgb=True)
    _wait(application, lambda: not viewer._busy and viewer._pending is None)
    assert "[0, 1]" in viewer.last_error
    assert viewer.frame is frame
    assert viewer.canvas._frame is frame
    viewer.update("A", Volume(torch.zeros(1, 3, 3, 3, 3)))
    _settled(application, viewer)
    assert viewer.frame.source.rgb


def test_detached_registration_close_and_reopen(application):
    leaf = torch.ones(1, 1, 3, 4, 5, requires_grad=True)
    computed = leaf * 2
    original = weakref.ref(computed)
    volume = Volume(computed)
    viewer = view(volume, block=False)
    del volume, computed
    gc.collect()
    assert original() is None
    _settled(application, viewer)
    assert viewer.sources["A"].volume.tensor.grad_fn is None
    retained = weakref.ref(viewer.sources["A"].volume.tensor)
    viewer.close()
    application.processEvents()
    gc.collect()
    assert not viewer._thread.isRunning()
    assert retained() is None
    assert viewer.frame is None and not viewer.sources
    assert viewer not in window._viewers
    with pytest.raises(RuntimeError, match="closed"):
        viewer.update("A", Volume(leaf))
    another = view(Volume(leaf.detach()), block=False)
    _settled(application, another)
    another.close()


def test_owned_event_loop_exits_and_can_restart(application):
    for _ in range(2):
        viewer = view(Volume(torch.zeros(1, 1, 3, 4, 5)), block=False)
        QtCore.QTimer.singleShot(200, viewer.close)
        assert window.run() == 0
        assert viewer._closed
        assert not viewer._thread.isRunning()
    assert window.run() == 0


@pytest.mark.parametrize("open_viewer", [view, window.view])
def test_blocking_view_releases_original_graph_before_event_loop_returns(application, open_viewer):
    originals, released = [], []

    def temporary_result():
        result = torch.ones(1, 1, 3, 4, 5, requires_grad=True) * 2
        originals.append(weakref.ref(result))
        return Volume(result)

    def inspect_and_close():
        gc.collect()
        released.extend(reference() is None for reference in originals)
        for viewer in tuple(window._viewers):
            viewer.close()

    QtCore.QTimer.singleShot(100, inspect_and_close)
    viewer = open_viewer(temporary_result(), temporary_result(), block=True)
    assert released == [True, True]
    assert viewer._closed


def test_close_during_preparation_releases_queued_result(application, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    prepare = window.prepare_frame

    def slow_prepare(request):
        entered.set()
        assert release.wait(5)
        return prepare(request)

    monkeypatch.setattr(window, "prepare_frame", slow_prepare)
    viewer = view(Volume(torch.ones(1, 1, 3, 4, 5)), block=False)
    _wait(application, entered.is_set)
    retained = weakref.ref(viewer.sources["A"].volume.tensor)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    viewer.close()
    timer.join()
    gc.collect()
    assert not viewer._thread.isRunning()
    # No later event-loop iteration should be necessary to drop the data.
    assert retained() is None
