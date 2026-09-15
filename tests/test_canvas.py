"""Coordinate and real-OpenGL checks for native slice display."""

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vispy")
pytest.importorskip("PySide6")

from vol5dkit import Volume
from vol5dkit.viewer._data import Request, make_source, prepare_frame
from vol5dkit.viewer.canvas import PLANE_AXES, VolumeCanvas, plane_transform, volume_transform


def test_image_centers_match_world_coordinates():
    angle = 0.37
    direction = ((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, -1))
    volume = Volume(torch.empty(1, 1, 5, 7, 11), spacing=(2.5, 0.7, 1.3), origin=(11, -5, 9), direction=direction)
    indices = (0, 0, 2, 3, 4)
    for name, (row, col, fixed) in PLANE_AXES.items():
        transform = plane_transform(volume, name, indices)
        for r, c in ((0, 0), (1, 2), (volume.shape[row + 2] - 1, volume.shape[col + 2] - 1)):
            shw = np.array(indices[2:], dtype=float)
            shw[row], shw[col] = r, c
            result = transform.map((c + 0.5, r + 0.5, 0))[:3]
            np.testing.assert_allclose(result, volume.index_to_world(shw), atol=1e-6)


def test_volume_preview_preserves_first_center_and_strided_spacing():
    volume = Volume(torch.empty(1, 1, 6, 7, 8), spacing=(3, 2, 1), origin=(7, 11, 13))
    transform = volume_transform(volume, stride=3)
    np.testing.assert_allclose(transform.map((0, 0, 0))[:3], volume.origin)
    np.testing.assert_allclose(transform.map((2, 1, 1))[:3], volume.index_to_world((3, 3, 6)))


@pytest.fixture
def canvas():
    if os.environ.get("VOL5DKIT_TEST_GUI") != "1":
        pytest.skip("set VOL5DKIT_TEST_GUI=1 to run actual OpenGL rendering")
    from PySide6 import QtCore
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    from vol5dkit.viewer.window import _app
    application = _app()
    # Match production: an embedded QOpenGLWidget, with an application whose
    # lifetime outlasts every canvas. Delete closed Qt contexts before another
    # test renders, instead of leaving their destruction to Python's GC.
    parent = QWidget()
    layout = QVBoxLayout(parent)
    canvas = VolumeCanvas()
    canvas.set_crosshair_visible(False)
    layout.addWidget(canvas.widget)
    parent.resize(800, 640)
    parent.show()
    application.processEvents()
    yield canvas
    canvas.close()
    parent.close()
    parent.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()


def _frame(tensor, *, rgb=False, volume_3d=False, preview_stride=1):
    return prepare_frame(Request(1, "A", make_source(Volume(tensor), rgb), (0, 0, 0, 0), clim=(0, 1), volume_3d=volume_3d, preview_stride=preview_stride))


def _sample(canvas, screenshot, name, column, row):
    mapped = canvas._images[name].get_transform(map_from="visual", map_to="canvas").map((column, row))
    x, y = mapped[:2] / mapped[3]
    scale_x = screenshot.shape[1] / canvas._canvas.size[0]
    scale_y = screenshot.shape[0] / canvas._canvas.size[1]
    return screenshot[int(y * scale_y), int(x * scale_x), :3]


@pytest.mark.gui
def test_nearest_and_explicit_linear_rendering(canvas):
    tensor = torch.tensor([[[[[0., 1.], [1., 0.]], [[0., 1.], [1., 0.]]]]])
    canvas.set_frame(_frame(tensor), mode_3d="hidden")
    assert canvas._images["HW"]._texture.internalformat == "r32f"
    nearest = canvas.render()
    np.testing.assert_array_equal(_sample(canvas, nearest, "HW", 0.75, 0.5), (0, 0, 0))
    np.testing.assert_array_equal(_sample(canvas, nearest, "HW", 1.25, 0.5), (255, 255, 255))
    canvas.set_interpolation("linear")
    linear = canvas.render()
    pixel = _sample(canvas, linear, "HW", 0.75, 0.5)
    assert np.all((pixel > 40) & (pixel < 90)), pixel


@pytest.mark.gui
def test_nan_magenta_and_native_cursor_coordinates(canvas):
    tensor = torch.zeros(1, 1, 2, 2, 2)
    tensor[0, 0, 0, 0, 0] = float("nan")
    canvas.set_frame(_frame(tensor), mode_3d="hidden")
    screenshot = canvas.render()
    np.testing.assert_array_equal(_sample(canvas, screenshot, "HW", 0.5, 0.5), (255, 0, 255))
    pixel = canvas._images["HW"].get_transform(map_from="visual", map_to="canvas").map((1.5, 0.5))
    position = canvas._cursor_position(SimpleNamespace(pos=pixel[:2] / pixel[3]))
    assert position[0] == "HW"
    np.testing.assert_allclose(position[1:], (0, 1), atol=1e-6)


@pytest.mark.gui
def test_rgb_nonfinite_pixels_are_magenta_without_changing_raw_data(canvas):
    tensor = torch.zeros(1, 3, 2, 2, 3)
    tensor[0, 0, 0, 0, 0] = float("nan")
    tensor[0, 1, 0, 0, 1] = float("inf")
    tensor[0, :, 0, 0, 2] = torch.tensor((0.0, 1.0, 0.0))
    frame = _frame(tensor, rgb=True)
    canvas.set_frame(frame, mode_3d="slices")
    screenshot = canvas.render()
    for column in (0.5, 1.5):
        np.testing.assert_array_equal(_sample(canvas, screenshot, "HW", column, 0.5), (255, 0, 255))
    np.testing.assert_array_equal(_sample(canvas, screenshot, "HW", 2.5, 0.5), (0, 255, 0))
    assert torch.isnan(frame.planes[0].raw[0, 0, 0])
    assert torch.isposinf(frame.planes[0].raw[0, 1, 1])
    assert np.isnan(frame.planes[0].image[0, :2]).all()
    assert frame.planes[0].raw.dtype == tensor.dtype
    for name in PLANE_AXES:
        assert canvas._images[name]._data is canvas._cutplanes[name]._data
    canvas.set_frame(_frame(torch.full((1, 3, 2, 2, 2), float("nan")), rgb=True))
    np.testing.assert_array_equal(_sample(canvas, canvas.render(), "HW", 0.5, 0.5), (255, 0, 255))


@pytest.mark.gui
def test_rgb_volume_modes_and_repeated_relative_zoom(canvas):
    from vispy.geometry import Rect
    first = _frame(torch.rand(1, 1, 4, 6, 8))
    second = _frame(torch.rand(1, 1, 8, 12, 16))
    canvas.set_frame(first)
    camera = canvas._views["HW"].camera
    camera.rect = Rect(1.5, 1, 4, 3)
    before = Rect(camera.rect)
    for _ in range(3):
        canvas.set_frame(second)
        canvas.set_frame(first)
    np.testing.assert_allclose(camera.rect.pos, before.pos)
    np.testing.assert_allclose(camera.rect.size, before.size)
    canvas.set_frame(_frame(torch.rand(1, 1, 8, 8, 8), volume_3d=True, preview_stride=2), mode_3d="volume")
    assert canvas.render().shape[-1] == 4
    assert canvas._volume_visual.visible
    canvas.set_frame(_frame(torch.rand(1, 3, 4, 4, 4), rgb=True))
    assert canvas._volume_visual is None
    assert canvas._images["HW"]._texture.internalformat == "rgb32f"
    assert canvas.render().shape[-1] == 4


def _canvas_pos(canvas, name, column, row):
    mapped = canvas._images[name].get_transform(map_from="visual", map_to="canvas").map((column, row))
    return mapped[:2] / mapped[3]


def _mouse(canvas, event_type, pos, **kwargs):
    from vispy.app.canvas import MouseEvent
    event = MouseEvent(event_type, pos=pos, **kwargs)
    getattr(canvas._canvas.events, event_type)(event)
    return event


@pytest.mark.gui
def test_left_selection_drag_and_shift_pan(canvas):
    from vispy.geometry import Rect
    from vispy.util import keys
    canvas.set_frame(_frame(torch.zeros(1, 1, 4, 6, 8)))
    canvas.render()
    picked = []
    canvas._on_pick = lambda *args: picked.append(args)
    start, end = _canvas_pos(canvas, "HW", 1.5, 1.5), _canvas_pos(canvas, "HW", 5.5, 4.5)
    camera = canvas._views["HW"].camera
    before = Rect(camera.rect)
    press = _mouse(canvas, "mouse_press", start, button=1, buttons=[1])
    move = _mouse(canvas, "mouse_move", end, button=1, buttons=[1], press_event=press, last_event=press)
    _mouse(canvas, "mouse_release", end, button=1, press_event=press, last_event=move)
    assert all(p[0] == "HW" for p in picked)
    np.testing.assert_allclose(picked[0][1:], (1, 1), atol=1e-6)
    np.testing.assert_allclose(picked[-1][1:], (4, 5), atol=1e-6)
    assert camera.rect == before
    assert canvas._canvas._mouse_handler is None
    picked.clear()
    press = _mouse(canvas, "mouse_press", start, button=1, buttons=[1], modifiers=(keys.SHIFT,))
    move = _mouse(canvas, "mouse_move", start + (20, 15), button=1, buttons=[1], modifiers=(keys.SHIFT,), press_event=press, last_event=press)
    _mouse(canvas, "mouse_release", start + (20, 15), button=1, modifiers=(keys.SHIFT,), press_event=press, last_event=move)
    assert not picked
    assert camera.rect != before
    np.testing.assert_allclose(camera.rect.size, before.size)
    before = Rect(camera.rect)
    press = _mouse(canvas, "mouse_press", start, button=2, buttons=[2])
    move = _mouse(canvas, "mouse_move", start + (20, 15), button=2, buttons=[2], press_event=press, last_event=press)
    _mouse(canvas, "mouse_release", start + (20, 15), button=2, press_event=press, last_event=move)
    assert not picked
    assert camera.rect.size != before.size


@pytest.mark.gui
def test_selection_stays_in_initial_plane_and_source(canvas):
    canvas.set_frame(_frame(torch.zeros(1, 1, 4, 6, 8)))
    canvas.render()
    picked = []
    canvas._on_pick = lambda *args: picked.append(args)
    start = _canvas_pos(canvas, "HW", 1.5, 1.5)
    other = _canvas_pos(canvas, "SW", 2.5, 1.5)
    press = _mouse(canvas, "mouse_press", start, button=1, buttons=[1])
    _mouse(canvas, "mouse_move", other, button=1, buttons=[1], press_event=press, last_event=press)
    assert len(picked) == 1
    canvas.set_frame(_frame(torch.zeros(1, 1, 4, 6, 8)))
    _mouse(canvas, "mouse_release", start, button=1, press_event=press, last_event=press)
    assert len(picked) == 1
    # Starting on letterbox padding captures selection, so dragging into the
    # image cannot accidentally trigger the underlying camera's left pan.
    canvas.reset_view()
    canvas.render()
    view = canvas._views["HW"]
    background = view.node_transform(canvas._canvas.scene).map((2, 2))
    background = background[:2] / background[3]
    before = tuple(view.camera.rect.pos)
    press = _mouse(canvas, "mouse_press", background, button=1, buttons=[1])
    start = _canvas_pos(canvas, "HW", 1.5, 1.5)
    move = _mouse(canvas, "mouse_move", start, button=1, buttons=[1], press_event=press, last_event=press)
    _mouse(canvas, "mouse_release", start, button=1, press_event=press, last_event=move)
    assert picked[-1][0] == "HW"
    np.testing.assert_allclose(picked[-1][1:], (1, 1), atol=1e-6)
    assert tuple(view.camera.rect.pos) == before


@pytest.mark.gui
def test_wheel_navigates_without_zoom_and_ctrl_wheel_zooms(canvas):
    from vispy.geometry import Rect
    from vispy.util import keys
    frame = _frame(torch.zeros(1, 1, 4, 6, 8))
    canvas.set_frame(frame)
    canvas.render()
    scrolled = []
    canvas._on_scroll = lambda *args: scrolled.append(args)
    pos = _canvas_pos(canvas, "HW", 2.5, 2.5)
    camera = canvas._views["HW"].camera
    before = Rect(camera.rect)
    for _ in range(5):
        event = _mouse(canvas, "mouse_wheel", pos, delta=(0, 0.2))
        assert event.blocked
    assert scrolled == [("HW", 1)]
    assert camera.rect == before
    _mouse(canvas, "mouse_wheel", pos, delta=(0, -2))
    assert scrolled[-1] == ("HW", -2)
    _mouse(canvas, "mouse_wheel", pos, delta=(0, 1), modifiers=(keys.CONTROL,))
    assert len(scrolled) == 2
    assert camera.rect.width < before.width
    # A partial wheel tick belongs to its source/slot, not the next A/B input.
    _mouse(canvas, "mouse_wheel", pos, delta=(0, 0.5))
    canvas.set_frame(replace(frame, slot="B"))
    _mouse(canvas, "mouse_wheel", pos, delta=(0, 0.5))
    assert len(scrolled) == 2
    # Navigation also works over the panel's background outside image bounds.
    view = canvas._views["HW"]
    background = view.node_transform(canvas._canvas.scene).map((2, 2))
    _mouse(canvas, "mouse_wheel", background[:2] / background[3], delta=(0, 0.5))
    assert scrolled[-1] == ("HW", 1)


@pytest.mark.gui
def test_crosshair_geometry_has_native_voxel_gap_and_can_hide(canvas):
    volume = Volume(torch.zeros(1, 1, 5, 7, 9), spacing=(3, 2, 0.5))
    frame = prepare_frame(Request(1, "A", make_source(volume), (0, 0.5, 0.5, 0.5), clim=(0, 1)))
    canvas.set_crosshair_visible(True)
    canvas.set_frame(frame)
    for name, (row, col, _) in PLANE_AXES.items():
        line = canvas._crosshairs[name]
        r, c = frame.indices[2 + row], frame.indices[2 + col]
        np.testing.assert_allclose(line.pos[[1, 2]], ((c, r + 0.5), (c + 1, r + 0.5)))
        np.testing.assert_allclose(line.pos[[5, 6]], ((c + 0.5, r), (c + 0.5, r + 1)))
        center = line.transform.map((c + 0.5, r + 0.5))
        np.testing.assert_allclose(center[:2], (c * volume.spacing[col], r * volume.spacing[row]))
        assert line.visible and not line.antialias
    screenshot = canvas.render()
    np.testing.assert_array_equal(_sample(canvas, screenshot, "HW", 4.5, 3.5), (0, 0, 0))
    canvas.set_crosshair_visible(False)
    canvas.set_frame(frame)
    assert not any(line.visible for line in canvas._crosshairs.values())
