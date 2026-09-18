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
from vol5dkit.viewer.canvas import (
    PLANE_AXES, VolumeCanvas, _sample_colormap, available_colormaps,
    plane_transform, volume_transform,
)


def test_all_builtin_colormaps_have_provider_ids_and_finite_rgba():
    choices = available_colormaps()
    assert ("vispy:grays", "grays · VisPy") in choices
    samples = np.linspace(0, 1, 256)
    for name, _ in choices:
        colors = _sample_colormap(name, samples)
        assert colors.shape == (256, 4)
        assert np.isfinite(colors).all()
        np.testing.assert_array_equal(colors, _sample_colormap(name.removeprefix("vispy:"), samples))


def test_missing_optional_palette_libraries_keep_builtin_choices(monkeypatch):
    from vol5dkit.viewer import canvas as module
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    assert available_colormaps(extended=True) == available_colormaps()


def test_broken_installed_palette_library_reports_its_error(monkeypatch):
    from vol5dkit.viewer import canvas as module
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())

    def broken(name):
        raise ImportError("installed library has a missing dependency")

    monkeypatch.setattr(module.importlib, "import_module", broken)
    with pytest.raises(ImportError, match="missing dependency"):
        available_colormaps(extended=True)
    assert available_colormaps()  # Basic choices do not import extras.


def test_matplotlib_colors_use_requested_provider_and_bare_names():
    mpl = pytest.importorskip("matplotlib")
    samples = np.linspace(0, 1, 256)
    colors = _sample_colormap("mpl:hot", samples)
    np.testing.assert_array_equal(colors, mpl.colormaps["hot"](samples))
    assert not np.allclose(colors, _sample_colormap("vispy:hot", samples))
    np.testing.assert_array_equal(_sample_colormap("plasma", samples), mpl.colormaps["plasma"](samples))
    choices = dict(available_colormaps(extended=True))
    assert choices["mpl:hot"] == "hot · Matplotlib"


def test_seaborn_named_continuous_maps_and_reverses_match_provider():
    sns = pytest.importorskip("seaborn")
    samples = np.linspace(0, 1, 256)
    choices = dict(available_colormaps(extended=True))
    for name in ("rocket", "mako", "icefire", "vlag", "flare", "crest"):
        for suffix in ("", "_r"):
            key = name + suffix
            assert f"sns:{key}" in choices
            np.testing.assert_array_equal(
                _sample_colormap(f"sns:{key}", samples), sns.color_palette(key, as_cmap=True)(samples),
            )
    with pytest.raises(ValueError, match="continuous Seaborn"):
        _sample_colormap("sns:deep", samples)


@pytest.mark.parametrize("name", ["other:hot", "vispy:does-not-exist"])
def test_invalid_colormap_provider_or_name(name):
    with pytest.raises(ValueError):
        _sample_colormap(name, np.linspace(0, 1, 256))


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


def test_volume_transform_preserves_native_voxel_centers():
    volume = Volume(torch.empty(1, 1, 6, 7, 8), spacing=(3, 2, 1), origin=(7, 11, 13))
    transform = volume_transform(volume)
    np.testing.assert_allclose(transform.map((0, 0, 0))[:3], volume.origin)
    np.testing.assert_allclose(transform.map((2, 1, 1))[:3], volume.index_to_world((1, 1, 2)))


@pytest.fixture(scope="session")
def application():
    if os.environ.get("VOL5DKIT_TEST_GUI") != "1":
        pytest.skip("set VOL5DKIT_TEST_GUI=1 to run actual OpenGL rendering")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(application):
    if os.environ.get("VOL5DKIT_TEST_GUI") != "1":
        pytest.skip("set VOL5DKIT_TEST_GUI=1 to run actual OpenGL rendering")
    from PySide6 import QtCore
    from PySide6.QtWidgets import QWidget, QVBoxLayout
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


def _frame(tensor, *, rgb=False, volume_3d=False):
    return prepare_frame(Request(1, 0, make_source(Volume(tensor), rgb), (0, 0, 0, 0), window=(0, 1), volume_3d=volume_3d))


def _sample(canvas, screenshot, name, column, row):
    mapped = canvas._images[name].get_transform(map_from="visual", map_to="canvas").map((column, row))
    x, y = mapped[:2] / mapped[3]
    scale_x = screenshot.shape[1] / canvas._canvas.size[0]
    scale_y = screenshot.shape[0] / canvas._canvas.size[1]
    return screenshot[int(y * scale_y), int(x * scale_x), :3]


@pytest.mark.gui
@pytest.mark.parametrize("rgb", [False, True])
def test_native_planes_screen_orientation_and_cursor_roundtrip(canvas, rgb):
    s, h, w = torch.meshgrid(torch.arange(5), torch.arange(7), torch.arange(9), indexing="ij")
    channels = torch.stack((s / 4, h / 6, w / 8))
    tensor = channels[None] if rgb else (channels * torch.tensor((0.6, 0.25, 0.15))[:, None, None, None]).sum(0)[None, None]
    volume = Volume(tensor, spacing=(3, 1.5, 0.75))
    frame = prepare_frame(Request(1, 0, make_source(volume, rgb), (0, 0.5, 0.5, 0.5), window=(0, 1)))
    canvas.set_frame(frame, mode_3d="hidden")
    screenshot = canvas.render()

    # Native array rows are H/S/S. On screen H grows down, while S grows up.
    for name, rows, cols, top_row in (("HW", 7, 9, 0), ("SW", 5, 9, 4), ("SH", 5, 7, 4)):
        first = _canvas_pos(canvas, name, 0.5, 0.5)
        last = _canvas_pos(canvas, name, cols - 0.5, rows - 0.5)
        assert first[0] < last[0]
        assert (first[1] < last[1]) == (top_row == 0)
        for row, col in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
            shw = {"HW": (2, row, col), "SW": (row, 3, col), "SH": (row, col, 4)}[name]
            expected = tensor[(0, slice(None), *shw)].numpy() * 255
            if not rgb:
                expected = np.repeat(expected, 3)
            np.testing.assert_allclose(_sample(canvas, screenshot, name, col + 0.5, row + 0.5), expected, atol=2)
            position = _canvas_pos(canvas, name, col + 0.5, row + 0.5)
            picked = canvas._cursor_position(SimpleNamespace(pos=position))
            assert picked[0] == name
            np.testing.assert_allclose(picked[1:], (row, col), atol=1e-6)


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
def test_all_colormaps_render_native_planes_and_scalar_volume(canvas):
    from vispy.color import get_colormap
    frame = _frame(torch.full((1, 1, 4, 4, 4), 0.5), volume_3d=True)
    canvas.set_frame(frame)
    for name in ("grays", "viridis", "hot", "coolwarm", "fire"):
        canvas.set_colormap(name)
        canvas.set_3d_mode("slices")
        screenshot = canvas.render()
        expected = get_colormap(name)[0.5].rgba[0, :3] * 255
        np.testing.assert_allclose(_sample(canvas, screenshot, "HW", 1.5, 1.5), expected, atol=2)
        assert all(image.visible for image in canvas._cutplanes.values())
        canvas.set_3d_mode("volume")
        assert canvas._volume_visual.visible
        assert canvas.render().shape[-1] == 4
        alpha = canvas._volume_visual.cmap.colors.rgba[:, 3]
        np.testing.assert_allclose(alpha, np.linspace(0, canvas._opacity, len(alpha)), atol=1e-7)


@pytest.mark.gui
@pytest.mark.parametrize("name,dependency", [("mpl:hot", "matplotlib"), ("sns:rocket", "seaborn")])
def test_optional_colormaps_render_planes_and_volume(canvas, name, dependency):
    pytest.importorskip(dependency)
    frame = _frame(torch.full((1, 1, 4, 4, 4), 0.5), volume_3d=True)
    canvas.set_frame(frame)
    canvas.set_colormap(name)
    for mode in ("slices", "volume"):
        canvas.set_3d_mode(mode)
        screenshot = canvas.render()
        # The display intentionally samples each provider into a 256-color LUT.
        expected = _sample_colormap(name, np.array([0.5]))[0, :3] * 255
        np.testing.assert_allclose(_sample(canvas, screenshot, "HW", 1.5, 1.5), expected, atol=2)
        assert canvas._volume_visual.visible == (mode == "volume")


@pytest.mark.gui
def test_failed_colormap_preparation_leaves_all_visuals_unchanged(canvas, monkeypatch):
    canvas.set_frame(_frame(torch.full((1, 1, 4, 4, 4), 0.5), volume_3d=True), mode_3d="volume")
    before = canvas.render()
    maps = [image.cmap for image in (*canvas._images.values(), *canvas._cutplanes.values(), canvas._volume_visual)]
    from vol5dkit.viewer import canvas as module
    colormap = module.Colormap
    calls = 0

    def fail_volume(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("broken volume colormap")
        return colormap(*args, **kwargs)

    monkeypatch.setattr(module, "Colormap", fail_volume)
    with pytest.raises(ValueError, match="broken volume"):
        canvas.set_colormap("hot")
    assert canvas.colormap == "grays"
    assert maps == [image.cmap for image in (*canvas._images.values(), *canvas._cutplanes.values(), canvas._volume_visual)]
    np.testing.assert_array_equal(canvas.render(), before)


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
    canvas.set_frame(_frame(torch.rand(1, 1, 8, 8, 8), volume_3d=True), mode_3d="volume")
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
@pytest.mark.parametrize("name", ["HW", "SW", "SH"])
def test_left_selection_drag_and_shift_pan(canvas, name):
    from vispy.geometry import Rect
    from vispy.util import keys
    canvas.set_frame(_frame(torch.zeros(1, 1, 5, 7, 9)))
    canvas.render()
    picked = []
    canvas._on_pick = lambda *args: picked.append(args)
    start, end = _canvas_pos(canvas, name, 1.5, 1.5), _canvas_pos(canvas, name, 5.5, 4.5)
    camera = canvas._views[name].camera
    before = Rect(camera.rect)
    press = _mouse(canvas, "mouse_press", start, button=1, buttons=[1])
    move = _mouse(canvas, "mouse_move", end, button=1, buttons=[1], press_event=press, last_event=press)
    _mouse(canvas, "mouse_release", end, button=1, press_event=press, last_event=move)
    assert all(p[0] == name for p in picked)
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
    canvas.render()
    np.testing.assert_allclose(_canvas_pos(canvas, name, 1.5, 1.5) - start, (20, 15), atol=1e-5)
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
    # A partial wheel tick belongs to the displayed input.
    _mouse(canvas, "mouse_wheel", pos, delta=(0, 0.5))
    canvas.set_frame(replace(frame, input_index=1))
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
    frame = prepare_frame(Request(1, 0, make_source(volume), (0, 0.5, 0.5, 0.5), window=(0, 1)))
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


@pytest.mark.gui
def test_prepared_palette_samples_once_and_shares_all_image_maps(canvas, monkeypatch):
    from vol5dkit.viewer import canvas as module
    sample = module._sample_colormap
    calls = []

    def record(name, samples):
        calls.append(name)
        return sample(name, samples)

    monkeypatch.setattr(module, "_sample_colormap", record)
    palette = canvas.prepare_colormap("fire")
    frame = _frame(torch.full((1, 1, 4, 4, 4), 0.5), volume_3d=True)
    canvas.set_frame(frame, mode_3d="volume", palette=palette)
    assert calls == ["fire"]
    assert all(image.cmap is palette.scalar for image in (*canvas._images.values(), *canvas._cutplanes.values()))
    assert canvas._volume_visual.cmap is palette.volume
    canvas.set_opacity(0.3)
    adjusted = canvas.palette
    canvas.set_frame(frame, mode_3d="volume", opacity=0.3, palette=palette)
    assert canvas.palette is adjusted
    assert calls == ["fire"]
    assert canvas._volume_visual._last_data is frame.volume_buffer
