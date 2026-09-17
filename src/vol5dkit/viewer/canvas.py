"""VisPy drawing only: native image planes and optional 3D volume rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import importlib.util
import itertools

import numpy as np
from vispy import app, scene
from vispy.color import Colormap, get_colormap, get_colormaps
from vispy.geometry import Rect
from vispy.util import keys
from vispy.visuals.transforms import MatrixTransform, STTransform


# (row, column, fixed) in SHW order; ImageVisual's local axes are (column, row).
PLANE_AXES = {"HW": (1, 2, 0), "SW": (0, 2, 1), "SH": (0, 1, 2)}

_SEABORN_COLORMAPS = ("rocket", "mako", "icefire", "vlag", "flare", "crest")


def available_colormaps(extended=False):
    """Return (provider-qualified name, display label), loading extras on demand.

    An absent extra is skipped. A broken installed library raises instead of
    silently hiding its installation error; the caller can retain its list.
    """
    result = [(f"vispy:{name}", f"{name} · VisPy") for name in get_colormaps()]
    if extended:
        if importlib.util.find_spec("matplotlib") is not None:
            mpl = importlib.import_module("matplotlib")
            result.extend((f"mpl:{name}", f"{name} · Matplotlib") for name in mpl.colormaps)
        if importlib.util.find_spec("seaborn") is not None:
            importlib.import_module("seaborn")
            result.extend(
                (f"sns:{name}{suffix}", f"{name}{suffix} · Seaborn")
                for name in _SEABORN_COLORMAPS for suffix in ("", "_r")
            )
    return result


def _sample_colormap(name, samples):
    """Sample the named provider directly; never substitute a namesake map."""
    provider, separator, key = name.partition(":")
    if not separator:
        key = name
        provider = "vispy" if name in get_colormaps() else "mpl"
    if provider == "vispy":
        if key not in get_colormaps():
            raise ValueError(f"unknown VisPy colormap: {key!r}")
        # Public lookup supplies column-shaped samples for analytic maps such
        # as hot/fire, whose map() methods cannot accept a 1D sample array.
        colors = get_colormap(key)[samples].rgba
    elif provider == "mpl":
        mpl = importlib.import_module("matplotlib")
        colors = mpl.colormaps[key](samples)
    elif provider == "sns":
        base = key[:-2] if key.endswith("_r") else key
        if base not in _SEABORN_COLORMAPS:
            raise ValueError(f"unknown continuous Seaborn colormap: {key!r}")
        sns = importlib.import_module("seaborn")
        colors = sns.color_palette(key, as_cmap=True)(samples)
    else:
        raise ValueError("colormap provider must be 'vispy', 'mpl', or 'sns'")
    return np.array(colors, copy=True)


def plane_transform(volume, name, indices):
    """Map image corners to world XYZ, with integer indices at voxel centers."""
    row, col, fixed = PLANE_AXES[name]
    basis = np.asarray(volume.direction, dtype=float)[:, ::-1] * volume.spacing
    start = np.zeros(3)
    start[row] = start[col] = -0.5
    start[fixed] = indices[2 + fixed]
    matrix = np.eye(4)
    matrix[:3, :3] = basis[:, (col, row, fixed)]
    matrix[:3, 3] = np.asarray(volume.origin) + basis @ start
    # MatrixTransform stores a row-vector matrix, unlike the column-vector
    # coordinate equation used by Volume.
    return MatrixTransform(matrix.T)


def volume_transform(volume):
    """VisPy VolumeVisual already centers its first voxel at index zero."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(volume.direction) * np.asarray(volume.spacing)[::-1]
    matrix[:3, 3] = volume.origin
    return MatrixTransform(matrix.T)


@dataclass(frozen=True, slots=True)
class _Palette:
    name: str
    scalar: Colormap
    volume: Colormap
    opacity: float


class VolumeCanvas:
    """A 2x2 native-slice canvas, independent of Qt controls and data workers.

    Hover/pick callbacks receive ``(plane_name, continuous_row, continuous_col)``.
    Left clicks and drags pick; Shift+left drags pan. Unmodified wheel events call
    ``on_scroll(plane_name, integer_steps)``; Ctrl+wheel retains camera zoom.
    Only CPU display frames may be passed to this object.
    """

    def __init__(self, on_hover=None, on_pick=None, on_scroll=None):
        app.use_app("pyside6")
        self._canvas = scene.SceneCanvas(
            keys=None, bgcolor="#101218", size=(1000, 800),
            config={"samples": 0}, show=False,
        )
        self.widget = self._canvas.native
        self._on_hover = on_hover
        self._on_pick = on_pick
        self._on_scroll = on_scroll
        self._selection = None
        self._pan = None
        self._scroll_remainders = dict.fromkeys(PLANE_AXES, 0.0)
        self._frame = None
        self._interpolation = "nearest"
        self._mode_3d = "slices"
        self._opacity = 0.15
        self._palette = self.prepare_colormap("grays")
        self._rgb = None
        self._images = {}
        self._cutplanes = {}
        self._crosshairs = {}
        self._crosshair_visible = True
        self._bounds = {}
        self._world_bounds = None
        self._volume_visual = None
        self._volume_data = None
        self._views = {}
        self._labels = {}
        grid = self._canvas.central_widget.add_grid(spacing=5, margin=5)
        for name, (row, col) in zip(PLANE_AXES, ((0, 0), (0, 1), (1, 0))):
            panel = grid.add_grid(row=row, col=col, spacing=0)
            label = scene.Label(name, color="#d4d9e3", font_size=10)
            label.height_max = 24
            panel.add_widget(label, row=0, col=0)
            view = panel.add_view(row=1, col=0, border_color="#343b49")
            view.camera = scene.PanZoomCamera(aspect=1)
            view.camera.flip = (False, True, False)
            self._views[name] = view
            self._labels[name] = label
            crosshair = scene.visuals.Line(
                parent=view.scene, connect="segments", method="gl",
                color=(0.2, 0.8, 1.0, 1.0), width=1, antialias=False,
            )
            crosshair.order = 10
            crosshair.set_gl_state("opaque", depth_test=False)
            self._crosshairs[name] = crosshair
        panel = grid.add_grid(row=1, col=1, spacing=0)
        label = scene.Label("3D · native planes", color="#d4d9e3", font_size=10)
        label.height_max = 24
        panel.add_widget(label, row=0, col=0)
        self._labels["3D"] = label
        self._view3d = panel.add_view(row=1, col=0, border_color="#343b49")
        self._view3d.camera = scene.TurntableCamera(
            fov=0, elevation=30, azimuth=40, up="+z",
        )
        # Intercept before SceneCanvas routes the same event to its camera.
        # handled alone does not stop SceneCanvas's routing; blocked does.
        for name, handler in (
            ("mouse_press", self._mouse_press), ("mouse_move", self._mouse_move),
            ("mouse_release", self._mouse_release), ("mouse_wheel", self._mouse_wheel),
        ):
            getattr(self._canvas.events, name).connect(
                handler, position="first",
            )

    def _make_images(self, rgb):
        # Recreate only when scalar/RGB changes the texture's channel format.
        for image in (*self._images.values(), *self._cutplanes.values()):
            image.parent = None
        self._images.clear()
        self._cutplanes.clear()
        blank = np.zeros((1, 1, 3) if rgb else (1, 1), dtype=np.float32)
        for name in PLANE_AXES:
            for target, parent, depth in (
                (self._images, self._views[name].scene, False),
                (self._cutplanes, self._view3d.scene, True),
            ):
                image = scene.visuals.Image(
                    blank, parent=parent, method="subdivide", clim=(0, 1),
                    # VisPy expands the 'r' prefix to 'rgb' for HWC data.
                    texture_format="r32f",
                    interpolation=self._interpolation,
                    cmap=self._palette.scalar,
                )
                image.set_gl_state(
                    "opaque", depth_test=depth, cull_face=False,
                )
                target[name] = image
        self._rgb = rgb

    @property
    def palette(self):
        """Prepared scalar/volume maps currently displayed by the canvas."""
        return self._palette

    @property
    def colormap(self):
        return self._palette.name

    def prepare_colormap(self, name):
        """Sample a provider once and validate both maps without changing pixels."""
        samples = np.linspace(0, 1, 256)
        colors = _sample_colormap(name, samples)
        colors[:, 3] = 1.0
        scalar = Colormap(colors.copy(), bad_color=(1, 0, 1, 1))
        colors[:, 3] = samples * self._opacity
        volume = Colormap(colors, bad_color=(1, 0, 1, self._opacity))
        return _Palette(name, scalar, volume, self._opacity)

    def set_colormap(self, value):
        """Apply a name or an already prepared palette to every scalar visual."""
        palette = self.prepare_colormap(value) if isinstance(value, str) else value
        if palette.opacity != self._opacity:
            palette = self._palette_opacity(palette, self._opacity)
        for image in (*self._images.values(), *self._cutplanes.values()):
            image.cmap = palette.scalar
        if self._volume_visual is not None:
            self._volume_visual.cmap = palette.volume
        self._palette = palette
        self._canvas.update()

    @staticmethod
    def _palette_opacity(palette, opacity):
        colors = palette.scalar.colors.rgba.copy()
        colors[:, 3] = np.linspace(0, opacity, len(colors))
        return replace(palette, volume=Colormap(colors, bad_color=(1, 0, 1, opacity)), opacity=opacity)

    def set_interpolation(self, value):
        """Change display sampling without modifying or resampling source data."""
        if value not in ("nearest", "linear"):
            raise ValueError("interpolation must be 'nearest' or 'linear'")
        if value == self._interpolation:
            return
        self._interpolation = value
        for image in (*self._images.values(), *self._cutplanes.values()):
            image.interpolation = value
        if self._volume_visual is not None:
            self._volume_visual.interpolation = value
        self._canvas.update()

    def set_3d_mode(self, value):
        if value not in ("slices", "volume", "hidden"):
            raise ValueError("3D mode must be 'slices', 'volume', or 'hidden'")
        self._mode_3d = value
        for image in self._cutplanes.values():
            image.visible = value == "slices"
        if self._volume_visual is not None:
            self._volume_visual.visible = value == "volume" and self._volume_data is not None
        captions = {"slices": "3D · native planes", "volume": "3D · scalar volume", "hidden": "3D · hidden"}
        self._labels["3D"].text = captions[value]
        self._canvas.update()

    def set_opacity(self, value):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("opacity must be between 0 and 1")
        if self._opacity == value:
            return
        palette = self._palette_opacity(self._palette, value)
        if self._volume_visual is not None:
            self._volume_visual.cmap = palette.volume
        self._palette = palette
        self._opacity = value
        self._canvas.update()

    def set_crosshair_visible(self, value):
        """Show the shared SHW point without changing native image pixels."""
        self._crosshair_visible = bool(value)
        for crosshair in self._crosshairs.values():
            crosshair.visible = self._crosshair_visible
        self._canvas.update()

    def set_frame(self, frame, *, interpolation="nearest", mode_3d="slices", opacity=0.15, palette=None):
        """Replace image content and labels atomically on the GUI thread."""
        self.set_interpolation(interpolation)
        self.set_opacity(opacity)
        if palette is not None and palette.scalar is not self._palette.scalar:
            self.set_colormap(palette)
        volume = frame.source.volume
        if self._frame is None or self._frame.source is not frame.source or self._frame.input_index != frame.input_index:
            self._scroll_remainders = dict.fromkeys(PLANE_AXES, 0.0)
        if self._rgb != frame.source.rgb:
            self._make_images(frame.source.rgb)
        for plane in frame.planes:
            row, col, fixed = PLANE_AXES[plane.name]
            image = self._images[plane.name]
            pixels = plane.image
            if frame.source.rgb:
                # RGB bypasses the scalar colormap (and its NaN color). Mark
                # the whole invalid pixel before upload, leaving raw values
                # and the worker's frame buffer unchanged.
                bad = ~np.isfinite(pixels).all(axis=-1)
                if bad.any():
                    pixels = pixels.copy()
                    pixels[bad] = (1, 0, 1)
            image.set_data(pixels, copy=False)
            dx, dy = volume.spacing[col], volume.spacing[row]
            image.transform = STTransform(scale=(dx, dy, 1), translate=(-0.5 * dx, -0.5 * dy, 0))
            height, width = plane.image.shape[:2]
            bounds = Rect(-0.5 * dx, -0.5 * dy, width * dx, height * dy)
            old = self._bounds.get(plane.name)
            camera = self._views[plane.name].camera
            if old is None:
                camera.rect = bounds
            elif bounds != old:
                rect = camera.rect
                camera.rect = Rect(
                    bounds.left + (rect.left - old.left) / old.width * bounds.width,
                    bounds.bottom + (rect.bottom - old.bottom) / old.height * bounds.height,
                    rect.width / old.width * bounds.width,
                    rect.height / old.height * bounds.height,
                )
            self._bounds[plane.name] = bounds
            # The central gap spans the selected voxel, preserving its color.
            r, c = frame.indices[2 + row], frame.indices[2 + col]
            x, y = c + 0.5, r + 0.5
            crosshair = self._crosshairs[plane.name]
            crosshair.set_data(pos=np.array(
                ((0, y), (c, y), (c + 1, y), (width, y),
                 (x, 0), (x, r), (x, r + 1), (x, height)), dtype=np.float32,
            ))
            crosshair.transform = image.transform
            crosshair.visible = self._crosshair_visible
            cutplane = self._cutplanes[plane.name]
            cutplane.set_data(pixels, copy=False)
            cutplane.transform = plane_transform(volume, plane.name, frame.indices)
            fixed_name = "SHW"[fixed]
            self._labels[plane.name].text = f"{plane.name} · {fixed_name}={frame.indices[2 + fixed]}"
        self._update_world_bounds(volume)
        if frame.volume_buffer is not None:
            if self._volume_visual is None:
                self._volume_visual = scene.visuals.Volume(
                    frame.volume_buffer, parent=self._view3d.scene, clim=(0, 1),
                    method="translucent", texture_format="r32f",
                    interpolation=self._interpolation,
                    cmap=self._palette.volume,
                )
            elif self._volume_data is not frame.volume_buffer:
                self._volume_visual.set_data(frame.volume_buffer, clim=(0, 1), copy=False)
            self._volume_visual.transform = volume_transform(volume)
        elif self._volume_visual is not None and self._volume_data is not None:
            # Release the full-volume CPU/GPU allocation when returning to slices.
            self._volume_visual.parent = None
            self._volume_visual = None
        self._volume_data = frame.volume_buffer
        self._frame = frame
        self.set_3d_mode(mode_3d)
        self._canvas.update()

    def _update_world_bounds(self, volume):
        corners = np.array(list(itertools.product(*[(-0.5, n - 0.5) for n in volume.shape[2:]])))
        xyz = (corners * volume.spacing)[:, ::-1] @ np.asarray(volume.direction).T + volume.origin
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        center = (low + high) * 0.5
        extent = float(np.max(high - low))
        camera = self._view3d.camera
        if self._world_bounds is None:
            camera.center = tuple(center)
            camera.scale_factor = extent * 1.25
        else:
            old_center, old_extent = self._world_bounds
            if not np.array_equal(center, old_center) or extent != old_extent:
                camera.center = tuple(center + (np.asarray(camera.center) - old_center) * extent / old_extent)
                camera.scale_factor *= extent / old_extent
        self._world_bounds = (center, extent)

    def reset_view(self):
        """Fit all planes and reset the 3D orbit."""
        for name, bounds in self._bounds.items():
            self._views[name].camera.rect = bounds
        if self._world_bounds is not None:
            center, extent = self._world_bounds
            camera = self._view3d.camera
            camera.center = tuple(center)
            camera.scale_factor = extent * 1.25
            camera.azimuth, camera.elevation, camera.roll = 40, 30, 0
        self._canvas.update()

    def _view_at(self, pos):
        if self._frame is None or pos is None:
            return None
        for name, view in self._views.items():
            local = view.node_transform(self._canvas.scene).imap(pos)
            if 0 <= local[0] < view.width and 0 <= local[1] < view.height:
                return name
        return None

    def _cursor_position(self, event):
        name = self._view_at(event.pos)
        if name is None:
            return None
        image = self._images[name]
        pixel = image.get_transform(map_from="canvas", map_to="visual").map(event.pos)
        pixel = pixel[:2] / pixel[3]
        width, height = image.size
        if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
            return name, float(pixel[1] - 0.5), float(pixel[0] - 0.5)
        return None

    def _mouse_press(self, event):
        self._selection = self._pan = None
        name = self._view_at(event.pos)
        if event.button != 1 or name is None:
            return
        if len(event.modifiers) == 1 and keys.SHIFT in event.modifiers:
            self._pan = (name, np.asarray(event.pos, dtype=float))
            event.handled = event.blocked = True
        elif not event.modifiers:
            event.handled = event.blocked = True
            self._selection = (name, self._frame.source, self._frame.input_index)
            self._select_at(event)

    def _mouse_move(self, event):
        if 1 in event.buttons:
            if self._selection is not None:
                event.handled = event.blocked = True
                self._select_at(event)
            elif self._pan is not None:
                event.handled = event.blocked = True
                name, previous = self._pan
                transform = self._views[name].scene.node_transform(self._canvas.scene)
                before, after = transform.imap(previous), transform.imap(event.pos)
                self._views[name].camera.pan(before[:2] / before[3] - after[:2] / after[3])
                self._pan = (name, np.asarray(event.pos, dtype=float))
        position = self._cursor_position(event)
        if position is not None and self._on_hover is not None:
            self._on_hover(*position)

    def _mouse_release(self, event):
        if event.button != 1:
            return
        if self._selection is not None:
            event.handled = event.blocked = True
            self._select_at(event)
        elif self._pan is not None:
            event.handled = event.blocked = True
        self._selection = self._pan = None

    def _select_at(self, event):
        name, source, input_index = self._selection
        if self._frame is None or self._frame.source is not source or self._frame.input_index != input_index:
            return
        position = self._cursor_position(event)
        if position is not None and position[0] == name and self._on_pick is not None:
            self._on_pick(*position)

    def _mouse_wheel(self, event):
        name = self._view_at(event.pos)
        if name is None or event.modifiers:
            return
        event.handled = event.blocked = True
        delta = float(event.delta[1])
        if not np.isfinite(delta):
            return
        total = self._scroll_remainders[name] + delta
        steps = int(np.trunc(total + np.copysign(1e-9, total)))
        remainder = total - steps
        self._scroll_remainders[name] = 0.0 if abs(remainder) < 1e-9 else remainder
        if steps and self._on_scroll is not None:
            self._on_scroll(name, steps)

    def render(self):
        """Return an RGBA screenshot, useful for renderer regression checks."""
        return self._canvas.render(alpha=True)

    def close(self):
        self._frame = None
        self._volume_data = None
        self._on_hover = self._on_pick = self._on_scroll = None
        self._selection = self._pan = None
        for image in (*self._images.values(), *self._cutplanes.values(), *self._crosshairs.values()):
            image.parent = None
        self._images.clear()
        self._cutplanes.clear()
        self._crosshairs.clear()
        if self._volume_visual is not None:
            self._volume_visual.parent = None
            self._volume_visual = None
        self._canvas.close()
