# vol5dkit

Lightweight **T,C,S,H,W** tensor coordinates and a local viewer for native
volume slices. Compute with ordinary PyTorch, attach coordinates to the results
you want to keep, and compare them with nearest-neighbor display by default.

The core has one data class, `Volume`, and depends on PyTorch and NumPy. The GUI
is optional. There are no tensor dispatch hooks, automatic metadata propagation
rules, medical readers, or processing pipelines.

## Installation

Python 3.10 or newer is required. Install the PyTorch build appropriate for your
CPU/CUDA environment first; this package does not select a CUDA wheel index.

From this checkout:

```bash
python -m pip install .                 # data core
python -m pip install ".[gui]"          # core and desktop viewer
python -m pip install -e ".[gui,test]"  # editable development installation
```

After a release has been published to PyPI, the corresponding installation
commands will be:

```bash
python -m pip install vol5dkit
python -m pip install "vol5dkit[gui]"
```

Importing `vol5dkit` does not import Qt, VisPy, or VolTensor. The optional GUI uses
PySide6-Essentials and VisPy and requires a local display with working OpenGL.

## Compute with tensors, keep results with coordinates

```python
import torch
import torch.nn.functional as F
import vol5dkit as v5

x = torch.rand(2, 1, 32, 64, 64, device="cpu")  # use device="cuda" if available
a = v5.Volume(x, spacing=(2.0, 1.0, 1.0), times=(0.0, 0.1))

# T acts as batch for this operation. No vol5dkit code runs inside the operation.
y = F.avg_pool3d(F.pad(a.tensor, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)
b = a.with_data(y)

v5.view(a, b)  # local blocking desktop window; nearest is the default
```

`with_data()` declares that a result uses the same time and spatial sampling
grid. It checks T and SHW, permits changes to C, dtype, and device, and shares
the immutable coordinate tuples. Shape equality cannot prove that a warp or
flip preserved coordinates; that decision stays with the caller.

For a different grid, construct a new `Volume` with its actual coordinates.
If only index-space inspection matters, `Volume(result)` uses default index
coordinates. The package does not infer a super-resolution model's output grid.

## Data and coordinate contract

```text
Volume(
    tensor,                         # native dense torch.Tensor or writable NumPy array
    *,
    spacing=(1.0, 1.0, 1.0),        # S, H, W
    origin=(0.0, 0.0, 0.0),         # X, Y, Z of the first voxel center
    direction=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    times=None,                     # defaults to 0, 1, ..., T-1
)
```

- Input must have exactly five nonempty dimensions in TCSHW order. Axes are not
  guessed, squeezed, or reordered. Noncontiguous strided tensors are supported.
- Spacing is finite and positive. Direction is orthogonal; reflections are
  allowed and shear is rejected. Columns describe the world directions of the
  XYZ index axes.
- Times are finite, have length T, and can be nonuniform or nonmonotonic. Units
  are caller-defined; values are not implicitly millimeters or seconds.
- Metadata is stored as small immutable CPU tuples. The tensor remains a native
  tensor with its original dtype, device, strides, and autograd graph.

Integer spatial indices denote voxel centers. For an SHW index `(s, h, w)`:

```text
world_xyz = origin_xyz + direction_xyz @ [w * spacing_W,
                                          h * spacing_H,
                                          s * spacing_S]
```

`index_to_world(shw)` and `world_to_index(xyz)` accept arrays of shape `(..., 3)`
and return CPU float64 NumPy arrays. They neither round nor clamp coordinates.

```python
world = a.index_to_world([3, 10, 20])
index = a.world_to_index(world)

cropped = a.crop(
    t=slice(0, 2),
    s=slice(4, 20, 2),
    h=slice(8, 40),
    w=slice(8, 48),
)
```

Crop uses exclusive stops and positive strides, retains all five dimensions,
updates origin to the first selected voxel center, multiplies spacing by the
spatial strides, and slices times. It returns a tensor view when PyTorch slicing
does. Integer indexing, negative strides, and empty crops are rejected.

For super-resolution, two common one-dimensional coordinate conventions differ:

| Grid preserved, input N → output M | Output spacing | Output origin |
| --- | --- | --- |
| First and last voxel centers | `d * (N - 1) / (M - 1)` | `o` |
| Outer voxel boundaries | `d * N / M` | `o + (d_out - d) / 2` |

The center-preserving formula requires N and M greater than one. In 3D, apply
direction to the XYZ origin offset. The SR example demonstrates explicit
boundary-preserving coordinates for `F.interpolate(..., align_corners=False)`.

### Ownership and conversion

| Operation | Data behavior |
| --- | --- |
| `Volume(tensor)` | Retains that native tensor; no clone, detach, or device transfer |
| `Volume(array)` | Shares compatible writable NumPy storage |
| `v.with_data(result)` | Wraps the result and preserves grid metadata |
| `v.to(...)` | Follows `Tensor.to()` and preserves metadata and autograd semantics |
| `v.clone()` | Clones tensor storage and preserves metadata |
| `v.numpy(copy=False)` | Detaches; shares CPU storage when possible, copies CUDA data to host |
| `v.numpy(copy=True)` | Returns independent NumPy storage |
| `Volume.from_dlpack(obj, **metadata)` | Shares external storage; does not import an external autograd graph |

Unsupported NumPy dtypes are not silently converted. Read-only or negative-stride
NumPy inputs require an explicit copy. NumPy export does not silently convert
unsupported tensor dtypes; for example, cast bfloat16 explicitly when exporting.

The frozen wrapper prevents field reassignment, not tensor writes. Structural
in-place changes such as `resize_()`, `set_()`, and `transpose_()` are unsupported
after wrapping. Ordinary PyTorch operations go through `.tensor`; there are no
implicit arithmetic, indexing, or NumPy forwarding hooks.

## Desktop comparison viewer

```python
viewer = v5.view(
    a,
    b=None,
    rgb=False,
    interpolation="nearest",  # "linear" is an explicit display option
    clim=None,                # scalar display limits, shared between A/B
    block=True,
)
```

All arguments after `b` are keyword-only. To update results from a script:

```python
viewer = v5.view(a, b, interpolation="nearest", clim=(0.0, 1.0), block=False)
viewer.update("B", a.with_data(new_result))
viewer.update("B", colored_volume, rgb=True)
v5.run()  # starts the event loop for nonblocking script windows
# viewer.close() releases this window and its worker
```

`update(slot, volume, *, rgb=None)` accepts `"A"` or `"B"`. Omitted `rgb` keeps
that input's display mode. B can be added after opening a single-input window.
Create and update viewers on the GUI/main thread.

- The three views show native HW, SW, and SH slices, fixed at integer S, H, and W
  indices. These correspond to axial/coronal/sagittal only when the input axes
  have that interpretation. There is no automatic anatomical reorientation.
- **Left-click or left-drag** to move the shared SHW crosshair. A click in HW changes H/W,
  a click in SW changes S/W, and a click in SH changes S/H. All three planes and
  their crosshairs update together. The crosshair leaves the selected voxel
  unobscured and can be hidden with the **Crosshair** checkbox.
- **Wheel over a 2D panel** to move its perpendicular axis: HW → S, SW → H,
  SH → W. One upward wheel notch increases the index by one; downward decreases
  it. Movement stops at the first/last slice. **Ctrl+wheel** zooms, **Shift+left-drag**
  pans, and **right-drag** zooms. The 3D panel
  keeps its orbit and zoom controls.
- **Nearest is the default.** Slices are extracted by direct indexing, with no
  preprocessing resampling. Linear interpolation changes only screen display.
  Spacing controls the displayed aspect ratio. Zoom in to examine individual
  voxels; a reduced screen image cannot show every voxel at once.
- A/B shares relative T/S/H/W position and zoom region. For a relative position
  `p`, an axis of length N selects `floor(p * (N - 1) + 0.5)`. Switching inputs
  does not change p. This is relative navigation, not physical registration or
  timestamp matching.
- Scalar mode selects one channel and offers a colormap and shared display
  limits. Initial limits use the exact finite min/max of the first selected
  scalar T/C, then remain fixed across navigation and updates. Constants display
  at mid-gray; nonfinite slice pixels are marked distinctly.
- RGB is explicit and requires C=3. uint8 uses `[0, 255]`; floating input uses
  `[0, 1]`. Normalize other ranges explicitly. RGB is never inferred from C=3.
- Pixel values come from owned CPU copies in the original dtype, not interpolated
  screen colors or repeated CUDA scalar reads. Scalar windowing happens before
  conversion to a float32 texture, preserving narrow contrasts.
- The 3D view places the current planes in their physical coordinates. Scalar
  volume rendering is optional and has color and opacity controls. Preview
  stride is explicit and does not reduce the native 2D slices.

The viewer holds detached tensor handles sharing the input storage, so it does
not keep the original autograd graph alive. CPU display buffers are independently
owned. Do not concurrently write into registered storage while the viewer reads
it. Register a clone if that storage will be reused; send replacement results
through `update()`.

CUDA registration records an event on the calling thread's current stream. The
worker waits for that event before reading the data. With a custom producer
stream, call `view()`/`update()` inside its stream context on the main thread, or
make the main thread's current stream wait for the producer first. Only selected
slices are copied to CPU in slice mode; the first scalar range reduction occurs
on the source device. Full SHW transfer happens only for explicit volume rendering.

### Local Jupyter / IPython

```python
%gui qt6
import vol5dkit as v5
viewer = v5.view(a, b, block=False)
```

This opens a desktop window on the same machine; it is not a browser-inline or
remote notebook widget. Existing `QApplication` instances are reused. A debugger
breakpoint that stops the GUI thread also stops the window; no separate GUI
process or IPC is started.

## Optional VolTensor compatibility

Install the reference VolTensor library separately. The adapter imports it only
when used; medical dependencies are not added to vol5dkit.

```python
a = v5.Volume.from_voltensor(vt)
exported = a.to_voltensor()
restored = v5.Volume.from_voltensor(exported, times=a.times)
```

Import time values use explicit `times`, then `vt.vol.source_meta.time_axis`,
then frame indices. Invalid existing time values are rejected. Export carries
data and spatial coordinates; retain `.times` separately for a time-preserving
round trip. No medical provenance or DICOM identifiers are fabricated.

The adapter maps the reference VolTensor 3.8.1 ZYX/LPS convention to this
package's XYZ convention. It interprets world XYZ as LPS when exporting; callers
using another world basis must convert that basis explicitly. Data axes are not
reordered, and storage is shared when the reference library permits it.

## Examples and measurements

```bash
python examples/compare_denoising.py --device cuda
python examples/compare_super_resolution.py --device cpu
python examples/compare_denoising.py --device cuda --smoke
python benchmarks/benchmark.py --device cuda --output benchmark.json
python benchmarks/benchmark.py --device cpu --cases rgb64 --gui --iterations 5
```

Examples use synthetic inputs and ordinary torch operations. `--smoke` waits for
prepared data, renders a real frame, and closes the window; `--screenshot path.png`
can save that frame. These checks require the GUI extra and working OpenGL.

The benchmark separates first preparation (including initial scalar limits) from
warm slice preparation with fixed limits. It reports p50/p95 latency, D2H payload
bytes, owned CPU frame memory, CUDA peak allocation, and a small wrapper overhead
measurement. `--gui` separately times rendering plus framebuffer readback; that
measurement is not an interactive FPS guarantee. Large cases run sequentially,
and optional `--volume-3d` measures explicit SHW data preparation as well.

## Development and release preparation

Development is committed and pushed to `dev-internal`, then merged into
`master`. Release tags such as `v0.0.0.1` are created on the resulting `master`
merge commit. GitHub Releases include a wheel and source distribution.

The implementation is grouped by responsibility: `volume.py` owns the data
contract, `_voltensor.py` the optional adapter, and `viewer/` contains the Qt
window, VisPy canvas, and Qt-free display preparation. No GUI import is required
for core or display-data tests.

```bash
python -m pip install -e ".[gui,dev]"
python -m pytest
```

GUI tests are opt-in on a machine with OpenGL:

```powershell
$env:VOL5DKIT_TEST_GUI = "1"
python -m pytest tests/test_viewer_gui.py
```

CUDA and reference-VolTensor tests skip when their runtime is unavailable. Build
both distributions and validate their metadata before any release:

```bash
python -m build
python -m twine check dist/*
```

These commands build and inspect artifacts; they do not publish to PyPI. The
project uses `pyproject.toml` and the standard setuptools build backend; a separate
`setup.py` is unnecessary. Releases should install and test the built wheel in a
clean environment before publication.

GitHub Actions CI checks the core on Linux/Windows, the GUI with Xvfb/Mesa,
and both distribution formats. The `Publish to PyPI` workflow is manually
dispatched. Before the first release, configure the GitHub environment `pypi`
and a PyPI Trusted Publisher for owner `jiwoosong`, repository `vol5dkit`,
workflow `release.yml`, environment `pypi`. Publishing uses OIDC rather than a
stored API token. Keep the version in `pyproject.toml` and `__init__.py` equal;
the release workflow checks this before building.

## License

MIT; see [LICENSE](LICENSE). PyTorch, NumPy, Qt/PySide6, VisPy, and optional
VolTensor remain under their respective licenses.
