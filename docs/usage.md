# Usage

## Data and coordinates

A volume has exactly five nonempty dimensions: time, channels, and spatial
S/H/W axes. Tensor inputs must be native dense strided PyTorch tensors; writable
NumPy arrays with supported dtype and strides are also accepted. There is no
axis guessing, automatic dimension insertion, or tensor-subclass dispatch.

```python
import torch
import vol5dkit as v5

x = torch.zeros(2, 1, 8, 16, 16)
a = v5.Volume(
    x,
    spacing=(2, 1, 1),       # S, H, W
    origin=(10, 20, 30),     # X, Y, Z of voxel (0, 0, 0)
    direction=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    times=(0.0, 0.1),
)
```

Defaults are unit spacing, zero origin, identity direction, and frame-index
times. Spacing must be finite and positive. Direction is orthogonal, with
reflections allowed and shear rejected; its columns are the world directions
of the XYZ index axes. Times must be finite and have length T; they may be
nonuniform, nonmonotonic, or repeated. Physical units and world basis are
caller-defined. All T/C entries share one regular spatial grid.

Coordinates are small immutable CPU tuples. Integer spatial indices are voxel
centers, with the mapping:

```text
world_xyz = origin_xyz + direction @ [w * spacing_W,
                                      h * spacing_H,
                                      s * spacing_S]
```

`a.index_to_world(shw)` and `a.world_to_index(xyz)` accept shape `(..., 3)` and
return CPU float64 NumPy arrays without rounding or clamping.

### Referencing a result

```python
y = a.tensor.square()          # ordinary PyTorch; no metadata hooks
b = v5.Volume(y, ref=a)        # coordinates are attached explicitly
```

| Result compared with `ref` | Behavior |
| --- | --- |
| Same T and SHW | Reuses the immutable coordinate tuples |
| Same T, changed SHW | Adjusts spacing and origin to preserve outer voxel bounds and center; retains direction and times |
| Changed T | Raises an error; time values are not inferred |
| Changed C, dtype, or device | Allowed; the result tensor is retained as supplied |

`ref` declares a spatial relationship; it does not establish one from tensor
values. Do not use it for a crop or warp that changes the represented region.
Use the corresponding coordinate-aware operation or supply explicit coordinates.
Combining `ref` with any explicit coordinate argument is an error, including
`times=None`. Omitting `ref`, or passing `ref=None`, selects ordinary construction.

For reference SHW size `N`, result size `M`, and spacing `d`, the calculation is:

```text
spacing_new = d * N / M
origin_new = origin + direction @ ((spacing_new - d) / 2)[::-1]
```

The operations are elementwise except for the matrix product; reversal changes
SHW to XYZ. This preserves all outer voxel corners and the overall center,
including rotated, reflected, anisotropic, and singleton grids.

```python
import torch.nn.functional as F

y = F.interpolate(a.tensor, scale_factor=2, mode="trilinear", align_corners=False)
upsampled = v5.Volume(y, ref=a)
```

Interpolation above is an explicit processing operation. Wrapping the result
only assigns coordinates. A model may use another output-grid convention:

| One-dimensional output grid | Spacing | Origin |
| --- | --- | --- |
| Preserve outer voxel boundaries | `d * N / M` | `o + (d_out - d) / 2` |
| Preserve first and last voxel centers | `d * (N - 1) / (M - 1)` | `o` |

The second formula requires N and M greater than one. Supply explicit spacing
and origin when the reference convention does not describe the output grid.

### Spatial operations

```python
cropped = a.crop(t=slice(0, 2), s=slice(1, 7, 2), h=slice(2, 12))
flipped = a.flip_spatial("s", "w")
reordered = a.permute_spatial("w", "s", "h")
```

Crop uses Python's exclusive-stop slices and positive strides, retains five
dimensions, moves origin to the first selected voxel center, multiplies spacing
by spatial strides, and slices times. Integer selection, negative strides, and
empty results are rejected.

Flip and permutation preserve each voxel's world position while changing its
index. Axis names are lowercase `"s"`, `"h"`, and `"w"`; duplicates are rejected.
Permutation requires each spatial axis once. An empty flip and identity
permutation return the original volume. T/C roles and times remain unchanged;
vector-valued channel components are not rotated automatically.

Crop and permutation use views where possible. Flip copies as `torch.flip`
does. These operations do not interpolate, force contiguity, or change dtype or
device. Resampling, warping, and padding are left to processing code.

### Ownership and conversion

| Operation | Data behavior |
| --- | --- |
| `Volume(tensor)` / `Volume(result, ref=a)` | Retains the native tensor, strides, device, dtype, and autograd graph |
| `Volume(array)` | Shares compatible writable NumPy storage |
| `a.to(...)` | Follows `Tensor.to()` and retains coordinates |
| `a.clone()` | Copies tensor storage and retains coordinates |
| `a.numpy(copy=False)` | Detaches; shares CPU storage where possible, copies other devices to CPU |
| `a.numpy(copy=True)` | Returns independent NumPy storage |
| `Volume.from_dlpack(obj, **coordinates)` | Shares external storage without promising to import an external autograd graph |

Read-only or negative-stride NumPy inputs require an explicit copy. Unsupported
dtypes are not silently converted; cast bfloat16 explicitly before NumPy export.
Lazy conjugate or negative views may need materialization during NumPy export.

The frozen wrapper prevents field reassignment, not tensor writes. Structural
in-place changes such as `resize_()`, `set_()`, and `transpose_()` are unsupported
after wrapping. Use `.tensor` for computation; arithmetic, indexing, and NumPy
conversion are never forwarded implicitly.

External coordinates and changed grids are validated at construction. Exact
base `Volume` objects on the same grid share already validated tuples after
checking tensor structure and time count, without rescanning all times. A
reference object is not retained. Subclass references are validated normally.

Dataclass subclasses may add fields and validation. Their generated constructors
do not automatically acquire the base constructor's `ref` argument. Existing
`.to()`, `.clone()`, and spatial operations use `dataclasses.replace()` for
subclasses, preserving extra fields and running their validation.

## Display options

```python
residual = v5.Volume(b.tensor - a.tensor, ref=a)
viewer = v5.view(
    v5.Display(a, name="original", window=(0, 1)),
    v5.Display(b, name="processed", window=(0, 1)),
    v5.Display(residual, name="difference", cmap="mpl:coolwarm"),
)
```

`view(*inputs, window=None, interpolation="nearest")` accepts one or more
`Volume` objects, native Torch tensors, NumPy arrays, or `Display` objects.
Raw tensors and arrays must already be nonempty TCSHW; they receive default
index coordinates. Input types can be mixed.

`Display(data, *, name=None, window=None, cmap="grays", rgb=False)` adds options
to one input. It is a small frozen dataclass. Default names are `Volume 1`,
`Volume 2`, and so on. Channel starts at 0 for each input. RGB is always explicit
and requires C=3: uint8 uses `[0, 255]`, floating inputs use `[0, 1]`; normalize
other ranges explicitly.

### Scalar windows and colormaps

| Setting | Behavior |
| --- | --- |
| Input `window=None` | Lazily computes that input's finite min/max over all TCSHW, then retains it |
| `Display(data, window=(low, high))` | Sets that input's individual window |
| `view(..., window=(low, high))` | Sets a common window and enables Share window |
| Enable Share window | Applies the current input's window to all scalar inputs |
| Disable Share window | Restores saved individual windows |

Sharing does not overwrite individual settings. Range editing and **Auto**
affect the common window while sharing, and the selected input otherwise.
Auto uses the selected input's whole TCSHW range, not a union across inputs;
its target remains fixed if input selection changes while calculation runs.
Ranges stay fixed during T/C navigation. Constants display at the colormap
midpoint. All-nonfinite inputs use `(0, 1)`; nonfinite pixels have a distinct
color and retain their original values for inspection.

The default GUI provides VisPy maps. The `colormaps` extra adds Matplotlib's
registry and Seaborn's continuous rocket, mako, icefire, vlag, flare, crest maps
and reversed variants. The searchable selector labels each source. Extra
libraries load when the selector first opens or an extra map is requested;
no pyplot figure is created.

Use `vispy:hot`, `mpl:hot`, or `sns:rocket` to identify the exact provider.
Bare VisPy names retain their meaning; other bare names use Matplotlib's
registry. Each provider is sampled directly into 256 RGBA colors. Failed
selections leave the previous display unchanged and show an error. Scalar
windows and colormaps are disabled in RGB mode.

### Navigation and rendering

The 2×2 display contains native HW, SW, SH slices and a 3D panel. The toolbar
selects input, Scalar/RGB, and channel; Fit resets the view. The resizable,
collapsible right panel contains sliders and numeric T/S/H/W controls, display
settings, and 3D settings. It scrolls when window height is limited. The status
area shows indices, world coordinates, original values, and processing state.

- Left-click or drag moves the linked SHW crosshair. HW changes H/W, SW changes
  S/W, and SH changes S/H. Crosshair visibility is optional.
- Wheel over a slice moves the perpendicular axis: HW → S, SW → H, SH → W.
  Up increases the index; down decreases it, stopping at the bounds.
- Ctrl+wheel and right-drag zoom; Shift+left-drag pans. The 3D panel retains
  orbit and zoom controls. Tab cycles inputs; A/B select the first two.
- Inputs share relative T/S/H/W position, zoom, and display interpolation.
  Initial time is 0 and spatial position is centered. An axis of length N uses
  `floor(p * (N - 1) + 0.5)` for relative position p; singleton axes use 0.
  Switching inputs preserves p. This is not registration or timestamp matching.
- Input name, channel, Scalar/RGB, colormap, and individual window are retained
  separately for each input.

Slices use direct integer indexing. Nearest is the default; linear is explicit
and affects display sampling only. Spacing determines aspect ratio without
resampling the slice. Original values come from owned CPU buffers in the input
dtype. Scalar windowing precedes float32 texture conversion so that narrow
float64 contrasts and large integer values are not lost to early quantization.

**Planes** places the current slices in world coordinates. **Volume** enables
scalar volume rendering, and **Off** hides the 3D display. Native 2D slices remain
available. The former voxel-step control is removed: every voxel is used.

Volume opacity defaults to 0.15 and scales per-sample opacity by normalized
scalar value. Opacity accumulates along the ray, so this is not uniform 15%
transparency. It is shown only in Volume mode. The current 3D buffer is reused
for spatial crosshair movement; input, time, channel, or window changes rebuild
it. Leaving Volume mode releases it. RGB volume rendering is not provided.

## Independent snapshots and debuggers

Each `view()` call creates a new process and returns after snapshot saving and
process launch, without waiting for first rendering or window closure:

```python
viewer = v5.view(a, b)
print(viewer.pid)
print(viewer.poll())       # None while running, otherwise the exit code
print(viewer.log_path)     # diagnostics when child startup fails
viewer.wait(timeout=60)    # optional; timeout raises subprocess.TimeoutExpired
# viewer.close()          # stop this window and clean its snapshot files
```

The handle provides process status, waiting, and closure only. There is no live
update channel: call `view()` again for new results.

The parent detaches and copies each **whole input** to an independent contiguous
CPU tensor, then writes it to temporary files. Inputs are staged sequentially,
so extra staging RAM is bounded by the largest input. CUDA inputs incur one
full initial device-to-host transfer. The child memory-maps saved tensors and
prepares CPU slices. Values, dtype, shape, and coordinates are preserved;
strides and autograd graphs are not transferred.

After return, modifying or releasing source tensors cannot change the snapshot.
Do not write concurrently during snapshot creation. For custom CUDA streams,
call inside the producer's stream context or make the calling stream wait for
the producer first.

Qt and the worker run in the child, so a parent breakpoint can leave the viewer
responsive. Importing `Display` or calling `view()` does not load Qt/VisPy into
the parent. A supported already-loaded pydevd disables subprocess argument
patching only for this launch; other debuggers may require their own settings.
Initial copying and file writing may exceed a debugger's evaluation timeout,
so a 3-second warning remains possible. Terminating the entire process tree
from an IDE can also terminate the viewer.

Normal closure and `close()` clean snapshot files. Failed startup preserves its
log while releasing snapshot data. Forced process termination may leave files.

Local Jupyter/IPython works with ordinary `view(...)`; `%gui qt6` is unnecessary
because the GUI event loop runs separately. This is a local desktop window,
not an inline or remote notebook widget.

See [API migration and development](development.md) and
[optional interoperability](interop.md).
