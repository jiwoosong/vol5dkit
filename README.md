# vol5dkit

Lightweight **TCSHW volumes and a desktop comparison viewer** for sampled fields,
voxel grids, simulations, and image-processing results. Compute with native
PyTorch tensors, attach coordinates when needed, and inspect original slices.

The data core is one `Volume` class with PyTorch and NumPy as dependencies.
The optional viewer opens an independent snapshot window. Display interpolation
is **nearest** by default.

## Install

Python 3.10+ and PyTorch 2.3+ are required. Install the PyTorch build for your
CPU/CUDA environment first; vol5dkit does not select a CUDA wheel index.

Install the tagged release from GitHub:

```bash
python -m pip install "vol5dkit @ git+https://github.com/jiwoosong/vol5dkit.git@v0.0.0.3"
python -m pip install "vol5dkit[gui] @ git+https://github.com/jiwoosong/vol5dkit.git@v0.0.0.3"
```

Use `[gui,colormaps]` to include Matplotlib and Seaborn colormaps.
From a local checkout:

```bash
python -m pip install .                  # data core
python -m pip install ".[gui]"           # core and desktop viewer
python -m pip install ".[gui,colormaps]" # also Matplotlib and Seaborn colormaps
```

The desktop viewer needs a local display and working OpenGL. Importing the
package does not load Qt or VisPy into your Python process.

On Ubuntu/Debian, the X11 (`xcb`) backend also requires system libraries,
including `libxcb-cursor0`; see [Linux viewer troubleshooting](docs/usage.md#linux-viewer-troubleshooting).

## Compute, then compare

```python
import torch
import torch.nn.functional as F
import vol5dkit as v5d

# T = time, C = channels, S/H/W = spatial axes. CUDA tensors work too.
x = torch.rand(2, 1, 32, 64, 64)
a = v5d.Volume(x, spacing=(2, 1, 1), times=(0.0, 0.1))

y = F.avg_pool3d(a.tensor, kernel_size=3, stride=1, padding=1)
b = v5d.Volume(y, ref=a)

viewer = v5d.view(
    v5d.Display(a, name="original", window=(0, 1)),
    v5d.Display(b, name="filtered", window=(0, 1)),
)
viewer.wait()  # optional: wait for this window to close
```

`Volume(x)` retains the native tensor, including its device, dtype, strides, and
autograd graph. Computation uses `.tensor`; there are no operation hooks or
automatic metadata propagation rules.

`Volume(y, ref=a)` assigns the result the same outer spatial bounds and times as
`a`. A changed SHW size adjusts spacing and origin; it does not interpolate data.
T must match. Use explicit coordinates for results with another spatial mapping.

Device and memory changes return a new `Volume` with the same coordinates:

```python
cpu_result = b.detach().cpu()     # detach the graph, move to CPU if needed
owned_result = b.detach().clone() # also own independent tensor storage
```

If coordinates and individual display options are unnecessary, pass native
TCSHW tensors or NumPy arrays directly:

```python
viewer = v5d.view(x, y, window=(0, 1))
```

`view()` accepts any number of inputs. `Display` adds optional `name`, `window`,
`cmap`, and `rgb` settings to one input. Without a window, each scalar input uses
its own finite min/max over the entire TCSHW tensor. A common `window` enables
**Share window**; disabling sharing restores each input's individual setting.

Every call copies the full inputs to CPU snapshot files before returning a
process handle. Later tensor changes do not affect an open window. The window
can remain responsive while your Python debugger is paused; initial copying
and startup still take time. Use `viewer.close()` to close it from Python or
open a new snapshot with another `view()` call.

## Inspect native slices

- The HW, SW, and SH panels show directly indexed slices, with spacing reflected
  in their aspect ratios. **Nearest** is the default; **linear** is opt-in.
- Click or drag to move the linked crosshair. Scroll over a slice to move its
  perpendicular axis. Ctrl+wheel zooms; Shift+drag pans; **Fit** resets the view.
- Choose an input from the toolbar or press **Tab**. **A/B** select the first two
  inputs. Relative T/S/H/W position and zoom are shared across resolutions.
- Use the right panel for navigation, scalar window, colormap, and crosshair.
  **Share window** links scalar ranges; **Auto** uses the current input's range.
- **Planes**, **Volume**, and **Off** control the 3D panel. Volume rendering uses
  every voxel; its opacity control defaults to 0.15.
- RGB is explicit: `v5d.Display(color, rgb=True)` requires C=3. Scalar maps include
  `vispy:hot`; the optional colormaps extra adds `mpl:hot` and `sns:rocket`.

Names and axes are generic: units, world basis, and channel meaning belong to
your application. The viewer does not perform registration or infer anatomy.

## More

- [Usage](docs/usage.md): coordinates, tensor state, array exchange, and viewing.
- [Denoising example](examples/compare_denoising.py) and
  [resolution comparison](examples/compare_super_resolution.py): edit the device
  variable at the top, then run the script.
- [Development](docs/development.md): tests, benchmarks, API migration, and releases.
- [Optional interoperability](docs/interop.md).

MIT; see [LICENSE](LICENSE). Dependencies retain their respective licenses.
