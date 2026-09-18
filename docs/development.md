# Development

## Setup and tests

Install the PyTorch build appropriate for your environment, then install the
editable package and development tools:

```bash
python -m pip install -e ".[gui,colormaps,dev]"
python -m pytest
```

GUI tests are opt-in and require working OpenGL. In PowerShell:

```powershell
$env:VOL5DKIT_TEST_GUI = "1"
python -m pytest
```

The actual breakpoint integration test also needs `debugpy>=1.8.20`. CUDA,
GUI, optional-colormap, debugger, and adapter cases skip if their requirements
are unavailable. The core and display-data tests do not need GUI imports.

With CuPy installed for your CUDA environment, run
`python -m pytest -q tests/test_volume.py -k cupy` to check DLPack storage sharing,
coordinates, and stream interoperability. CuPy remains an optional test dependency.

Core tests cover ownership, autograd, geometry, subclass behavior, and validation
reuse. Viewer tests cover exact range calculations, raw-value buffers, rendering,
per-input settings, stale worker results, 3D buffer reuse, and snapshot/process
cleanup. The debugpy test evaluates public `view()` at a real breakpoint and
checks rendering and navigation before allowing the parent to resume.

CI covers Python 3.10–3.13 on Linux, Python 3.11 on Windows, minimum Torch 2.3
compatibility, and GUI rendering under Xvfb/Mesa. Distribution checks build a
wheel from the sdist, install it in a fresh environment, and run
`tests/smoke_installed.py` outside the checkout with `PYTHONPATH` unset and
Python's `-I` flag. Tests should check behavior and ownership; timing ratios
are not CI pass/fail thresholds.

## Examples and benchmarks

The examples use public APIs only. Change their top-level device variable,
then run either script and close its window when finished:

```bash
python examples/compare_denoising.py
python examples/compare_super_resolution.py
```

The benchmark keeps setup, data preparation, and rendering costs separate:

```bash
python benchmarks/benchmark.py --device cpu --cases scalar256 --iterations 10
python benchmarks/benchmark.py --device cuda --cases scalar256 scalar512 rgb64 time128 --gui --output benchmark.json
python benchmarks/benchmark.py --device cuda --cases scalar256 --volume-3d
python benchmarks/benchmark.py --device cuda --core-only
```

Use `--help` for the case names and CLI options. Inputs are processed sequentially.
The public viewer always copies whole inputs into CPU snapshot files before
launch, even in slice mode; subsequent navigation prepares CPU data from those
snapshots. Report snapshot creation separately from CPU slice preparation.
`--gui` also measures child startup, first rendering, and navigation. Rendering
measurements include framebuffer readback and are not an interactive FPS claim.

`--volume-3d` compares initial/rebuilt 3D preparation with spatial navigation
that reuses the current normalized 3D buffer. Reuse is limited to one input,
T, C, and window; leaving scalar Volume mode releases it.

`--core-only` separates ordinary construction, same-grid reference construction,
spatial operations, and model forward. It varies T to expose metadata-validation
costs and compares direct native tensor input against `.tensor` input, checking
output and gradient equivalence. CUDA timings use warm-up and synchronization.
Always report device, versions, shape, dtype, and measurement boundaries with
results; do not compare startup and steady-state numbers as if they were equal.

## API migration

The constructor and display API now use one path for each operation. These are
intentional API changes; the old names and calling forms have no aliases.

| Previous form | Current form |
| --- | --- |
| `Volume.from_tensor(y)` | `Volume(y)` |
| `Volume.from_tensor(y, ref=a)` | `Volume(y, ref=a)` |
| `a.with_data(y)` | `Volume(y, ref=a)` |
| `(a, ViewOptions(name="A", win=(0, 1)))` | `Display(a, name="A", window=(0, 1))` |
| `view(a, b, win=(0, 1))` | `view(a, b, window=(0, 1))` |
| `view(a, b, clim=(0, 1))` | `view(a, b, window=(0, 1))` |
| `view(a, rgb=True)` | `view(Display(a, rgb=True))` |
| `view(a, b, block=True)` | `view(a, b).wait()` |
| `view(a, b, block=False)` | `view(a, b)` |
| `viewer.update(...)` | Open another snapshot with `view(...)` |
| Public `run()` | No event-loop call; optionally `viewer.wait()` |
| 3D voxel-step control | Removed; all voxels are used |

Unlike the former strict `with_data`, reference construction permits changed
SHW and preserves outer bounds and center. It still requires the same T.
When an operation requires unchanged SHW, assert that condition in the calling
code. A reference cannot be combined with explicit coordinate arguments.
Dataclass subclass constructors do not automatically acquire `ref`; their
existing transform methods retain fields and validation using `replace()`.

`view()` accepts any number of Volume, native Tensor, NumPy, and Display inputs.
Omit absent inputs rather than passing the old optional `b=None`. Tensor/array
inputs require nonempty TCSHW and use index coordinates. Display options belong
to each input; window is the only common scalar-range keyword.

## Build and release

The project uses `pyproject.toml` with setuptools; a separate `setup.py` is not
needed. Build both distributions and validate their metadata:

```bash
python -m build
python -m twine check --strict dist/*
```

Install the wheel into a clean environment and exercise the core and optional
GUI before publication. Building and checking artifacts does not publish them.
The package uses MIT; dependency licenses remain separate.

Development goes to `dev-internal`, then merges into `master`. Release tags
such as `v0.0.0.2` belong on the resulting master merge commit. GitHub Releases
include the wheel and source distribution. Keep the versions in `pyproject.toml`
and `src/vol5dkit/__init__.py` equal; the release workflow verifies them.

The `Publish to PyPI` workflow is manually dispatched. Configure the GitHub
`pypi` environment and a PyPI Trusted Publisher for owner `jiwoosong`, repository
`vol5dkit`, workflow `release.yml`, environment `pypi`. Publishing uses OIDC,
without a stored API token.
