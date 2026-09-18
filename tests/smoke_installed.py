"""Run with an installed wheel, outside the checkout and without PYTHONPATH."""

import importlib.metadata
import os
from pathlib import Path
import sys

import numpy as np
import torch

import vol5dkit as v5d
import vol5dkit.viewer._data


package_path = Path(v5d.__file__).resolve()
distribution = importlib.metadata.distribution("vol5dkit")
assert "PYTHONPATH" not in os.environ
assert Path(sys.prefix).resolve() in package_path.parents, package_path
assert package_path == Path(distribution.locate_file("vol5dkit/__init__.py")).resolve()
assert v5d.__version__ == distribution.version

tensor = torch.arange(48.0).reshape(2, 1, 2, 3, 4).requires_grad_()
volume = v5d.Volume(tensor, spacing=(2, 3, 4), origin=(5, 6, 7), times=(0, 0.5))
assert volume.tensor is tensor
result = v5d.Volume(volume.tensor * 2, ref=volume)
for name in ("spacing", "origin", "direction", "times"):
    assert getattr(result, name) is getattr(volume, name)
result.tensor.sum().backward()
assert torch.equal(tensor.grad, torch.full_like(tensor, 2))
np.testing.assert_allclose(volume.world_to_index(volume.index_to_world((1, 2, 3))), (1, 2, 3))

detached = result.detach()
assert not detached.tensor.requires_grad
assert detached.tensor.data_ptr() == result.tensor.data_ptr()
assert detached.cpu().tensor is detached.tensor
assert detached.contiguous().tensor is detached.tensor
assert detached.clone().tensor.data_ptr() != detached.tensor.data_ptr()
assert detached.numpy().ctypes.data == detached.tensor.data_ptr()
assert not np.shares_memory(detached.numpy(copy=True), detached.numpy())

display = v5d.Display(volume, name="smoke", window=(0, 47))
assert display.window == (0, 47) and callable(v5d.view)
try:
    handle = v5d.view(display)
except ImportError as exc:
    assert "gui extra" in str(exc), exc
else:
    handle.close()
    raise AssertionError("core-only installation must reject the viewer before launch")
for name in ("PySide6", "vispy", "OpenGL", "matplotlib", "seaborn", "voltensor"):
    assert name not in sys.modules, name
assert not torch.cuda.is_initialized()
print(f"Installed wheel smoke passed: {package_path}")
