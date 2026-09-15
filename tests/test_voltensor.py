"""Optional real-library compatibility tests (reference: VolTensor 3.8.1).

To run integration cases, install the reference or add its root to PYTHONPATH.
No stand-in Tensor class is used to claim compatibility with the reference.
"""

import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vol5dkit
from vol5dkit import Volume


@pytest.fixture
def reference():
    try:
        import voltensor
    except ModuleNotFoundError as exc:
        if exc.name == "voltensor":
            pytest.skip("optional VolTensor reference is not installed")
        raise
    return voltensor


def test_core_and_adapter_import_without_importing_optional_packages():
    code = """
import sys
import vol5dkit
import vol5dkit._voltensor
for name in ('voltensor', 'SimpleITK', 'nibabel', 'nrrd', 'torchio', 'PySide6', 'vispy'):
    assert name not in sys.modules, name
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(vol5dkit.__file__).parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_voltensor_reports_the_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def without_voltensor(name, *args, **kwargs):
        if name == "voltensor":
            raise ModuleNotFoundError("No module named 'voltensor'", name="voltensor")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_voltensor)
    volume = Volume(torch.zeros(1, 1, 1, 1, 1))
    with pytest.raises(ImportError, match="separately installed voltensor"):
        volume.to_voltensor()
    with pytest.raises(ImportError, match="separately installed voltensor"):
        Volume.from_voltensor(object())


def test_missing_nested_dependency_is_not_misreported(monkeypatch):
    original_import = builtins.__import__
    missing = ModuleNotFoundError("No module named 'SimpleITK'", name="SimpleITK")

    def broken_voltensor(name, *args, **kwargs):
        if name == "voltensor":
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_voltensor)
    with pytest.raises(ModuleNotFoundError) as exc:
        Volume(torch.zeros(1, 1, 1, 1, 1)).to_voltensor()
    assert exc.value is missing


def test_real_reference_round_trip_preserves_storage_geometry_and_autograd(reference):
    x = torch.randn(2, 2, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    y = (2 * x).transpose(-1, -2)
    direction = ((0.0, -1.0, 0.0), (0.6, 0.0, -0.8), (0.8, 0.0, 0.6))
    volume = Volume(y, spacing=(2, 3, 4), origin=(10, 20, 30), direction=direction, times=[0.2, 2.5])
    vt = volume.to_voltensor()
    assert isinstance(vt, reference.VolTensor)
    assert vt.as_tensor().data_ptr() == y.data_ptr()
    assert vt.as_tensor().stride() == y.stride()
    assert vt.vol.spacing == [2.0, 3.0, 4.0]
    assert vt.vol.origin == [30.0, 20.0, 10.0]
    assert vt.vol.source_meta is None
    assert vt.vol.base_tags == {}
    restored = Volume.from_voltensor(vt, times=volume.times)
    assert type(restored.tensor) is torch.Tensor
    assert restored.tensor.data_ptr() == y.data_ptr()
    assert restored.spacing == volume.spacing
    assert restored.origin == volume.origin
    assert restored.direction == volume.direction
    assert restored.times is volume.times
    restored.tensor.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))

    # This checks the real library's independent physical-coordinate API,
    # including asymmetric rotations and the XYZ versus SHW axis convention.
    for index in ([0, 0, 0], [1.5, 2, 3], [3, 5, 4]):
        point = vt.TransformContinuousIndexToPhysicalPoint(index[::-1], order="xyz")
        np.testing.assert_allclose(volume.index_to_world(index), point, atol=1e-12)
    cropped = volume.crop(s=slice(1, 4, 2), h=slice(1, None, 2), w=slice(2, None))
    cropped_vt = cropped.to_voltensor()
    np.testing.assert_allclose(
        cropped_vt.TransformContinuousIndexToPhysicalPoint([1, 1, 1], order="xyz"),
        cropped.index_to_world([1, 1, 1]),
        atol=1e-12,
    )


def test_real_reference_time_precedence_and_invalid_existing_time(reference):
    vt = reference.VolTensor(torch.zeros(3, 1, 2, 3, 4))
    assert Volume.from_voltensor(vt).times == (0.0, 1.0, 2.0)
    vt.vol.source_meta = SimpleNamespace(time_axis=(1.0, 1.4, 3.0))
    assert Volume.from_voltensor(vt).times == (1.0, 1.4, 3.0)
    assert Volume.from_voltensor(vt, times=[7, 2, 7]).times == (7.0, 2.0, 7.0)
    vt.vol.source_meta.time_axis = [1, 2]
    with pytest.raises(ValueError, match="times"):
        Volume.from_voltensor(vt)
    assert Volume.from_voltensor(vt, times=[4, 5, 6]).times == (4.0, 5.0, 6.0)
    with pytest.raises(ValueError, match="times"):
        Volume.from_voltensor(vt, times=[])


def test_real_reference_missing_geometry_defaults_independently(reference):
    vt = reference.VolTensor(torch.zeros(1, 1, 2, 3, 4))
    volume = Volume.from_voltensor(vt)
    assert volume.spacing == (1.0, 1.0, 1.0)
    assert volume.origin == (0.0, 0.0, 0.0)
    assert volume.direction == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    vt.vol.origin = [30, 20, 10]
    assert Volume.from_voltensor(vt).origin == (10.0, 20.0, 30.0)
    assert Volume.from_voltensor(vt).spacing == (1.0, 1.0, 1.0)
    del vt.vol.origin
    assert Volume.from_voltensor(vt).origin == (0.0, 0.0, 0.0)
    del vt.vol
    assert Volume.from_voltensor(vt).spacing == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("spacing", [1, 0, 1]),
        ("spacing", [1, 2]),
        ("origin", [1, 2]),
        ("origin", [1, 2, float("nan")]),
        ("direction", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("direction", [1, 0.2, 0, 0, 1, 0, 0, 0, 1]),
        ("direction", [1, 0, 0]),
    ],
)
def test_real_reference_malformed_geometry_is_not_replaced(reference, name, value):
    vt = reference.VolTensor(torch.zeros(1, 1, 2, 3, 4))
    setattr(vt.vol, name, value)
    with pytest.raises(ValueError):
        Volume.from_voltensor(vt)


def test_real_reference_rejects_unrelated_input(reference):
    with pytest.raises(TypeError, match="VolTensor"):
        Volume.from_voltensor(torch.zeros(1, 1, 2, 3, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_reference_cuda_round_trip(reference):
    x = torch.randn(1, 2, 3, 4, 5, device="cuda", requires_grad=True)
    volume = Volume(x, spacing=(2, 1, 0.5))
    restored = Volume.from_voltensor(volume.to_voltensor())
    assert restored.device == x.device
    assert restored.dtype == x.dtype
    assert restored.tensor.data_ptr() == x.data_ptr()
    restored.tensor.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
