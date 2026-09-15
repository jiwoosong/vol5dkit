"""Ownership, coordinate and conversion contracts of the data wrapper."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from vol5dkit import Volume


def test_native_tensor_and_metadata_are_retained_without_copy_or_detach():
    x = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    x = x.transpose(-1, -2).requires_grad_()
    spacing = (2.0, 1.5, 0.5)
    origin = (3.0, -2.0, 7.0)
    direction = ((0.0, -1.0, 0.0), (0.6, 0.0, -0.8), (0.8, 0.0, 0.6))
    times = (7.0, 3.0)
    volume = Volume(x, spacing=spacing, origin=origin, direction=direction, times=times)
    assert volume.tensor is x
    assert volume.tensor.stride() == x.stride()
    assert volume.spacing is spacing
    assert volume.origin is origin
    assert volume.direction is direction
    assert volume.times is times
    assert volume.shape == x.shape
    assert volume.dtype == x.dtype
    assert volume.device == x.device
    assert not volume.tensor.is_contiguous()
    with pytest.raises(FrozenInstanceError):
        volume.tensor = x.clone()
    with pytest.raises(AttributeError):
        volume.__dict__


def test_coordinate_inputs_are_normalized_and_default_times_are_indices():
    volume = Volume(
        torch.zeros(3, 1, 2, 3, 4),
        spacing=[2, 3, 4],
        origin=np.array([1, 2, 3]),
        direction=np.diag([-1.0, 1.0, 1.0]),
    )
    assert volume.spacing == (2.0, 3.0, 4.0)
    assert all(type(x) is float for x in volume.spacing + volume.origin + volume.times)
    assert volume.times == (0.0, 1.0, 2.0)
    assert volume.direction[0] == (-1.0, 0.0, 0.0)
    assert isinstance(volume.direction, tuple)
    assert "tensor(" not in repr(volume)
    assert "shape=(3, 1, 2, 3, 4)" in repr(volume)
    assert "time_count=3" in repr(volume)


@pytest.mark.parametrize(
    "metadata",
    [
        {"spacing": (1, 0, 1)},
        {"spacing": (-1, 1, 1)},
        {"spacing": (1, 1)},
        {"spacing": (1, float("inf"), 1)},
        {"origin": (0, float("nan"), 0)},
        {"origin": 0},
        {"direction": ((1, 0.1, 0), (0, 1, 0), (0, 0, 1))},
        {"direction": ((1, 0, 0), (0, 1, 0))},
        {"direction": (1, 0, 0)},
        {"direction": ((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1))},
        {"times": (0.0,)},
        {"times": (0.0, float("inf"))},
        {"times": 3},
    ],
)
def test_invalid_metadata_is_rejected(metadata):
    with pytest.raises(ValueError):
        Volume(torch.zeros(2, 1, 3, 4, 5), **metadata)


def test_duplicate_and_nonmonotonic_times_are_preserved():
    assert Volume(torch.zeros(3, 1, 1, 1, 1), times=[2, 0, 2]).times == (2.0, 0.0, 2.0)


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 1, 0, 3, 4), (0, 1, 2, 3, 4)])
def test_only_nonempty_five_dimensional_data_is_accepted(shape):
    with pytest.raises(ValueError, match="T,C,S,H,W"):
        Volume(torch.zeros(shape))


def test_sparse_tensor_subclass_and_implicit_forwarding_are_rejected():
    x = torch.zeros(1, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="dense"):
        Volume(x.to_sparse())
    with pytest.raises(TypeError, match="subclasses"):
        Volume(torch.nn.Parameter(x))
    with pytest.raises(TypeError):
        Volume([[[[[1]]]]])
    volume = Volume(x)
    with pytest.raises(TypeError):
        volume[0]
    with pytest.raises(TypeError):
        volume + 1
    with pytest.raises(AttributeError):
        volume.clamp


def test_with_data_preserves_autograd_grid_and_can_change_channels_or_dtype():
    x = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    volume = Volume(x, spacing=(2, 3, 4), times=[0.5, 2.5])
    y = (x.square().sum(dim=1, keepdim=True)).to(torch.float64)
    result = volume.with_data(y)
    assert result.tensor is y
    assert result.shape[1] == 1
    assert result.dtype == torch.float64
    assert result.spacing is volume.spacing
    assert result.origin is volume.origin
    assert result.direction is volume.direction
    assert result.times is volume.times
    result.tensor.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    for shape in [(1, 2, 3, 4, 5), (2, 2, 6, 4, 5)]:
        with pytest.raises(ValueError, match="same T and SHW"):
            volume.with_data(torch.zeros(shape))


def test_clone_and_to_follow_tensor_semantics():
    x = torch.ones(1, 1, 2, 3, 4, requires_grad=True)
    volume = Volume(x)
    unchanged = volume.to("cpu")
    assert unchanged.tensor is x
    copied = volume.clone()
    assert copied.tensor.data_ptr() != x.data_ptr()
    assert copied.tensor.grad_fn is not None
    converted = volume.to(dtype=torch.float64)
    assert converted.dtype == torch.float64
    assert converted.spacing is volume.spacing
    converted.tensor.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_numpy_sharing_noncontiguous_inputs_and_explicit_copy():
    array = np.arange(2 * 3 * 4, dtype=np.float64).reshape(1, 1, 2, 3, 4)
    array = array.swapaxes(-1, -2)
    volume = Volume(array)
    assert np.shares_memory(volume.numpy(), array)
    volume.tensor[0, 0, 0, 0, 0] = 71
    assert array[0, 0, 0, 0, 0] == 71
    copied = volume.numpy(copy=True)
    assert not np.shares_memory(copied, array)
    copied[...] = -1
    assert array[0, 0, 0, 0, 0] == 71
    volume.tensor.requires_grad_()
    assert np.shares_memory(volume.numpy(), array)
    with pytest.raises(ValueError, match="negative strides"):
        Volume(array[..., ::-1])
    array.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        Volume(array)
    with pytest.raises(TypeError):
        Volume(torch.zeros(1, 1, 2, 3, 4, dtype=torch.bfloat16)).numpy()


def test_numpy_conversion_resolves_conjugate_values():
    x = torch.full((1, 1, 1, 1, 2), 1 + 2j, dtype=torch.complex64)
    np.testing.assert_array_equal(Volume(x.conj()).numpy(), np.full(x.shape, 1 - 2j))


def test_dlpack_shares_storage_without_graph():
    x = torch.ones(1, 1, 2, 3, 4)
    volume = Volume.from_dlpack(x, spacing=(2, 3, 4))
    assert volume.tensor.data_ptr() == x.data_ptr()
    assert volume.spacing == (2.0, 3.0, 4.0)
    volume.tensor.add_(2)
    torch.testing.assert_close(x, torch.full_like(x, 3))


def test_continuous_coordinates_match_the_defined_transform_and_round_trip():
    direction = np.array([[0, -1, 0], [0.6, 0, -0.8], [0.8, 0, 0.6]])
    volume = Volume(
        torch.zeros(2, 1, 8, 9, 10),
        spacing=(2, 3, 4),
        origin=(10, 20, 30),
        direction=direction,
    )
    indices = np.array([[[0.0, 0.0, 0.0], [1.5, 2.0, -1.0]], [[3, 4, 5], [9, 12, 15]]])
    expected = np.einsum("ij,...j->...i", direction, (indices * [2, 3, 4])[..., ::-1]) + [10, 20, 30]
    np.testing.assert_allclose(volume.index_to_world(indices), expected)
    np.testing.assert_allclose(volume.world_to_index(expected), indices, atol=1e-12)
    assert volume.index_to_world([0, 0, 0]).dtype == np.float64
    with pytest.raises(ValueError, match="shape"):
        volume.index_to_world([1, 2])
    with pytest.raises(ValueError, match="shape"):
        volume.world_to_index(1)


def test_crop_stride_preserves_world_coordinates_times_storage_and_autograd():
    x = torch.arange(4 * 3 * 8 * 9 * 10, dtype=torch.float32).reshape(4, 3, 8, 9, 10).requires_grad_()
    volume = Volume(
        x,
        spacing=(2, 3, 4),
        origin=(10, 20, 30),
        direction=((0, -1, 0), (0.6, 0, -0.8), (0.8, 0, 0.6)),
        times=[0, 0.1, 0.4, 1.2],
    )
    cropped = volume.crop(t=slice(1, None, 2), c=slice(1, 2), s=slice(1, 8, 2), h=slice(-7, 8, 3), w=slice(2, 9, 2))
    assert cropped.shape == (2, 1, 4, 2, 4)
    assert cropped.times == (0.1, 1.2)
    assert cropped.spacing == (4.0, 9.0, 8.0)
    assert cropped.tensor.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    for index in ([0, 0, 0], [1.5, 1, 3]):
        original_index = np.array(index) * [2, 3, 2] + [1, 2, 2]
        np.testing.assert_allclose(cropped.index_to_world(index), volume.index_to_world(original_index))
    cropped.tensor.sum().backward()
    expected = torch.zeros_like(x)
    expected[1::2, 1:2, 1:8:2, 2:8:3, 2:9:2] = 1
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("selection", [slice(2, 2), slice(4, 1), slice(None, None, -1), slice(None, None, 0)])
def test_crop_rejects_empty_or_nonpositive_strides(selection):
    with pytest.raises(ValueError):
        Volume(torch.zeros(1, 1, 5, 6, 7)).crop(s=selection)


def test_crop_integer_index_is_rejected_and_full_slice_is_valid():
    volume = Volume(torch.zeros(1, 1, 1, 1, 1))
    with pytest.raises(TypeError, match="slice"):
        volume.crop(s=0)
    assert volume.crop().shape == volume.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_storage_autograd_numpy_and_dlpack():
    x = torch.randn(2, 3, 4, 5, 6, device="cuda", requires_grad=True).transpose(-1, -2)
    volume = Volume(x)
    assert volume.tensor is x
    assert volume.device.type == "cuda"
    array = volume.numpy()
    np.testing.assert_array_equal(array, x.detach().cpu().numpy())
    array[...] = 0
    assert torch.count_nonzero(x).item() > 0
    shared = Volume.from_dlpack(x.detach())
    assert shared.tensor.data_ptr() == x.data_ptr()
    result = volume.with_data(x * 2)
    assert result.tensor.grad_fn is not None
    assert volume.to("cpu").device.type == "cpu"
