"""Ownership, coordinate and conversion contracts of the data wrapper."""

from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
from itertools import combinations, permutations, product
import weakref

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


def test_reference_preserves_autograd_grid_and_can_change_channels_or_dtype():
    x = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    volume = Volume(x, spacing=(2, 3, 4), times=[0.5, 2.5])
    y = (x.square().sum(dim=1, keepdim=True)).to(torch.float64)
    result = Volume(y, ref=volume)
    assert result.tensor is y
    assert result.shape[1] == 1
    assert result.dtype == torch.float64
    assert result.spacing is volume.spacing
    assert result.origin is volume.origin
    assert result.direction is volume.direction
    assert result.times is volume.times
    result.tensor.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    with pytest.raises(ValueError, match="same T"):
        Volume(torch.zeros(1, 2, 3, 4, 5), ref=volume)


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
    result = Volume(x * 2, ref=volume)
    assert result.tensor.grad_fn is not None
    assert volume.to("cpu").device.type == "cpu"


def test_reference_defaults_and_same_grid_retain_tensor_metadata_and_graph():
    x = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    reference = Volume(x, spacing=(2, 3, 4), origin=(7, -3, 2), times=(4, 1))
    y = x[:, :1].square().to(torch.float64).transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not y.is_contiguous()
    result = Volume(y, ref=reference)
    assert result.tensor is y
    assert result.tensor.stride() == y.stride()
    assert result.dtype == y.dtype
    for name in ("spacing", "origin", "direction", "times"):
        assert getattr(result, name) is getattr(reference, name)
    result.tensor.sum().backward()
    expected = torch.zeros_like(x)
    expected[:, :1] = 2 * x.detach()[:, :1]
    torch.testing.assert_close(x.grad, expected)
    default = Volume(y)
    assert default.tensor is y
    assert default.spacing == (1.0, 1.0, 1.0)
    assert default.origin == (0.0, 0.0, 0.0)
    assert default.times == (0.0, 1.0)


@pytest.mark.parametrize("shape,output_shape", [((3, 4, 7), (6, 9, 2)), ((1, 4, 7), (5, 1, 1)), ((1, 1, 1), (3, 4, 5))])
@pytest.mark.parametrize("reflection", [False, True])
def test_reference_preserves_all_outer_corners_and_center(shape, output_shape, reflection):
    direction, _ = np.linalg.qr(np.array([[2.0, -1.0, 4.0], [3.0, 5.0, -2.0], [7.0, 1.0, 3.0]]))
    if reflection:
        direction[:, 0] *= -1
    reference = Volume(
        torch.zeros(2, 1, *shape), spacing=(2, 0.7, 3.5),
        origin=(11, -7, 2), direction=direction, times=(7, 2),
    )
    x = torch.randn(2, 3, *output_shape, dtype=torch.float64, requires_grad=True)
    result = Volume(x, ref=reference)
    assert result.tensor is x
    assert result.times is reference.times
    assert result.direction is reference.direction
    expected_spacing = np.array(reference.spacing) * shape / output_shape
    np.testing.assert_allclose(result.spacing, expected_spacing)
    reference_corners = list(product(*[(-0.5, size - 0.5) for size in shape]))
    result_corners = list(product(*[(-0.5, size - 0.5) for size in output_shape]))
    np.testing.assert_allclose(result.index_to_world(result_corners), reference.index_to_world(reference_corners), atol=1e-12)
    np.testing.assert_allclose(
        result.index_to_world((np.array(output_shape) - 1) / 2),
        reference.index_to_world((np.array(shape) - 1) / 2), atol=1e-12,
    )
    result.tensor.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_reference_validates_t_input_and_ref_without_retaining_reference_graph():
    leaf = torch.ones(2, 1, 3, 4, 5, requires_grad=True)
    source = leaf.square()
    source_handle, leaf_handle = weakref.ref(source), weakref.ref(leaf)
    reference = Volume(source)
    result = Volume(torch.zeros_like(source), ref=reference)
    with pytest.raises(ValueError, match="same T"):
        Volume(torch.zeros(1, 1, 3, 4, 5), ref=reference)
    with pytest.raises(TypeError, match="ref must be a Volume"):
        Volume(source, ref=source)
    del source, leaf, reference
    assert source_handle() is None
    assert leaf_handle() is None
    assert not result.tensor.requires_grad


@pytest.mark.parametrize(
    "metadata",
    [{"spacing": (1, 1, 1)}, {"origin": (0, 0, 0)}, {"direction": np.eye(3)}, {"times": None}, {"times": (0,)}],
)
def test_reference_cannot_be_combined_with_explicit_coordinates(metadata):
    reference = Volume(torch.zeros(1, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="ref cannot be combined"):
        Volume(reference.tensor, ref=reference, **metadata)
    # An absent reference leaves explicit coordinates and default times usable.
    volume = Volume(reference.tensor, ref=None, **metadata)
    assert volume.times == (0.0,)


def test_volume_has_only_five_stored_fields_and_one_tensor_constructor():
    assert [item.name for item in fields(Volume)] == ["tensor", "spacing", "origin", "direction", "times"]
    assert not hasattr(Volume, "from_tensor")
    assert not hasattr(Volume, "with_data")


def test_same_grid_reuse_does_not_repeat_coordinate_validation(monkeypatch):
    import vol5dkit.volume as module

    reference = Volume(torch.zeros(10000, 1, 1, 1, 1))

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("same-grid metadata must not be rescanned")

    monkeypatch.setattr(module, "_float_tuple", unexpected_validation)
    for result in (Volume(reference.tensor, ref=reference), reference.clone(), reference.to(torch.float64)):
        assert result.times is reference.times
        assert result.direction is reference.direction


def test_base_constructor_validates_coordinates_from_subclass_references():
    @dataclass(frozen=True, slots=True, eq=False)
    class UncheckedVolume(Volume):
        def __post_init__(self):
            pass

    tensor = torch.ones(1, 1, 2, 3, 4)
    invalid = UncheckedVolume(tensor, spacing=(1.0, -1.0, 1.0), times=(0.0,))
    with pytest.raises(ValueError, match="spacing must be positive"):
        Volume(tensor, ref=invalid)
    valid = UncheckedVolume(tensor, spacing=[2, 3, 4], times=[7])
    result = Volume(tensor, ref=valid)
    assert type(result) is Volume
    assert result.tensor is tensor
    assert result.spacing == (2.0, 3.0, 4.0)
    assert result.times == (7.0,)


@pytest.mark.parametrize("invalid", ["subclass", "sparse", "empty", "ndim"])
def test_same_grid_reuse_still_validates_native_dense_nonempty_tensor(invalid):
    reference = Volume(torch.zeros(1, 1, 2, 3, 4))
    data = {
        "subclass": lambda: torch.nn.Parameter(reference.tensor),
        "sparse": lambda: reference.tensor.to_sparse(),
        "empty": lambda: torch.empty(1, 0, 2, 3, 4),
        "ndim": lambda: torch.empty(1, 2, 3, 4),
    }[invalid]()
    with pytest.raises((TypeError, ValueError)):
        Volume(data, ref=reference)


def test_same_grid_methods_share_all_coordinate_tuples():
    reference = Volume(torch.ones(3, 1, 2, 3, 4), times=(1, 7, -3))
    for result in (reference.to(torch.float64), reference.clone(), Volume(reference.tensor.detach(), ref=reference)):
        assert result is not reference
        for name in ("spacing", "origin", "direction", "times"):
            assert getattr(result, name) is getattr(reference, name)
    # Check the time count even when unsupported external mutation changed T.
    reference.tensor.resize_(1, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="times must match T"):
        Volume(reference.tensor, ref=reference)


def test_subclass_extra_fields_and_validation_are_preserved():
    @dataclass(frozen=True, slots=True, eq=False)
    class LabeledVolume(Volume):
        label: str = field(default="result", kw_only=True)
        maximum: float = field(default=5.0, kw_only=True)

        def __post_init__(self):
            Volume.__post_init__(self)
            if self.tensor.max() > self.maximum:
                raise ValueError("maximum exceeded")

    reference = LabeledVolume(torch.ones(1, 1, 2, 3, 4), label="A", maximum=3)
    results = (
        replace(reference, tensor=reference.tensor + 1), reference.to(torch.float64), reference.clone(),
        reference.crop(s=slice(1, None)), reference.flip_spatial("s"), reference.permute_spatial("w", "s", "h"),
    )
    for result in results:
        assert type(result) is LabeledVolume
        assert result.label == "A"
        assert result.maximum == 3
    with pytest.raises(ValueError, match="maximum exceeded"):
        replace(reference, tensor=reference.tensor + 3)
    # Ordinary dataclass subclass constructors keep their generated signature.
    with pytest.raises(TypeError, match="ref"):
        LabeledVolume(reference.tensor, ref=reference)
    reference.tensor.fill_(4)
    with pytest.raises(ValueError, match="maximum exceeded"):
        reference.clone()
    with pytest.raises(ValueError, match="maximum exceeded"):
        reference.to(torch.float64)
    assert type(Volume(reference.tensor, ref=reference)) is Volume


@pytest.fixture(params=[(2, 3, 4), (1, 3, 1)])
def spatial_volume(request):
    shape = request.param
    x = torch.arange(4 * np.prod(shape), dtype=torch.float32).reshape(2, 2, *shape).requires_grad_()
    return Volume(
        x, spacing=(2, 0.7, 3.5), origin=(11, -7, 2), times=(2, 0),
        direction=((0, -1, 0), (-0.6, 0, -0.8), (-0.8, 0, 0.6)),
    )


@pytest.mark.parametrize("axes", [axes for count in range(4) for axes in combinations("shw", count)])
def test_all_spatial_flips_preserve_values_world_positions_and_autograd(spatial_volume, axes):
    reference = spatial_volume
    result = reference.flip_spatial(*axes)
    index_axes = tuple("shw".index(axis) for axis in axes)
    if axes:
        assert result.tensor.untyped_storage().data_ptr() != reference.tensor.untyped_storage().data_ptr()
    else:
        assert result is reference
    output_indices = np.indices(result.shape[2:]).reshape(3, -1).T
    input_indices = output_indices.copy()
    for axis in index_axes:
        input_indices[:, axis] = reference.shape[axis + 2] - 1 - input_indices[:, axis]
    np.testing.assert_allclose(result.index_to_world(output_indices), reference.index_to_world(input_indices), atol=1e-12)
    np.testing.assert_allclose(result.world_to_index(reference.index_to_world(input_indices)), output_indices, atol=1e-12)
    for output, original in zip(output_indices, input_indices):
        torch.testing.assert_close(result.tensor[(slice(None), slice(None), *output)], reference.tensor[(slice(None), slice(None), *original)])
    assert result.spacing is reference.spacing
    assert result.times is reference.times
    result.tensor.sum().backward()
    torch.testing.assert_close(reference.tensor.grad, torch.ones_like(reference.tensor))


@pytest.mark.parametrize("order", list(permutations("shw")))
def test_all_spatial_permutations_preserve_values_world_positions_and_storage(spatial_volume, order):
    reference = spatial_volume
    result = reference.permute_spatial(*order)
    index_axes = tuple("shw".index(axis) for axis in order)
    if order == ("s", "h", "w"):
        assert result is reference
    assert result.tensor.untyped_storage().data_ptr() == reference.tensor.untyped_storage().data_ptr()
    assert result.tensor.stride() == tuple(reference.tensor.stride()[axis] for axis in (0, 1, *(i + 2 for i in index_axes)))
    output_indices = np.indices(result.shape[2:]).reshape(3, -1).T
    input_indices = np.empty_like(output_indices)
    input_indices[:, index_axes] = output_indices
    np.testing.assert_allclose(result.index_to_world(output_indices), reference.index_to_world(input_indices), atol=1e-12)
    for output, original in zip(output_indices, input_indices):
        torch.testing.assert_close(result.tensor[(slice(None), slice(None), *output)], reference.tensor[(slice(None), slice(None), *original)])
    assert result.origin is reference.origin
    assert result.times is reference.times
    result.tensor.square().sum().backward()
    torch.testing.assert_close(reference.tensor.grad, 2 * reference.tensor.detach())


@pytest.mark.parametrize("axes", [("s", "s"), ("S",), ("t",), ("c",), (0,), (["s"],)])
def test_flip_rejects_duplicate_or_invalid_spatial_axes(axes):
    with pytest.raises(ValueError, match="spatial axes"):
        Volume(torch.zeros(1, 1, 2, 3, 4)).flip_spatial(*axes)


@pytest.mark.parametrize("order", [(), ("s",), ("s", "h"), ("s", "h", "h"), ("s", "h", "t"), (2, 3, 4), ("s", "h", "W")])
def test_permutation_rejects_incomplete_duplicate_or_invalid_spatial_axes(order):
    with pytest.raises(ValueError, match="spatial order"):
        Volume(torch.zeros(1, 1, 2, 3, 4)).permute_spatial(*order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_reference_and_spatial_operations_preserve_cuda_tensor_properties():
    x = torch.randn(2, 2, 3, 4, 5, device="cuda", requires_grad=True)
    reference = Volume(torch.zeros(x.shape), spacing=(2, 3, 4))
    result = Volume(x, ref=reference)
    assert result.tensor is x
    assert result.device.type == "cuda"
    transformed = result.flip_spatial("w").permute_spatial("w", "s", "h")
    assert transformed.device == x.device
    assert transformed.dtype == x.dtype
    transformed.tensor.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
