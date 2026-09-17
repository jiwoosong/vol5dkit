"""Display-data correctness tests that do not import Qt or OpenGL."""

import numpy as np
import pytest
import torch

from vol5dkit import Volume
from vol5dkit.viewer._data import Request, make_source, prepare_frame, relative_index


def frame_for(tensor, *, rgb=False, **options):
    volume = tensor if isinstance(tensor, Volume) else Volume(tensor)
    return prepare_frame(
        Request(7, 0, make_source(volume, rgb), options.pop("positions", (0, 0.5, 0.5, 0.5)), **options)
    )


def test_relative_index_uses_half_up_rounding_and_singletons():
    assert [relative_index(p, 4) for p in (0, 1 / 6, 0.5, 5 / 6, 1)] == [0, 1, 2, 3, 3]
    assert [relative_index(p, 1) for p in (0, 0.5, 1)] == [0, 0, 0]
    position = 2 / 7
    assert [relative_index(position, n) for n in (8, 101, 8, 101, 8)] == [2, 29, 2, 29, 2]


@pytest.mark.parametrize("position,length", [(float("nan"), 3), (-0.01, 3), (1.01, 3), (0.5, 0), (0.5, 1.5)])
def test_invalid_relative_indices(position, length):
    with pytest.raises(ValueError):
        relative_index(position, length)


def test_source_uses_loaded_cpu_snapshot_without_another_wrapper_or_copy():
    original = Volume(torch.ones((1, 1, 2, 3, 4)))
    source = make_source(original)
    assert source.volume is original


def test_display_preparation_rejects_non_cpu_sources():
    with pytest.raises(ValueError, match="CPU snapshot"):
        make_source(Volume(torch.empty((1, 1, 2, 3, 4), device="meta")))


def test_native_plane_orientation_and_owned_cpu_buffers():
    # Values encode T, C, S, H, W independently, so transposed planes fail.
    t, c, s, h, w = torch.meshgrid(
        torch.arange(2), torch.arange(2), torch.arange(3), torch.arange(4), torch.arange(5), indexing="ij"
    )
    tensor = 10000 * t + 1000 * c + 100 * s + 10 * h + w
    frame = frame_for(tensor, positions=(1, 0.5, 1 / 3, 0.75), channel=1)
    assert frame.indices == (1, 1, 1, 1, 3)
    assert [p.name for p in frame.planes] == ["HW", "SW", "SH"]
    assert frame.planes[0].raw.tolist() == [[11100 + 10 * i + j for j in range(5)] for i in range(4)]
    assert frame.planes[1].raw.tolist() == [[11010 + 100 * i + j for j in range(5)] for i in range(3)]
    assert frame.planes[2].raw.tolist() == [[11003 + 100 * i + 10 * j for j in range(4)] for i in range(3)]
    before = [p.raw.clone() for p in frame.planes]
    images = [p.image.copy() for p in frame.planes]
    tensor.zero_()
    for plane, raw, image in zip(frame.planes, before, images):
        assert torch.equal(plane.raw, raw)
        np.testing.assert_array_equal(plane.image, image)
        assert plane.raw.is_contiguous()
        assert plane.image.flags.c_contiguous
    assert frame.volume_buffer is None


def test_noncontiguous_tensor_keeps_native_values():
    x = torch.arange(120, dtype=torch.float32).reshape(1, 1, 4, 5, 6).transpose(-1, -2)
    frame = frame_for(x, positions=(0, 0, 0, 0))
    assert frame.planes[0].raw.shape == (6, 5)
    assert torch.equal(frame.planes[0].raw, x[0, 0, 0])


def test_initial_window_uses_entire_time_and_channel_extent():
    x = torch.arange(48, dtype=torch.float64).reshape(2, 2, 2, 2, 3)
    frame = frame_for(x, positions=(1, 0, 0, 0), channel=0)
    assert frame.window == (0.0, 47.0)
    # Explicit display limits survive a different selected frame.
    updated = frame_for(x, positions=(0, 0, 0, 0), window=frame.window)
    assert updated.window == frame.window
    np.testing.assert_allclose(updated.planes[0].image, x[0, 0, 0].numpy() / 47)


def test_shared_autorange_uses_bound_input_without_changing_displayed_pixels():
    displayed = Volume(torch.full((1, 1, 2, 3, 4), 0.5))
    limits = torch.ones(2, 2, 2, 3, 4)
    limits[1, 1, -1, -1, -1] = -1
    bound = make_source(Volume(limits))
    frame = frame_for(displayed, range_source=bound)
    assert frame.window == (-1, 1)
    assert frame.source.volume.tensor.data_ptr() == displayed.tensor.data_ptr()
    assert frame.planes[0].raw.eq(0.5).all()
    np.testing.assert_array_equal(frame.planes[0].image, 0.75)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint16, torch.uint32, torch.uint64])
def test_integer_autorange_uses_bounded_views_and_exact_endpoints(monkeypatch, dtype):
    from vol5dkit.viewer import _data
    monkeypatch.setattr(_data, "_CHUNK_ELEMENTS", 7)
    maximum = 1 if dtype == torch.bool else torch.iinfo(dtype).max
    # A stepped/transposed input must not be flattened to an entire copy.
    array = np.full((2, 2, 3, 4, 8), maximum, dtype=str(dtype).removeprefix("torch."))
    array[0, 0, 0, 0, 0] = maximum - 1
    tensor = torch.from_numpy(array)[..., ::2].transpose(-1, -2)
    assert not tensor.is_contiguous()
    chunks = [chunk for _, chunk in _data._chunks(tensor)]
    assert sum(chunk.numel() for chunk in chunks) == tensor.numel()
    assert all(chunk.numel() <= 7 for chunk in chunks)
    assert all(chunk.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr() for chunk in chunks)
    calls = []
    aminmax = torch.aminmax

    def bounded_aminmax(chunk):
        calls.append(chunk.numel())
        assert chunk.numel() <= 7
        return aminmax(chunk)

    monkeypatch.setattr(torch, "aminmax", bounded_aminmax)
    assert _data._finite_range(tensor) == (maximum - 1, maximum)
    assert len(calls) > 1


def test_nonfinite_autorange_bounds_mask_allocations(monkeypatch):
    from vol5dkit.viewer import _data
    monkeypatch.setattr(_data, "_CHUNK_ELEMENTS", 7)
    x = torch.full((2, 2, 3, 4, 8), float("nan"), dtype=torch.float64)
    x[0, 1, 0, 0, 0] = -13
    x[1, 0, 2, 3, 6] = 29
    x[0, 0, 0, 0, 0] = float("inf")
    x[1, 1, 0, 0, 0] = -float("inf")
    tensor = x[..., ::2].transpose(-1, -2)
    isfinite = torch.isfinite
    calls = []

    def bounded_isfinite(chunk):
        calls.append(chunk.numel())
        assert chunk.numel() <= 7
        return isfinite(chunk)

    monkeypatch.setattr(torch, "isfinite", bounded_isfinite)
    assert _data._finite_range(tensor) == (-13, 29)
    assert len(calls) > 1


def test_float64_small_contrast_is_windowed_before_float32_conversion():
    x = torch.tensor([1000 + n * 1e-8 for n in range(4)], dtype=torch.float64).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x)
    assert frame.planes[0].raw.dtype == torch.float64
    assert frame.planes[0].image.dtype == np.float32
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 1 / 3, 2 / 3, 1], atol=5e-6)


@pytest.mark.parametrize("offset", [2**60, -(2**60)])
def test_wide_integer_low_bits_survive_windowing(offset):
    x = torch.tensor([offset + n for n in range(4)], dtype=torch.int64).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x)
    assert frame.window == (offset, offset + 3)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 1 / 3, 2 / 3, 1])


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64])
def test_integer_full_range_is_monotonic(dtype):
    info = torch.iinfo(dtype)
    values = [info.min, info.min + (info.max - info.min) // 2, info.max]
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x)
    assert frame.window == (info.min, info.max)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 0.5, 1], atol=0.003)


def test_uint64_low_bits_above_signed_range_survive_windowing():
    offset = 2**64 - 4
    x = torch.tensor([offset + n for n in range(4)], dtype=torch.uint64).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x)
    assert frame.window == (offset, offset + 3)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 1 / 3, 2 / 3, 1])


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_integer_narrow_window_can_extend_outside_dtype_range(dtype, side):
    info = torch.iinfo(dtype)
    if side == "lower":
        values, window = [info.min + n for n in range(4)], (info.min - 1, info.min + 3)
    else:
        values, window = [info.max - 3 + n for n in range(4)], (info.max - 3, info.max + 1)
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x, window=window)
    expected = [(v - window[0]) / (window[1] - window[0]) for v in values]
    np.testing.assert_allclose(frame.planes[0].image[0], expected)


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("window", ["wide", "huge", "below", "above"])
def test_integer_windows_wider_than_or_disjoint_from_dtype_range(dtype, window):
    info = torch.iinfo(dtype)
    values = [info.min, info.min + (info.max - info.min) // 2, info.max]
    windows = {
        "wide": (info.min - 2**65, info.max + 2**65),
        "huge": (-(10**308), 10**308),
        "below": (info.min - 4, info.min - 1),
        "above": (info.max + 1, info.max + 4),
    }
    lo, hi = windows[window]
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x, window=(lo, hi))
    expected = [(min(max(v, lo), hi) - lo) / (hi - lo) for v in values]
    np.testing.assert_allclose(frame.planes[0].image[0], expected)


def test_numpy_integer_limit_width_does_not_overflow():
    lo, hi = np.int64(-(2**63)), np.int64(2**63 - 1)
    x = torch.tensor([int(lo), 0, int(hi)], dtype=torch.int64).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x, window=(lo, hi))
    np.testing.assert_array_equal(frame.planes[0].image[0], [0, 0.5, 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.bool])
def test_real_scalar_display_dtypes(dtype):
    x = torch.ones((1, 1, 2, 3, 4), dtype=dtype)
    frame = frame_for(x)
    assert frame.planes[0].raw.dtype == dtype
    assert np.all(frame.planes[0].image == 0.5)


def test_constant_nonfinite_and_empty_finite_range():
    x = torch.tensor([3, float("nan"), float("inf"), -float("inf")]).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x)
    assert frame.window == (3, 3)
    assert frame.planes[0].image[0, 0] == 0.5
    assert np.isnan(frame.planes[0].image[0, 1:]).all()
    x.fill_(float("nan"))
    invalid = frame_for(x)
    assert invalid.window == (0, 1)
    assert np.isnan(invalid.planes[0].image).all()


def test_extreme_finite_float64_limits_do_not_overflow_window():
    x = torch.tensor([-1e308, 0, 1e308], dtype=torch.float64).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x)
    np.testing.assert_array_equal(frame.planes[0].image, [[0, 0.5, 1]])


def test_rgb_plane_channel_order_and_uint8_normalization():
    x = torch.zeros((1, 3, 2, 3, 4), dtype=torch.uint8)
    x[:, 0] = 255
    x[:, 1] = 128
    frame = frame_for(x, rgb=True)
    assert frame.window is None
    assert frame.planes[0].raw.shape == (3, 4, 3)
    np.testing.assert_allclose(frame.planes[0].image[0, 0], [1, 128 / 255, 0])
    assert frame.planes[0].raw[0, 0].tolist() == [255, 128, 0]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64, torch.bfloat16])
def test_rgb_image_owns_storage_separately_from_raw_and_input(dtype):
    x = torch.ones((1, 3, 2, 3, 4), dtype=dtype)
    frame = frame_for(x, rgb=True)
    plane = frame.planes[0]
    image = plane.image.copy()
    x.zero_()
    plane.raw.zero_()
    np.testing.assert_array_equal(plane.image, image)
    assert plane.image.dtype == np.float32
    assert plane.image.flags.c_contiguous


def test_rgb_validation_is_local_to_requested_pixels():
    x = torch.zeros((2, 3, 2, 3, 4), dtype=torch.float64)
    x[1] = 2
    source = make_source(Volume(x), rgb=True)  # no full tensor range scan
    prepare_frame(Request(0, 0, source, (0, 0, 0, 0)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_frame(Request(1, 0, source, (1, 0, 0, 0)))
    x[0, 0, 0, 0, 0] = float("nan")
    invalid = prepare_frame(Request(2, 0, source, (0, 0, 0, 0)))
    assert np.isnan(invalid.planes[0].image[0, 0]).all()


@pytest.mark.parametrize("shape,dtype,rgb", [((1, 1, 2, 2, 2), torch.complex64, False), ((1, 1, 2, 2, 2), torch.float32, True), ((1, 3, 2, 2, 2), torch.int32, True)])
def test_invalid_display_sources(shape, dtype, rgb):
    with pytest.raises((TypeError, ValueError)):
        make_source(Volume(torch.zeros(shape, dtype=dtype)), rgb)


def test_opt_in_volume_uses_selected_time_and_channel_at_full_resolution():
    x = torch.arange(2 * 2 * 5 * 7 * 9, dtype=torch.float32).reshape(2, 2, 5, 7, 9)
    frame = frame_for(x, positions=(1, 0.5, 0.5, 0.5), channel=1, volume_3d=True)
    expected = (x[1, 1].numpy() - frame.window[0]) / (frame.window[1] - frame.window[0])
    assert frame.volume_buffer.shape == (5, 7, 9)
    np.testing.assert_allclose(frame.volume_buffer, expected)
    assert frame.planes[0].raw.shape == (7, 9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.uint64, torch.bfloat16])
def test_volume_normalization_uses_bounded_chunks_without_copying_raw_volume(monkeypatch, dtype):
    from vol5dkit.viewer import _data

    monkeypatch.setattr(_data, "_CHUNK_ELEMENTS", 7)
    values = torch.arange(2 * 3 * 4).reshape(1, 1, 2, 3, 4)
    if dtype == torch.float64:
        x = values.double() * 1e-8 + 1000
    elif dtype == torch.uint64:
        x = torch.tensor([2**64 - 24 + i for i in range(24)], dtype=dtype).reshape(values.shape)
    elif dtype == torch.int64:
        x = values + 2**60
    else:
        x = values.to(dtype)
    x = x.transpose(-1, -2)
    visited = []
    original_chunks = _data._chunks

    def bounded_chunks(raw):
        if raw.ndim == 3:
            assert raw.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for index, chunk in original_chunks(raw):
            assert chunk.numel() <= 7
            visited.append((raw.ndim, chunk.numel()))
            yield index, chunk

    monkeypatch.setattr(_data, "_chunks", bounded_chunks)
    frame = frame_for(x, volume_3d=True)
    expected = values[0, 0].transpose(-1, -2).float().numpy() / 23
    np.testing.assert_allclose(frame.volume_buffer, expected, atol=5e-6)
    assert sum(n for ndim, n in visited if ndim == 3) == x.numel()
    assert frame.volume_buffer.dtype == np.float32
    assert frame.volume_buffer.flags.c_contiguous


def test_volume_nonfinite_voxels_are_background_while_slices_keep_nan(monkeypatch):
    from vol5dkit.viewer import _data

    monkeypatch.setattr(_data, "_CHUNK_ELEMENTS", 2)
    x = torch.tensor([1, float("nan"), float("inf"), -float("inf")]).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x, volume_3d=True)
    np.testing.assert_array_equal(frame.volume_buffer, [[[0.5, 0, 0, 0]]])
    assert np.isnan(frame.planes[0].image[0, 1:]).all()


def test_spatial_navigation_reuses_supplied_volume_buffer_without_normalizing_again(monkeypatch):
    from vol5dkit.viewer import _data

    source = make_source(Volume(torch.arange(120).reshape(1, 1, 4, 5, 6)))
    first = prepare_frame(Request(0, 0, source, (0, 0, 0, 0), volume_3d=True))
    normalize = _data._window

    def planes_only(raw, window, **kwargs):
        assert raw.ndim == 2
        return normalize(raw, window, **kwargs)

    monkeypatch.setattr(_data, "_window", planes_only)
    request = Request(1, 0, source, (0, 1, 1, 1), window=first.window, volume_3d=True)
    moved = prepare_frame(request, volume_buffer=first.volume_buffer)
    assert moved.volume_buffer is first.volume_buffer
    assert moved.indices == (0, 0, 3, 4, 5)
    assert moved.planes[0].raw[0, 0].item() == 90
    planes = prepare_frame(
        Request(2, 0, source, (0, 1, 1, 1), window=first.window),
        volume_buffer=first.volume_buffer,
    )
    assert planes.volume_buffer is None


@pytest.mark.parametrize("options", [{"channel": -1}, {"channel": 1}, {"window": (1, 0)}, {"window": (0, float("inf"))}])
def test_invalid_requests(options):
    with pytest.raises((TypeError, ValueError)):
        frame_for(torch.ones((1, 1, 2, 3, 4)), **options)
