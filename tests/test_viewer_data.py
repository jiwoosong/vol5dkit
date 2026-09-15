"""Display-data correctness tests that do not import Qt or OpenGL."""

from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import numpy as np
import pytest
import torch

from vol5dkit import Volume
from vol5dkit.viewer._data import Request, make_source, prepare_frame, relative_index


def frame_for(tensor, *, rgb=False, **options):
    volume = tensor if isinstance(tensor, Volume) else Volume(tensor)
    return prepare_frame(
        Request(7, "A", make_source(volume, rgb), options.pop("positions", (0, 0.5, 0.5, 0.5)), **options)
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


def test_source_detaches_without_retaining_original_graph_or_tensor_object():
    leaf = torch.ones((1, 1, 2, 3, 4), requires_grad=True)
    intermediate = leaf * 3
    intermediate_ref = weakref.ref(intermediate)
    original = Volume(intermediate)
    source = make_source(original)
    assert source.volume.tensor.data_ptr() == intermediate.data_ptr()
    assert source.volume.tensor.grad_fn is None
    assert not source.volume.tensor.requires_grad
    del original, intermediate
    gc.collect()
    assert intermediate_ref() is None
    assert source.volume.spacing == (1, 1, 1)


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
    assert frame.volume is None
    assert frame.transfer_bytes == 0


def test_noncontiguous_tensor_keeps_native_values():
    x = torch.arange(120, dtype=torch.float32).reshape(1, 1, 4, 5, 6).transpose(-1, -2)
    frame = frame_for(x, positions=(0, 0, 0, 0))
    assert frame.planes[0].raw.shape == (6, 5)
    assert torch.equal(frame.planes[0].raw, x[0, 0, 0])


def test_initial_clim_uses_only_selected_time_and_channel():
    x = torch.arange(48, dtype=torch.float64).reshape(2, 2, 2, 2, 3)
    frame = frame_for(x, positions=(1, 0, 0, 0), channel=0)
    assert frame.clim == (24.0, 35.0)
    # Explicit display limits survive a different selected frame.
    updated = frame_for(x, positions=(0, 0, 0, 0), clim=frame.clim)
    assert updated.clim == frame.clim
    assert np.all(updated.planes[0].image == 0)


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
    assert frame.clim == (offset, offset + 3)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 1 / 3, 2 / 3, 1])


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64])
def test_integer_full_range_is_monotonic(dtype):
    info = torch.iinfo(dtype)
    values = [info.min, info.min + (info.max - info.min) // 2, info.max]
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x)
    assert frame.clim == (info.min, info.max)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 0.5, 1], atol=0.003)


def test_uint64_low_bits_above_signed_range_survive_windowing():
    offset = 2**64 - 4
    x = torch.tensor([offset + n for n in range(4)], dtype=torch.uint64).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x)
    assert frame.clim == (offset, offset + 3)
    np.testing.assert_allclose(frame.planes[0].image[0], [0, 1 / 3, 2 / 3, 1])


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_integer_narrow_window_can_extend_outside_dtype_range(dtype, side):
    info = torch.iinfo(dtype)
    if side == "lower":
        values, clim = [info.min + n for n in range(4)], (info.min - 1, info.min + 3)
    else:
        values, clim = [info.max - 3 + n for n in range(4)], (info.max - 3, info.max + 1)
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 4)
    frame = frame_for(x, clim=clim)
    expected = [(v - clim[0]) / (clim[1] - clim[0]) for v in values]
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
    frame = frame_for(x, clim=(lo, hi))
    expected = [(min(max(v, lo), hi) - lo) / (hi - lo) for v in values]
    np.testing.assert_allclose(frame.planes[0].image[0], expected)


def test_numpy_integer_limit_width_does_not_overflow():
    lo, hi = np.int64(-(2**63)), np.int64(2**63 - 1)
    x = torch.tensor([int(lo), 0, int(hi)], dtype=torch.int64).reshape(1, 1, 1, 1, 3)
    frame = frame_for(x, clim=(lo, hi))
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
    assert frame.clim == (3, 3)
    assert frame.planes[0].image[0, 0] == 0.5
    assert np.isnan(frame.planes[0].image[0, 1:]).all()
    x.fill_(float("nan"))
    invalid = frame_for(x)
    assert invalid.clim == (0, 1)
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
    assert frame.clim is None
    assert frame.planes[0].raw.shape == (3, 4, 3)
    np.testing.assert_allclose(frame.planes[0].image[0, 0], [1, 128 / 255, 0])
    assert frame.planes[0].raw[0, 0].tolist() == [255, 128, 0]


def test_rgb_validation_is_local_to_requested_pixels():
    x = torch.zeros((2, 3, 2, 3, 4), dtype=torch.float64)
    x[1] = 2
    source = make_source(Volume(x), rgb=True)  # no full tensor range scan
    prepare_frame(Request(0, "A", source, (0, 0, 0, 0)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_frame(Request(1, "A", source, (1, 0, 0, 0)))
    x[0, 0, 0, 0, 0] = float("nan")
    invalid = prepare_frame(Request(2, "A", source, (0, 0, 0, 0)))
    assert np.isnan(invalid.planes[0].image[0, 0]).all()


@pytest.mark.parametrize("shape,dtype,rgb", [((1, 1, 2, 2, 2), torch.complex64, False), ((1, 1, 2, 2, 2), torch.float32, True), ((1, 3, 2, 2, 2), torch.int32, True)])
def test_invalid_display_sources(shape, dtype, rgb):
    with pytest.raises((TypeError, ValueError)):
        make_source(Volume(torch.zeros(shape, dtype=dtype)), rgb)


def test_opt_in_volume_uses_only_selected_time_channel_and_stride():
    x = torch.arange(2 * 2 * 5 * 7 * 9, dtype=torch.float32).reshape(2, 2, 5, 7, 9)
    frame = frame_for(x, positions=(1, 0.5, 0.5, 0.5), channel=1, volume_3d=True, preview_stride=2)
    raw_preview = x[1, 1, ::2, ::2, ::2].numpy()
    expected = (raw_preview - frame.clim[0]) / (frame.clim[1] - frame.clim[0])
    assert frame.volume.shape == (3, 4, 5)
    np.testing.assert_allclose(frame.volume, expected)
    assert frame.preview_stride == 2
    assert frame.planes[0].raw.shape == (7, 9)  # preview does not reduce 2D slices


@pytest.mark.parametrize("options", [{"channel": -1}, {"channel": 1}, {"preview_stride": 0}, {"preview_stride": 1.5}, {"clim": (1, 0)}, {"clim": (0, float("inf"))}])
def test_invalid_requests(options):
    with pytest.raises((TypeError, ValueError)):
        frame_for(torch.ones((1, 1, 2, 3, 4)), **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_plane_transfers_are_bounded_and_wait_for_producer_stream():
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        x = torch.arange(2 * 1 * 5 * 7 * 9, device="cuda", dtype=torch.float64).reshape(2, 1, 5, 7, 9)
        x = x + 123
        source = make_source(Volume(x))
    request = Request(9, "B", source, (1, 0.5, 0.5, 0.5), clim=(0, 1000))
    with ThreadPoolExecutor(max_workers=1) as executor:
        frame = executor.submit(prepare_frame, request).result(timeout=30)
    assert frame.transfer_bytes == (7 * 9 + 5 * 9 + 5 * 7) * 8
    assert frame.planes[0].raw[0, 0].item() == 123 + 315 + 2 * 63
    assert frame.planes[0].raw.device.type == "cpu"
    assert frame.volume is None
