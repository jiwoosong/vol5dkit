"""Tensor-to-display preparation, with no Qt or OpenGL dependencies."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import torch

from ..volume import Volume


@dataclass(frozen=True, slots=True)
class Source:
    volume: Volume
    rgb: bool = False
    ready: torch.cuda.Event | None = None


@dataclass(frozen=True, slots=True)
class Request:
    revision: int
    slot: str
    source: Source
    positions: tuple[float, float, float, float]
    channel: int = 0
    clim: tuple[float, float] | None = None
    volume_3d: bool = False
    preview_stride: int = 1


@dataclass(frozen=True, slots=True)
class Plane:
    name: str
    raw: torch.Tensor
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class Frame:
    revision: int
    slot: str
    source: Source
    indices: tuple[int, int, int, int, int]
    planes: tuple[Plane, ...]
    clim: tuple[float, float] | None
    volume: np.ndarray | None
    preview_stride: int
    transfer_bytes: int = 0


def relative_index(position: float, length: int) -> int:
    """Map a relative axis position to the nearest native voxel center."""
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
        raise ValueError("axis length must be a positive integer")
    if not math.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("relative position must be finite and within [0, 1]")
    return math.floor(position * (length - 1) + 0.5)


def make_source(volume: Volume, rgb: bool = False) -> Source:
    """Retain shared storage without retaining the caller's autograd graph.

    Call this on the producing CUDA stream, or make the calling stream wait
    for that producer before registering the data. Display preparation will
    wait on the recorded event. Floating RGB ranges are checked only for the
    pixels requested for display, never by scanning the full input here.
    """
    if not isinstance(volume, Volume):
        raise TypeError("viewer inputs must be Volume instances")
    tensor = volume.tensor
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError("the viewer supports CPU and CUDA tensors")
    if tensor.is_complex() or tensor.is_quantized:
        raise TypeError("the viewer requires real, non-quantized data")
    if not isinstance(rgb, bool):
        raise TypeError("rgb must be a bool")
    if rgb:
        if tensor.shape[1] != 3:
            raise ValueError("RGB display requires exactly three channels")
        if tensor.dtype != torch.uint8 and not tensor.is_floating_point():
            raise TypeError("RGB display requires uint8 or floating point data")
    detached = volume.with_data(tensor.detach())
    ready = None
    if tensor.device.type == "cuda":
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
    return Source(detached, rgb, ready)


def _finite_range(tensor: torch.Tensor) -> tuple[float, float]:
    """Reduce on the source device; transfer only the two scalar endpoints."""
    if tensor.dtype == torch.uint64:
        # Flip the sign bit so a signed reduction has unsigned ordering.
        encoded = tensor.view(torch.int64) ^ torch.iinfo(torch.int64).min
        low, high = torch.aminmax(encoded)
        return low.item() + 2**63, high.item() + 2**63
    if tensor.dtype in (torch.bool, torch.uint16, torch.uint32):
        tensor = tensor.to(torch.int64)
    low, high = torch.aminmax(tensor)
    lo, hi = low.item(), high.item()
    if math.isfinite(lo) and math.isfinite(hi):
        return lo, hi
    finite = torch.isfinite(tensor)
    lo = torch.where(finite, tensor, math.inf).amin().item()
    hi = torch.where(finite, tensor, -math.inf).amax().item()
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else (0.0, 1.0)


def _window(raw: torch.Tensor, clim: tuple[float, float]) -> np.ndarray:
    """Window in original precision before creating a float32 texture."""
    array = raw.to(torch.float64).numpy() if raw.dtype == torch.bfloat16 else raw.numpy()
    lo, hi = clim
    finite = np.isfinite(array)
    if lo == hi:
        output = np.full(array.shape, 0.5, dtype=np.float32)
    else:
        # Preserve narrow contrasts around large integer baselines. Casting
        # int64/uint64 directly to float64 first would lose low-order bits.
        integral_limits = isinstance(lo, Integral) and isinstance(hi, Integral)
        integer_input = array.dtype.kind in "iu"
        if integer_input and integral_limits:
            lo, hi = int(lo), int(hi)
            limits = np.iinfo(array.dtype)
            if hi <= limits.min:
                output = np.ones(array.shape, dtype=np.float32)
            elif lo >= limits.max:
                output = np.zeros(array.shape, dtype=np.float32)
            else:
                base, stop = max(lo, limits.min), min(hi, limits.max)
                clipped = np.clip(array, base, stop)
                # Unsigned subtraction gives the exact nonnegative distance
                # even across the full signed-int64 range. Keep out-of-dtype
                # offsets and the width as Python integers until division.
                baseline = np.array(base, dtype=array.dtype).astype(np.uint64)
                distance = clipped.astype(np.uint64) - baseline
                width = hi - lo
                normalized = distance.astype(np.float64) * (1 / width) + (base - lo) / width
                output = np.asarray(normalized, dtype=np.float32)
        else:
            work = array.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                width = float(hi) - float(lo)
                if math.isfinite(width):
                    normalized = (work - lo) / width
                else:
                    normalized = (work / 2 - lo / 2) / (hi / 2 - lo / 2)
            output = np.asarray(normalized, dtype=np.float32)
        np.clip(output, 0, 1, out=output)
    output[~finite] = np.nan
    return np.ascontiguousarray(output)


def _rgb_image(raw: torch.Tensor) -> np.ndarray:
    image = raw.to(torch.float32).numpy()
    # Keep image separately owned even if raw already has float32 dtype.
    image = np.array(image, dtype=np.float32, order="C", copy=True)
    if raw.dtype == torch.uint8:
        image /= 255.0
    else:
        finite = torch.isfinite(raw)
        if torch.any(finite & ((raw < 0) | (raw > 1))).item():
            raise ValueError("floating RGB values must be in [0, 1]; normalize the input explicitly")
        image[~np.isfinite(image).all(axis=-1)] = np.nan
    return image


def prepare_frame(request: Request) -> Frame:
    """Copy only the requested native slices, plus an opt-in 3D preview.

    This can run on a worker thread. The returned raw buffers own their CPU
    storage even for CPU inputs. ``transfer_bytes`` counts image data copied
    from CUDA to host, excluding the tiny initial range-reduction scalars.
    """
    source = request.source
    tensor = source.volume.tensor
    if len(request.positions) != 4:
        raise ValueError("positions must contain T, S, H, W relative positions")
    t, s, h, w = (
        relative_index(p, n)
        for p, n in zip(request.positions, (tensor.shape[0], *tensor.shape[2:]))
    )
    if isinstance(request.channel, bool) or not isinstance(request.channel, Integral):
        raise TypeError("channel must be an integer")
    c = 0 if source.rgb else int(request.channel)
    if not 0 <= c < tensor.shape[1]:
        raise ValueError("channel index is out of range")
    stride = request.preview_stride
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("preview_stride must be a positive integer")
    if request.volume_3d and source.rgb:
        raise ValueError("volume rendering supports scalar inputs only")
    if source.ready is not None:
        torch.cuda.current_stream(tensor.device).wait_event(source.ready)

    selected = tensor[t] if source.rgb else tensor[t, c]
    clim = request.clim
    if not source.rgb:
        if clim is None:
            clim = _finite_range(selected)
        elif len(clim) != 2 or not all(math.isfinite(v) for v in clim) or clim[0] > clim[1]:
            raise ValueError("clim must contain two finite values with low <= high")
        clim = tuple(clim)
    if source.rgb:
        slices = (
            selected[:, s, :, :].permute(1, 2, 0),
            selected[:, :, h, :].permute(1, 2, 0),
            selected[:, :, :, w].permute(1, 2, 0),
        )
    else:
        slices = (selected[s, :, :], selected[:, h, :], selected[:, :, w])

    planes = []
    copied_bytes = 0
    for name, sliced in zip(("HW", "SW", "SH"), slices):
        raw = sliced.to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        copied_bytes += raw.numel() * raw.element_size()
        image = _rgb_image(raw) if source.rgb else _window(raw, clim)
        planes.append(Plane(name, raw, image))

    volume = None
    if request.volume_3d:
        preview = selected[::stride, ::stride, ::stride].to(
            device="cpu", copy=True, memory_format=torch.contiguous_format
        )
        copied_bytes += preview.numel() * preview.element_size()
        volume = _window(preview, clim)
        # Nonfinite voxels are transparent/background in volume rendering;
        # slice views still expose them as NaNs with their original values.
        np.nan_to_num(volume, copy=False, nan=0.0)

    return Frame(
        request.revision,
        request.slot,
        source,
        (t, c, s, h, w),
        tuple(planes),
        clim,
        volume,
        int(stride),
        copied_bytes if tensor.device.type == "cuda" else 0,
    )
