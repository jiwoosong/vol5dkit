"""Tensor-to-display preparation, with no Qt or OpenGL dependencies."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import torch

from ..volume import Volume
from .._view import _validate_window


@dataclass(frozen=True, slots=True)
class Source:
    volume: Volume
    rgb: bool = False


@dataclass(frozen=True, slots=True)
class Request:
    revision: int
    input_index: int
    source: Source
    positions: tuple[float, float, float, float]
    channel: int = 0
    window: tuple[int | float, int | float] | None = None
    volume_3d: bool = False
    range_source: Source | None = None


@dataclass(frozen=True, slots=True)
class Plane:
    name: str
    raw: torch.Tensor
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class Frame:
    revision: int
    input_index: int
    source: Source
    indices: tuple[int, int, int, int, int]
    planes: tuple[Plane, ...]
    window: tuple[int | float, int | float] | None
    volume_buffer: np.ndarray | None


def relative_index(position: float, length: int) -> int:
    """Map a relative axis position to the nearest native voxel center."""
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
        raise ValueError("axis length must be a positive integer")
    if not math.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("relative position must be finite and within [0, 1]")
    return math.floor(position * (length - 1) + 0.5)


def make_source(volume: Volume, rgb: bool = False) -> Source:
    """Validate a loaded CPU snapshot without copying or scanning its data."""
    if not isinstance(volume, Volume):
        raise TypeError("viewer inputs must be Volume instances")
    tensor = volume.tensor
    if tensor.device.type != "cpu":
        raise ValueError("display preparation requires a CPU snapshot")
    if tensor.is_complex() or tensor.is_quantized:
        raise TypeError("the viewer requires real, non-quantized data")
    if not isinstance(rgb, bool):
        raise TypeError("rgb must be a bool")
    if rgb:
        if tensor.shape[1] != 3:
            raise ValueError("RGB display requires exactly three channels")
        if tensor.dtype != torch.uint8 and not tensor.is_floating_point():
            raise TypeError("RGB display requires uint8 or floating point data")
    return Source(volume, rgb)


_CHUNK_ELEMENTS = 1 << 20


def _chunks(tensor: torch.Tensor):
    """Yield slice indices and bounded views, including noncontiguous inputs."""
    pending = [tuple(slice(0, n) for n in tensor.shape)]
    while pending:
        index = pending.pop()
        chunk = tensor[index]
        if chunk.numel() <= _CHUNK_ELEMENTS:
            yield index, chunk
            continue
        axis = max(range(chunk.ndim), key=lambda i: chunk.shape[i])
        start, stop = index[axis].start, index[axis].stop
        middle = start + (stop - start) // 2
        pending.append((*index[:axis], slice(middle, stop), *index[axis + 1:]))
        pending.append((*index[:axis], slice(start, middle), *index[axis + 1:]))


def _finite_range(tensor: torch.Tensor) -> tuple[int | float, int | float]:
    """Reduce all TCSHW while bounding temporary allocations by chunk.

    Ordinary finite data need only aminmax. The fallback masks and unsupported
    integer casts stay bounded even for a large, noncontiguous volume. Chunk
    endpoints retain the original integer precision.
    """
    unsigned64 = tensor.dtype == torch.uint64
    cast_integer = tensor.dtype in (torch.bool, torch.uint16, torch.uint32)
    if not unsigned64 and not cast_integer:
        low, high = torch.aminmax(tensor)
        lo, hi = low.item(), high.item()
        if math.isfinite(lo) and math.isfinite(hi):
            return lo, hi

    low = high = None
    for _, chunk in _chunks(tensor):
        if unsigned64:
            # Same-width view plus sign-bit flip preserves unsigned ordering.
            encoded = chunk.view(torch.int64) ^ torch.iinfo(torch.int64).min
            chunk_low, chunk_high = torch.aminmax(encoded)
        elif cast_integer:
            chunk_low, chunk_high = torch.aminmax(chunk.to(torch.int64))
        else:
            finite = torch.isfinite(chunk)
            chunk_low = torch.where(finite, chunk, math.inf).amin()
            chunk_high = torch.where(finite, chunk, -math.inf).amax()
        low = chunk_low if low is None else torch.minimum(low, chunk_low)
        high = chunk_high if high is None else torch.maximum(high, chunk_high)

    lo, hi = low.item(), high.item()
    if unsigned64:
        return lo + 2**63, hi + 2**63
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else (0.0, 1.0)


def _window(
    raw: torch.Tensor, window: tuple[int | float, int | float], *, nonfinite: float = np.nan
) -> np.ndarray:
    """Create an owned float32 texture with bounded precision-preserving work."""
    result = np.empty(tuple(raw.shape), dtype=np.float32)
    for index, chunk in _chunks(raw):
        array = chunk.float().numpy() if raw.dtype == torch.bfloat16 else chunk.numpy()
        output = result[index]
        lo, hi = window
        finite = np.isfinite(array)
        if lo == hi:
            output.fill(0.5)
            output[~finite] = nonfinite
            continue
        # Preserve narrow contrasts around large integer baselines. Casting
        # int64/uint64 directly to float64 first would lose low-order bits.
        integral_limits = isinstance(lo, Integral) and isinstance(hi, Integral)
        integer_input = array.dtype.kind in "iu"
        if integer_input and integral_limits:
            lo, hi = int(lo), int(hi)
            limits = np.iinfo(array.dtype)
            if hi <= limits.min:
                output.fill(1)
            elif lo >= limits.max:
                output.fill(0)
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
                output[...] = normalized
        else:
            work = array.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                width = float(hi) - float(lo)
                if math.isfinite(width):
                    normalized = (work - lo) / width
                else:
                    normalized = (work / 2 - lo / 2) / (hi / 2 - lo / 2)
            output[...] = normalized
        np.clip(output, 0, 1, out=output)
        output[~finite] = nonfinite
    return result


def _rgb_image(raw: torch.Tensor) -> np.ndarray:
    # One cast/copy also owns storage when raw already has float32 dtype.
    image = raw.to(dtype=torch.float32, copy=True, memory_format=torch.contiguous_format).numpy()
    if raw.dtype == torch.uint8:
        image /= 255.0
    else:
        finite = torch.isfinite(raw)
        if torch.any(finite & ((raw < 0) | (raw > 1))).item():
            raise ValueError("floating RGB values must be in [0, 1]; normalize the input explicitly")
        image[~np.isfinite(image).all(axis=-1)] = np.nan
    return image


def prepare_frame(request: Request, *, volume_buffer: np.ndarray | None = None) -> Frame:
    """Prepare owned native slices and optionally normalize the selected 3D data.

    A worker may supply its cached 3D buffer for the same input, T, C and
    window. All preparation reads a CPU snapshot; CUDA transfer happens only
    in the parent when that snapshot is created.
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
    if request.volume_3d and source.rgb:
        raise ValueError("volume rendering supports scalar inputs only")
    selected = tensor[t] if source.rgb else tensor[t, c]
    window = _validate_window(request.window)
    if not source.rgb:
        if window is None:
            # A shared Auto action stays tied to the input selected when it
            # was requested, even if newer navigation now displays another.
            range_source = source if request.range_source is None else request.range_source
            range_tensor = range_source.volume.tensor
            window = _finite_range(range_tensor)
    if source.rgb:
        slices = (
            selected[:, s, :, :].permute(1, 2, 0),
            selected[:, :, h, :].permute(1, 2, 0),
            selected[:, :, :, w].permute(1, 2, 0),
        )
    else:
        slices = (selected[s, :, :], selected[:, h, :], selected[:, :, w])

    planes = []
    for name, sliced in zip(("HW", "SW", "SH"), slices):
        raw = sliced.clone(memory_format=torch.contiguous_format)
        image = _rgb_image(raw) if source.rgb else _window(raw, window)
        planes.append(Plane(name, raw, image))

    if not request.volume_3d:
        volume_buffer = None
    elif volume_buffer is None:
        # Nonfinite voxels are transparent/background in volume rendering;
        # slice views still expose them as NaNs with their original values.
        volume_buffer = _window(selected, window, nonfinite=0.0)

    return Frame(
        request.revision,
        request.input_index,
        source,
        (t, c, s, h, w),
        tuple(planes),
        window,
        volume_buffer,
    )
