"""A tensor and its sampling coordinates, without tensor dispatch hooks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np
import torch


_UNSET = object()
_SPACING = (1.0, 1.0, 1.0)
_ORIGIN = (0.0, 0.0, 0.0)
_DIRECTION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _float_tuple(values: Any, length: int, name: str) -> tuple[float, ...]:
    """Canonicalize small CPU metadata, retaining already canonical tuples."""
    try:
        items = values if type(values) is tuple else tuple(values)
        if len(items) != length:
            raise ValueError(f"{name} must contain {length} values")
        result = items if all(type(x) is float for x in items) else tuple(float(x) for x in items)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _as_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        if not data.flags.writeable:
            raise ValueError("NumPy input must be writable; pass an explicit copy")
        if any(stride < 0 for stride in data.strides):
            raise ValueError("NumPy input cannot have negative strides; pass an explicit copy")
        data = torch.from_numpy(data)
    if type(data) is not torch.Tensor:
        raise TypeError("data must be a native torch.Tensor or NumPy array; convert tensor subclasses explicitly")
    if data.layout != torch.strided:
        raise ValueError("data must be a dense strided tensor")
    if data.ndim != 5 or any(size == 0 for size in data.shape):
        raise ValueError("data must have nonempty T,C,S,H,W dimensions")
    return data


@dataclass(frozen=True, slots=True, eq=False, repr=False, init=False)
class Volume:
    """A native TCSHW tensor with immutable sampling metadata.

    Tensor inputs are retained exactly. Compatible writable NumPy inputs share
    storage. Use ``tensor`` for ordinary PyTorch operations. ``Volume(y, ref=a)``
    assigns a result the reference's outer voxel bounds and times. If SHW
    changes, spacing and origin preserve those bounds and their center; values
    are never resampled. T must match, while C may change. Cropped or warped
    data should instead use the coordinates defined by that operation.

    ``spacing`` is SHW; ``origin`` is XYZ at voxel index (0, 0, 0).
    Columns of ``direction`` are the world directions of XYZ index axes.
    Integer spatial indices denote voxel centers. Units are caller-defined.

    Freezing prevents field reassignment, not tensor writes. Structural in-place
    mutations such as resize_(), set_(), and transpose_() are unsupported.
    State changes return new wrappers; tensor copies follow PyTorch semantics.
    """

    tensor: torch.Tensor
    spacing: tuple[float, float, float] = field(default=_SPACING, kw_only=True)
    origin: tuple[float, float, float] = field(default=_ORIGIN, kw_only=True)
    direction: tuple[tuple[float, float, float], ...] = field(default=_DIRECTION, kw_only=True)
    times: tuple[float, ...] | None = field(default=None, kw_only=True)

    def __init__(
        self,
        tensor: torch.Tensor | np.ndarray,
        *,
        ref: Volume | None = None,
        spacing: Any = _UNSET,
        origin: Any = _UNSET,
        direction: Any = _UNSET,
        times: Any = _UNSET,
    ) -> None:
        reuse = False
        if ref is not None:
            if not isinstance(ref, Volume):
                raise TypeError("ref must be a Volume")
            if any(value is not _UNSET for value in (spacing, origin, direction, times)):
                raise ValueError("ref cannot be combined with explicit spacing, origin, direction, or times")
            tensor = _as_tensor(tensor)
            if tensor.shape[0] != ref.shape[0]:
                raise ValueError("ref requires the same T; supply explicit times without ref")
            if len(ref.times) != tensor.shape[0]:
                raise ValueError("times must match T; structural in-place tensor mutations are unsupported")
            spacing, origin, direction, times = ref.spacing, ref.origin, ref.direction, ref.times
            if tensor.shape[2:] == ref.shape[2:]:
                # Only exact base instances can reuse validation: a subclass
                # may have extra fields or additional metadata constraints.
                reuse = type(self) is Volume and type(ref) is Volume
            else:
                spacing = tuple(d * n / m for d, n, m in zip(ref.spacing, ref.shape[2:], tensor.shape[2:]))
                offset_xyz = ((np.asarray(spacing) - np.asarray(ref.spacing)) / 2)[::-1]
                origin = tuple(np.asarray(ref.origin) + np.asarray(direction) @ offset_xyz)
        else:
            spacing = _SPACING if spacing is _UNSET else spacing
            origin = _ORIGIN if origin is _UNSET else origin
            direction = _DIRECTION if direction is _UNSET else direction
            times = None if times is _UNSET else times
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "times", times)
        if not reuse:
            self.__post_init__()

    def __post_init__(self) -> None:
        tensor = _as_tensor(self.tensor)
        spacing = _float_tuple(self.spacing, 3, "spacing")
        if any(x <= 0 for x in spacing):
            raise ValueError("spacing must be positive")
        origin = _float_tuple(self.origin, 3, "origin")
        try:
            rows = tuple(self.direction)
            if len(rows) != 3:
                raise ValueError("direction must be a 3 x 3 matrix")
            direction = tuple(_float_tuple(row, 3, "direction row") for row in rows)
        except TypeError as exc:
            raise ValueError("direction must be a 3 x 3 matrix") from exc
        if type(self.direction) is tuple and all(a is b for a, b in zip(direction, self.direction)):
            direction = self.direction
        matrix = np.asarray(direction)
        if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1e-6):
            raise ValueError("direction must be orthogonal (reflections are allowed)")
        times = (
            tuple(float(i) for i in range(tensor.shape[0]))
            if self.times is None
            else _float_tuple(self.times, tensor.shape[0], "times")
        )
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "times", times)

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __repr__(self) -> str:
        return (
            f"Volume(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, spacing={self.spacing}, time_count={len(self.times)})"
        )

    def to(self, *args: Any, **kwargs: Any) -> Volume:
        """Apply Tensor.to(), retaining coordinates and its autograd semantics."""
        tensor = self.tensor.to(*args, **kwargs)
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def clone(self, **kwargs: Any) -> Volume:
        """Clone tensor storage with Tensor.clone(), retaining coordinates."""
        tensor = self.tensor.clone(**kwargs)
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def detach(self) -> Volume:
        """Return a new volume detached from autograd, sharing tensor storage."""
        tensor = self.tensor.detach()
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def cpu(self, *, memory_format: torch.memory_format = torch.preserve_format) -> Volume:
        """Return a CPU volume, retaining coordinates and autograd.

        Tensor.cpu() copies only when device or memory format requires it.
        The wrapper is new even when the tensor is unchanged.
        """
        tensor = self.tensor.cpu(memory_format=memory_format)
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def cuda(
        self,
        device: torch.device | str | int | None = None,
        non_blocking: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Volume:
        """Return a CUDA volume using Tensor.cuda() device and copy semantics.

        Coordinates and autograd are retained. The wrapper is new even when
        the tensor is unchanged; non_blocking has the native Torch meaning.
        """
        tensor = self.tensor.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format)
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def contiguous(self, *, memory_format: torch.memory_format = torch.contiguous_format) -> Volume:
        """Return a new volume with the requested contiguous tensor layout.

        Tensor.contiguous() copies only when needed. Coordinates and autograd
        are retained, including when the tensor is already in this layout.
        """
        tensor = self.tensor.contiguous(memory_format=memory_format)
        return Volume(tensor, ref=self) if type(self) is Volume else replace(self, tensor=tensor)

    def numpy(self, copy: bool = False) -> np.ndarray:
        """Detach and convert to NumPy without implicit dtype conversion.

        CPU storage is shared where possible unless copy=True. Other devices
        require a host copy. Lazy conjugate/negative views may also require a
        copy. The returned array never retains the autograd graph.
        """
        tensor = self.tensor.detach().to(device="cpu", copy=copy)
        return tensor.resolve_conj().resolve_neg().numpy()

    def crop(
        self,
        *,
        t: slice = slice(None),
        c: slice = slice(None),
        s: slice = slice(None),
        h: slice = slice(None),
        w: slice = slice(None),
    ) -> Volume:
        """Slice each axis with a positive stride, preserving five dimensions.

        Uses Python's exclusive-stop slice rules, including negative endpoints.
        Integer indexing and negative strides are intentionally unsupported.
        """
        slices = []
        for name, selection, length in zip("tcshw", (t, c, s, h, w), self.shape):
            if not isinstance(selection, slice):
                raise TypeError(f"{name} must be a slice, not an integer index")
            start, stop, step = selection.indices(length)
            if step <= 0:
                raise ValueError("crop only supports positive strides")
            if start >= stop:
                raise ValueError(f"crop would produce an empty {name} dimension")
            slices.append(slice(start, stop, step))
        origin = tuple(self.index_to_world([selection.start for selection in slices[2:]]))
        spacing = tuple(value * selection.step for value, selection in zip(self.spacing, slices[2:]))
        return replace(
            self,
            tensor=self.tensor[tuple(slices)],
            origin=origin,
            spacing=spacing,
            times=self.times[slices[0]],
        )

    def flip_spatial(self, *axes: str) -> Volume:
        """Reverse named spatial axes while preserving each voxel's world position.

        Axes are distinct names from ``s``, ``h``, ``w``. Like ``torch.flip``,
        a nonempty flip copies tensor storage and retains autograd. No channel
        components are transformed. An empty flip returns this volume.
        """
        if any(not isinstance(axis, str) or axis not in ("s", "h", "w") for axis in axes):
            raise ValueError("spatial axes must be 's', 'h', or 'w'")
        if len(set(axes)) != len(axes):
            raise ValueError("spatial axes must not contain duplicates")
        if not axes:
            return self
        indices = tuple("shw".index(axis) for axis in axes)
        start = [self.shape[axis + 2] - 1 if axis in indices else 0 for axis in range(3)]
        direction = np.array(self.direction)
        direction[:, [2 - axis for axis in indices]] *= -1
        return replace(
            self,
            tensor=self.tensor.flip(tuple(axis + 2 for axis in indices)),
            origin=tuple(self.index_to_world(start)),
            direction=tuple(tuple(row) for row in direction),
        )

    def permute_spatial(self, *order: str) -> Volume:
        """Reorder SHW axes and their coordinates using a tensor view.

        ``order`` contains ``s``, ``h``, ``w`` exactly once, naming the original
        axes in the desired output order. T/C and channel components are kept.
        The identity order returns this volume without constructing a wrapper.
        """
        if (
            len(order) != 3
            or any(not isinstance(axis, str) or axis not in ("s", "h", "w") for axis in order)
            or len(set(order)) != 3
        ):
            raise ValueError("spatial order must contain 's', 'h', and 'w' exactly once")
        if order == ("s", "h", "w"):
            return self
        indices = tuple("shw".index(axis) for axis in order)
        direction = np.asarray(self.direction)[:, [2 - axis for axis in reversed(indices)]]
        return replace(
            self,
            tensor=self.tensor.permute(0, 1, *(axis + 2 for axis in indices)),
            spacing=tuple(self.spacing[axis] for axis in indices),
            direction=tuple(tuple(row) for row in direction),
        )

    def index_to_world(self, shw: Any) -> np.ndarray:
        """Convert continuous (..., 3) SHW indices to XYZ world coordinates.

        Returns a CPU float64 NumPy array. Does not round or clamp indices.
        """
        indices = np.asarray(shw, dtype=np.float64)
        if indices.ndim == 0 or indices.shape[-1] != 3:
            raise ValueError("indices must have shape (..., 3) in SHW order")
        local_xyz = (indices * np.asarray(self.spacing))[..., ::-1]
        return local_xyz @ np.asarray(self.direction).T + np.asarray(self.origin)

    def world_to_index(self, xyz: Any) -> np.ndarray:
        """Convert continuous (..., 3) XYZ coordinates to SHW indices.

        Returns a CPU float64 NumPy array. Does not round or clamp indices.
        """
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("coordinates must have shape (..., 3) in XYZ order")
        local_xyz = (points - np.asarray(self.origin)) @ np.asarray(self.direction)
        return local_xyz[..., ::-1] / np.asarray(self.spacing)

    @classmethod
    def from_dlpack(cls, obj: Any, **metadata: Any) -> Volume:
        """Share external storage through DLPack, without importing its graph."""
        return cls(torch.utils.dlpack.from_dlpack(obj), **metadata)

    @classmethod
    def from_voltensor(cls, vt: Any, times: Any = None) -> Volume:
        """Import data, geometry and optional time values from VolTensor."""
        from ._voltensor import from_voltensor

        return from_voltensor(vt, times=times)

    def to_voltensor(self) -> Any:
        """Export data and geometry; retain times separately for round trips."""
        from ._voltensor import to_voltensor

        return to_voltensor(self)
