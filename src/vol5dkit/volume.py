"""A tensor and its sampling coordinates, without tensor dispatch hooks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np
import torch


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


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Volume:
    """A native TCSHW tensor with immutable sampling metadata.

    Tensor inputs are retained exactly. Compatible writable NumPy inputs share
    storage. No construction, processing, or conversion automatically infers a
    changed sampling grid. Use ``tensor`` for ordinary PyTorch operations.

    ``spacing`` is SHW; ``origin`` is XYZ at voxel index (0, 0, 0).
    Columns of ``direction`` are the world directions of XYZ index axes.
    Integer spatial indices denote voxel centers. Units are caller-defined.

    Freezing prevents field reassignment, not tensor writes. Structural in-place
    mutations such as resize_(), set_(), and transpose_() are unsupported.
    """

    tensor: torch.Tensor
    spacing: tuple[float, float, float] = field(default=(1.0, 1.0, 1.0), kw_only=True)
    origin: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), kw_only=True)
    direction: tuple[tuple[float, float, float], ...] = field(
        default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        kw_only=True,
    )
    times: tuple[float, ...] | None = field(default=None, kw_only=True)

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

    def with_data(self, tensor: torch.Tensor | np.ndarray) -> Volume:
        """Declare that a result uses this sampling grid; C may change.

        Shape checks catch accidental T/SHW changes, but cannot prove that an
        operation preserved coordinates. This declaration is the caller's.
        """
        tensor = _as_tensor(tensor)
        if tensor.shape[0] != self.shape[0] or tensor.shape[2:] != self.shape[2:]:
            raise ValueError("with_data requires the same T and SHW sampling grid")
        return replace(self, tensor=tensor)

    def to(self, *args: Any, **kwargs: Any) -> Volume:
        """Apply Tensor.to(), retaining coordinates and its autograd semantics."""
        return replace(self, tensor=self.tensor.to(*args, **kwargs))

    def clone(self, **kwargs: Any) -> Volume:
        """Clone tensor storage with Tensor.clone(), retaining coordinates."""
        return replace(self, tensor=self.tensor.clone(**kwargs))

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
