"""Optional VolTensor 3.8.1 interop; no medical dependencies at core import."""

from __future__ import annotations

from typing import Any

import numpy as np

from .volume import Volume


def _library():
    try:
        import voltensor
    except ModuleNotFoundError as exc:
        if exc.name != "voltensor":
            raise
        raise ImportError(
            "VolTensor conversion requires the separately installed voltensor package"
        ) from exc
    return voltensor


def from_voltensor(vt: Any, times: Any = None) -> Volume:
    """Read a VolTensor's native tensor, XYZ/LPS geometry, and optional times.

    Missing geometric fields use Volume defaults independently. Existing but
    malformed fields are rejected. This adapter interprets world XYZ as LPS;
    callers must convert other coordinate conventions explicitly.
    """
    library = _library()
    if not isinstance(vt, library.VolTensor):
        raise TypeError("from_voltensor requires a voltensor.VolTensor")
    properties = getattr(vt, "vol", None)
    metadata = {}
    spacing = getattr(properties, "spacing", None)
    origin = getattr(properties, "origin", None)
    direction = getattr(properties, "direction", None)
    if spacing is not None:
        metadata["spacing"] = spacing
    if origin is not None:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("VolTensor origin must contain three ZYX values")
        metadata["origin"] = origin[::-1]
    if direction is not None:
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (9,):
            raise ValueError("VolTensor direction must contain nine flat values")
        metadata["direction"] = direction[::-1].reshape(3, 3)
    if times is None:
        source = getattr(properties, "source_meta", None)
        times = getattr(source, "time_axis", None)
    return Volume(vt.as_tensor(), times=times, **metadata)


def to_voltensor(volume: Volume) -> Any:
    """Share tensor data and export geometry, without synthesizing provenance.

    VolTensor has no general time-axis constructor argument. Keep volume.times
    separately and pass it to from_voltensor() when round-tripping.
    """
    library = _library()
    return library.VolTensor(
        volume.tensor,
        spacing=list(volume.spacing),
        origin=list(volume.origin[::-1]),
        direction=np.asarray(volume.direction).reshape(-1)[::-1].tolist(),
    )
