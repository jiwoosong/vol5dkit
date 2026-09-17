# Optional interoperability

The data core requires only PyTorch and NumPy. Optional adapters load their
external libraries when called and do not add them as installation dependencies.
The `Volume` constructor accepts native tensors; convert tensor subclasses
explicitly or use an adapter that understands their coordinate conventions.

## VolTensor

Install the reference VolTensor library separately. The existing adapter targets
VolTensor 3.8.1 and maps its spatial coordinates to the `Volume` convention.
Importing `vol5dkit` alone does not import VolTensor or its dependencies.

```python
from vol5dkit import Volume

a = Volume.from_voltensor(vt)
exported = a.to_voltensor()
restored = Volume.from_voltensor(exported, times=a.times)
```

The adapter obtains a native tensor with `vt.as_tensor()` and does not reorder
data axes. Input must already be a nonempty TCSHW tensor. Storage is shared when
the reference library permits it; the conversion does not detach the native
tensor or move it between devices.

### Spatial coordinates

| Coordinate | Import from VolTensor |
| --- | --- |
| Spacing | Retain the SHW order |
| Origin | Reverse ZYX values to obtain XYZ |
| Direction | Reverse the entire flat nine-value sequence, then reshape to 3 × 3 |

Export applies the inverse conversions. Missing spatial attributes use the
corresponding `Volume` defaults; existing malformed values are rejected. The
adapter uses the reference library's LPS world convention. If your `Volume`
uses another world basis, convert origin and direction to LPS explicitly before
exporting. This convention belongs to the adapter; the core imposes no world
basis or physical units.

### Time and other metadata

Import chooses time values in this order:

1. Explicit `times`, when supplied.
2. `vt.vol.source_meta.time_axis`, when present.
3. Frame indices `0, 1, ..., T - 1`.

Existing time values must be finite and have length T. Invalid values are not
silently replaced. Export carries the tensor and spatial coordinates, but the
reference constructor has no general-purpose time-axis argument. Keep `.times`
separately and supply it on import for a time-preserving round trip, as above.

File headers, medical provenance, and DICOM identifiers are not transferred or
fabricated. Keep any such application metadata separately. The adapter and its
compatibility tests remain optional; tests skip when the reference runtime is
unavailable. Each external library retains its own license.
