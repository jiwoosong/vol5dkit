"""Snapshot transport and the public viewer API; no GUI imports in the caller."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .volume import Volume


def _validate_window(value):
    if value is None:
        return None
    try:
        low, high = value
    except (TypeError, ValueError) as exc:
        raise ValueError("window must be a (low, high) pair") from exc
    endpoints = []
    for endpoint in (low, high):
        if isinstance(endpoint, bool) or not isinstance(endpoint, Real) or not math.isfinite(endpoint):
            raise ValueError("window endpoints must be finite real numbers")
        # Keep large integer endpoints exact through JSON and windowing.
        endpoints.append(int(endpoint) if isinstance(endpoint, Integral) else float(endpoint))
    if endpoints[0] > endpoints[1]:
        raise ValueError("window requires low <= high")
    return tuple(endpoints)


@dataclass(frozen=True, slots=True, eq=False)
class Display:
    """One viewer input with its name and display settings.

    ``data`` is a Volume, native TCSHW tensor, or NumPy array.
    ``window=None`` uses the finite range of the entire input on first use.
    Colormap names may specify ``vispy:``, ``mpl:`` or ``sns:`` providers.
    """

    data: Volume | torch.Tensor | np.ndarray
    name: str | None = field(default=None, kw_only=True)
    window: tuple[float | int, float | int] | None = field(default=None, kw_only=True)
    cmap: str = field(default="grays", kw_only=True)
    rgb: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.data, (Volume, torch.Tensor, np.ndarray)):
            raise TypeError("Display data must be a Volume, torch.Tensor, or NumPy array")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        if not isinstance(self.cmap, str) or not self.cmap:
            raise ValueError("cmap must be a nonempty colormap name")
        if not isinstance(self.rgb, bool):
            raise TypeError("rgb must be a bool")
        object.__setattr__(self, "window", _validate_window(self.window))


def _normalize_inputs(inputs):
    normalized = []
    for item in inputs:
        display = item if isinstance(item, Display) else Display(item)
        volume = display.data if isinstance(display.data, Volume) else Volume(display.data)
        tensor = volume.tensor
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError("the viewer supports CPU and CUDA tensors")
        if tensor.is_complex() or tensor.is_quantized:
            raise TypeError("the viewer requires real, non-quantized data")
        if display.rgb:
            if tensor.shape[1] != 3:
                raise ValueError("RGB display requires exactly three channels")
            if tensor.dtype != torch.uint8 and not tensor.is_floating_point():
                raise TypeError("RGB display requires uint8 or floating point data")
        normalized.append(display if volume is display.data else replace(display, data=volume))
    if not normalized:
        raise ValueError("view requires at least one input")
    return normalized


# Torch 2.3 cannot serialize these storage types. Reinterpret the same bits,
# without converting values, and restore their dtype after mmap loading.
_UNSIGNED_STORAGE = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}
_OWNER_MARKER = "vol5dkit snapshot format 2"


def _write_snapshot(inputs, directory, *, window=None, interpolation="nearest", started=None):
    """Serialize one compact independent CPU staging tensor at a time."""
    directory = Path(directory)
    data_dir = directory / "data"
    data_dir.mkdir()
    (directory / ".snapshot-owner").write_text(_OWNER_MARKER, encoding="ascii")
    manifest = {
        "format": 2,
        "window": window,
        "interpolation": interpolation,
        "inputs": [],
        "started": time.perf_counter() if started is None else started,
        "snapshot_copy_ms": 0.0,
        "snapshot_write_ms": 0.0,
    }
    for index, display in enumerate(inputs):
        volume = display.data
        tick = time.perf_counter()
        cpu = volume.tensor.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        # Materialize lazy conjugate/negative views only when required; ordinary
        # CPU inputs and noncontiguous views use the single allocation above.
        cpu = cpu.resolve_conj().resolve_neg()
        manifest["snapshot_copy_ms"] += (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        filename = f"volume-{index}.pt"
        torch.save({
            "tensor": cpu.view(_UNSIGNED_STORAGE[cpu.dtype]) if cpu.dtype in _UNSIGNED_STORAGE else cpu,
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "spacing": volume.spacing,
            "origin": volume.origin,
            "direction": volume.direction,
            "times": volume.times,
        }, data_dir / filename)
        manifest["snapshot_write_ms"] += (time.perf_counter() - tick) * 1000
        # Do not use dataclasses.asdict: it would deep-copy the input tensor.
        manifest["inputs"].append({"file": filename, "display": {
            "name": display.name, "window": display.window,
            "cmap": display.cmap, "rgb": display.rgb,
        }})
        del cpu
    path = directory / "manifest.json"
    manifest["snapshot_ready"] = time.perf_counter()
    path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    return path


def _load_snapshot(manifest_path):
    """Load safe tensor payloads with file-backed storage, without importing Qt."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != 2:
        raise ValueError("unsupported snapshot format")
    inputs = []
    for index, entry in enumerate(manifest["inputs"]):
        if entry["file"] != f"volume-{index}.pt":
            raise ValueError("invalid snapshot tensor filename")
        payload = torch.load(
            manifest_path.parent / "data" / entry["file"],
            map_location="cpu", mmap=True, weights_only=True,
        )
        tensor = payload.pop("tensor")
        dtype = getattr(torch, payload.pop("dtype"))
        if tensor.dtype != dtype:
            if dtype not in _UNSIGNED_STORAGE or tensor.dtype != _UNSIGNED_STORAGE[dtype]:
                raise ValueError("invalid snapshot storage dtype")
            tensor = tensor.view(dtype)
        inputs.append(Display(Volume(tensor, **payload), **entry["display"]))
    return _normalize_inputs(inputs), manifest


def _cleanup_snapshot(directory, *, keep_log=False):
    directory = Path(directory)
    try:
        if (directory / ".snapshot-owner").read_text(encoding="ascii") != _OWNER_MARKER:
            return
    except (OSError, UnicodeError):
        return
    # Only files in this call's private directory are removed. The log is kept
    # on failure so that a launched child never fails silently.
    shutil.rmtree(directory / "data", ignore_errors=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    if not keep_log:
        try:
            (directory / "viewer.log").unlink(missing_ok=True)
        except PermissionError:
            pass  # A still-exiting Windows child can retain its log handle.
        if not (directory / "viewer.log").exists() and not (directory / "data").exists():
            (directory / ".snapshot-owner").unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        pass


def _open_log(path):
    if os.name != "nt":
        return path.open("wb")
    import _winapi
    import msvcrt

    # Some native libraries duplicate stderr. Share deletion so a successful
    # child can unlink its log without closing another library's handles.
    handle = _winapi.CreateFile(str(path), _winapi.GENERIC_WRITE, 7, 0, 2, 128, 0)
    # 7 = FILE_SHARE_READ | WRITE | DELETE; 2 = CREATE_ALWAYS; 128 = NORMAL.
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY)
    except OSError:
        _winapi.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "wb")


class ViewerProcess:
    """A snapshot window's process handle, with no Qt objects or source tensors.

    A failed child writes its traceback to ``log_path``. ``poll()`` and ``wait``
    return its exit code; ``wait`` uses subprocess.TimeoutExpired on timeout.
    Closing or discarding this handle does not affect the caller's tensors.
    """

    __slots__ = ("_process", "_directory", "log_path")

    def __init__(self, process, directory):
        self._process = process
        self._directory = Path(directory)
        self.log_path = self._directory / "viewer.log"

    @property
    def pid(self):
        return self._process.pid

    def poll(self):
        code = self._process.poll()
        if code is not None:
            _cleanup_snapshot(self._directory, keep_log=code != 0)
        return code

    def wait(self, timeout=None):
        code = self._process.wait(timeout=timeout)
        _cleanup_snapshot(self._directory, keep_log=code != 0)
        return code

    def close(self):
        was_running = self._process.poll() is None
        if was_running:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass  # It can exit between poll() and terminate().
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        _cleanup_snapshot(self._directory, keep_log=not was_running and self._process.returncode not in (0, None))


def _launch(inputs, *, window=None, interpolation="nearest", smoke=False, screenshot=None, report=None, iterations=1, volume_3d=False):
    """Private diagnostic flags serve benchmarks and process integration tests."""
    started = time.perf_counter()
    inputs = _normalize_inputs(inputs)
    window = _validate_window(window)
    if interpolation not in ("nearest", "linear"):
        raise ValueError("interpolation must be 'nearest' or 'linear'")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("diagnostic iterations must be a positive integer")
    directory = Path(tempfile.mkdtemp(prefix="vol5dkit-"))
    try:
        manifest = _write_snapshot(inputs, directory, window=window, interpolation=interpolation, started=started)
        command = [sys.executable, "-m", "vol5dkit.viewer", str(manifest)]
        if smoke or screenshot or report or volume_3d:
            command.extend(("--smoke", "--iterations", str(iterations)))
        if volume_3d:
            command.append("--volume-3d")
        if screenshot:
            command.extend(("--screenshot", str(Path(screenshot).resolve())))
        if report:
            command.extend(("--report", str(Path(report).resolve())))
        # pydevd's local context bypasses its subprocess injection without
        # importing a debugger or changing its process-wide configuration.
        skip_patch = nullcontext
        for module_name in ("pydevd", "_pydev_bundle.pydev_monkey"):
            candidate = getattr(sys.modules.get(module_name), "skip_subprocess_arg_patch", None)
            if callable(candidate):
                skip_patch = candidate
                break
        with _open_log(directory / "viewer.log") as output, skip_patch():
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    except BaseException:
        _cleanup_snapshot(directory)
        raise
    return ViewerProcess(process, directory)


def view(*inputs, window=None, interpolation="nearest"):
    """Open an independent snapshot window and return its process handle.

    Inputs may be Volumes, native TCSHW tensors, NumPy arrays, or Display
    objects. Bare tensors and arrays use index coordinates. ``window`` sets
    a shared initial range; otherwise each input uses its own Display settings.

    Inputs are copied sequentially to independent CPU snapshot files before
    returning. A CUDA producer must use the calling stream or establish its
    dependency first. Later changes to the input do not change this window.
    The child runs its own Qt event loop, including while the caller is paused
    in a supported debugger. Call ``handle.wait()`` to wait for window closure.
    """
    return _launch(inputs, window=window, interpolation=interpolation)
