"""Snapshot transport and process behavior, mostly without a GUI runtime."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from vol5dkit import Display, Volume, view
from vol5dkit import _view


def sample():
    return Volume(torch.arange(48).reshape(2, 1, 2, 3, 4), spacing=(2, 3, 4),
                  origin=(1, -3, 9), direction=((0, 1, 0), (1, 0, 0), (0, 0, -1)), times=(3, -2))


def test_display_are_small_immutable_and_preserve_integer_endpoints():
    options = Display(sample(), window=[np.uint64(2**64 - 3), np.uint64(2**64 - 1)], cmap="mpl:hot")
    assert options.window == (2**64 - 3, 2**64 - 1)
    assert not hasattr(options, "__dict__")
    with pytest.raises(FrozenInstanceError):
        options.rgb = True


def test_display_repr_does_not_read_data_values(monkeypatch):
    def forbidden_repr(self):
        raise AssertionError("Display repr must not format data values")

    class Array(np.ndarray):
        __repr__ = forbidden_repr

    volume = sample()
    array = volume.numpy().view(Array)
    monkeypatch.setattr(torch.Tensor, "__repr__", forbidden_repr)
    monkeypatch.setattr(Volume, "__repr__", forbidden_repr)
    for data in (volume, volume.tensor, array):
        result = repr(Display(data, name="original", window=(0, 1)))
        assert "shape=(2, 1, 2, 3, 4)" in result
        assert "dtype=" in result and "device=cpu" in result
        assert "name='original'" in result and "window=(0, 1)" in result


@pytest.mark.parametrize("missing", ["PySide6", "vispy", "OpenGL"])
def test_missing_gui_fails_before_staging_or_starting_process(monkeypatch, missing):
    def unexpected(*args, **kwargs):
        raise AssertionError("missing GUI must fail before snapshot or process creation")

    monkeypatch.setattr(_view, "find_spec", lambda name: None if name == missing else object())
    monkeypatch.setattr(_view.tempfile, "mkdtemp", unexpected)
    monkeypatch.setattr(_view, "_write_snapshot", unexpected)
    monkeypatch.setattr(_view.subprocess, "Popen", unexpected)
    with pytest.raises(ImportError, match=f"missing {missing}"):
        view(sample())


@pytest.mark.parametrize("kwargs", [
    {"name": 3}, {"rgb": 1}, {"cmap": ""}, {"window": (2, 1)}, {"window": (0, float("nan"))},
    {"window": (0, float("inf"))}, {"window": (True, 1)}, {"window": (1,)}, {"window": "ab"},
])
def test_invalid_display(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Display(sample(), **kwargs)


def test_input_normalization_and_early_validation():
    a = sample()
    options = Display(a, name="original")
    normalized = _view._normalize_inputs([a, options, a.tensor, a.numpy()])
    assert normalized[0].data is a and normalized[1] is options
    assert normalized[2].data.tensor is a.tensor
    assert normalized[2].data.spacing == (1, 1, 1)
    assert normalized[2].data.times == (0, 1)
    assert np.shares_memory(normalized[3].data.numpy(), a.numpy())
    for inputs in ([], [a.tensor[0]], [(a, {})], [None]):
        with pytest.raises((ValueError, TypeError)):
            _view._normalize_inputs(inputs)
    with pytest.raises(ValueError, match="three channels"):
        _view._normalize_inputs([Display(a, rgb=True)])
    with pytest.raises(TypeError, match="RGB"):
        _view._normalize_inputs([Display(a.tensor.expand(-1, 3, -1, -1, -1), rgb=True)])
    with pytest.raises(TypeError, match="real"):
        _view._normalize_inputs([Volume(a.tensor.to(torch.complex64))])


@pytest.mark.parametrize("dtype", [
    torch.bool, torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
])
def test_mmap_roundtrip_preserves_values_dtype_geometry_and_options(tmp_path, dtype):
    a = sample().to(dtype=dtype)
    options = Display(a, name="field", window=(0, 2**64 - 1), cmap="sns:rocket")
    path = _view._write_snapshot([options], tmp_path)
    inputs, manifest = _view._load_snapshot(path)
    restored = inputs[0]
    b = restored.data
    assert b.dtype == a.dtype
    # Conversion here is only for equality of unsigned dtypes on Torch 2.3.
    assert torch.equal(b.tensor.view(torch.uint8), a.tensor.view(torch.uint8))
    assert b.tensor.data_ptr() != a.tensor.data_ptr()
    assert b.tensor.is_contiguous()
    assert (restored.name, restored.window, restored.cmap, restored.rgb) == (options.name, options.window, options.cmap, options.rgb)
    for attr in ("spacing", "origin", "direction", "times"):
        assert getattr(b, attr) == getattr(a, attr)
    assert manifest["snapshot_copy_ms"] >= 0 and manifest["snapshot_write_ms"] > 0


def test_unsigned_storage_uses_signed_bits_in_serialized_payload(tmp_path):
    tensor = torch.from_numpy(np.array([2**64 - 1, 2**63 + 1], dtype=np.uint64)).reshape(1, 1, 1, 1, 2)
    path = _view._write_snapshot([Display(Volume(tensor))], tmp_path)
    payload = torch.load(tmp_path / "data" / "volume-0.pt", weights_only=True, mmap=True)
    assert payload["tensor"].dtype == torch.int64
    assert payload["tensor"].flatten().tolist() == [-1, -(2**63) + 1]
    inputs, _ = _view._load_snapshot(path)
    assert inputs[0].data.tensor.flatten().tolist() == [2**64 - 1, 2**63 + 1]


def test_snapshot_compacts_large_partial_storage_and_preserves_autograd_source(tmp_path):
    leaf = torch.arange(100_000.0, requires_grad=True)
    computed = leaf * 2
    intermediate = weakref.ref(computed)
    tensor = computed[:120].reshape(2, 1, 3, 4, 5).transpose(-1, -2)
    volume = Volume(tensor)
    path = _view._write_snapshot([Display(volume)], tmp_path)
    inputs, _ = _view._load_snapshot(path)
    restored = inputs[0].data.tensor
    assert restored.untyped_storage().nbytes() == restored.numel() * restored.element_size()
    assert restored.is_contiguous() and restored.grad_fn is None and not restored.requires_grad
    assert torch.equal(restored, tensor)
    tensor.sum().backward()
    assert leaf.grad[:120].eq(2).all()
    with torch.no_grad():
        tensor.fill_(-1)
    assert not restored.eq(-1).any()
    del tensor, volume, computed
    gc.collect()
    assert intermediate() is None


def test_staging_is_released_between_inputs(tmp_path, monkeypatch):
    previous = []
    original_save = torch.save

    def save(payload, file):
        if previous:
            assert previous[-1]() is None
        previous.append(weakref.ref(payload["tensor"]))
        return original_save(payload, file)

    monkeypatch.setattr(torch, "save", save)
    _view._write_snapshot([Display(sample()) for _ in range(3)], tmp_path)
    assert all(item() is None for item in previous)


def test_parent_import_is_gui_free():
    result = subprocess.run([sys.executable, "-c", "import sys, vol5dkit; "
        "from vol5dkit import Display, view; "
        "assert not any(k.split('.')[0] in ('PySide6','vispy','matplotlib','seaborn') for k in sys.modules); "
        "assert not hasattr(vol5dkit, 'run')"], check=True, capture_output=True, text=True)
    assert result.returncode == 0


def test_public_window_and_removed_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(_view, "_launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    view(sample(), window=(1, 3))
    assert calls[0][1]["window"] == (1, 3)
    for keyword in ("win", "clim", "block", "rgb"):
        with pytest.raises(TypeError):
            view(sample(), **{keyword: None})
    import vol5dkit
    assert not hasattr(vol5dkit, "ViewOptions")


class FakeProcess:
    pid = 98765
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -15


def test_launch_uses_local_debugger_context_and_owned_log(tmp_path, monkeypatch):
    state = {"inside": False}
    process = FakeProcess()

    @contextmanager
    def skip_patch():
        state["inside"] = True
        try:
            yield
        finally:
            state["inside"] = False

    def popen(command, **kwargs):
        assert state["inside"]
        assert command[:3] == [sys.executable, "-m", "vol5dkit.viewer"]
        assert (tmp_path / "viewer.log").exists()
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["creationflags"] == (subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return process

    monkeypatch.setitem(sys.modules, "_pydev_bundle.pydev_monkey", SimpleNamespace(skip_subprocess_arg_patch=skip_patch))
    monkeypatch.setattr(_view, "find_spec", lambda name: object())
    monkeypatch.setattr(_view.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(_view.subprocess, "Popen", popen)
    handle = _view._launch([sample()])
    assert handle.pid == 98765 and handle.poll() is None and not state["inside"]
    with pytest.raises(subprocess.TimeoutExpired):
        handle.wait(timeout=0.01)
    handle.close()
    assert not tmp_path.exists()
    handle.close()


def test_failure_logs_survive_handle_cleanup(tmp_path):
    _view._write_snapshot([Display(sample())], tmp_path)
    (tmp_path / "viewer.log").write_text("initialization failed", encoding="utf-8")
    process = FakeProcess()
    process.returncode = 1
    handle = _view.ViewerProcess(process, tmp_path)
    assert handle.wait() == 1
    assert handle.log_path.read_text(encoding="utf-8") == "initialization failed"
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert handle.poll() == 1


def test_public_debugger_context_is_preferred(tmp_path, monkeypatch):
    calls = []

    @contextmanager
    def public_context():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    def unexpected_private_context():
        raise AssertionError("the loaded public debugger capability takes precedence")

    monkeypatch.setitem(sys.modules, "pydevd", SimpleNamespace(skip_subprocess_arg_patch=public_context))
    monkeypatch.setitem(sys.modules, "_pydev_bundle.pydev_monkey", SimpleNamespace(skip_subprocess_arg_patch=unexpected_private_context))
    monkeypatch.setattr(_view, "find_spec", lambda name: object())
    monkeypatch.setattr(_view.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(_view.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    handle = _view._launch([sample()])
    assert calls == ["enter", "exit"]
    handle.close()


def test_parent_launch_failure_removes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(_view, "find_spec", lambda name: object())
    monkeypatch.setattr(_view.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("could not execute Python")

    monkeypatch.setattr(_view.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="execute"):
        _view._launch([sample()])
    assert not tmp_path.exists()


def test_cleanup_does_not_touch_an_unowned_directory(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "important.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("invalid", encoding="utf-8")
    _view._cleanup_snapshot(tmp_path)
    assert (tmp_path / "data" / "important.txt").read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "manifest.json").exists()


def test_cleanup_can_retry_a_temporarily_locked_data_file(tmp_path, monkeypatch):
    _view._write_snapshot([Display(sample())], tmp_path)
    original = _view.shutil.rmtree
    monkeypatch.setattr(_view.shutil, "rmtree", lambda *args, **kwargs: None)
    _view._cleanup_snapshot(tmp_path)
    assert (tmp_path / ".snapshot-owner").exists()
    monkeypatch.setattr(_view.shutil, "rmtree", original)
    _view._cleanup_snapshot(tmp_path)
    assert not tmp_path.exists()


def test_child_can_remove_its_redirected_log_without_parent_cleanup(tmp_path):
    log = tmp_path / "test.log"
    command = [sys.executable, "-c", "import pathlib,sys,torch; "
               "path=pathlib.Path(sys.argv[1]); path.unlink()", str(log)]
    with _view._open_log(log) as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    assert process.wait(timeout=30) == 0
    assert not log.exists()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_snapshot_waits_for_calling_stream(tmp_path):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        tensor = torch.arange(120.0, device="cuda").reshape(2, 1, 3, 4, 5).transpose(-1, -2)
        tensor.mul_(3)
        path = _view._write_snapshot([Display(Volume(tensor))], tmp_path)
    inputs, _ = _view._load_snapshot(path)
    assert torch.equal(inputs[0].data.tensor, torch.arange(120.0).reshape(2, 1, 3, 4, 5).transpose(-1, -2) * 3)


@pytest.mark.gui
@pytest.mark.skipif(os.environ.get("VOL5DKIT_TEST_GUI") != "1", reason="explicit GUI test opt-in required")
def test_real_child_smoke_is_independent_and_cleans_files(tmp_path):
    report = tmp_path / "report.json"
    screenshot = tmp_path / "snapshot.png"
    handle = _view._launch(
        [sample(), sample().numpy(), Display(torch.full((1, 3, 2, 3, 4), 0.5), rgb=True)],
        smoke=True, report=report, screenshot=screenshot,
    )
    directory = handle.log_path.parent
    assert handle.pid != os.getpid()
    # Let the child remove its own files, without using handle.poll() cleanup.
    code = handle._process.wait(timeout=60)
    assert code == 0, handle.log_path.read_text(encoding="utf-8")
    info = json.loads(report.read_text(encoding="utf-8"))
    assert info["trace_active"] is False
    assert info["first_render_ms"] > 0
    assert info["navigation_prepare_ms"] > 0
    assert screenshot.stat().st_size > 0
    assert not directory.exists()


@pytest.mark.gui
@pytest.mark.skipif(os.environ.get("VOL5DKIT_TEST_GUI") != "1", reason="explicit GUI test opt-in required")
def test_real_child_startup_failure_retains_traceback_and_releases_snapshot():
    handle = _view._launch([Display(sample(), cmap="vispy:no-such-colormap")])
    try:
        assert handle.wait(timeout=60) != 0
        assert "no-such-colormap" in handle.log_path.read_text(encoding="utf-8")
        assert not (handle.log_path.parent / "data").exists()
        assert not (handle.log_path.parent / "manifest.json").exists()
    finally:
        handle.close()
        _view._cleanup_snapshot(handle.log_path.parent)


@pytest.mark.skipif(
    any(_view.find_spec(name) is None for name in ("PySide6", "vispy", "OpenGL")),
    reason="GUI dependencies are required to launch the child",
)
def test_real_handle_close_during_startup_is_idempotent():
    for _ in range(2):
        handle = view(sample())
        directory = handle.log_path.parent
        handle.close()
        handle.close()
        assert handle.poll() is not None
        assert not directory.exists()
