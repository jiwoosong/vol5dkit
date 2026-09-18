"""Exercise the public snapshot API while a real debugpy target is suspended."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(os.environ.get("VOL5DKIT_TEST_GUI") != "1", reason="requires a local GUI"),
    pytest.mark.skipif(importlib.util.find_spec("debugpy") is None, reason="optional debugpy integration"),
]


class _DAP:
    """The small subset of Debug Adapter Protocol needed for this test."""

    def __init__(self, address):
        self.socket = socket.create_connection(tuple(address), timeout=30)
        self.stream = self.socket.makefile("rb")
        self.sequence = 0
        self.messages = []

    def send(self, command, **arguments):
        self.sequence += 1
        data = json.dumps({"seq": self.sequence, "type": "request", "command": command,
                           "arguments": arguments}).encode()
        self.socket.sendall(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        return self.sequence

    def receive(self, predicate):
        for index, message in enumerate(self.messages):
            if predicate(message):
                return self.messages.pop(index)
        while True:
            headers = {}
            while True:
                line = self.stream.readline()
                if not line:
                    raise RuntimeError("debug adapter disconnected")
                if line == b"\r\n":
                    break
                key, value = line.decode().split(":", 1)
                headers[key.lower()] = value.strip()
            message = json.loads(self.stream.read(int(headers["content-length"])))
            if predicate(message):
                return message
            self.messages.append(message)

    def response(self, sequence):
        message = self.receive(lambda item: item.get("request_seq") == sequence)
        assert message["success"], message
        return message.get("body", {})

    def event(self, name):
        return self.receive(lambda item: item.get("event") == name).get("body", {})

    def close(self):
        self.stream.close()
        self.socket.close()


def _wait_for_file(path, process, log):
    deadline = time.monotonic() + 60
    while not path.exists():
        assert process.poll() is None, log.read_text(encoding="utf-8", errors="replace")
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out waiting for {path}: {log.read_text(encoding='utf-8', errors='replace')}")
        time.sleep(0.02)


def test_public_view_navigates_and_renders_while_debuggee_is_at_breakpoint(tmp_path):
    target = tmp_path / "target.py"
    address = tmp_path / "address.json"
    report = tmp_path / "viewer.json"
    resumed = tmp_path / "resumed"
    target_log = tmp_path / "target.log"
    handle_log = tmp_path / "handle-log.txt"
    target.write_text('''
import json
from pathlib import Path
import sys
import debugpy
import torch
import vol5dkit as v5d
import vol5dkit._view as process_view

directory = Path(sys.argv[1])
original_launch = process_view._launch
def diagnostic_launch(inputs, **kwargs):
    handle = original_launch(inputs, smoke=True, iterations=3,
                             report=directory / "viewer.json", **kwargs)
    (directory / "handle-log.txt").write_text(str(handle.log_path), encoding="utf-8")
    return handle
process_view._launch = diagnostic_launch
a = v5d.Volume(torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 1, 3, 4, 5))
b = v5d.Volume(a.tensor * 0.5, ref=a)
endpoint = debugpy.listen(("127.0.0.1", 0))
(directory / "address.json").write_text(json.dumps(endpoint), encoding="utf-8")
debugpy.wait_for_client()
debugpy.breakpoint()
(directory / "resumed").write_text("continued", encoding="utf-8")
assert viewer.wait(timeout=30) == 0
assert all(name not in sys.modules for name in ("PySide6", "vispy", "OpenGL"))
''', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    client = None
    with target_log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, str(target), str(tmp_path)], env=env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    try:
        _wait_for_file(address, process, target_log)
        client = _DAP(json.loads(address.read_text(encoding="utf-8")))
        client.response(client.send("initialize", adapterID="debugpy", clientID="vol5dkit-test",
                                    pathFormat="path", linesStartAt1=True, columnsStartAt1=True))
        attached = client.send("attach", justMyCode=False, subProcess=True)
        client.event("initialized")
        client.response(client.send("configurationDone"))
        client.response(attached)
        stopped = client.event("stopped")
        stack = client.response(client.send("stackTrace", threadId=stopped["threadId"]))
        frame = next(item for item in stack["stackFrames"]
                     if Path(item.get("source", {}).get("path", "")) == target)
        # This invokes the actual public view at a breakpoint. The private
        # launch wrapper only enables child-owned diagnostics and auto-close.
        client.response(client.send("evaluate", expression="viewer = v5d.view(a, b)",
                                    frameId=frame["id"], context="repl"))
        _wait_for_file(report, process, target_log)
        results = json.loads(report.read_text(encoding="utf-8"))
        assert not resumed.exists(), "parent must still be suspended during child navigation"
        assert not results["trace_active"]
        assert len(results["navigation_prepare_samples_ms"]) == 3
        assert len(results["navigation_render_samples_ms"]) == 3
        assert min(results["framebuffer_shape"][:2]) > 1
        client.response(client.send("continue", threadId=stopped["threadId"]))
        assert process.wait(timeout=30) == 0, target_log.read_text(encoding="utf-8", errors="replace")
        assert resumed.exists()
    finally:
        if client is not None:
            client.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        # The diagnostic child auto-closes on success or after its 30 s deadline.
        if handle_log.exists() and not report.exists():
            log = Path(handle_log.read_text(encoding="utf-8"))
            if log.exists():
                print(log.read_text(encoding="utf-8", errors="replace"))
