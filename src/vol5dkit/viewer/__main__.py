"""Private child entry point for an independent snapshot window."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

from .._view import _cleanup_snapshot, _load_snapshot


def _run(args):
    tick = time.perf_counter()
    inputs, manifest = _load_snapshot(args.manifest)
    report = {
        "snapshot_copy_ms": manifest["snapshot_copy_ms"],
        "snapshot_write_ms": manifest["snapshot_write_ms"],
        "child_load_ms": (time.perf_counter() - tick) * 1000,
        "child_start_ms": (tick - manifest["snapshot_ready"]) * 1000,
        "trace_active": sys.gettrace() is not None,
    }
    try:
        from PySide6 import QtWidgets
        from .window import Viewer
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "vispy"}:
            raise ImportError('The viewer requires: pip install "vol5dkit[gui]"') from exc
        raise
    application = QtWidgets.QApplication([])
    application.setApplicationName("vol5dkit")
    tick = time.perf_counter()
    viewer = Viewer(inputs, window=manifest["window"], interpolation=manifest["interpolation"])
    del inputs
    viewer.show()
    if args.volume_3d:
        viewer.view3d_box.setCurrentText("Volume")
    if args.smoke:
        from PySide6 import QtCore

        timer = QtCore.QTimer()
        timer.setInterval(1)
        timer.setTimerType(QtCore.Qt.PreciseTimer)
        report["diagnostic_poll_interval_ms"] = 1
        report["navigation_prepare_samples_ms"] = []
        report["navigation_render_samples_ms"] = []
        errors = []
        deadline = time.monotonic() + (120 if args.volume_3d else 30)
        first_revision = None
        navigation_started = None
        navigation_axis = None
        previous_volume_id = None
        report["navigation_volume_reused"] = []

        def move_next():
            nonlocal first_revision, navigation_started
            axis, tensor_axis = navigation_axis
            first_revision = viewer.frame.revision
            navigation_started = time.perf_counter()
            length = viewer.frame.source.volume.shape[tensor_axis]
            viewer._move_axis(axis, (viewer.frame.indices[tensor_axis] + 1) % length)

        def inspect():
            nonlocal first_revision, navigation_started, navigation_axis, previous_volume_id
            try:
                if viewer.last_error is not None:
                    raise RuntimeError(viewer.last_error)
                if time.monotonic() > deadline:
                    raise TimeoutError("viewer did not finish diagnostics before the deadline")
                if viewer.frame is None or viewer._busy or viewer._pending is not None:
                    return
                if args.volume_3d and viewer.frame.volume_buffer is None:
                    raise RuntimeError("3D diagnostics require a scalar input")
                if first_revision is None:
                    report["initial_prepare_ms"] = (time.perf_counter() - tick) * 1000
                    render_started = time.perf_counter()
                    pixels = viewer.canvas.render()
                    report["first_render_ms"] = (time.perf_counter() - render_started) * 1000
                    report["total_to_first_frame_ms"] = (time.perf_counter() - manifest["started"]) * 1000
                    report["framebuffer_shape"] = list(pixels.shape)
                    if pixels.ndim != 3 or min(pixels.shape[:2]) < 2:
                        raise RuntimeError("OpenGL did not produce a valid framebuffer")
                    if args.screenshot:
                        from vispy.io import write_png

                        destination = Path(args.screenshot)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        write_png(str(destination), pixels)
                    first_revision = viewer.frame.revision
                    if viewer.frame.volume_buffer is not None:
                        previous_volume_id = id(viewer.frame.volume_buffer)
                        report["volume_buffer_bytes"] = viewer.frame.volume_buffer.nbytes
                    source = viewer.frame.source.volume
                    navigation_axis = next(((a, i) for a, i in (("S", 2), ("H", 3), ("W", 4), ("T", 0))
                                            if source.shape[i] > 1), None)
                    if navigation_axis is not None:
                        move_next()
                        return
                elif navigation_started is not None:
                    if viewer.frame.revision == first_revision:
                        return
                    if previous_volume_id is not None:
                        report["navigation_volume_reused"].append(id(viewer.frame.volume_buffer) == previous_volume_id)
                        previous_volume_id = id(viewer.frame.volume_buffer)
                    report["navigation_prepare_ms"] = (time.perf_counter() - navigation_started) * 1000
                    report["navigation_prepare_samples_ms"].append(report["navigation_prepare_ms"])
                    render_started = time.perf_counter()
                    viewer.canvas.render()
                    report["navigation_render_ms"] = (time.perf_counter() - render_started) * 1000
                    report["navigation_render_samples_ms"].append(report["navigation_render_ms"])
                    if len(report["navigation_render_samples_ms"]) < args.iterations:
                        move_next()
                        return
                if args.report:
                    destination = Path(args.report)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
                timer.stop()
                viewer.close()
            except Exception as exc:
                errors.append(exc)
                timer.stop()
                viewer.close()

        timer.timeout.connect(inspect)
        timer.start()
    try:
        result = application.exec()
        if args.smoke and errors:
            raise errors[0]
        return result
    finally:
        viewer.close()
        from PySide6 import QtCore

        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--volume-3d", action="store_true")
    parser.add_argument("--screenshot")
    parser.add_argument("--report")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    status = 1
    try:
        status = _run(args)
    except Exception:
        traceback.print_exc()
    finally:
        # Release mmap references before removing files for portable cleanup.
        # _run's frame is gone; collect Qt callback cycles as well.
        gc.collect()
        _cleanup_snapshot(args.manifest.parent, keep_log=status != 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
