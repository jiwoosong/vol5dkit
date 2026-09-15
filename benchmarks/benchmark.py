"""Measure preparation, wrapper cost, and optional real OpenGL render/readback."""

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import vol5dkit as v5
from vol5dkit.viewer._data import Request, make_source, prepare_frame


CASES = {
    "scalar256": ((1, 1, 256, 256, 256), False),
    "scalar512": ((1, 1, 512, 512, 512), False),
    "rgb64": ((1, 3, 64, 64, 64), True),
    "time128": ((3, 1, 128, 128, 128), False),
}


def summary(samples):
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def wrapper_benchmark(device, iterations):
    small = v5.Volume(torch.rand((1, 1, 16, 32, 32), device=device))
    repetitions = 200
    wrapped, native, endpoint = [], [], []
    for _ in range(20):
        small.with_data((small.tensor * 1.01 + 0.01).clamp(0, 1))
    synchronize(device)
    for _ in range(iterations):
        start = time.perf_counter()
        for _ in range(repetitions):
            result = small.with_data(small.tensor)
        wrapped.append((time.perf_counter() - start) * 1000 / repetitions)
        for samples, attach in ((native, False), (endpoint, True)):
            synchronize(device)
            start = time.perf_counter()
            for _ in range(repetitions):
                processed = (small.tensor * 1.01 + 0.01).clamp(0, 1)
                if attach:
                    result = small.with_data(processed)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000 / repetitions)
    return {
        "shape": list(small.shape),
        "repetitions_per_sample": repetitions,
        "with_data_only": summary(wrapped),
        "native_processing": summary(native),
        "processing_with_one_result_wrapper": summary(endpoint),
        "note": "CPU wall time per call; processing samples include device completion. No wrapper is inserted inside torch operations.",
    }


def gui_benchmark(volume, rgb, iterations):
    from PySide6.QtWidgets import QApplication

    started = time.perf_counter()
    viewer = v5.view(volume, rgb=rgb, block=False)
    app = QApplication.instance()
    try:
        deadline = time.monotonic() + 60
        while viewer.frame is None and viewer.last_error is None:
            if time.monotonic() > deadline:
                raise TimeoutError("GUI initial preparation timed out")
            app.processEvents()
            time.sleep(0.001)
        if viewer.last_error is not None:
            raise RuntimeError(viewer.last_error)
        app.processEvents()
        first_start = time.perf_counter()
        image = viewer.canvas.render()
        first_render_ms = (time.perf_counter() - first_start) * 1000
        initial_ms = (time.perf_counter() - started) * 1000
        samples = []
        for _ in range(iterations):
            app.processEvents()
            before = time.perf_counter()
            image = viewer.canvas.render()
            samples.append((time.perf_counter() - before) * 1000)
        return {
            "initial_window_prepare_and_render_ms": initial_ms,
            "first_explicit_render_readback_ms": first_render_ms,
            "warm_render_readback": summary(samples),
            "framebuffer_shape": list(image.shape),
            "note": "Real OpenGL drawing and framebuffer readback for a retained frame; not interactive FPS or GPU-only draw time.",
        }
    finally:
        viewer.close()
        app.processEvents()


def benchmark_case(name, device, iterations, gui=False, volume_3d=False):
    shape, rgb = CASES[name]
    torch.manual_seed(0)
    tensor = torch.rand(shape, device=device, dtype=torch.float32)
    volume = v5.Volume(tensor)
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        baseline_gpu = torch.cuda.memory_allocated()
    else:
        baseline_gpu = None
    started = time.perf_counter()
    source = make_source(volume, rgb=rgb)
    registration_ms = (time.perf_counter() - started) * 1000
    request = Request(0, "A", source, (0, 0.5, 0.5, 0.5), volume_3d=volume_3d and not rgb)
    started = time.perf_counter()
    frame = prepare_frame(request)
    initial_ms = (time.perf_counter() - started) * 1000
    clim = frame.clim
    samples = []
    for index in range(iterations):
        fraction = (index + 1) / (iterations + 1)
        request = Request(
            index + 1, "A", source, (fraction, fraction, 1 - fraction, 0.5),
            clim=clim, volume_3d=volume_3d and not rgb,
        )
        started = time.perf_counter()
        frame = prepare_frame(request)
        samples.append((time.perf_counter() - started) * 1000)
    cpu_bytes = sum(plane.raw.numel() * plane.raw.element_size() + plane.image.nbytes for plane in frame.planes)
    if frame.volume is not None:
        cpu_bytes += frame.volume.nbytes
    result = {
        "case": name,
        "shape_tcshw": list(shape),
        "rgb": rgb,
        "dtype": str(tensor.dtype),
        "input_bytes": tensor.numel() * tensor.element_size(),
        "registration_ms": registration_ms,
        "initial_prepare_ms": initial_ms,
        "warm_prepare": summary(samples),
        "clim": clim,
        "volume_3d": request.volume_3d,
        "d2h_image_bytes_per_frame": frame.transfer_bytes,
        "owned_cpu_frame_bytes": cpu_bytes,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "extra_gpu_peak_bytes": torch.cuda.max_memory_allocated() - baseline_gpu if device == "cuda" else None,
    }
    if gui:
        result["gui"] = gui_benchmark(volume, rgb, iterations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="also measure an actual local OpenGL render and readback")
    parser.add_argument("--volume-3d", action="store_true", help="explicitly transfer scalar SHW previews")
    parser.add_argument("--output", type=Path, help="write the same JSON report to this path")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vol5dkit": v5.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "iterations": args.iterations,
        "wrapper": wrapper_benchmark(args.device, args.iterations),
        "cases": [],
    }
    for case in args.cases:
        report["cases"].append(benchmark_case(case, args.device, args.iterations, args.gui, args.volume_3d))
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
