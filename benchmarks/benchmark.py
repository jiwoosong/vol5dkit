"""Measure preparation, wrapper cost, and optional real OpenGL render/readback."""

import argparse
import gc
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

import vol5dkit as v5d
from vol5dkit._view import _load_snapshot, _normalize_inputs, _write_snapshot
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


def core_benchmark(device, iterations):
    """Separate coordinate validation, result attachment, reindexing and kernels."""
    torch.manual_seed(0)

    def measure(operation, repetitions):
        for _ in range(10):
            operation()
        samples = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            for _ in range(repetitions):
                operation()
            synchronize(device)
            samples.append((time.perf_counter() - started) * 1000 / repetitions)
        return summary(samples)

    coordinates = []
    for frames in (1, 1000, 100000):
        tensor = torch.empty((frames, 1, 1, 1, 1), device=device)
        ref = v5d.Volume(tensor)
        coordinates.append({
            "T": frames,
            "construct_default_times": measure(lambda: v5d.Volume(tensor), 10),
            "construct_existing_coordinates": measure(
                lambda: v5d.Volume(tensor, spacing=ref.spacing, origin=ref.origin,
                                  direction=ref.direction, times=ref.times), 10
            ),
            "reference_same_grid": measure(lambda: v5d.Volume(tensor, ref=ref), 200),
        })

    ref = v5d.Volume(torch.rand((1, 1, 16, 32, 32), device=device), spacing=(2, 1, 0.5))
    resized = torch.empty((1, 1, 32, 64, 64), device=device)
    spatial = {
        "shape_tcshw": list(ref.shape),
        "attach_changed_grid": measure(lambda: v5d.Volume(resized, ref=ref), 200),
        "crop_view": measure(lambda: ref.crop(s=slice(1, None, 2)), 200),
        "permute_view": measure(lambda: ref.permute_spatial("w", "s", "h"), 200),
        "flip_copy": measure(lambda: ref.flip_spatial("s", "w"), 200),
    }

    # Both paths pass the very same native Tensor to this ordinary torch model.
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv3d(4, 1, 3, padding=1),
    ).to(device).eval()
    tensor = ref.tensor.detach().requires_grad_()
    volume = v5d.Volume(tensor, ref=ref)
    direct = model(tensor)
    via_volume = model(volume.tensor)
    torch.testing.assert_close(direct, via_volume)
    direct_grad = torch.autograd.grad(direct.sum(), tensor)[0]
    volume_grad = torch.autograd.grad(via_volume.sum(), tensor)[0]
    torch.testing.assert_close(direct_grad, volume_grad)
    direct_samples, volume_samples = [], []
    with torch.inference_mode():
        for _ in range(10):
            model(tensor)
            model(volume.tensor)
        for index in range(iterations):
            # Alternate order to reduce systematic warm-up / clock bias.
            paths = [(direct_samples, lambda: model(tensor)),
                     (volume_samples, lambda: model(volume.tensor))]
            for samples, forward in paths[::1 if index % 2 == 0 else -1]:
                synchronize(device)
                started = time.perf_counter()
                for _ in range(20):
                    forward()
                synchronize(device)
                samples.append((time.perf_counter() - started) * 1000 / 20)
    return {
        "coordinates": coordinates,
        "spatial": spatial,
        "forward": {
            "model": "Conv3d(1,4,3)-ReLU-Conv3d(4,1,3), padding=1, eval",
            "same_tensor_object": volume.tensor is tensor,
            "outputs_and_input_gradients_match": True,
            "native_tensor": summary(direct_samples),
            "volume_tensor": summary(volume_samples),
        },
        "note": "Per-call wall time in milliseconds, with warm-up and CUDA synchronization. Model timings use inference_mode; output/gradient correctness is checked separately. No timing ratio is a pass/fail threshold.",
    }


def gui_benchmark(volume, rgb, iterations, volume_3d=False):
    from vol5dkit._view import _launch

    with tempfile.TemporaryDirectory(prefix="vol5dkit-benchmark-") as directory:
        report_path = Path(directory) / "timings.json"
        started = time.perf_counter()
        viewer = _launch(
            [v5d.Display(volume, rgb=rgb)],
            smoke=True, report=report_path, iterations=iterations, volume_3d=volume_3d,
        )
        returned_ms = (time.perf_counter() - started) * 1000
        try:
            code = viewer.wait(timeout=120)
            if code:
                raise RuntimeError(f"Snapshot viewer exited with code {code}; see {viewer.log_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        finally:
            viewer.close()
    report["view_return_ms"] = returned_ms
    for key in ("navigation_prepare", "navigation_render"):
        samples = report.get(f"{key}_samples_ms", [])
        if samples:
            report[key] = summary(samples)
    report["snapshot_bytes"] = volume.tensor.numel() * volume.tensor.element_size()
    report["note"] = (
        "Independent child with a full CPU snapshot; initial copy and file write are separate. "
        "Navigation timings include the child's Qt event loop and diagnostic timer polling. "
        "Rendering includes framebuffer readback, not GPU-only drawing."
    )
    return report


def benchmark_case(name, device, iterations, gui=False, volume_3d=False):
    shape, rgb = CASES[name]
    torch.manual_seed(0)
    volume = v5d.Volume(torch.rand(shape, device=device, dtype=torch.float32))
    synchronize(device)
    volume_3d = volume_3d and not rgb
    with tempfile.TemporaryDirectory(prefix="vol5dkit-benchmark-") as directory:
        # Exercise the public transport: complete CPU snapshot, file write, mmap.
        manifest = _write_snapshot(_normalize_inputs([v5d.Display(volume, rgb=rgb)]), directory)
        started = time.perf_counter()
        inputs, transport = _load_snapshot(manifest)
        load_ms = (time.perf_counter() - started) * 1000
        source = make_source(inputs[0].data, rgb)
        request = Request(0, 0, source, (0, 0.5, 0.5, 0.5), volume_3d=volume_3d)
        started = time.perf_counter()
        frame = prepare_frame(request)
        initial_ms = (time.perf_counter() - started) * 1000
        window = frame.window
        buffer = frame.volume_buffer
        uncached, cached = [], []
        for index in range(iterations):
            fraction = (index + 1) / (iterations + 1)
            request = Request(index + 1, 0, source, (0, fraction, 1 - fraction, 0.5),
                              window=window, volume_3d=volume_3d)
            if volume_3d:
                started = time.perf_counter()
                uncached_frame = prepare_frame(request)
                uncached.append((time.perf_counter() - started) * 1000)
                assert uncached_frame.volume_buffer is not buffer
                del uncached_frame
            started = time.perf_counter()
            frame = prepare_frame(request, volume_buffer=buffer)
            cached.append((time.perf_counter() - started) * 1000)
            assert frame.volume_buffer is buffer
        plane_bytes = sum(p.raw.numel() * p.raw.element_size() + p.image.nbytes for p in frame.planes)
        result = {
            "case": name, "shape_tcshw": list(shape), "rgb": rgb,
            "dtype": str(volume.dtype), "input_bytes": volume.tensor.numel() * volume.tensor.element_size(),
            "snapshot_copy_ms": transport["snapshot_copy_ms"],
            "snapshot_write_ms": transport["snapshot_write_ms"], "mmap_load_ms": load_ms,
            "initial_prepare_ms": initial_ms, "window": window, "volume_3d": volume_3d,
            "spatial_prepare": summary(cached),
            "spatial_prepare_without_buffer_reuse": summary(uncached) if uncached else None,
            "owned_plane_bytes": plane_bytes,
            "volume_buffer_bytes": buffer.nbytes if buffer is not None else 0,
            "volume_buffer_reused": volume_3d,
            "note": "Full snapshot from the selected device; frame preparation on CPU. Spatial navigation keeps T/C/window fixed. Owned buffer sizes exclude mapped input and bounded normalization workspace.",
        }
        del inputs, source, request, frame, buffer
        gc.collect()
    if gui:
        result["gui"] = gui_benchmark(volume, rgb, iterations, volume_3d)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="also measure independent snapshot launch, preparation and real OpenGL rendering")
    parser.add_argument("--volume-3d", action="store_true", help="measure full scalar 3D preparation and buffer reuse")
    parser.add_argument("--core-only", action="store_true", help="measure coordinate and model paths without large display cases")
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
            "vol5dkit": v5d.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "iterations": args.iterations,
        "core": core_benchmark(args.device, args.iterations),
        "cases": [],
    }
    for case in (() if args.core_only else args.cases):
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
