"""Compare native grids using explicit boundary-preserving upsampling geometry."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import vol5dkit as v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="render one complete frame and close")
    parser.add_argument("--screenshot", type=Path, help="save the smoke frame as a PNG (implies --smoke)")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")

    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, 24, device=args.device),
        torch.linspace(-1, 1, 48, device=args.device),
        torch.linspace(-1, 1, 48, device=args.device),
        indexing="ij",
    )
    frames = torch.stack([
        (0.6 * ((x - shift).abs() < 0.45) * (y.abs() < 0.45) * (z.abs() < 0.45)
         + 0.4 * torch.exp(-30 * ((x + 0.35) ** 2 + (y - shift) ** 2 + z**2))).clamp(0, 1)
        for shift in (0.0, 0.15)
    ]).unsqueeze(1)
    a = v5.Volume(frames, spacing=(2.0, 1.0, 1.0), times=(0.0, 0.1))

    # This is an interpolation baseline, not a learned SR model. Interpolation
    # happens in the explicit processing step; the viewer still uses nearest.
    result = F.interpolate(a.tensor, scale_factor=2, mode="trilinear", align_corners=False)
    spacing = np.asarray(a.spacing) * np.asarray(a.shape[2:]) / np.asarray(result.shape[2:])
    offset_xyz = ((spacing - np.asarray(a.spacing)) / 2)[::-1]
    origin = np.asarray(a.origin) + np.asarray(a.direction) @ offset_xyz
    b = v5.Volume(result, spacing=spacing, origin=origin, direction=a.direction, times=a.times)

    viewer = v5.view(a, b, clim=(0.0, 1.0), interpolation="nearest", block=False)
    if not (args.smoke or args.screenshot):
        v5.run()
        return

    from PySide6.QtWidgets import QApplication
    from vispy.io import write_png

    app = QApplication.instance()
    try:
        deadline = time.monotonic() + 30
        while viewer.frame is None and viewer.last_error is None:
            if time.monotonic() > deadline:
                raise TimeoutError("viewer did not prepare the first frame")
            app.processEvents()
            time.sleep(0.005)
        if viewer.last_error is not None:
            raise RuntimeError(viewer.last_error)
        app.processEvents()
        pixels = viewer.canvas.render()
        if pixels.ndim != 3 or pixels.shape[0] < 2 or pixels.shape[1] < 2:
            raise RuntimeError("OpenGL did not produce a valid framebuffer")
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            write_png(str(args.screenshot), pixels)
        print(f"Rendered SR baseline on {args.device}: {tuple(a.shape)} -> {tuple(b.shape)}, framebuffer {pixels.shape}")
    finally:
        viewer.close()
        app.processEvents()


if __name__ == "__main__":
    main()
