"""Compare a synthetic noisy volume and a simple torch denoising result."""

import argparse
from pathlib import Path
import time

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

    torch.manual_seed(7)
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, 48, device=args.device),
        torch.linspace(-1, 1, 96, device=args.device),
        torch.linspace(-1, 1, 96, device=args.device),
        indexing="ij",
    )
    clean = torch.stack([
        (0.7 * torch.exp(-7 * ((x - shift) ** 2 + y**2 + z**2))
         + 0.3 * ((x.abs() < 0.6) & (y.abs() < 0.35) & (z.abs() < 0.15))).clamp(0, 1)
        for shift in (0.0, 0.15)
    ]).unsqueeze(1)
    noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
    a = v5.Volume(noisy, spacing=(2.0, 1.0, 1.0), times=(0.0, 0.1))

    # T is the batch dimension for avg_pool3d; C is preserved.
    padded = F.pad(a.tensor, (1, 1, 1, 1, 1, 1), mode="replicate")
    b = a.with_data(F.avg_pool3d(padded, kernel_size=3, stride=1))
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
        print(f"Rendered denoising comparison on {args.device}: {tuple(a.shape)}, framebuffer {pixels.shape}")
    finally:
        viewer.close()
        app.processEvents()


if __name__ == "__main__":
    main()
