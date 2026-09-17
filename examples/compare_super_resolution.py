"""Compare resolutions using reference coordinates that preserve volume bounds."""

import torch
import torch.nn.functional as F

import vol5dkit as v5


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, 24, device=device),
        torch.linspace(-1, 1, 48, device=device),
        torch.linspace(-1, 1, 48, device=device),
        indexing="ij",
    )
    frames = torch.stack([
        (0.6 * ((x - shift).abs() < 0.45) * (y.abs() < 0.45) * (z.abs() < 0.45)
         + 0.4 * torch.exp(-30 * ((x + 0.35) ** 2 + (y - shift) ** 2 + z**2))).clamp(0, 1)
        for shift in (0.0, 0.15)
    ]).unsqueeze(1)
    a = v5.Volume(frames, spacing=(2.0, 1.0, 1.0), times=(0.0, 0.1))

    # Interpolation baseline; the viewer still displays with nearest sampling.
    result = F.interpolate(a.tensor, scale_factor=2, mode="trilinear", align_corners=False)
    # Attach the original outer bounds and center without resampling again.
    b = v5.Volume(result, ref=a)

    viewer = v5.view(
        v5.Display(a, name="Input"),
        v5.Display(b, name="Upsampled"),
        window=(0, 1),
    )
    viewer.wait()


if __name__ == "__main__":
    main()
