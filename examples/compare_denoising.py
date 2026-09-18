"""Compare a synthetic noisy volume and a simple torch denoising result."""

import torch
import torch.nn.functional as F

import vol5dkit as v5d


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(7)
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, 48, device=device),
        torch.linspace(-1, 1, 96, device=device),
        torch.linspace(-1, 1, 96, device=device),
        indexing="ij",
    )
    clean = torch.stack([
        (0.7 * torch.exp(-7 * ((x - shift) ** 2 + y**2 + z**2))
         + 0.3 * ((x.abs() < 0.6) & (y.abs() < 0.35) & (z.abs() < 0.15))).clamp(0, 1)
        for shift in (0.0, 0.15)
    ]).unsqueeze(1)
    noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
    a = v5d.Volume(noisy, spacing=(2.0, 1.0, 1.0), times=(0.0, 0.1))

    # T is the batch dimension for avg_pool3d; C is preserved.
    padded = F.pad(a.tensor, (1, 1, 1, 1, 1, 1), mode="replicate")
    result = F.avg_pool3d(padded, kernel_size=3, stride=1)
    b = v5d.Volume(result, ref=a)
    residual = v5d.Volume(result - noisy, ref=a)
    viewer = v5d.view(
        v5d.Display(a, name="Noisy", window=(0, 1)),
        v5d.Display(b, name="Denoised", window=(0, 1)),
        v5d.Display(residual, name="Residual", window=(-0.2, 0.2), cmap="vispy:coolwarm"),
    )
    viewer.wait()


if __name__ == "__main__":
    main()
