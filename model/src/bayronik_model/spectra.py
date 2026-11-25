"""
Power spectra of projected density maps in physical units.

Conventions (2D periodic map, side L in Mpc/h, N pixels per side):

    delta(x) = Sigma(x) / <Sigma> - 1
    delta_k  = sum_x delta(x) exp(-i k.x) * (L/N)^2
    P(k)     = |delta_k|^2 / L^2                    [(Mpc/h)^2]
    k        = 2 pi |n| / L                         [h/Mpc]

Modes are binned by round(|n|), so bin i holds the annulus around
k = i * 2 pi / L, from the fundamental (0.251 h/Mpc for L = 25) to the
Nyquist frequency (32.2 h/Mpc for N = 256). Everything is differentiable, so
the same code serves the training loss and the validation report.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from .fields import BOX_SIZE, MAP_RESOLUTION


class PowerSpectrum2D(nn.Module):
    """Binned auto and cross power spectra of batches of (B, 1, N, N) fields."""

    def __init__(self, resolution: int = MAP_RESOLUTION, box_size: float = BOX_SIZE):
        super().__init__()
        n = resolution
        self.resolution = n
        self.box_size = box_size
        self.n_bins = n // 2

        nx = torch.fft.fftfreq(n, d=1.0 / n)
        ny = torch.fft.rfftfreq(n, d=1.0 / n)
        radius = torch.sqrt(nx[:, None] ** 2 + ny[None, :] ** 2)
        shell = torch.round(radius).long()

        # rfft2 keeps half the plane: a mode with 0 < ny < N/2 stands for
        # itself and its complex conjugate.
        weight = torch.full_like(radius, 2.0)
        weight[:, 0] = 1.0
        if n % 2 == 0:
            weight[:, -1] = 1.0

        # Bin 0 collects the mean mode and everything beyond the Nyquist
        # circle; it is dropped on output.
        keep = (shell >= 1) & (shell <= self.n_bins)
        shell = torch.where(keep, shell, torch.zeros_like(shell))
        weight = torch.where(keep, weight, torch.zeros_like(weight))

        counts = torch.zeros(self.n_bins + 1).index_add_(0, shell.flatten(), weight.flatten())
        k_sum = torch.zeros(self.n_bins + 1).index_add_(
            0, shell.flatten(), (weight * radius).flatten()
        )

        self.register_buffer("shell", shell.flatten(), persistent=False)
        self.register_buffer("weight", weight.flatten(), persistent=False)
        self.register_buffer("counts", counts[1:], persistent=False)
        self.register_buffer(
            "k", 2 * math.pi / box_size * k_sum[1:] / counts[1:], persistent=False
        )
        # |FFT|^2 -> P(k): (L/N)^4 / L^2
        self.norm = box_size ** 2 / n ** 4

    def _bin(self, mode_power: torch.Tensor) -> torch.Tensor:
        flat = (mode_power.flatten(1) * self.weight).float()
        out = torch.zeros(flat.shape[0], self.n_bins + 1, device=flat.device, dtype=flat.dtype)
        out = out.index_add(1, self.shell, flat)
        return out[:, 1:] / self.counts * self.norm

    def modes(self, field: torch.Tensor) -> torch.Tensor:
        """rfft2 of (B, 1, N, N) or (B, N, N) fields, always in float32."""
        if field.dim() == 4:
            field = field.squeeze(1)
        return torch.fft.rfft2(field.float())

    def auto(self, field: torch.Tensor) -> torch.Tensor:
        """(B, n_bins) auto power spectrum."""
        f = self.modes(field)
        return self._bin(f.real ** 2 + f.imag ** 2)

    def cross(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (P_aa, P_bb, P_ab), each (B, n_bins)."""
        fa, fb = self.modes(a), self.modes(b)
        p_aa = self._bin(fa.real ** 2 + fa.imag ** 2)
        p_bb = self._bin(fb.real ** 2 + fb.imag ** 2)
        p_ab = self._bin(fa.real * fb.real + fa.imag * fb.imag)
        return p_aa, p_bb, p_ab

    def filter(self, field: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
        """Multiply each mode by transfer(k) and transform back.

        `transfer` is (n_bins,) or (B, n_bins). Modes outside the binned range
        (the mean, the corners beyond Nyquist) are left unchanged.
        """
        squeeze = field.dim() == 4
        f = self.modes(field)
        if transfer.dim() == 1:
            transfer = transfer[None]
        full = torch.cat([torch.ones_like(transfer[:, :1]), transfer], dim=1)
        gain = full[:, self.shell].reshape(-1, *f.shape[-2:])
        out = torch.fft.irfft2(f * gain, s=(self.resolution, self.resolution))
        return out[:, None] if squeeze else out


def density_contrast(density: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
    """delta = density / <reference> - 1, means taken per map.

    Pass the target as `reference` when comparing a prediction against it, so
    that an overall amplitude error in the prediction is not normalised away.
    """
    ref = density if reference is None else reference
    mean = ref.float().mean(dim=(-2, -1), keepdim=True)
    return density.float() / mean - 1.0


def correlation_coefficient(p_aa: torch.Tensor, p_bb: torch.Tensor, p_ab: torch.Tensor) -> torch.Tensor:
    """r(k) = P_ab / sqrt(P_aa P_bb)."""
    return p_ab / torch.sqrt(p_aa * p_bb).clamp_min(1e-30)
