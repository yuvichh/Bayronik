"""
Units and normalisation contract for the N-body -> hydro task.

Everything the network sees goes through this module, so training, validation
and the exported runtime cannot disagree about units.

Maps
    A map is a projected mass surface density Sigma in (Msun/h) / (Mpc/h)^2:
    a 25 x 25 x 5 (Mpc/h)^3 slab of a CAMELS box on a 256^2 grid.

    CMD hydro maps (Mtot, Mcdm, ...) are already surface densities. CMD N-body
    maps store the mass per pixel, so they are multiplied by the pixel density
    (256 / 25)^2 to get the same units (checked: per-map total mass of the
    N-body and hydro maps then agrees to 1.000 +- 0.006).

    The network works on u = (log10 Sigma - 11) / 0.5 for input and target
    alike, so that u_target - u_input is the baryonic correction in units of
    half a dex.

Parameters
    The six CAMELS-IllustrisTNG parameters are mapped to [-1, 1]: Omega_m and
    sigma_8 linearly, the four feedback amplitudes logarithmically (that is how
    the Latin hypercube samples them). The fiducial model is the origin.
"""

import math
from typing import Sequence

import numpy as np
import torch

BOX_SIZE = 25.0        # Mpc/h, side of a CAMELS box
SLAB_DEPTH = 5.0       # Mpc/h, projection depth of one CMD map
MAP_RESOLUTION = 256

LOG10_SIGMA_MEAN = 11.0
LOG10_SIGMA_STD = 0.5

# Factor that turns each input field into a surface density.
INPUT_FIELD_UNITS = {
    "Mtot_Nbody": (MAP_RESOLUTION / BOX_SIZE) ** 2,  # gravity-only run, mass per pixel
    "Mcdm": 1.0,                                     # dark matter of the hydro run (legacy)
}

PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PARAM_LO = (0.1, 0.6, 0.25, 0.25, 0.5, 0.5)
PARAM_HI = (0.5, 1.0, 4.0, 4.0, 2.0, 2.0)
PARAM_LOG = (False, False, True, True, True, True)
FIDUCIAL_PARAMS = (0.3, 0.8, 1.0, 1.0, 1.0, 1.0)


def encode_field(sigma):
    """Surface density -> network units. Accepts numpy arrays or tensors."""
    log10 = torch.log10 if isinstance(sigma, torch.Tensor) else np.log10
    return (log10(sigma) - LOG10_SIGMA_MEAN) / LOG10_SIGMA_STD


def decode_field(u):
    """Network units -> surface density."""
    return 10.0 ** (u * LOG10_SIGMA_STD + LOG10_SIGMA_MEAN)


def relative_density(u: torch.Tensor) -> torch.Tensor:
    """Sigma / 10^LOG10_SIGMA_MEAN, computed in float32.

    Enough for anything that only needs ratios (mass ratios, density
    contrast), and it stays far from the float32 range limit.
    """
    return torch.exp(u.float() * (LOG10_SIGMA_STD * math.log(10.0)))


def encode_params(params):
    """Physical parameters (..., 6) -> [-1, 1]. Accepts numpy arrays or tensors."""
    is_tensor = isinstance(params, torch.Tensor)
    xp = torch if is_tensor else np
    cols = []
    for i in range(len(PARAM_NAMES)):
        p, lo, hi = params[..., i], PARAM_LO[i], PARAM_HI[i]
        if PARAM_LOG[i]:
            p, lo, hi = xp.log(p), math.log(lo), math.log(hi)
        cols.append(2.0 * (p - lo) / (hi - lo) - 1.0)
    return xp.stack(cols, -1)


def mean_surface_density(omega_m: float) -> float:
    """Cosmic mean of a CMD map: Omega_m * rho_crit * slab depth."""
    rho_crit = 2.775e11  # (Msun/h) / (Mpc/h)^3
    return omega_m * rho_crit * SLAB_DEPTH


def as_param_array(params: Sequence[float]) -> np.ndarray:
    arr = np.asarray(params, dtype=np.float32)
    if arr.shape[-1] != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} parameters {PARAM_NAMES}, got shape {arr.shape}")
    return arr
