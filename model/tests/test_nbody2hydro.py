"""The N-body -> hydro pipeline: units, spectra, data pairing, model symmetries.

These run on synthetic maps, so they need neither CAMELS files nor weights.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bayronik_model.dataset import NbodyHydroMaps, sim_split
from bayronik_model.fields import (
    FIDUCIAL_PARAMS,
    INPUT_FIELD_UNITS,
    PARAM_HI,
    PARAM_LO,
    decode_field,
    encode_field,
    encode_params,
    mean_surface_density,
)
from bayronik_model.losses import NbodyHydroLoss
from bayronik_model.spectra import PowerSpectrum2D, correlation_coefficient
from bayronik_model.ufno import PeriodicUFNO2d


def _small_model(**kwargs) -> PeriodicUFNO2d:
    torch.manual_seed(0)
    return PeriodicUFNO2d(base_channels=8, modes=8, depth=3, **kwargs)


def _train_head(model: PeriodicUFNO2d) -> None:
    """Give the zero-initialised output and FiLM layers some weight."""
    torch.manual_seed(1)
    with torch.no_grad():
        model.project[-1].weight.normal_(0, 0.1)
        for m in model.modules():
            if hasattr(m, "film"):
                m.film.weight.normal_(0, 0.1)


# --- units -----------------------------------------------------------------


def test_field_encoding_roundtrip():
    sigma = torch.tensor([1e10, 4.163e11, 1e15], dtype=torch.float64)
    assert torch.allclose(decode_field(encode_field(sigma)), sigma, rtol=1e-10)
    assert encode_field(np.array([1e11]))[0] == pytest.approx(0.0)


def test_param_encoding_range_and_fiducial():
    lo = encode_params(np.array(PARAM_LO, dtype=np.float64))
    hi = encode_params(np.array(PARAM_HI, dtype=np.float64))
    fid = encode_params(torch.tensor(FIDUCIAL_PARAMS, dtype=torch.float64))
    assert np.allclose(lo, -1.0) and np.allclose(hi, 1.0)
    assert torch.allclose(fid, torch.zeros(6, dtype=torch.float64), atol=1e-12)


def test_mean_surface_density_matches_cmd_maps():
    # measured mean of the fiducial hydro Mtot maps
    assert mean_surface_density(0.3) == pytest.approx(4.163e11, rel=2e-3)


# --- spectra ---------------------------------------------------------------


def test_power_spectrum_single_mode():
    n, box = 64, 25.0
    power = PowerSpectrum2D(n, box)
    x = torch.arange(n) * box / n
    mode, amp = 5, 0.3
    delta = amp * torch.cos(2 * math.pi * mode * x / box)[None, :].expand(n, n)
    p = power.auto(delta[None, None])[0]

    assert int(torch.argmax(p)) == mode - 1
    # a bin's k is the mean |k| of the lattice modes in its annulus
    assert abs(power.k[mode - 1].item() / (2 * math.pi / box) - mode) < 0.5
    # variance = sum over modes of P / L^2
    variance = (p * power.counts).sum() / box ** 2
    assert variance.item() == pytest.approx(amp ** 2 / 2, rel=1e-4)


def test_power_spectrum_white_noise_level():
    n, box, sigma = 128, 25.0, 0.7
    power = PowerSpectrum2D(n, box)
    delta = sigma * torch.randn(16, 1, n, n, generator=torch.Generator().manual_seed(0))
    p = power.auto(delta).mean(0)
    expected = sigma ** 2 * (box / n) ** 2
    assert torch.allclose(p[10:], torch.full_like(p[10:], expected), rtol=0.1)


def test_cross_correlation_and_filter():
    n = 64
    power = PowerSpectrum2D(n, 25.0)
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 1, n, n, generator=g)
    r_same = correlation_coefficient(*power.cross(a, 3.0 * a))
    r_flip = correlation_coefficient(*power.cross(a, -a))
    assert torch.allclose(r_same, torch.ones_like(r_same), atol=1e-5)
    assert torch.allclose(r_flip, -torch.ones_like(r_flip), atol=1e-5)

    transfer = torch.linspace(1.0, 0.5, power.n_bins)
    filtered = power.filter(a, transfer)
    ratio = power.auto(filtered) / power.auto(a)
    assert torch.allclose(ratio, (transfer ** 2).expand_as(ratio), rtol=1e-4)
    assert torch.allclose(filtered.mean(dim=(-2, -1)), a.mean(dim=(-2, -1)), atol=1e-6)


# --- data ------------------------------------------------------------------


def _write_dummy(tmp_path, n_sims=6, res=32):
    rng = np.random.default_rng(1)
    n = n_sims * NbodyHydroMaps.MAPS_PER_SIM
    hydro = (1e11 * rng.lognormal(0.0, 1.0, (n, res, res))).astype(np.float32)
    # N-body file: same field, 5% lower, stored as mass per pixel
    nbody = (hydro / 1.05 / INPUT_FIELD_UNITS["Mtot_Nbody"]).astype(np.float32)
    np.save(tmp_path / "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy", hydro)
    np.save(tmp_path / "Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy", nbody)
    params = np.column_stack([rng.uniform(lo, hi, n_sims) for lo, hi in zip(PARAM_LO, PARAM_HI)])
    np.savetxt(tmp_path / "params_LH_IllustrisTNG.txt", params)
    return hydro, params


def test_dataset_units_pairing_and_params(tmp_path):
    hydro, params = _write_dummy(tmp_path)
    sims = np.array([4, 1])
    data = NbodyHydroMaps(str(tmp_path), sims=sims)

    assert len(data) == 2 * 15
    # sims are sorted, so the first 15 maps belong to sim 1
    assert torch.allclose(decode_field(data.y[0, 0].double()), torch.from_numpy(hydro[15]).double(), rtol=1e-4)
    # after the unit factor, target / input is the 1.05 put in above
    offset = (data.y - data.x) * 0.5
    assert torch.allclose(offset, torch.full_like(offset, math.log10(1.05)), atol=1e-4)
    expected = encode_params(params[[1, 4]])
    assert np.allclose(data.params[0].numpy(), expected[0], atol=1e-5)
    assert np.allclose(data.params[15].numpy(), expected[1], atol=1e-5)


def test_dataset_rejects_unknown_input_field(tmp_path):
    _write_dummy(tmp_path)
    with pytest.raises(ValueError):
        NbodyHydroMaps(str(tmp_path), input_field="Mgas")


def test_augmentation_keeps_pairs_aligned(tmp_path):
    _write_dummy(tmp_path)
    data = NbodyHydroMaps(str(tmp_path))
    g = torch.Generator().manual_seed(3)
    seen_change = False
    for x, y, _ in data.batches(8, shuffle=True, augment=True, generator=g):
        offset = (y - x) * 0.5
        assert torch.allclose(offset, torch.full_like(offset, math.log10(1.05)), atol=1e-4)
        seen_change = True
    assert seen_change
    # augmentation permutes pixels, it does not change them
    x0 = data.x[:4].clone()
    xa, _ = NbodyHydroMaps.augment(x0.clone(), x0.clone(), torch.Generator().manual_seed(5))
    assert torch.allclose(xa.flatten(1).sort(dim=1).values, x0.flatten(1).sort(dim=1).values)
    assert not torch.equal(xa, x0)


def test_sim_split_is_disjoint_and_stable():
    a = sim_split(1000, 0.1, 0.1, seed=42)
    b = sim_split(1000, 0.1, 0.1, seed=42)
    sets = [set(a[k].tolist()) for k in ("train", "val", "test")]
    assert [len(s) for s in sets] == [800, 100, 100]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(np.array_equal(a[k], b[k]) for k in a)


# --- model -----------------------------------------------------------------


def test_untrained_model_is_identity():
    model = _small_model()
    x = torch.randn(2, 1, 64, 64)
    p = torch.rand(2, 6) * 2 - 1
    assert torch.equal(model(x, p), x)


def test_model_is_periodic_shift_equivariant():
    model = _small_model().eval()
    _train_head(model)
    x = torch.randn(1, 1, 64, 64)
    p = torch.zeros(1, 6)
    stride = 2 ** (model.depth - 1)
    with torch.no_grad():
        base = model(x, p)
        for shift in ((stride, 0), (0, 3 * stride), (5 * stride, 2 * stride)):
            moved = model(torch.roll(x, shift, dims=(-2, -1)), p)
            assert torch.allclose(moved, torch.roll(base, shift, dims=(-2, -1)), atol=2e-4)


def test_model_responds_to_parameters():
    model = _small_model().eval()
    _train_head(model)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a = model(x, torch.zeros(1, 6))
        b = model(x, torch.ones(1, 6))
    assert (a - b).abs().max() > 1e-4


def test_model_traces_to_torchscript():
    model = _small_model().eval()
    _train_head(model)
    x, p = torch.randn(1, 1, 64, 64), torch.zeros(1, 6)
    traced = torch.jit.trace(model, (x, p))
    with torch.no_grad():
        assert torch.allclose(traced(x, p), model(x, p), atol=1e-5)


# --- loss ------------------------------------------------------------------


def test_loss_zero_at_truth_and_sees_mass_error():
    loss = NbodyHydroLoss(resolution=64)
    y = torch.randn(2, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    at_truth = loss(y, y)
    assert at_truth["total"].item() == pytest.approx(0.0, abs=1e-6)

    # +0.02 in network units is +0.01 dex everywhere: 2.33% more mass
    shifted = loss(y + 0.02, y)
    assert shifted["mass"].item() == pytest.approx(10 ** 0.01 - 1, rel=1e-3)
    assert shifted["spectral"].item() == pytest.approx((2 * 0.01 * math.log(10)) ** 2, rel=1e-2)


def test_loss_spectral_term_sees_lost_small_scale_power():
    loss = NbodyHydroLoss(resolution=64)
    y = 0.5 * torch.randn(2, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    blurred = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(y, (1, 1, 1, 1), mode="circular"), 3, stride=1
    )
    out = loss(blurred, y)
    assert out["spectral"].item() > 0.1
    assert out["total"].requires_grad is False


def test_model_pre_projection_activations_are_bounded():
    model = PeriodicUFNO2d(base_channels=16, modes=8, depth=4, num_conditions=6)
    x = torch.randn(2, 1, 64, 64, generator=torch.Generator().manual_seed(0)) * 2.0
    p = torch.zeros(2, 6)
    captured = {}
    handle = model.project[-1].register_forward_pre_hook(
        lambda _m, inp: captured.update(std=inp[0].std().item(), mx=inp[0].abs().max().item())
    )
    try:
        with torch.no_grad():
            model(x, p)
    finally:
        handle.remove()
    assert captured["std"] < 5.0
    assert captured["mx"] < 50.0


def test_loss_stays_finite_on_extreme_outlier():
    loss = NbodyHydroLoss(resolution=64)
    y = torch.randn(2, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    pred = y.clone().requires_grad_(True)
    with torch.no_grad():
        pred[0, 0, 0, 0] = 100.0  # would overflow float32 exp() without clamping
    out = loss(pred, y)
    for key, val in out.items():
        assert torch.isfinite(val), f"{key} was non-finite: {val}"
    out["total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 0, 0, 0].item() > 0.0  # pixel MSE pulls outlier back down


# --- training utilities ----------------------------------------------------


def test_gradient_clipping_handles_complex_weights():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train import clip_grad_norm

    real = torch.nn.Parameter(torch.zeros(3))
    cplx = torch.nn.Parameter(torch.zeros(2, dtype=torch.cfloat))
    real.grad = torch.tensor([3.0, 0.0, 0.0])
    cplx.grad = torch.tensor([0.0 + 4.0j, 0.0 + 0.0j])

    total = clip_grad_norm([real, cplx], max_norm=1.0)
    assert total.item() == pytest.approx(5.0)
    clipped = math.sqrt(real.grad.pow(2).sum().item() + cplx.grad.abs().pow(2).sum().item())
    assert clipped == pytest.approx(1.0, rel=1e-4)
