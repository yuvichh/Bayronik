#!/usr/bin/env python3
"""
Benchmark for the N-body -> hydro task, with the baselines a result must beat.

Evaluated on simulations the model never saw: the held-out LH test sims and
the CV set. All spectra are of the density contrast, in physical k (h/Mpc).

Methods compared
    identity        the N-body map itself ("no baryons")
    amp_mean        N-body modes scaled by sqrt of the mean suppression S(k)
                    of the training sims (one curve for all parameters)
    amp_emulated    the same with S(k | parameters) from a small MLP fitted on
                    the training sims: an amplitude-only emulator
    amp_oracle      the same with each map's own true S(k): the ceiling of any
                    amplitude-only correction
    model           the trained network (with --weights)

An amplitude-only correction cannot change the phases, so its
cross-correlation r(k) with the truth equals the identity's. That, and the
pixel-level error, is where a field-level model has to earn its keep.

Usage:
    uv run python benchmarks/nbody2hydro.py                       # baselines only
    uv run python benchmarks/nbody2hydro.py --weights weights/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bayronik_model.dataset import NbodyHydroMaps, sim_split  # noqa: E402
from bayronik_model.fields import BOX_SIZE, LOG10_SIGMA_STD, relative_density  # noqa: E402
from bayronik_model.spectra import PowerSpectrum2D  # noqa: E402

# Report bins, as ranges of the integer shell index n (k = n * 2 pi / L)
SHELL_EDGES = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, 65, 86, 113, 129)
SUMMARY_K = (1.0, 5.0, 10.0, 20.0)
BATCH = 64


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def contrast(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Density contrast of a field in network units, and its mean density."""
    rho = relative_density(u)
    mean = rho.mean(dim=(-2, -1), keepdim=True)
    return rho / mean - 1.0, mean


@torch.no_grad()
def suppression_per_map(power: PowerSpectrum2D, data: NbodyHydroMaps) -> torch.Tensor:
    """S(k) = P_hydro / P_nbody for every map, on the linear shells."""
    out = []
    for x, y, _ in data.batches(BATCH):
        out.append(power.auto(contrast(y)[0]) / power.auto(contrast(x)[0]))
    return torch.cat(out)


def fit_amplitude_emulator(params: torch.Tensor, log_s: torch.Tensor, steps: int = 600) -> nn.Module:
    """Small MLP for ln S(k | parameters): an amplitude-only emulator."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(params.shape[1], 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, log_s.shape[1]),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((net(params) - log_s) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return net.eval()


@torch.no_grad()
def rescale_amplitudes(power: PowerSpectrum2D, x: torch.Tensor, s_of_k: torch.Tensor):
    """Scale the modes of the N-body contrast by sqrt(S(k)); back to network units.

    A filtered density is not guaranteed positive: the ringing around halos
    can push void pixels below zero. Those pixels are floored at the emptiest
    pixel of the input map. Returns the field and the floored fraction, which
    is what makes the log-field error of these baselines large.
    """
    delta, mean = contrast(x)
    rho = mean * (1.0 + power.filter(delta, torch.sqrt(s_of_k.clamp_min(0.0))))
    floor = relative_density(x).amin(dim=(-2, -1), keepdim=True)
    clipped = (rho < floor).float().mean().item()
    rho = torch.maximum(rho, floor)
    return torch.log(rho) / (LOG10_SIGMA_STD * np.log(10.0)), clipped


@torch.no_grad()
def model_predictions(weights: Path, data: NbodyHydroMaps, device: torch.device) -> tuple[torch.Tensor, dict]:
    from train import create_model, predict

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"], config.get("model_kwargs"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    out = []
    for x, _, params in data.batches(16):
        out.append(predict(model, config["model"], x.to(device), params.to(device)).float().cpu())
    return torch.cat(out), config


def rebin(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Mode-count weighted average of (maps, shells) into the report bins."""
    cols = []
    for lo, hi in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]):
        w = counts[lo - 1 : hi - 1]
        cols.append((values[:, lo - 1 : hi - 1] * w).sum(1) / w.sum())
    return torch.stack(cols, dim=1)


@torch.no_grad()
def evaluate(power: PowerSpectrum2D, pred: torch.Tensor, data: NbodyHydroMaps) -> dict:
    """All metrics of one method on one evaluation set."""
    p_pred, p_true, p_cross, p_in = [], [], [], []
    mse, mass = [], []
    edges = torch.linspace(-3.0, 7.0, 81)
    hist_pred = torch.zeros(80)
    hist_true = torch.zeros(80)

    for start in range(0, len(data), BATCH):
        u = pred[start : start + BATCH]
        x, y = data.x[start : start + BATCH], data.y[start : start + BATCH]
        d_pred, mean_pred = contrast(u)
        d_true, mean_true = contrast(y)
        a, b, c = power.cross(d_pred, d_true)
        p_pred.append(a); p_true.append(b); p_cross.append(c)
        p_in.append(power.auto(contrast(x)[0]))
        mse.append(((u - y) ** 2).mean(dim=(1, 2, 3)) * LOG10_SIGMA_STD ** 2)
        mass.append((mean_pred / mean_true).flatten())
        hist_pred += torch.histc(u, bins=80, min=-3.0, max=7.0)
        hist_true += torch.histc(y, bins=80, min=-3.0, max=7.0)

    counts = power.counts
    p_pred, p_true, p_cross, p_in = (rebin(torch.cat(v), counts) for v in (p_pred, p_true, p_cross, p_in))
    mse, mass = torch.cat(mse), torch.cat(mass)

    ratio = p_pred / p_true
    r = p_cross / torch.sqrt(p_pred * p_true)
    k = rebin(power.k[None], counts)[0]

    quantile = lambda v, q: torch.quantile(v, q, dim=0).tolist()
    result = {
        "n_maps": len(data),
        "log_mse_dex2": mse.mean().item(),
        "mass_ratio_mean": mass.mean().item(),
        "mass_ratio_std": mass.std().item(),
        "pdf_l1": ((hist_pred / hist_pred.sum() - hist_true / hist_true.sum()).abs().sum() / 2).item(),
        "k": k.tolist(),
        "power_ratio_mean": ratio.mean(0).tolist(),
        "power_ratio_p16": quantile(ratio, 0.16),
        "power_ratio_p84": quantile(ratio, 0.84),
        "power_ratio_rms_error": torch.sqrt(((ratio - 1) ** 2).mean(0)).tolist(),
        "r_mean": r.mean(0).tolist(),
        "suppression_mean": (p_pred / p_in).mean(0).tolist(),
        "suppression_true_mean": (p_true / p_in).mean(0).tolist(),
    }
    below_10 = k <= 10.0
    result["max_power_bias_k_le_10"] = (ratio.mean(0)[below_10] - 1).abs().max().item()
    result["max_power_rms_error_k_le_10"] = torch.sqrt(((ratio - 1) ** 2).mean(0))[below_10].max().item()
    for k_report in SUMMARY_K:
        i = int(torch.argmin((k - k_report).abs()))
        result[f"power_ratio_k{k_report:g}"] = ratio.mean(0)[i].item()
        result[f"r_k{k_report:g}"] = r.mean(0)[i].item()
    return result


def write_markdown(path: Path, report: dict) -> None:
    lines = ["# N-body -> hydro benchmark", ""]
    meta = report["meta"]
    lines += [
        f"- Input: `{meta['input_field']}`; target: hydro `Mtot` ({meta['suite']}), z = 0",
        f"- Weights: `{meta['weights'] or 'none (baselines only)'}`",
        f"- Split seed {meta['split_seed']}: {meta['n_train_sims']} train / {meta['n_test_sims']} held-out test sims",
        "- Spectra are of the density contrast, k in h/Mpc. P/P_true is the mean over maps.",
        "",
    ]
    for set_name, methods in report["sets"].items():
        n = next(iter(methods.values()))["n_maps"]
        lines += [f"## {set_name} ({n} maps)", ""]
        lines.append(
            "| Method | log-MSE (1e-3 dex²) | floored pixels | mass ratio | max \\|P/P_true − 1\\|, k ≤ 10 | "
            + " | ".join(f"P/P_true k={k:g}" for k in SUMMARY_K) + " | "
            + " | ".join(f"r k={k:g}" for k in SUMMARY_K) + " |"
        )
        lines.append("|---|" + "---:|" * (4 + 2 * len(SUMMARY_K)))
        for name, m in methods.items():
            lines.append(
                f"| {name} | {m['log_mse_dex2'] * 1e3:.3f} | {m['floored_pixel_fraction'] * 100:.2f}% | "
                f"{m['mass_ratio_mean']:.4f} ± {m['mass_ratio_std']:.4f} | "
                f"{m['max_power_bias_k_le_10']:.3f} | "
                + " | ".join(f"{m[f'power_ratio_k{k:g}']:.3f}" for k in SUMMARY_K) + " | "
                + " | ".join(f"{m[f'r_k{k:g}']:.4f}" for k in SUMMARY_K) + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def write_figure(path: Path, report: dict, set_name: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ink, secondary, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    styles = {
        "identity": ("N-body (no baryons)", secondary),
        "amp_emulated": ("amplitude emulator", "#eb6834"),
        "model": ("U-FNO", "#2a78d6"),
    }
    methods = report["sets"][set_name]
    k = np.array(next(iter(methods.values()))["k"])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": secondary, "axes.labelcolor": ink,
                         "xtick.color": secondary, "ytick.color": secondary})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=surface, layout="constrained")
    panels = (
        ("power_ratio_mean", r"$P_{\rm pred}(k)\,/\,P_{\rm hydro}(k)$", "Power relative to the hydro simulation"),
        ("r_mean", r"$r(k)$ with the hydro map", "Cross-correlation with the hydro simulation"),
        ("suppression_mean", r"$S(k) = P(k)\,/\,P_{\rm N\text{-}body}(k)$", "Baryonic suppression"),
    )
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        if key == "suppression_mean":
            truth = np.array(methods["identity"]["suppression_true_mean"])
            ax.plot(k, truth, color=ink, lw=2, label="hydro simulation")
        for name, (label, color) in styles.items():
            if name not in methods:
                continue
            ax.plot(k, methods[name][key], color=color, lw=2, label=label)
            if key == "power_ratio_mean" and name == "model":
                ax.fill_between(k, methods[name]["power_ratio_p16"], methods[name]["power_ratio_p84"],
                                color=color, alpha=0.15, lw=0)
        if key != "r_mean":
            ax.axhline(1.0, color=secondary, lw=0.8, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
        ax.set_xlabel(r"$k$  [$h$/Mpc]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.grid(True, color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, loc="lower left")
    n = next(iter(methods.values()))["n_maps"]
    fig.suptitle(f"{set_name}: {n} maps from simulations not used in training", x=0.01, ha="left", color=ink, fontsize=12)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=None, help="best.pt of a training run")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--suite", default="IllustrisTNG")
    parser.add_argument("--input-field", default=None, help="default: taken from the weights, else Mtot_Nbody")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fit-sims", type=int, default=0, help="training sims used to fit the amplitude baselines (0 = all)")
    parser.add_argument("--max-test-sims", type=int, default=0, help="evaluate on fewer held-out sims (0 = all)")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "nbody2hydro")
    args = parser.parse_args()

    device = get_device(args.device)
    input_field = args.input_field
    split_seed, val_split, test_split = args.split_seed, 0.10, 0.10
    if args.weights is not None:
        config = torch.load(args.weights, map_location="cpu", weights_only=False)["config"]
        input_field = input_field or config["input_field"]
        split_seed, val_split, test_split = config["split_seed"], config["val_split"], config["test_split"]
    input_field = input_field or "Mtot_Nbody"

    split = sim_split(1000, val_split, test_split, split_seed)
    fit_sims = split["train"][: args.fit_sims] if args.fit_sims else split["train"]
    test_sims = split["test"][: args.max_test_sims] if args.max_test_sims else split["test"]
    common = dict(data_dir=str(args.data_dir), suite=args.suite, input_field=input_field)

    power = PowerSpectrum2D(256, BOX_SIZE)

    # Amplitude baselines are fitted on training sims only
    t0 = time.time()
    s_train, p_train = [], []
    for chunk in np.array_split(fit_sims, max(1, len(fit_sims) // 50)):
        part = NbodyHydroMaps(dataset_type="LH", sims=chunk, **common)
        s_train.append(suppression_per_map(power, part))
        p_train.append(part.params)
    s_train, p_train = torch.cat(s_train), torch.cat(p_train)
    s_mean = s_train.mean(0)
    emulator = fit_amplitude_emulator(p_train, torch.log(s_train))
    print(f"Amplitude baselines fitted on {len(fit_sims)} training sims in {time.time() - t0:.0f}s")

    eval_sets = {"LH held-out test sims": NbodyHydroMaps(dataset_type="LH", sims=test_sims, **common)}
    cv_path = args.data_dir / f"Maps_Mtot_{args.suite}_CV_z=0.00.npy"
    if not args.skip_cv and cv_path.exists():
        eval_sets["CV (fiducial model)"] = NbodyHydroMaps(dataset_type="CV", **common)

    report = {
        "meta": {
            "input_field": input_field,
            "suite": args.suite,
            "weights": str(args.weights) if args.weights else None,
            "split_seed": split_seed,
            "n_train_sims": int(len(split["train"])),
            "n_fit_sims": int(len(fit_sims)),
            "n_test_sims": int(len(test_sims)),
            "shell_edges": list(SHELL_EDGES),
        },
        "sets": {},
    }

    for set_name, data in eval_sets.items():
        t0 = time.time()
        methods = {"identity": data.x}
        clipped = {}
        with torch.no_grad():
            s_emulated = torch.exp(emulator(data.params))
            s_oracle = suppression_per_map(power, data)
        for name, s_of_k in (
            ("amp_mean", s_mean.expand(len(data), -1)),
            ("amp_emulated", s_emulated),
            ("amp_oracle", s_oracle),
        ):
            fields, fractions = [], []
            for start in range(0, len(data), BATCH):
                field, fraction = rescale_amplitudes(power, data.x[start : start + BATCH], s_of_k[start : start + BATCH])
                fields.append(field); fractions.append(fraction)
            methods[name] = torch.cat(fields)
            clipped[name] = float(np.mean(fractions))
        if args.weights is not None:
            methods["model"], _ = model_predictions(args.weights, data, device)

        report["sets"][set_name] = {}
        for name, pred in methods.items():
            result = evaluate(power, pred, data)
            result["floored_pixel_fraction"] = clipped.get(name, 0.0)
            report["sets"][set_name][name] = result
        print(f"{set_name}: {len(data)} maps evaluated in {time.time() - t0:.0f}s")
        for name, m in report["sets"][set_name].items():
            print(
                f"  {name:13s} log-MSE {m['log_mse_dex2'] * 1e3:6.3f}e-3 dex²  "
                f"P/Ptrue k=1,5,10,20: {m['power_ratio_k1']:.3f} {m['power_ratio_k5']:.3f} "
                f"{m['power_ratio_k10']:.3f} {m['power_ratio_k20']:.3f}  "
                f"r k=10,20: {m['r_k10']:.4f} {m['r_k20']:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    write_markdown(args.out.with_suffix(".md"), report)
    for i, set_name in enumerate(report["sets"]):
        tag = "lh_test" if i == 0 else "cv"
        if write_figure(args.out.parent / f"{args.out.name}_{tag}.png", report, set_name):
            print(f"Figure: {args.out.parent / f'{args.out.name}_{tag}.png'}")
    print(f"Report: {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
