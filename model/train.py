#!/usr/bin/env python3
"""
Training script for the Bayronik field-level baryon emulator.

Task: map the total-matter map of a gravity-only (N-body) CAMELS box to the
total-matter map of its hydrodynamic twin, conditioned on cosmology and
feedback parameters. Units and normalisation live in bayronik_model/fields.py.

Features:
- Periodic residual U-FNO (ufno_v2) plus the older architectures for ablations
- Simulation-level train/val/test split, written next to the weights
- Whole dataset on the GPU, augmentation by exact symmetries of a periodic map
- Loss: log-field MSE + power spectrum of the density contrast + total mass
- Mixed precision, warmup + cosine schedule per step, gradient clipping
- Resumable checkpoints every epoch and an optional wall-clock budget
- Per-epoch validation against the "no baryons" identity baseline

Usage:
    uv run train.py --epochs 40
    uv run train.py --time-budget-min 50 --epochs 200      # fit the schedule to a rental
    uv run train.py --resume weights/ufno_v2_Mtot_Nbody_IllustrisTNG/last.pt
    uv run train.py --max-train-sims 20 --max-val-sims 5 --epochs 1   # smoke test
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bayronik_model import UNet, ResUNet, AttentionUNet
from bayronik_model.fno import FNO2d, FNO2dConditional, MultiscaleFNO2d
from bayronik_model.ufno import UFNO2d, UFNO2dConditional, AttentionUFNO2d, PeriodicUFNO2d
from bayronik_model.losses import NbodyHydroLoss
from bayronik_model.dataset import NbodyHydroMaps, sim_split
from bayronik_model.fields import (
    BOX_SIZE,
    INPUT_FIELD_UNITS,
    LOG10_SIGMA_MEAN,
    LOG10_SIGMA_STD,
    PARAM_NAMES,
    relative_density,
)
from bayronik_model.spectra import PowerSpectrum2D, correlation_coefficient


# Model registry
MODELS = {
    # Periodic residual U-FNO: the model for the N-body -> hydro task
    "ufno_v2": PeriodicUFNO2d,
    # Traditional architectures
    "unet": UNet,
    "resunet": ResUNet,
    "attention_unet": AttentionUNet,
    # FNO architectures
    "fno": FNO2d,
    "fno_cond": FNO2dConditional,
    "fno_multiscale": MultiscaleFNO2d,
    # U-FNO architectures
    "ufno": UFNO2d,
    "ufno_cond": UFNO2dConditional,
    "ufno_attention": AttentionUFNO2d,
}

# Default hyperparameters per model
MODEL_DEFAULTS = {
    "ufno_v2": {"base_channels": 32, "modes": 32, "depth": 4},
    "unet": {"base_features": 64},
    "resunet": {"base_features": 64},
    "attention_unet": {"base_features": 64},
    "fno": {"hidden_channels": 64, "modes_x": 32, "modes_y": 32, "num_layers": 4},
    "fno_cond": {"hidden_channels": 64, "modes_x": 32, "modes_y": 32, "num_layers": 4},
    "fno_multiscale": {"hidden_channels": 48, "modes": 16},
    "ufno": {"base_channels": 32, "modes": 32, "depth": 4},
    "ufno_cond": {"base_channels": 32, "modes": 32, "depth": 4},
    "ufno_attention": {"base_channels": 32, "modes": 32, "depth": 4},
}

CONDITIONAL_MODELS = {"ufno_v2", "fno_cond", "ufno_cond"}

# k values (h/Mpc) quoted in the per-epoch log line
REPORT_K = (1.0, 5.0, 10.0, 20.0)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(model_name: str, model_kwargs: Optional[dict] = None) -> nn.Module:
    """Create model from name."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    kwargs = MODEL_DEFAULTS.get(model_name, {}).copy()
    kwargs.update(model_kwargs or {})
    if model_name in CONDITIONAL_MODELS:
        kwargs["num_conditions"] = len(PARAM_NAMES)

    return MODELS[model_name](in_channels=1, out_channels=1, **kwargs)


def predict(model: nn.Module, model_name: str, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Run any registered model as input + correction, in network units.

    ufno_v2 is residual by construction. The older architectures predict a
    field, so they are wrapped the same way to keep ablations like for like.
    """
    if model_name == "ufno_v2":
        return model(x, params)
    if model_name in CONDITIONAL_MODELS:
        return x + model(x, params)
    return x + model(x)


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    """Linear warmup, then cosine decay to min_lr at total_steps."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Autocast context when mixed precision is on, a no-op otherwise."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=amp_dtype)


def clip_grad_norm(parameters, max_norm: float) -> torch.Tensor:
    """Global gradient-norm clipping that also handles complex parameters.

    The spectral weights are complex, and torch's own clip_grad_norm_ cannot
    take their norm on every backend (MPS), so the norm is taken over real and
    imaginary parts.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total = torch.sqrt(
        sum((torch.view_as_real(g) if g.is_complex() else g).float().pow(2).sum() for g in grads)
    )
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale)
    return total


def train_epoch(
    model: nn.Module,
    model_name: str,
    data: NbodyHydroMaps,
    criterion: NbodyHydroLoss,
    optimizer,
    scaler,
    device: torch.device,
    args,
    state: dict,
    generator: torch.Generator,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Train for one epoch. `state` carries the global step and schedule length."""
    model.train()

    sums: Dict[str, float] = {}
    num_batches = 0
    log_every = max(1, data.num_batches(args.batch_size, drop_last=True) // 10)

    for x, y, params in data.batches(
        args.batch_size, shuffle=True, augment=args.augment, generator=generator, drop_last=True
    ):
        x, y, params = x.to(device), y.to(device), params.to(device)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        lr = lr_at(state["step"], state["total_steps"], args.lr, args.warmup_steps, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # bf16 needs no GradScaler; fp16 does. The loss always runs in float32.
        with autocast(device, amp_dtype):
            pred = predict(model, model_name, x, params)
        losses = criterion(pred, y)
        loss = losses['total']

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state['step']}: {losses}")

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + value.item()
        num_batches += 1
        state["step"] += 1

        if args.verbose and num_batches % log_every == 0:
            print(f"    step {state['step']:6d}  loss {loss.item():.5f}  lr {lr:.2e}", flush=True)

    return {key: value / num_batches for key, value in sums.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    model_name: str,
    data: NbodyHydroMaps,
    criterion: NbodyHydroLoss,
    power: PowerSpectrum2D,
    device: torch.device,
    args,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Validation losses plus the numbers the result is judged on.

    Reports the log-field MSE in dex^2 for the model and for the identity
    ("no baryons") baseline, and at a few k the power ratio P_pred / P_true
    and the cross-correlation with the truth for both.
    """
    model.eval()

    sums: Dict[str, float] = {}
    spec = {name: torch.zeros(power.n_bins, device=device) for name in ("ratio", "r", "ratio_id", "r_id")}
    n_maps = 0

    for x, y, params in data.batches(args.batch_size):
        x, y, params = x.to(device), y.to(device), params.to(device)
        with autocast(device, amp_dtype):
            pred = predict(model, model_name, x, params)
        pred = pred.float()
        b = x.shape[0]

        losses = criterion(pred, y)
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + value.item() * b
        sums["mse_dex2"] = sums.get("mse_dex2", 0.0) + ((pred - y) ** 2).mean().item() * LOG10_SIGMA_STD ** 2 * b
        sums["mse_dex2_identity"] = (
            sums.get("mse_dex2_identity", 0.0) + ((x - y) ** 2).mean().item() * LOG10_SIGMA_STD ** 2 * b
        )

        rho_true = relative_density(y)
        mean_true = rho_true.mean(dim=(1, 2, 3), keepdim=True)
        d_true = rho_true / mean_true - 1
        for name, field in (("", pred), ("_id", x)):
            d = relative_density(field) / mean_true - 1
            p_aa, p_bb, p_ab = power.cross(d, d_true)
            spec["ratio" + name] += (p_aa / p_bb).sum(0)
            spec["r" + name] += correlation_coefficient(p_aa, p_bb, p_ab).sum(0)
        n_maps += b

    out = {key: value / n_maps for key, value in sums.items()}
    for k_report in REPORT_K:
        i = int(torch.argmin((power.k - k_report).abs()))
        for name, values in spec.items():
            out[f"{name}_k{k_report:g}"] = (values[i] / n_maps).item()
    return out


def save_atomic(obj, path: Path):
    """Write then rename, so an interrupted save never leaves a broken file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args):
    """Main training function."""
    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    run_name = args.run_name or f"{args.model}_{args.input_field}_{args.suite}"
    run_dir = Path(args.weights_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data. The split is at simulation granularity: the 15 maps of one box share
    # its large-scale structure, so they must stay on one side. The held-out
    # test sims are never loaded here.
    n_sims = 1000 if args.dataset == "LH" else 27
    split = sim_split(n_sims, args.val_split, args.test_split, args.split_seed)
    train_sims, val_sims = split["train"], split["val"]
    if args.max_train_sims:
        train_sims = train_sims[: args.max_train_sims]
    if args.max_val_sims:
        val_sims = val_sims[: args.max_val_sims]

    split_payload = {
        "seed": args.split_seed,
        "suite": args.suite,
        "dataset_type": args.dataset,
        "n_sims": n_sims,
        "maps_per_sim": NbodyHydroMaps.MAPS_PER_SIM,
        "train_sims": [int(s) for s in split["train"]],
        "val_sims": [int(s) for s in split["val"]],
        "test_sims": [int(s) for s in split["test"]],
    }
    (run_dir / "split.json").write_text(json.dumps(split_payload, indent=2))

    data_device = args.data_device
    if data_device == "auto":
        big_gpu = device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory > 20e9
        data_device = "cuda" if big_gpu else "cpu"

    t0 = time.time()
    common = dict(
        data_dir=args.data_dir, suite=args.suite, dataset_type=args.dataset,
        input_field=args.input_field, device=data_device,
    )
    train_data = NbodyHydroMaps(sims=train_sims, **common)
    val_data = NbodyHydroMaps(sims=val_sims, **common)
    print(
        f"Data: {args.input_field} -> Mtot ({args.suite} {args.dataset}), "
        f"{len(train_sims)} train / {len(val_sims)} val / {len(split['test'])} test sims, "
        f"{len(train_data)} train maps on {data_device}, loaded in {time.time() - t0:.0f}s"
    )

    # Model
    model = create_model(args.model).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    num_params = count_parameters(model)
    print(f"Model: {args.model} ({num_params:,} params)")

    criterion = NbodyHydroLoss(
        spectral_weight=args.spectral_weight,
        mass_weight=args.mass_weight,
        resolution=train_data.x.shape[-1],
        box_size=BOX_SIZE,
    ).to(device)
    power = PowerSpectrum2D(train_data.x.shape[-1], BOX_SIZE).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Mixed precision. bf16 is preferred on recent GPUs: no loss scaling, and
    # the spectral layers cast to float32 internally.
    amp_dtype: Optional[torch.dtype] = None
    scaler = None
    if device.type == "cuda":
        if args.amp == "bf16":
            amp_dtype = torch.bfloat16
        elif args.amp == "fp16":
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    print(f"AMP: {amp_dtype if amp_dtype is not None else 'disabled'}")

    steps_per_epoch = train_data.num_batches(args.batch_size, drop_last=True)
    state = {"step": 0, "total_steps": args.epochs * steps_per_epoch}
    start_epoch = 0
    best_val_loss = float("inf")
    planned_epochs = args.epochs

    config = {
        'model': args.model,
        'model_kwargs': MODEL_DEFAULTS.get(args.model, {}),
        'input_field': args.input_field,
        'input_unit_factor': INPUT_FIELD_UNITS[args.input_field],
        'log10_sigma_mean': LOG10_SIGMA_MEAN,
        'log10_sigma_std': LOG10_SIGMA_STD,
        'param_names': list(PARAM_NAMES),
        'dataset': args.dataset,
        'suite': args.suite,
        'split_seed': args.split_seed,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'spectral_weight': args.spectral_weight,
        'mass_weight': args.mass_weight,
        'num_params': num_params,
        'seed': args.seed,
    }

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    log_path = run_dir / "log.jsonl"

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        state = checkpoint['state']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        planned_epochs = checkpoint.get('planned_epochs', args.epochs)
        if args.extend_schedule:
            planned_epochs = args.epochs
            state["total_steps"] = args.epochs * steps_per_epoch
        print(f"Resumed from epoch {start_epoch} (step {state['step']}, {planned_epochs} epochs planned)")

    print(f"\n{'='*60}")
    print(f"Training {run_name}: up to {planned_epochs} epochs, {steps_per_epoch} steps each")
    if args.time_budget_min:
        print(f"Wall-clock budget: {args.time_budget_min:.0f} min")
    print(f"{'='*60}\n")

    run_start = time.time()
    epoch_seconds = None

    for epoch in range(start_epoch, planned_epochs):
        if args.time_budget_min and epoch_seconds is not None:
            remaining = args.time_budget_min * 60 - (time.time() - run_start)
            if remaining < epoch_seconds:
                print(f"Stopping before epoch {epoch+1}: {remaining/60:.1f} min left, an epoch takes {epoch_seconds/60:.1f}")
                break

        epoch_start = time.time()

        train_losses = train_epoch(
            model, args.model, train_data, criterion, optimizer, scaler,
            device, args, state, generator, amp_dtype,
        )
        val = validate(model, args.model, val_data, criterion, power, device, args, amp_dtype)

        epoch_seconds = time.time() - epoch_start

        # With a time budget, fit the schedule to what the hardware can do:
        # after the first epoch the epoch time is known, so set the number of
        # epochs so the cosine decay ends when the budget does.
        if args.time_budget_min and epoch == start_epoch:
            budget = args.time_budget_min * 60 - (time.time() - run_start)
            fit = epoch + 1 + max(0, int(budget // epoch_seconds))
            if fit < planned_epochs:
                planned_epochs = fit
                state["total_steps"] = planned_epochs * steps_per_epoch
                print(f"  epoch takes {epoch_seconds:.0f}s: schedule set to {planned_epochs} epochs")

        improved = val['total'] < best_val_loss
        if improved:
            best_val_loss = val['total']
            save_atomic({'model_state_dict': model.state_dict(), 'config': config, 'epoch': epoch, 'val': val}, best_path)

        save_atomic(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'state': state,
                'best_loss': best_val_loss,
                'planned_epochs': planned_epochs,
                'config': config,
            },
            last_path,
        )

        lr = optimizer.param_groups[0]['lr']
        marker = "*" if improved else ""
        print(
            f"Epoch {epoch+1:3d}/{planned_epochs} │ "
            f"train {train_losses['total']:.5f} │ "
            f"val {val['total']:.5f} │ "
            f"mse {val['mse_dex2']*1e3:.3f} (identity {val['mse_dex2_identity']*1e3:.3f}) e-3 dex² │ "
            f"P/Ptrue k=10: {val['ratio_k10']:.3f} (id {val['ratio_id_k10']:.3f}) │ "
            f"r k=10: {val['r_k10']:.4f} (id {val['r_id_k10']:.4f}) │ "
            f"lr {lr:.2e} │ {epoch_seconds:.0f}s {marker}",
            flush=True,
        )

        with log_path.open("a") as f:
            record = {"epoch": epoch + 1, "step": state["step"], "lr": lr, "seconds": epoch_seconds}
            record.update({f"train/{k}": v for k, v in train_losses.items()})
            record.update({f"val/{k}": v for k, v in val.items()})
            f.write(json.dumps(record) + "\n")

    print(f"\n{'='*60}")
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Weights: {best_path}")
    print(f"Evaluate on the held-out test sims with benchmarks/nbody2hydro.py --weights {best_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the Bayronik baryon emulator (N-body -> hydro total matter)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="LH", choices=["CV", "LH"], help="Dataset type")
    parser.add_argument("--suite", type=str, default="IllustrisTNG", help="Hydro suite of the target maps")
    parser.add_argument("--input-field", type=str, default="Mtot_Nbody", choices=list(INPUT_FIELD_UNITS),
                        help="Mtot_Nbody: gravity-only run. Mcdm: dark matter of the hydro run (old task, for comparison)")
    parser.add_argument("--val-split", type=float, default=0.10, help="Fraction of sims for validation")
    parser.add_argument("--test-split", type=float, default=0.10, help="Fraction of sims held out for final test")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed of the simulation split")
    parser.add_argument("--max-train-sims", type=int, default=0, help="Use only this many training sims (0 = all)")
    parser.add_argument("--max-val-sims", type=int, default=0, help="Use only this many validation sims (0 = all)")
    parser.add_argument("--data-device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Where the dataset tensors live; auto puts them on a GPU with > 20 GB")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable flips, transposes and shifts")

    # Model
    parser.add_argument("--model", type=str, default="ufno_v2", choices=list(MODELS.keys()), help="Model architecture")

    # Training
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs (upper bound with --time-budget-min)")
    parser.add_argument("--time-budget-min", type=float, default=0, help="Wall-clock budget in minutes; the schedule is fitted to it (0 = off)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--seed", type=int, default=0, help="Seed for initialisation, shuffling and augmentation")

    # Loss weights
    parser.add_argument("--spectral-weight", type=float, default=0.1, help="Power spectrum loss weight")
    parser.add_argument("--mass-weight", type=float, default=0.1, help="Mass conservation loss weight")

    # System
    parser.add_argument("--weights-dir", type=str, default="weights", help="Weights directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run directory name (default: model_input_suite)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a last.pt checkpoint")
    parser.add_argument("--extend-schedule", action="store_true", help="On resume, restart the schedule length from --epochs")
    parser.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"], help="Mixed precision on CUDA")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false", help="Keep NCHW memory layout")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Print the loss ten times per epoch")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
