"""Sim-level split correctness.

The train/val/test split must be disjoint at SIMULATION granularity:
projections of the same CAMELS box share large-scale structure, so a
map-level random split leaks training information into validation.
"""

from __future__ import annotations

import json

import numpy as np

from bayronik_model.dataset import create_dataloaders, infer_sim_layout


def _write_dummy_lh(tmp_path, n_sims: int = 10, maps_per_sim: int = 15, res: int = 32):
    n = n_sims * maps_per_sim
    rng = np.random.default_rng(1)
    maps = rng.exponential(1.0, (n, res, res)).astype(np.float32)
    np.save(tmp_path / "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy", maps)
    np.save(tmp_path / "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy", maps * 1.1)
    params = rng.uniform(0.1, 1.0, (n_sims, 6))
    np.savetxt(tmp_path / "params_LH_IllustrisTNG.txt", params)
    return n_sims, maps_per_sim


def test_infer_sim_layout_from_params(tmp_path):
    n_sims, maps_per_sim = _write_dummy_lh(tmp_path)
    n, m = infer_sim_layout(n_sims * maps_per_sim, tmp_path, "IllustrisTNG", "LH")
    assert (n, m) == (n_sims, maps_per_sim)


def test_sim_level_split_is_disjoint(tmp_path):
    n_sims, maps_per_sim = _write_dummy_lh(tmp_path)
    split_file = tmp_path / "split.json"

    train_loader, val_loader = create_dataloaders(
        data_dir=str(tmp_path),
        batch_size=4,
        num_workers=0,
        dataset_type="LH",
        val_split=0.2,
        test_split=0.2,
        split_file=str(split_file),
        return_params=True,
        augment_train=False,
    )

    payload = json.loads(split_file.read_text())
    train_s = set(payload["train_sims"])
    val_s = set(payload["val_sims"])
    test_s = set(payload["test_sims"])

    assert not train_s & val_s
    assert not train_s & test_s
    assert not val_s & test_s
    assert train_s | val_s | test_s == set(range(n_sims))
    assert payload["maps_per_sim"] == maps_per_sim

    # Every loader index must belong to a sim on the correct side of the
    # split, and test-sim maps must appear in neither loader.
    train_idx = list(train_loader.dataset.indices)
    val_idx = list(val_loader.dataset.indices)
    assert {i // maps_per_sim for i in train_idx} == train_s
    assert {i // maps_per_sim for i in val_idx} == val_s
    assert len(train_idx) == len(train_s) * maps_per_sim
    assert len(val_idx) == len(val_s) * maps_per_sim


def test_split_is_deterministic(tmp_path):
    _write_dummy_lh(tmp_path)
    f1 = tmp_path / "s1.json"
    f2 = tmp_path / "s2.json"
    for f in (f1, f2):
        create_dataloaders(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            dataset_type="LH",
            split_file=str(f),
            augment_train=False,
        )
    assert json.loads(f1.read_text()) == json.loads(f2.read_text())
