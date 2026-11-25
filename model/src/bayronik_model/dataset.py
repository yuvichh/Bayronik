"""
CAMELS dataset loader with advanced features for baryonic field emulation.

Features:
- Memory-mapping support for large datasets
- Data augmentation (rotation, flip, noise)
- Conditional inputs (physics parameters)
- Multiple simulation suite support
- Multi-redshift support
- Efficient caching
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class CAMELSDataset(Dataset):
    """
    Dataset for CAMELS 2D projected mass maps with conditional inputs.
    
    Supports:
    - CV (Cosmic Variance): 27 simulations, same cosmology
    - LH (Latin Hypercube): 1000 simulations, varied cosmology
    - Multiple simulation suites (IllustrisTNG, SIMBA)
    
    Args:
        data_dir: Directory containing .npy files
        suite: Simulation suite (IllustrisTNG, SIMBA)
        dataset_type: CV or LH
        redshift: Target redshift (0.00, 0.50, 1.00, etc.)
        mmap: Memory-map files instead of loading to RAM
        augment: Enable data augmentation
        return_params: Return physics parameters for conditioning
        normalize: Normalization method ('log1p', 'standardize', None)
    """

    # Physics parameters for LH simulations (example values)
    # In practice, load from CAMELS parameter files
    LH_PARAMS = {
        'Omega_m': (0.1, 0.5),      # Matter density
        'sigma_8': (0.6, 1.0),      # Clustering amplitude
        'A_SN1': (0.25, 4.0),       # SN feedback
        'A_AGN1': (0.25, 4.0),      # AGN feedback
        'A_SN2': (0.5, 2.0),        # SN wind speed
        'A_AGN2': (0.5, 2.0),       # AGN boost
    }

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "CV",
        redshift: str = "0.00",
        mmap: bool = False,
        augment: bool = False,
        return_params: bool = False,
        normalize: str = "log1p",
        cache_stats: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.dataset_type = dataset_type
        self.redshift = redshift
        self.augment = augment
        self.return_params = return_params
        self.normalize = normalize
        
        # Build file paths
        input_file = f"Maps_Mcdm_{suite}_{dataset_type}_z={redshift}.npy"
        target_file = f"Maps_Mtot_{suite}_{dataset_type}_z={redshift}.npy"
        
        input_path = self.data_dir / input_file
        target_path = self.data_dir / target_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target_path}")
        
        # Load data
        mmap_mode = "r" if mmap else None
        self.input_maps = np.load(input_path, mmap_mode=mmap_mode)
        self.target_maps = np.load(target_path, mmap_mode=mmap_mode)
        
        if self.input_maps.shape[0] != self.target_maps.shape[0]:
            raise ValueError("Input/target shape mismatch")
        
        self.num_samples = self.input_maps.shape[0]
        self.resolution = self.input_maps.shape[-1]
        
        # Load or generate physics parameters
        self.params = self._load_params() if return_params else None
        
        # Compute normalization statistics
        self.stats = self._compute_stats() if cache_stats else None

    def _load_params(self) -> Optional[np.ndarray]:
        """Load physics parameters for LH dataset.
        
        CAMELS maps have multiple projections per simulation (typically 15).
        The params file has one row per simulation, so we repeat each row
        to match the number of maps.
        """
        if self.dataset_type == "CV":
            params = np.tile([0.3, 0.8, 1.0, 1.0, 1.0, 1.0], (self.num_samples, 1))
            return params.astype(np.float32)
        
        # Try both naming conventions
        param_file = self.data_dir / f"params_{self.dataset_type}_{self.suite}.txt"
        if not param_file.exists():
            param_file = self.data_dir / f"params_{self.suite}_{self.dataset_type}.txt"
        
        if param_file.exists():
            params = np.loadtxt(param_file)
            n_sims = params.shape[0]
            
            if n_sims == self.num_samples:
                return params.astype(np.float32)
            
            if n_sims > self.num_samples:
                return params[:self.num_samples].astype(np.float32)
            
            # Multiple projections per simulation (e.g. 15000 maps / 1000 sims = 15)
            if self.num_samples % n_sims == 0:
                maps_per_sim = self.num_samples // n_sims
                print(f"Expanding {n_sims} params to {self.num_samples} maps "
                      f"({maps_per_sim} projections per sim)")
                params = np.repeat(params, maps_per_sim, axis=0)
                return params.astype(np.float32)
            
            # Partial match: take as many complete repetitions as possible
            print(f"Warning: params ({n_sims}) don't divide evenly into "
                  f"maps ({self.num_samples}), using nearest repeat")
            maps_per_sim = max(1, self.num_samples // n_sims)
            params = np.repeat(params, maps_per_sim, axis=0)
            return params[:self.num_samples].astype(np.float32)
        
        raise FileNotFoundError(
            f"No parameter file found for {self.dataset_type}_{self.suite}. "
            f"Looked for: params_{self.dataset_type}_{self.suite}.txt and "
            f"params_{self.suite}_{self.dataset_type}.txt in {self.data_dir}"
        )

    def _compute_stats(self) -> Dict[str, float]:
        """Compute dataset statistics for normalization."""
        # Sample a subset for efficiency (seeded so stats are reproducible)
        n_sample = min(100, self.num_samples)
        rng = np.random.default_rng(0)
        indices = rng.choice(self.num_samples, n_sample, replace=False)
        
        input_sample = np.stack([self.input_maps[i] for i in indices])
        target_sample = np.stack([self.target_maps[i] for i in indices])
        
        # Apply log1p before computing stats
        input_log = np.log1p(input_sample)
        target_log = np.log1p(target_sample)
        
        return {
            'input_mean': float(input_log.mean()),
            'input_std': float(input_log.std()),
            'target_mean': float(target_log.mean()),
            'target_std': float(target_log.std()),
        }

    def _normalize(self, x: np.ndarray, is_input: bool = True) -> np.ndarray:
        """Apply normalization."""
        if self.normalize == "log1p":
            return np.log1p(x.astype(np.float32))
        
        elif self.normalize == "standardize":
            x = np.log1p(x.astype(np.float32))
            if self.stats is not None:
                mean_key = 'input_mean' if is_input else 'target_mean'
                std_key = 'input_std' if is_input else 'target_std'
                x = (x - self.stats[mean_key]) / (self.stats[std_key] + 1e-8)
            return x
        
        return x.astype(np.float32)

    def _augment(
        self,
        inp: np.ndarray,
        tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(4)
        if k > 0:
            inp = np.rot90(inp, k, axes=(-2, -1)).copy()
            tgt = np.rot90(tgt, k, axes=(-2, -1)).copy()
        
        # Random flip
        if np.random.random() > 0.5:
            inp = np.flip(inp, axis=-1).copy()
            tgt = np.flip(tgt, axis=-1).copy()
        
        if np.random.random() > 0.5:
            inp = np.flip(inp, axis=-2).copy()
            tgt = np.flip(tgt, axis=-2).copy()
        
        return inp, tgt

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get a sample.
        
        Returns:
            If return_params=False: (input, target)
            If return_params=True: (input, target, params)
        """
        inp = self.input_maps[idx]
        tgt = self.target_maps[idx]
        
        # Augmentation (before normalization for proper rotation)
        if self.augment:
            inp, tgt = self._augment(inp, tgt)
        
        # Normalize
        inp = self._normalize(inp, is_input=True)
        tgt = self._normalize(tgt, is_input=False)
        
        # Convert to tensors and add channel dimension
        inp_tensor = torch.from_numpy(inp).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt).unsqueeze(0)
        
        if self.return_params and self.params is not None:
            params_tensor = torch.from_numpy(self.params[idx])
            return inp_tensor, tgt_tensor, params_tensor
        
        return inp_tensor, tgt_tensor


class MultiSuiteDataset(Dataset):
    """
    Combined dataset from multiple simulation suites.
    
    Useful for training models that generalize across
    different hydrodynamic codes and feedback implementations.
    """

    def __init__(
        self,
        data_dir: str,
        suites: List[str] = ["IllustrisTNG", "SIMBA"],
        dataset_type: str = "LH",
        redshift: str = "0.00",
        **kwargs,
    ):
        self.datasets = []
        self.suite_ids = []
        
        for i, suite in enumerate(suites):
            try:
                ds = CAMELSDataset(
                    data_dir=data_dir,
                    suite=suite,
                    dataset_type=dataset_type,
                    redshift=redshift,
                    **kwargs,
                )
                self.datasets.append(ds)
                self.suite_ids.extend([i] * len(ds))
            except FileNotFoundError:
                print(f"Warning: Suite {suite} not found, skipping")
        
        if not self.datasets:
            raise FileNotFoundError("No datasets found")
        
        # Build cumulative indices
        self.cumulative_sizes = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _find_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Find which dataset and local index."""
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                local_idx = idx - (self.cumulative_sizes[i-1] if i > 0 else 0)
                return i, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self._find_dataset_idx(idx)
        return self.datasets[ds_idx][local_idx]


class MultiRedshiftDataset(Dataset):
    """
    Dataset combining multiple redshifts for temporal generalization.
    """

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "LH",
        redshifts: List[str] = ["0.00", "0.50", "1.00"],
        return_redshift: bool = True,
        **kwargs,
    ):
        self.datasets = []
        self.redshift_values = []
        self.return_redshift = return_redshift
        
        for z in redshifts:
            try:
                ds = CAMELSDataset(
                    data_dir=data_dir,
                    suite=suite,
                    dataset_type=dataset_type,
                    redshift=z,
                    **kwargs,
                )
                self.datasets.append(ds)
                self.redshift_values.extend([float(z)] * len(ds))
            except FileNotFoundError:
                print(f"Warning: Redshift z={z} not found, skipping")
        
        if not self.datasets:
            raise FileNotFoundError("No datasets found")
        
        # Build cumulative indices
        self.cumulative_sizes = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _find_dataset_idx(self, idx: int) -> Tuple[int, int]:
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                local_idx = idx - (self.cumulative_sizes[i-1] if i > 0 else 0)
                return i, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self._find_dataset_idx(idx)
        result = self.datasets[ds_idx][local_idx]
        
        if self.return_redshift:
            z = torch.tensor([self.redshift_values[idx]], dtype=torch.float32)
            if isinstance(result, tuple) and len(result) == 3:
                # Has params, append z to params
                inp, tgt, params = result
                params = torch.cat([params, z])
                return inp, tgt, params
            else:
                inp, tgt = result
                return inp, tgt, z
        
        return result


def infer_sim_layout(
    num_maps: int,
    data_dir: Path,
    suite: str,
    dataset_type: str,
) -> Tuple[int, int]:
    """Return ``(n_sims, maps_per_sim)`` for a CAMELS map file.

    LH: the params file has one row per simulation and CAMELS Multifield
    ships ``num_maps = 15 * n_sims`` (15 projections per box).
    CV: no per-sim params file, but the 15-projections-per-box layout is
    the same.
    """
    for name in (
        f"params_{dataset_type}_{suite}.txt",
        f"params_{suite}_{dataset_type}.txt",
    ):
        param_file = data_dir / name
        if param_file.exists():
            n_sims = np.loadtxt(param_file).shape[0]
            if n_sims > 0 and num_maps % n_sims == 0:
                return n_sims, num_maps // n_sims
    if num_maps % 15 == 0:
        return num_maps // 15, 15
    return num_maps, 1


def sim_split(
    n_sims: int,
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Disjoint train/val/test simulation indices, reproducible from the seed."""
    order = np.random.default_rng(seed).permutation(n_sims)
    n_test = int(round(n_sims * test_split))
    n_val = max(1, int(round(n_sims * val_split)))
    return {
        "test": np.sort(order[:n_test]),
        "val": np.sort(order[n_test : n_test + n_val]),
        "train": np.sort(order[n_test + n_val :]),
    }


class NbodyHydroMaps:
    """Paired N-body -> hydro total-matter maps, held in memory in network units.

    The input is the total-matter map of the gravity-only twin of each CAMELS
    box (same initial conditions as the hydro run); the target is the hydro
    total-matter map. Both are stored as u = (log10 Sigma - 11) / 0.5, see
    ``fields.py``. Parameters are stored mapped to [-1, 1].

    The whole selection lives on ``device``, so batching and augmentation cost
    no data-loader time: 12,000 map pairs take 6.3 GB.

    Args:
        data_dir: Directory with the CMD .npy files and the params file
        suite: Hydro suite of the target maps
        dataset_type: LH or CV
        input_field: "Mtot_Nbody" (gravity-only run) or "Mcdm" (dark matter of
            the hydro run; only for comparison with the old task)
        sims: Simulation indices to load; None loads all
        device: Where the tensors live
    """

    MAPS_PER_SIM = 15

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "LH",
        input_field: str = "Mtot_Nbody",
        redshift: str = "0.00",
        sims: Optional[np.ndarray] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        from .fields import FIDUCIAL_PARAMS, INPUT_FIELD_UNITS, encode_field, encode_params

        if input_field not in INPUT_FIELD_UNITS:
            raise ValueError(f"input_field must be one of {list(INPUT_FIELD_UNITS)}, got {input_field!r}")

        data_dir = Path(data_dir)
        input_path = data_dir / f"Maps_{input_field}_{suite}_{dataset_type}_z={redshift}.npy"
        target_path = data_dir / f"Maps_Mtot_{suite}_{dataset_type}_z={redshift}.npy"
        for path in (input_path, target_path):
            if not path.exists():
                raise FileNotFoundError(f"Map file not found: {path}")

        inputs = np.load(input_path, mmap_mode="r")
        targets = np.load(target_path, mmap_mode="r")
        if inputs.shape != targets.shape:
            raise ValueError(f"Input/target shape mismatch: {inputs.shape} vs {targets.shape}")

        self.n_sims_total = inputs.shape[0] // self.MAPS_PER_SIM
        if inputs.shape[0] != self.n_sims_total * self.MAPS_PER_SIM:
            raise ValueError(f"{inputs.shape[0]} maps is not a multiple of {self.MAPS_PER_SIM} per simulation")
        self.sims = np.arange(self.n_sims_total) if sims is None else np.sort(np.asarray(sims))
        self.map_index = (
            self.sims[:, None] * self.MAPS_PER_SIM + np.arange(self.MAPS_PER_SIM)[None, :]
        ).reshape(-1)

        unit = INPUT_FIELD_UNITS[input_field]
        n = len(self.map_index)
        x = torch.empty(n, 1, *inputs.shape[1:], dtype=torch.float32)
        y = torch.empty_like(x)
        # One simulation at a time keeps peak host memory at the size of the
        # selection, even when the files are far larger than RAM.
        for j, sim in enumerate(self.sims):
            src = slice(sim * self.MAPS_PER_SIM, (sim + 1) * self.MAPS_PER_SIM)
            dst = slice(j * self.MAPS_PER_SIM, (j + 1) * self.MAPS_PER_SIM)
            x[dst, 0] = torch.from_numpy(encode_field(np.asarray(inputs[src], dtype=np.float64) * unit))
            y[dst, 0] = torch.from_numpy(encode_field(np.asarray(targets[src], dtype=np.float64)))

        params = self._load_params(data_dir, suite, dataset_type, FIDUCIAL_PARAMS)
        params = encode_params(params[self.sims].astype(np.float64)).astype(np.float32)

        self.input_field = input_field
        self.device = torch.device(device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.params = torch.from_numpy(np.repeat(params, self.MAPS_PER_SIM, axis=0)).to(self.device)

    def _load_params(self, data_dir: Path, suite: str, dataset_type: str, fiducial) -> np.ndarray:
        for name in (
            f"params_{dataset_type}_{suite}.txt",
            f"params_{dataset_type}_Nbody_{suite}.txt",
        ):
            if (data_dir / name).exists():
                params = np.atleast_2d(np.loadtxt(data_dir / name))[:, :6]
                if params.shape[0] != self.n_sims_total:
                    raise ValueError(
                        f"{name} has {params.shape[0]} rows for {self.n_sims_total} simulations"
                    )
                return params
        if dataset_type == "CV":
            return np.tile(np.asarray(fiducial), (self.n_sims_total, 1))
        raise FileNotFoundError(f"No parameter file for {dataset_type} {suite} in {data_dir}")

    def __len__(self) -> int:
        return self.x.shape[0]

    @staticmethod
    def augment(x: torch.Tensor, y: torch.Tensor, generator: torch.Generator):
        """Random symmetries of a periodic map, applied to the pair together.

        Flips, the transpose and periodic shifts are exact symmetries of the
        problem, so they add no bias.
        """
        b, n = x.shape[0], x.shape[-1]
        pair = torch.cat([x, y], dim=1)
        draw = torch.rand(3, b, generator=generator)
        for axis, row in ((-1, 0), (-2, 1)):
            sel = (draw[row] < 0.5).to(pair.device)
            pair[sel] = pair[sel].flip(axis)
        sel = (draw[2] < 0.5).to(pair.device)
        pair[sel] = pair[sel].transpose(-2, -1)
        shifts = torch.randint(0, n, (b, 2), generator=generator)
        pair = torch.stack(
            [torch.roll(pair[i], (int(sx), int(sy)), dims=(-2, -1)) for i, (sx, sy) in enumerate(shifts)]
        )
        return pair[:, :1], pair[:, 1:]

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        augment: bool = False,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
    ):
        """Yield (input, target, params) batches on ``self.device``."""
        n = len(self)
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        stop = n - (n % batch_size) if drop_last else n
        for start in range(0, stop, batch_size):
            idx = order[start : start + batch_size].to(self.device)
            x, y, p = self.x[idx], self.y[idx], self.params[idx]
            if augment:
                x, y = self.augment(x, y, generator)
            yield x, y, p

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        n = len(self)
        return n // batch_size if drop_last else -(-n // batch_size)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    suite: str = "IllustrisTNG",
    dataset_type: str = "LH",
    val_split: float = 0.10,
    test_split: float = 0.10,
    split_by_sim: bool = True,
    split_seed: int = 42,
    split_file: Optional[str] = None,
    augment_train: bool = True,
    return_params: bool = False,
    **kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    By default the split is done at the SIMULATION level: all projections of
    a given CAMELS box land on the same side of the split. A map-level random
    split leaks train information into validation because the 15 projections
    of one box share the same large-scale structure.

    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        suite: Simulation suite
        dataset_type: CV or LH
        val_split: Fraction of simulations held out for validation
        test_split: Fraction of simulations held out for the final test set.
            Test maps are never loaded here; the sim indices are recorded in
            ``split_file`` for the offline validation script.
        split_by_sim: Split at simulation granularity (default). Set False
            to reproduce the legacy leaky map-level split.
        split_seed: RNG seed for the sim permutation
        split_file: Optional JSON path where the train/val/test sim indices
            are persisted for downstream evaluation.
        augment_train: Enable augmentation for training
        return_params: Return physics parameters

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, Subset, random_split

    # Create dataset
    dataset = CAMELSDataset(
        data_dir=data_dir,
        suite=suite,
        dataset_type=dataset_type,
        augment=False,  # Augment only training
        return_params=return_params,
        **kwargs,
    )

    n_maps = len(dataset)

    if split_by_sim:
        n_sims, maps_per_sim = infer_sim_layout(
            n_maps, Path(data_dir), suite, dataset_type
        )
        split = sim_split(n_sims, val_split, test_split, split_seed)
        train_sims, val_sims, test_sims = split["train"], split["val"], split["test"]

        def expand(sims: np.ndarray) -> List[int]:
            return [
                int(s) * maps_per_sim + j for s in sims for j in range(maps_per_sim)
            ]

        train_ds = Subset(dataset, expand(train_sims))
        val_ds = Subset(dataset, expand(val_sims))

        print(
            f"Sim-level split (seed {split_seed}): "
            f"{len(train_sims)} train / {len(val_sims)} val / {len(test_sims)} test sims "
            f"({maps_per_sim} maps/sim; test maps excluded from training)"
        )

        if split_file is not None:
            payload = {
                "seed": split_seed,
                "suite": suite,
                "dataset_type": dataset_type,
                "n_sims": int(n_sims),
                "maps_per_sim": int(maps_per_sim),
                "train_sims": [int(s) for s in train_sims],
                "val_sims": [int(s) for s in val_sims],
                "test_sims": [int(s) for s in test_sims],
            }
            split_path = Path(split_file)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(payload, indent=2))
            print(f"Split indices written to {split_path}")
    else:
        # Legacy map-level split. Projections of the same simulation land in
        # both train and val, so val metrics overstate generalization.
        val_size = int(n_maps * val_split)
        train_size = n_maps - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(split_seed),
        )

    # Create augmented wrapper for training
    if augment_train:
        train_ds = AugmentedDataset(train_ds)
    
    # Determine pin_memory
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


class AugmentedDataset(Dataset):
    """Wrapper to add augmentation to a dataset subset."""
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        result = self.dataset[idx]
        
        # Unpack
        if len(result) == 3:
            inp, tgt, params = result
        else:
            inp, tgt = result
            params = None
        
        # Random rotation
        k = np.random.randint(4)
        if k > 0:
            inp = torch.rot90(inp, k, dims=(-2, -1))
            tgt = torch.rot90(tgt, k, dims=(-2, -1))
        
        # Random flips
        if np.random.random() > 0.5:
            inp = torch.flip(inp, dims=[-1])
            tgt = torch.flip(tgt, dims=[-1])
        
        if np.random.random() > 0.5:
            inp = torch.flip(inp, dims=[-2])
            tgt = torch.flip(tgt, dims=[-2])
        
        if params is not None:
            return inp, tgt, params
        return inp, tgt


if __name__ == "__main__":
    # Test dataset
    import sys
    
    print("Testing dataset module")
    print("-" * 50)
    
    # Create dummy data for testing
    test_dir = Path("/tmp/bayronik_test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Generate dummy maps
    n_samples = 10
    resolution = 256
    
    input_maps = np.random.exponential(1.0, (n_samples, resolution, resolution))
    target_maps = input_maps * 1.15 + np.random.normal(0, 0.1, input_maps.shape)
    
    np.save(test_dir / "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy", input_maps)
    np.save(test_dir / "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy", target_maps)
    
    # Test basic dataset
    ds = CAMELSDataset(
        data_dir=str(test_dir),
        suite="IllustrisTNG",
        dataset_type="CV",
        augment=True,
        return_params=True,
    )
    
    print(f"Dataset size: {len(ds)}")
    
    inp, tgt, params = ds[0]
    print(f"Input shape: {inp.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Params shape: {params.shape}")
    print(f"Params: {params}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        data_dir=str(test_dir),
        batch_size=4,
        num_workers=0,
        return_params=True,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for inp, tgt, params in train_loader:
        print(f"Batch input: {inp.shape}")
        print(f"Batch target: {tgt.shape}")
        print(f"Batch params: {params.shape}")
        break
    
    print("\nDataset tests passed!")
