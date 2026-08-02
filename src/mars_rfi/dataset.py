"""Dataset and collate helpers for MARS training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment_patch


def _resolve(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(root) / p


def _squeeze_patch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected patch shape HxW or 1xHxW, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


class NpyRFIDataset(Dataset):
    """Memory-mapped tanh-domain patch dataset.

    Samples return four tensors: ``x``, ``y``, ``weight``, and ``m_astro``.
    This keeps DataLoader shared-memory use predictable and avoids expensive
    dictionaries.
    """

    def __init__(
        self,
        patch_path: str | os.PathLike[str],
        mask_path: str | os.PathLike[str] | None,
        augmentation: dict[str, Any],
        root: str | os.PathLike[str] = ".",
        train: bool = True,
        seed: int = 0,
    ):
        self.patch_path = _resolve(patch_path, root)
        self.mask_path = None if mask_path is None else _resolve(mask_path, root)
        self.augmentation = dict(augmentation)
        self.train = bool(train)
        self.seed = int(seed)
        self._patches = None
        self._masks = None
        self._open_arrays()

    def _open_arrays(self) -> None:
        self._patches = np.load(self.patch_path, mmap_mode="r")
        if self.mask_path is None:
            self._masks = None
        else:
            self._masks = np.load(self.mask_path, mmap_mode="r")
            if len(self._masks) != len(self._patches):
                raise ValueError(
                    f"patch/mask length mismatch: {len(self._patches)} vs {len(self._masks)}"
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_patches"] = None
        state["_masks"] = None
        return state

    def _ensure_open(self) -> None:
        if self._patches is None:
            self._open_arrays()

    def __len__(self) -> int:
        self._ensure_open()
        return int(len(self._patches))

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.train:
            seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        else:
            seed = (self.seed + 1000003 * int(idx)) % (2**32 - 1)
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int):
        self._ensure_open()
        patch = _squeeze_patch(self._patches[idx])
        mask = None if self._masks is None else _squeeze_patch(self._masks[idx])

        cfg = self.augmentation
        if (not self.train) and (not bool(cfg.get("augment_val", False))):
            threshold = float(cfg.get("mask_threshold", 0.5))
            x_np = patch.astype(np.float32, copy=True)
            y_np = np.zeros_like(x_np, dtype=np.float32) if mask is None else (
                mask.astype(np.float32, copy=False) > threshold
            ).astype(np.float32)
            w_np = np.ones_like(x_np, dtype=np.float32)
            astro_np = np.zeros_like(x_np, dtype=np.float32)
        else:
            result = augment_patch(patch, mask, cfg, self._rng_for_index(idx))
            x_np, y_np, w_np, astro_np = (
                result.patch,
                result.mask,
                result.weight,
                result.astro_mask,
            )

        x = torch.from_numpy(x_np[None, :, :].copy())
        y = torch.from_numpy(y_np[None, :, :].copy())
        weight = torch.from_numpy(w_np[None, :, :].copy())
        m_astro = torch.from_numpy(astro_np[None, :, :].copy())
        return x, y, weight, m_astro


def packed_collate(batch):
    """Collate ``(x, y, weight, m_astro)`` into one contiguous shared tensor.

    PyTorch's default collate creates one shared storage per returned tensor.
    Packing the three tensors first reduces shared-memory pressure and has been
    more stable with multiprocessing workers.
    """

    xs, ys, ws, astros = zip(*batch)
    b = len(xs)
    shape = xs[0].shape
    packed = xs[0].new_empty((4, b, *shape))
    for i, (x, y, w, m_astro) in enumerate(batch):
        packed[0, i].copy_(x)
        packed[1, i].copy_(y)
        packed[2, i].copy_(w)
        packed[3, i].copy_(m_astro)
    return packed[0], packed[1], packed[2], packed[3]


def seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
