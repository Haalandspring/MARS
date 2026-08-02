#!/usr/bin/env python3
"""Evaluate a fixed patch dataset with pooled pixel-count metrics.

The script consumes already-generated paper candidates; it does not synthesize
RFI. Arrays may have shape ``(N, 512, 512)`` or ``(N, 1, 512, 512)``. Optional
JSONL metadata can contain ``family`` and ``sampling_time_us`` for stratified
micro metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mars_rfi.config import CONFIG
from mars_rfi.model import build_model, count_parameters
from mars_rfi.provenance import validate_training_identity


COUNT_KEYS = ("tp", "fp", "fn", "tn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = deepcopy(CONFIG)
    if isinstance(checkpoint, dict):
        checkpoint_config = checkpoint.get("config", {})
        validate_training_identity(
            checkpoint_config,
            source=f"checkpoint {checkpoint_path}",
        )
        config.update(checkpoint_config)
    else:
        raise ValueError(
            "The evaluation checkpoint must embed its resolved config and artifact identity."
        )
    model = build_model(config).to(device).eval()
    missing, unexpected = model.load_state_dict(extract_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    parameter_count = count_parameters(model)
    expected = config.get("expected_parameters")
    if expected is not None and parameter_count != int(expected):
        raise RuntimeError(
            f"model has {parameter_count:,} parameters; expected {int(expected):,}"
        )
    return model, config, parameter_count


def load_metadata(path: Path | None, count: int) -> list[dict]:
    if path is None:
        return [{} for _ in range(count)]
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"metadata line {line_number} is not an object")
            rows.append(row)
    if len(rows) != count:
        raise ValueError(f"metadata has {len(rows)} rows; arrays contain {count} cases")
    return rows


def add_counts(total: dict[str, int], pred: torch.Tensor, truth: torch.Tensor) -> None:
    total["tp"] += int((pred & truth).sum().item())
    total["fp"] += int((pred & ~truth).sum().item())
    total["fn"] += int((~pred & truth).sum().item())
    total["tn"] += int((~pred & ~truth).sum().item())


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def normalise_array_shape(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W); got {array.shape}")
    return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    patches = normalise_array_shape(np.load(args.patches, mmap_mode="r"), "patches")
    targets = normalise_array_shape(np.load(args.targets, mmap_mode="r"), "targets")
    if patches.shape != targets.shape:
        raise ValueError(f"patch/target shape mismatch: {patches.shape} vs {targets.shape}")
    if tuple(patches.shape[1:]) != (512, 512):
        raise ValueError(f"paper evaluation requires 512x512 patches; got {patches.shape[1:]}")
    metadata = load_metadata(args.metadata_jsonl, len(patches))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, model_config, parameter_count = load_model(args.checkpoint, device)

    overall = dict.fromkeys(COUNT_KEYS, 0)
    grouped = defaultdict(lambda: dict.fromkeys(COUNT_KEYS, 0))
    for start in range(0, len(patches), args.batch_size):
        stop = min(start + args.batch_size, len(patches))
        inputs = torch.from_numpy(
            np.asarray(patches[start:stop], dtype=np.float32).copy()
        ).unsqueeze(1).to(device)
        truth = torch.from_numpy(
            np.asarray(targets[start:stop] >= 0.5, dtype=np.bool_)
        ).to(device)
        pred = torch.sigmoid(model(inputs)).squeeze(1) >= float(args.threshold)
        add_counts(overall, pred, truth)

        for local_idx, row in enumerate(metadata[start:stop]):
            family = str(row.get("family", "unspecified"))
            cadence = str(row.get("sampling_time_us", "unspecified"))
            add_counts(grouped[(family, cadence)], pred[local_idx], truth[local_idx])

    result = {
        "schema_version": 1,
        "aggregation": "pooled_pixel_counts",
        "cases": len(patches),
        "threshold": float(args.threshold),
        "device": str(device),
        "trainable_parameters": parameter_count,
        "model_config": model_config,
        "artifacts": {
            "patches": {"path": str(args.patches), "sha256": sha256(args.patches)},
            "targets": {"path": str(args.targets), "sha256": sha256(args.targets)},
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256(args.checkpoint),
            },
        },
        "overall": metrics(overall),
        "strata": [
            {
                "family": family,
                "sampling_time_us": cadence,
                **metrics(counts),
            }
            for (family, cadence), counts in sorted(grouped.items())
        ],
    }
    if args.metadata_jsonl is not None:
        result["artifacts"]["metadata"] = {
            "path": str(args.metadata_jsonl),
            "sha256": sha256(args.metadata_jsonl),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["overall"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
