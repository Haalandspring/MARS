"""Train the paper-aligned MARS model or an explicit ablation.

Run from the repository root:

    mars-train --config configs/train/paper.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CONFIG
from .dataset import NpyRFIDataset, packed_collate, seed_worker
from .losses import astro_loss_from_logits, binary_metrics, segmentation_loss
from .model import build_model, count_parameters
from .provenance import (
    ARTIFACT_IDENTITY_KEYS,
    PAPER_ARTIFACT_ROLE,
    PAPER_EXPERIMENT_ID,
    training_fingerprint,
    validate_training_identity,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | os.PathLike[str], root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return root / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(config: dict, train: bool, root: Path) -> DataLoader:
    aug = deepcopy(config["augmentation"])
    if not train and not bool(aug.get("augment_val", False)):
        aug["enabled"] = False

    ds = NpyRFIDataset(
        patch_path=config["train_patch_path"] if train else config["val_patch_path"],
        mask_path=config["train_mask_path"] if train else config["val_mask_path"],
        augmentation=aug,
        root=root,
        train=train,
        seed=int(config["seed"]) + (0 if train else 991),
    )
    expected_key = "expected_train_examples" if train else "expected_val_examples"
    expected_examples = config.get(expected_key)
    if expected_examples is not None and len(ds) != int(expected_examples):
        split = "training" if train else "validation"
        raise ValueError(
            f"The paper {split} split requires {int(expected_examples):,} examples; "
            f"loaded {len(ds):,}. Set {expected_key} to null only for an explicitly "
            "identified non-paper experiment."
        )

    num_workers = int(config["train_num_workers"] if train else config["val_num_workers"])
    kwargs = {
        "batch_size": int(config["batch_size"]),
        "shuffle": bool(train),
        "drop_last": bool(train),
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", False)),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "collate_fn": packed_collate if bool(config.get("packed_collate", True)) else None,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 1))
        start_method = str(config.get("worker_start_method", "")).strip()
        if start_method:
            kwargs["multiprocessing_context"] = start_method
    return DataLoader(ds, **kwargs)


def _amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    config: dict,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_astro_loss = 0.0
    total_items = 0
    use_astro_loss = float(config.get("lambda_astro", 0.0)) > 0.0
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    bar = tqdm(loader, desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)
    for step, (x, y, weight, m_astro) in enumerate(bar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        m_astro = m_astro.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_enabled):
            logits = model(x)
            loss = segmentation_loss(
                logits,
                y,
                weight=weight,
                pos_weight=float(config["pos_weight"]),
                lambda_dice=float(config["lambda_dice"]),
                focal_gamma=float(config.get("focal_gamma", 0.0)),
                lambda_astro=float(config.get("lambda_astro", 0.0)),
                astro_mask=m_astro,
            )
            if use_astro_loss:
                astro_value = astro_loss_from_logits(logits.detach().float(), y.float(), m_astro.float())

        if amp_enabled:
            scaler.scale(loss).backward()
            if float(config.get("clip_grad_norm", 0.0)) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["clip_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(config.get("clip_grad_norm", 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["clip_grad_norm"]))
            optimizer.step()

        bs = int(x.shape[0])
        total_loss += float(loss.detach().item()) * bs
        if use_astro_loss:
            total_astro_loss += float(astro_value.detach().item()) * bs
        total_items += bs
        if step % int(config.get("log_every", 50)) == 0:
            postfix = {"loss": f"{total_loss / max(1, total_items):.4f}"}
            if use_astro_loss:
                postfix["astro"] = f"{total_astro_loss / max(1, total_items):.4f}"
            bar.set_postfix(postfix)
    stats = {"loss": total_loss / max(1, total_items)}
    if use_astro_loss:
        stats["astro_loss"] = total_astro_loss / max(1, total_items)
    return stats


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_astro_loss = 0.0
    total_items = 0
    use_astro_loss = float(config.get("lambda_astro", 0.0)) > 0.0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    bar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for x, y, weight, m_astro in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        m_astro = m_astro.to(device, non_blocking=True)
        with _amp_context(device, amp_enabled):
            logits = model(x)
            loss = segmentation_loss(
                logits,
                y,
                weight=weight,
                pos_weight=float(config["pos_weight"]),
                lambda_dice=float(config["lambda_dice"]),
                focal_gamma=float(config.get("focal_gamma", 0.0)),
                lambda_astro=float(config.get("lambda_astro", 0.0)),
                astro_mask=m_astro,
            )
            if use_astro_loss:
                astro_value = astro_loss_from_logits(logits.float(), y.float(), m_astro.float())
        bs = int(x.shape[0])
        total_loss += float(loss.detach().item()) * bs
        if use_astro_loss:
            total_astro_loss += float(astro_value.detach().item()) * bs
        total_items += bs
        batch_metrics = binary_metrics(logits.float(), y.float(), threshold=float(config["threshold"]))
        for key in counts:
            counts[key] += batch_metrics[key]

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    eps = 1.0e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    stats = {
        "loss": total_loss / max(1, total_items),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
    if use_astro_loss:
        stats["astro_loss"] = total_astro_loss / max(1, total_items)
    return stats


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, config, stats):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "config": config,
        "stats": stats,
    }
    torch.save(payload, path)


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="JSON object merged into the paper defaults before CLI overrides.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--lambda-astro", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--model",
        choices=("light_unet_v1", "light_unet_v2", "trt_fast_unet", "trt_shape_unet"),
    )
    parser.add_argument(
        "--decoder-horizontal-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--decoder-horizontal-refine-stages", type=str)
    parser.add_argument("--decoder-horizontal-refine-kernel", type=int)
    parser.add_argument(
        "--decoder-vertical-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--decoder-vertical-refine-stages", type=str)
    parser.add_argument("--decoder-vertical-refine-kernel", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--train-patches", type=str)
    parser.add_argument("--train-masks", type=str)
    parser.add_argument("--val-patches", type=str)
    parser.add_argument("--val-masks", type=str)
    return parser.parse_args(argv)


def apply_overrides(config: dict, args) -> dict:
    config = deepcopy(config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.pos_weight is not None:
        config["pos_weight"] = args.pos_weight
    if args.lambda_astro is not None:
        config["lambda_astro"] = args.lambda_astro
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model is not None:
        config["model"] = args.model
    if args.decoder_horizontal_refine_enabled is not None:
        config["decoder_horizontal_refine_enabled"] = bool(args.decoder_horizontal_refine_enabled)
    if args.decoder_horizontal_refine_stages is not None:
        config["decoder_horizontal_refine_stages"] = args.decoder_horizontal_refine_stages
    if args.decoder_horizontal_refine_kernel is not None:
        config["decoder_horizontal_refine_kernel"] = int(args.decoder_horizontal_refine_kernel)
    if args.decoder_vertical_refine_enabled is not None:
        config["decoder_vertical_refine_enabled"] = bool(args.decoder_vertical_refine_enabled)
    if args.decoder_vertical_refine_stages is not None:
        config["decoder_vertical_refine_stages"] = args.decoder_vertical_refine_stages
    if args.decoder_vertical_refine_kernel is not None:
        config["decoder_vertical_refine_kernel"] = int(args.decoder_vertical_refine_kernel)
    if args.num_workers is not None:
        config["train_num_workers"] = args.num_workers
    if args.out_dir is not None:
        config["out_dir"] = args.out_dir
    if args.train_patches is not None:
        config["train_patch_path"] = args.train_patches
    if args.train_masks is not None:
        config["train_mask_path"] = args.train_masks
    if args.val_patches is not None:
        config["val_patch_path"] = args.val_patches
    if args.val_masks is not None:
        config["val_mask_path"] = args.val_masks
    return config


def prepare_training_config(config: dict) -> dict:
    """Stamp a resolved config and prevent ablations from using paper identity."""

    resolved = deepcopy(config)
    resolved["training_fingerprint"] = training_fingerprint(resolved)
    validate_training_identity(resolved, source="resolved training config")

    experiment_id = str(resolved["experiment_id"])
    artifact_role = str(resolved["artifact_role"])
    root = _repo_root()
    paper_out_dir = _resolve(CONFIG["out_dir"], root).resolve()
    requested_out_dir = _resolve(resolved["out_dir"], root).resolve()

    if artifact_role == PAPER_ARTIFACT_ROLE:
        if experiment_id != PAPER_EXPERIMENT_ID:
            raise ValueError(
                f"Paper artifacts must use experiment_id={PAPER_EXPERIMENT_ID!r}."
            )
        if resolved["training_fingerprint"] != CONFIG["training_fingerprint"]:
            raise ValueError(
                "The paper training profile was modified. Use an explicit JSON config "
                "with a new experiment_id, artifact_role='ablation', and a non-paper "
                "out_dir instead of overriding paper loss/model/training fields."
            )
    else:
        if experiment_id == PAPER_EXPERIMENT_ID:
            raise ValueError("A non-paper artifact cannot reuse the paper experiment_id.")
        if requested_out_dir == paper_out_dir:
            raise ValueError("A non-paper experiment cannot write to the paper artifact directory.")

    return resolved


def validate_output_identity(out_dir: Path, config: dict) -> None:
    config_path = out_dir / "config.json"
    if not config_path.is_file():
        if out_dir.is_dir() and any(out_dir.glob("*.pt")):
            raise FileExistsError(
                f"Refusing to write into an unidentified checkpoint directory: {out_dir}"
            )
        return

    with config_path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    mismatches = [
        key
        for key in ARTIFACT_IDENTITY_KEYS
        if existing.get(key) != config.get(key)
    ]
    if mismatches:
        raise FileExistsError(
            f"Checkpoint directory {out_dir} belongs to a different experiment; "
            f"identity differs for {mismatches}."
        )


def run(config: dict) -> None:
    config = prepare_training_config(config)
    root = _repo_root()
    os.chdir(root)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sharing = str(config.get("sharing_strategy", "file_descriptor"))
    try:
        torch.multiprocessing.set_sharing_strategy(sharing)
    except RuntimeError as exc:
        print(f"[warn] could not set torch sharing strategy {sharing!r}: {exc}")

    seed_everything(int(config["seed"]))
    out_dir = _resolve(config["out_dir"], root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    parameter_count = count_parameters(model)
    expected_parameters = config.get("expected_parameters")
    if expected_parameters is not None and parameter_count != int(expected_parameters):
        raise ValueError(
            "Configured model has "
            f"{parameter_count:,} trainable parameters; expected "
            f"{int(expected_parameters):,}. Set expected_parameters to null for "
            "an intentional architecture ablation."
        )
    print(f"device: {device}")
    print(f"model: {config.get('model', 'trt_shape_unet')}")
    print(f"trainable params: {parameter_count:,}")
    print(f"out_dir: {out_dir}")

    validate_output_identity(out_dir, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    train_loader = make_loader(config, train=True, root=root)
    val_loader = make_loader(config, train=False, root=root)

    if str(config.get("optimizer", "AdamW")).lower() != "adamw":
        raise ValueError("Only the paper AdamW optimizer is implemented.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    if str(config.get("scheduler", "ReduceLROnPlateau")).lower() != "reducelronplateau":
        raise ValueError("Only the paper ReduceLROnPlateau scheduler is implemented.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        min_lr=float(config["scheduler_min_lr"]),
    )
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_f1 = -1.0
    best_loss = float("inf")
    history = []
    for epoch in tqdm(range(1, int(config["epochs"]) + 1), desc="epochs", dynamic_ncols=True):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
        val_stats = evaluate(model, val_loader, device, config)
        scheduler.step(float(val_stats["f1"]))

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_precision": val_stats["precision"],
            "val_recall": val_stats["recall"],
            "val_f1": val_stats["f1"],
            "val_iou": val_stats["iou"],
        }
        if float(config.get("lambda_astro", 0.0)) > 0.0:
            row["train_astro_loss"] = train_stats["astro_loss"]
            row["val_astro_loss"] = val_stats["astro_loss"]
        history.append(row)
        write_history(out_dir / "history.csv", history)

        msg = (
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"f1={row['val_f1']:.4f} iou={row['val_iou']:.4f} "
            f"p={row['val_precision']:.4f} r={row['val_recall']:.4f}"
        )
        if float(config.get("lambda_astro", 0.0)) > 0.0:
            msg += (
                f" train_astro={row['train_astro_loss']:.4f}"
                f" val_astro={row['val_astro_loss']:.4f}"
            )
        print(msg)

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, config, row)
        if row["val_f1"] > best_f1:
            best_f1 = row["val_f1"]
            save_checkpoint(out_dir / "best_f1.pt", model, optimizer, scheduler, scaler, epoch, config, row)
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            save_checkpoint(out_dir / "best_loss.pt", model, optimizer, scheduler, scaler, epoch, config, row)

    print(f"best F1: {best_f1:.4f}")
    print(f"best val loss: {best_loss:.4f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = deepcopy(CONFIG)
    if args.config is not None:
        config_path = Path(args.config)
        with config_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError(f"Training config must be a JSON object: {config_path}")
        config.update(loaded)
    run(apply_overrides(config, args))


if __name__ == "__main__":
    main(sys.argv[1:])
