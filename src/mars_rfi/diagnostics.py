#!/usr/bin/env python3
"""Generate patch-level diagnostics for MARS filterbank inference.

The command uses the same preprocessing functions as the public mitigation
pipeline. Input, checkpoint, and output paths are required CLI arguments so a
developer's local paths cannot silently select an experiment.

Run from the repository root:

    mars-diagnostics --input-fil observation.fil \
      --checkpoint artifacts/checkpoints/mars-paper/best_f1.pt \
      --output-dir outputs/diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mars-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mars-numba"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sigpyproc.readers import FilReader

from .config import CONFIG as TRAIN_CONFIG
from .model import build_model, count_parameters
from .preprocess import (
    replace_negative_blocks_2d,
    replace_negative_blocks_2d_per_segment,
    replace_continuous_negative_segments,
    replace_continuous_negative_segments_per_segment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
from .pipeline import (  # noqa: E402
    CONFIG as V6_CONFIG,
    _apply_raw_detector_mode,
    apply_nn_input_blank_mode,
    detect_raw_segment_persistent_rfi_gpu,
    split_segment_to_patches,
)


# --- User-editable controls ---
INPUT_FIL = None
CHECKPOINT = "artifacts/checkpoints/mars-paper/best_f1.pt"
OUTPUT_DIR = None

THRESHOLD = 0.5
HYS_ENABLED = False
HYS_SEED_THRESHOLD = None  # None means use THRESHOLD as the seed threshold.
HYS_SUPPORT_THRESHOLD = 0.35

N_VISUALIZE = 12
SELECTION_MODE = "residual_score"
VISUALIZE_INDICES = None
VISUALIZE_RESULT_INDICES = None
VISUALIZE_RESULT_FILES = None
VISUALIZE_RESULT_SUMMARY = None
BATCH_SIZE = 32


CONFIG = {
    # Input / output.
    "input_fil": INPUT_FIL,
    "output_dir": OUTPUT_DIR,

    # MARS checkpoint and inference.
    "checkpoint": CHECKPOINT,
    "model": str(TRAIN_CONFIG["model"]),
    "channels": tuple(TRAIN_CONFIG["channels"]),
    "in_channels": int(TRAIN_CONFIG["in_channels"]),
    "out_channels": int(TRAIN_CONFIG["out_channels"]),
    "output_bias_prior": float(TRAIN_CONFIG["output_bias_prior"]),
    "axis_reduction": int(TRAIN_CONFIG.get("axis_reduction", 4)),
    "anisotropic_kernel": int(TRAIN_CONFIG.get("anisotropic_kernel", 9)),
    "bottleneck_dilation": int(TRAIN_CONFIG.get("bottleneck_dilation", 2)),
    "shape_kernel": int(TRAIN_CONFIG.get("shape_kernel", 9)),
    "decoder_horizontal_refine_enabled": bool(
        TRAIN_CONFIG.get("decoder_horizontal_refine_enabled", False)
    ),
    "decoder_horizontal_refine_stages": tuple(
        TRAIN_CONFIG.get("decoder_horizontal_refine_stages", ("up1",))
    ),
    "decoder_horizontal_refine_kernel": int(
        TRAIN_CONFIG.get("decoder_horizontal_refine_kernel", 15)
    ),
    "decoder_vertical_refine_enabled": bool(
        TRAIN_CONFIG.get("decoder_vertical_refine_enabled", False)
    ),
    "decoder_vertical_refine_stages": tuple(
        TRAIN_CONFIG.get("decoder_vertical_refine_stages", ())
    ),
    "decoder_vertical_refine_kernel": int(
        TRAIN_CONFIG.get("decoder_vertical_refine_kernel", 9)
    ),
    "batch_size": BATCH_SIZE,
    "use_amp": True,
    "threshold": THRESHOLD,

    # Paper-aligned preprocessing controls.
    "target_segment_seconds": 2.0,
    "patch_size": 512,
    "sat_sigma": 6.0,
    "sat_ratio_segment": 0.5,
    "mad_const": 1.4826,
    "mad_min_valid": 0.5,
    "tanh_scale": 6.0,
    "negative_segment_preprocess_enabled": False,
    "negative_segment_min_len": 96,
    "negative_segment_low_threshold": 0.0,
    "negative_segment_low_fraction": 0.90,
    "negative_segment_replace_ceiling": 0.0,
    "negative_segment_context_abs_max": 0.35,
    "negative_segment_min_context_fraction": 0.20,
    "negative_segment_std_floor": 0.015,
    "negative_segment_std_scale": 1.0,
    "negative_block_preprocess_enabled": False,
    "negative_block_freq_window": 9,
    "negative_block_time_window": 33,
    "negative_block_low_threshold": 0.0,
    "negative_block_low_fraction": 0.75,
    "negative_block_replace_ceiling": 0.0,
    "negative_block_context_abs_max": 0.35,
    "negative_block_min_context_fraction": 0.20,
    "negative_block_std_floor": 0.015,
    "negative_block_std_scale": 1.0,
    "nn_input_blank_mode": "interp",
    "nn_input_blank_noise_std": 1.0,
    "raw_gpu_dtype": "float32",
    "raw_compute_dtype": "float16",
    "raw_segment_detector_enabled": True,
    "raw_segment_detector_mode": "guarded",
    "raw_segment_detector_max_live_fraction": 0.5,
    "raw_segment_detector_high_sigma": 5.0,
    "raw_segment_detector_low_sigma": 2.5,
    "raw_segment_detector_seed_occupancy": 0.10,
    "raw_segment_detector_support_occupancy": 0.02,
    "raw_segment_detector_density_window_chans": 49,
    "raw_segment_detector_density_threshold": 0.20,
    "raw_segment_detector_support_density_threshold": 0.35,
    "raw_segment_detector_dilate_chans": 4,

    # Hysteresis diagnostic mask.
    "hys_enabled": HYS_ENABLED,
    "hys_seed_threshold": HYS_SEED_THRESHOLD,
    "hys_support_threshold": HYS_SUPPORT_THRESHOLD,
    "hys_connect_time_radius": 5,
    "hys_connect_freq_radius": 1,
    "hys_connect_iterations": 1,
    "hys_close_time_radius": 3,
    "hys_dilate_freq_radius": 0,
    "hys_dilate_time_radius": 1,
    "hys_row_min_support_fraction": 0.01,
    "hys_max_patch_mask_fraction": 0.15,
    "hys_max_added_fraction": 0.02,
    "hys_verbose": True,

    # Patch selection.
    "selection_mode": SELECTION_MODE,
    "n_visualize": N_VISUALIZE,
    "residual_score_source": "cleaned",
    "residual_score_abs_z_threshold": 2.5,
    "residual_score_row_quantile": 98.0,
    "residual_score_min_time_patch_gap": 2,
    "visualize_indices": VISUALIZE_INDICES,
    # Optional: revisit previous result_N diagnostics.
    # This is useful when comparing checkpoints on exactly the same patches:
    # keep VISUALIZE_RESULT_SUMMARY pointing to the reference run's
    # diagnostic_summary.json, then change CHECKPOINT and OUTPUT_DIR for the
    # checkpoint being diagnosed.
    # Use exactly one of these modes:
    #   visualize_indices: original patch_index values.
    #   visualize_result_indices + visualize_result_summary: previous result_N ids.
    #   visualize_result_files: previous result_*.png paths; summary is inferred.
    "visualize_result_indices": VISUALIZE_RESULT_INDICES,
    "visualize_result_files": VISUALIZE_RESULT_FILES,
    "visualize_result_summary": VISUALIZE_RESULT_SUMMARY,
    "max_segments": None,

    # Outputs.
    "save_selected_arrays": True,
    "dpi": 150,
    "tensorrt_path": None,
}


def _repo_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _parse_index_values(values, *, label: str) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]

    text = " ".join(str(value) for value in values).strip()
    if not text:
        return []

    indices = []
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        result_match = re.search(r"(?:^|[/\\])result_(\d+)(?:\.[A-Za-z0-9]+)?$", token)
        if result_match:
            indices.append(int(result_match.group(1)))
            continue
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Could not parse {label} index token: {token!r}") from exc
    return indices


def _default_summary_from_result_files(result_files) -> Path | None:
    if not result_files:
        return None
    first = _repo_path(result_files[0])
    summary = first.parent / "diagnostic_summary.json"
    for item in result_files[1:]:
        item_path = _repo_path(item)
        if item_path.parent != first.parent:
            raise ValueError(
                "--result-files must all come from the same diagnostics directory "
                "unless --result-summary is provided explicitly"
            )
    return summary


def _patch_indices_from_result_summary(
    summary_path: str | os.PathLike[str],
    result_indices: list[int],
) -> tuple[list[int], dict[int, dict]]:
    path = _repo_path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic summary not found: {path}")

    with open(path) as f:
        summary = json.load(f)

    selected = summary.get("selected", [])
    by_result = {
        int(record["result_index"]): record
        for record in selected
        if "result_index" in record and "patch_index" in record
    }

    patch_indices = []
    extra_by_patch = {}
    missing = []
    for result_idx in result_indices:
        record = by_result.get(int(result_idx))
        if record is None:
            missing.append(int(result_idx))
            continue
        patch_idx = int(record["patch_index"])
        patch_indices.append(patch_idx)
        extra_by_patch[patch_idx] = {
            "source_result_index": int(result_idx),
            "source_result_path": record.get("path"),
            "source_diagnostic_summary": str(path),
        }

    if missing:
        available = sorted(by_result)
        raise KeyError(
            f"result indices not found in {path}: {missing}; available={available}"
        )

    return patch_indices, extra_by_patch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype_from_name(name, *, kind: str):
    name = str(name).lower()
    if kind == "raw" and name in ("uint8", "u8", "byte"):
        return torch.uint8, "uint8"
    if name in ("float16", "fp16", "half"):
        return torch.float16, "float16"
    if name in ("float32", "fp32", "float"):
        return torch.float32, "float32"
    raise ValueError(f"Unsupported {kind} dtype: {name}")


def _load_filterbank_to_gpu(cfg: dict, device: torch.device):
    fil = FilReader(cfg["input_fil"])
    block = fil.read_block(0, fil.header.nsamples)
    raw_dtype, raw_dtype_name = _dtype_from_name(cfg["raw_gpu_dtype"], kind="raw")
    raw = torch.from_numpy(block.data).to(device=device, dtype=raw_dtype)
    del block
    cfg["_tsamp"] = fil.header.tsamp
    cfg["_raw_gpu_dtype"] = raw_dtype_name
    cfg["_raw_gpu_element_size"] = torch.empty((), dtype=raw_dtype).element_size()
    return raw, fil.header


def _patch_infos_for_meta(meta, segment_idx: int, seg_start_sample: int, patch_size: int):
    infos = []
    n_tp = meta["n_time_patches"]
    nchans = meta["nchans"]
    if meta.get("small_channel_packing", False):
        per_patch = int(meta["time_blocks_per_patch"])
        for patch_idx in range(int(meta["patch_count"])):
            first_time = patch_idx * per_patch
            packed_count = min(per_patch, n_tp - first_time)
            t0 = seg_start_sample + first_time * patch_size
            infos.append(
                {
                    "segment": int(segment_idx),
                    "channel_start": 0,
                    "channel_end": int(nchans),
                    "sample_start": int(t0),
                    "sample_end": int(t0 + packed_count * patch_size),
                    "small_channel_packing": True,
                    "packed_time_blocks": int(packed_count),
                }
            )
        return infos
    for ch_block in range(meta["full_chan_blocks"]):
        ch0 = ch_block * patch_size
        ch1 = ch0 + patch_size
        for t_patch in range(n_tp):
            t0 = seg_start_sample + t_patch * patch_size
            infos.append(
                {
                    "segment": int(segment_idx),
                    "channel_start": int(ch0),
                    "channel_end": int(ch1),
                    "sample_start": int(t0),
                    "sample_end": int(t0 + patch_size),
                }
            )
    if meta["has_overlap"]:
        ch0 = nchans - patch_size
        ch1 = nchans
        for t_patch in range(n_tp):
            t0 = seg_start_sample + t_patch * patch_size
            infos.append(
                {
                    "segment": int(segment_idx),
                    "channel_start": int(ch0),
                    "channel_end": int(ch1),
                    "sample_start": int(t0),
                    "sample_end": int(t0 + patch_size),
                    "overlap_block": True,
                }
            )
    return infos


def _preprocess_filterbank_to_nn_patches(raw, header, cfg: dict, device: torch.device):
    nchans, ntime = raw.shape
    patch_size = int(cfg["patch_size"])
    tsamp = float(header.tsamp)
    _, raw_compute_dtype_name = _dtype_from_name(cfg["raw_compute_dtype"], kind="compute")
    raw_compute_dtype = torch.float16 if raw_compute_dtype_name == "float16" else torch.float32

    target_samples = cfg["target_segment_seconds"] / tsamp
    n_patches_per_seg = max(1, int(round(target_samples / patch_size)))
    seg_len = n_patches_per_seg * patch_size
    n_full_segs_total = ntime // seg_len
    if cfg.get("max_segments") is not None:
        n_full_segs = min(n_full_segs_total, int(cfg["max_segments"]))
    else:
        n_full_segs = n_full_segs_total
    usable_time = n_full_segs * seg_len
    tail_len = 0 if n_full_segs < n_full_segs_total else ntime - usable_time

    if n_full_segs <= 0 and tail_len < patch_size:
        raise ValueError("Filterbank is too short to produce a 512-sample diagnostic patch")

    print(
        f"  Segment length: {seg_len} samples ({seg_len * tsamp:.3f}s), "
        f"using {n_full_segs} full segments"
        + (f" + {tail_len} tail samples" if tail_len else "")
    )

    nn_global_blank = torch.zeros(nchans, dtype=torch.bool, device=device)
    all_patches = []
    all_infos = []

    if n_full_segs > 0:
        raw_segs = raw[:, :usable_time].reshape(nchans, n_full_segs, seg_len)
        if raw_segs.dtype != raw_compute_dtype:
            raw_segs = raw_segs.to(raw_compute_dtype)

        mad_const = float(cfg["mad_const"])
        seg_medians = raw_segs.median(dim=2, keepdim=True).values
        seg_mads = (raw_segs - seg_medians).abs().median(dim=2, keepdim=True).values * mad_const
        science_vars, _ = torch.var_mean(raw_segs, dim=2, keepdim=True, unbiased=True)
        seg_stds = torch.sqrt(torch.clamp(science_vars, min=0.0))
        seg_dead = seg_stds.squeeze(2) < 1e-6

        seg_ch_medians = seg_medians.squeeze(2)
        seg_global_median = seg_ch_medians.median(dim=0).values
        seg_global_mad = (
            (seg_ch_medians - seg_global_median).abs().median(dim=0).values
            * mad_const
        )
        seg_sat_thresh = seg_global_median + cfg["sat_sigma"] * seg_global_mad
        seg_sat_fraction = (
            raw_segs > seg_sat_thresh.unsqueeze(0).unsqueeze(2)
        ).float().mean(dim=2)
        seg_sat_unusable = seg_sat_fraction > cfg["sat_ratio_segment"]

        if cfg.get("raw_segment_detector_enabled", False):
            seg_raw_persistent_rfi = detect_raw_segment_persistent_rfi_gpu(
                seg_ch_medians,
                seg_mads.squeeze(2),
                seg_sat_fraction,
                cfg,
                label="rfi-light-unet-diagnostic",
            )
            seg_raw_persistent_rfi = _apply_raw_detector_mode(
                seg_raw_persistent_rfi,
                seg_dead,
                cfg,
                label="rfi-light-unet-diagnostic",
            )
        else:
            seg_raw_persistent_rfi = torch.zeros_like(seg_sat_unusable)

        seg_unusable = seg_sat_unusable | seg_raw_persistent_rfi
        combined_unusable = nn_global_blank.unsqueeze(1) | seg_unusable

        use_std = seg_mads < cfg["mad_min_valid"]
        scale = torch.where(use_std, seg_stds, seg_mads)
        safe_scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        z_segs = (raw_segs - seg_medians) / safe_scale
        degenerate = (scale.squeeze(2) < 1e-6) | combined_unusable
        z_segs = apply_nn_input_blank_mode(z_segs, degenerate, cfg)
        z_segs, neg_seg_stats = replace_continuous_negative_segments_per_segment(
            z_segs,
            cfg,
            segment_dim=1,
            label="diagnostics-full-segments",
        )
        z_segs, neg_block_stats = replace_negative_blocks_2d_per_segment(
            z_segs,
            cfg,
            segment_dim=1,
            label="diagnostics-full-segments",
        )
        tanh_data = torch.tanh(z_segs / cfg["tanh_scale"])
        if neg_seg_stats.get("enabled", 0.0):
            print(
                "  Negative-segment NN-input cleanup (pre-tanh full): "
                f"replaced={neg_seg_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_seg_stats.get('rows_touched', 0))}/"
                f"{int(neg_seg_stats.get('n_rows', 0))}"
            )
        if neg_block_stats.get("enabled", 0.0):
            print(
                "  Negative-block NN-input cleanup (pre-tanh full): "
                f"replaced={neg_block_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_block_stats.get('rows_touched', 0))}/"
                f"{int(neg_block_stats.get('n_rows', 0))}, "
                f"windows={int(neg_block_stats.get('windows', 0))}"
            )

        for s in range(n_full_segs):
            patches, meta = split_segment_to_patches(tanh_data[:, s, :], patch_size)
            all_patches.append(patches)
            all_infos.extend(_patch_infos_for_meta(meta, s, s * seg_len, patch_size))

        del raw_segs, seg_medians, seg_mads, seg_stds, seg_dead
        del seg_ch_medians, seg_global_median, seg_global_mad, seg_sat_thresh
        del seg_sat_fraction, seg_sat_unusable, seg_raw_persistent_rfi
        del seg_unusable, combined_unusable, scale, safe_scale, z_segs, tanh_data

    if tail_len >= patch_size:
        tail_start = usable_time
        tail_seg = raw[:, tail_start:ntime]
        if tail_seg.dtype != raw_compute_dtype:
            tail_seg = tail_seg.to(raw_compute_dtype)

        mad_const = float(cfg["mad_const"])
        tail_median = tail_seg.median(dim=1, keepdim=True).values
        tail_mad = (tail_seg - tail_median).abs().median(dim=1, keepdim=True).values * mad_const
        tail_std = tail_seg.std(dim=1, keepdim=True)
        tail_dead = tail_std.squeeze(1) < 1e-6
        tail_use_std = tail_mad < cfg["mad_min_valid"]
        tail_scale = torch.where(tail_use_std, tail_std, tail_mad)
        safe_tail_scale = torch.where(
            tail_scale > 1e-6,
            tail_scale,
            torch.ones_like(tail_scale),
        )
        z_tail = (tail_seg - tail_median) / safe_tail_scale

        tail_ch_medians = tail_median.squeeze(1)
        tail_global_median = tail_ch_medians.median()
        tail_global_mad = (tail_ch_medians - tail_global_median).abs().median() * mad_const
        tail_sat_thresh = tail_global_median + cfg["sat_sigma"] * tail_global_mad
        tail_sat_fraction = (tail_seg > tail_sat_thresh).float().mean(dim=1)
        tail_sat_unusable = tail_sat_fraction > cfg["sat_ratio_segment"]

        if cfg.get("raw_segment_detector_enabled", False):
            tail_raw_persistent_rfi = detect_raw_segment_persistent_rfi_gpu(
                tail_ch_medians.unsqueeze(1),
                tail_mad.squeeze(1).unsqueeze(1),
                tail_sat_fraction.unsqueeze(1),
                cfg,
                label="rfi-light-unet-diagnostic-tail",
            )
            tail_raw_persistent_rfi = _apply_raw_detector_mode(
                tail_raw_persistent_rfi,
                tail_dead.unsqueeze(1),
                cfg,
                label="rfi-light-unet-diagnostic-tail",
            ).squeeze(1)
        else:
            tail_raw_persistent_rfi = torch.zeros_like(tail_sat_unusable)

        tail_unusable = tail_sat_unusable | tail_raw_persistent_rfi
        tail_degen = (
            (tail_scale.squeeze(1) < 1e-6)
            | nn_global_blank
            | tail_unusable
        )
        z_tail = apply_nn_input_blank_mode(
            z_tail.unsqueeze(1),
            tail_degen.unsqueeze(1),
            cfg,
        ).squeeze(1)
        usable_tail = (tail_len // patch_size) * patch_size
        z_tail_for_nn, neg_tail_stats = replace_continuous_negative_segments(
            z_tail[:, :usable_tail],
            cfg,
            label="diagnostics-tail",
        )
        z_tail_for_nn, neg_tail_block_stats = replace_negative_blocks_2d(
            z_tail_for_nn,
            cfg,
            label="diagnostics-tail",
        )
        tanh_tail = torch.tanh(z_tail_for_nn / cfg["tanh_scale"])
        del z_tail_for_nn
        if neg_tail_stats.get("enabled", 0.0):
            print(
                "  Negative-segment NN-input cleanup (pre-tanh tail): "
                f"replaced={neg_tail_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_tail_stats.get('rows_touched', 0))}/"
                f"{int(neg_tail_stats.get('n_rows', 0))}"
            )
        if neg_tail_block_stats.get("enabled", 0.0):
            print(
                "  Negative-block NN-input cleanup (pre-tanh tail): "
                f"replaced={neg_tail_block_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_tail_block_stats.get('rows_touched', 0))}/"
                f"{int(neg_tail_block_stats.get('n_rows', 0))}, "
                f"windows={int(neg_tail_block_stats.get('windows', 0))}"
            )
        patches, meta = split_segment_to_patches(tanh_tail, patch_size)
        all_patches.append(patches)
        all_infos.extend(_patch_infos_for_meta(meta, n_full_segs, tail_start, patch_size))

        del tail_seg, tail_median, tail_mad, tail_std, tail_dead
        del tail_scale, safe_tail_scale, z_tail, tanh_tail
        del tail_ch_medians, tail_global_median, tail_global_mad
        del tail_sat_thresh, tail_sat_fraction, tail_sat_unusable
        del tail_raw_persistent_rfi, tail_unusable, tail_degen

    return torch.cat(all_patches, dim=0), all_infos


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def load_mars_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    ckpt_path = _repo_path(cfg["checkpoint"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            "MARS checkpoint does not exist. Train it first or pass "
            f"--checkpoint explicitly: {ckpt_path}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = dict(cfg)
    model_cfg.update(ckpt_config)
    model = build_model(model_cfg).to(device)

    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match mars_rfi.model. "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(
        f"  {model_cfg.get('model', 'trt_shape_unet')} loaded: {ckpt_path} "
        f"(epoch {epoch}, params {count_parameters(model):,})"
    )
    return model


@torch.inference_mode()
def infer_patch_prob(model: torch.nn.Module, patches: torch.Tensor, cfg: dict, device: torch.device):
    batch_size = int(cfg["batch_size"])
    probs_out = []
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    for i in range(0, patches.shape[0], batch_size):
        batch = patches[i:i + batch_size]
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(batch)
                probs = torch.sigmoid(logits).float()
        else:
            logits = model(batch)
            probs = torch.sigmoid(logits)
        probs_out.append(probs.squeeze(1).float())
    return torch.cat(probs_out, dim=0)


def _dilate_bool(mask: torch.Tensor, freq_radius: int, time_radius: int) -> torch.Tensor:
    freq_radius = max(0, int(freq_radius))
    time_radius = max(0, int(time_radius))
    if freq_radius == 0 and time_radius == 0:
        return mask
    kernel = (2 * freq_radius + 1, 2 * time_radius + 1)
    padding = (freq_radius, time_radius)
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=padding)
    return y.squeeze(1) > 0.0


def _erode_bool(mask: torch.Tensor, freq_radius: int, time_radius: int) -> torch.Tensor:
    return ~_dilate_bool(~mask, freq_radius, time_radius)


def _close_bool(mask: torch.Tensor, freq_radius: int, time_radius: int) -> torch.Tensor:
    return _erode_bool(
        _dilate_bool(mask, freq_radius, time_radius),
        freq_radius,
        time_radius,
    )


def apply_hysteresis_mask(prob: torch.Tensor, cfg: dict) -> torch.Tensor:
    seed_threshold = cfg.get("hys_seed_threshold", None)
    if seed_threshold is None:
        seed_threshold = cfg["threshold"]
    seed_threshold = float(seed_threshold)
    support_threshold = float(cfg["hys_support_threshold"])
    if support_threshold > seed_threshold:
        raise ValueError(
            "hys_support_threshold must be <= hys_seed_threshold/threshold "
            f"(got support={support_threshold}, seed={seed_threshold})"
        )

    hard_seed = prob > seed_threshold
    seed = hard_seed
    support = prob > support_threshold

    row_min = float(cfg.get("hys_row_min_support_fraction", 0.0))
    if row_min > 0.0:
        row_support = support.float().mean(dim=2, keepdim=True)
        row_gate = row_support >= row_min
        seed = seed & row_gate
        support = support & row_gate

    connected = seed & support
    for _ in range(int(cfg.get("hys_connect_iterations", 0))):
        grown = _dilate_bool(
            connected,
            cfg.get("hys_connect_freq_radius", 1),
            cfg.get("hys_connect_time_radius", 8),
        )
        connected = grown & support

    close_time = int(cfg.get("hys_close_time_radius", 0))
    if close_time > 0:
        connected = connected | _close_bool(connected, 0, close_time)

    final = _dilate_bool(
        connected,
        cfg.get("hys_dilate_freq_radius", 0),
        cfg.get("hys_dilate_time_radius", 0),
    )
    final = final | hard_seed

    max_patch_fraction = cfg.get("hys_max_patch_mask_fraction", None)
    max_added_fraction = cfg.get("hys_max_added_fraction", None)
    if max_patch_fraction is not None or max_added_fraction is not None:
        final_fraction = final.float().mean(dim=(1, 2))
        added_fraction = (final & ~hard_seed).float().mean(dim=(1, 2))
        keep = torch.ones_like(final_fraction, dtype=torch.bool)
        if max_patch_fraction is not None:
            keep = keep & (final_fraction <= float(max_patch_fraction))
        if max_added_fraction is not None:
            keep = keep & (added_fraction <= float(max_added_fraction))
        final = torch.where(keep.view(-1, 1, 1), final, hard_seed)

    if cfg.get("hys_verbose", False):
        seed_frac = hard_seed.float().mean().item()
        support_frac = support.float().mean().item()
        final_frac = final.float().mean().item()
        print(
            "  Hysteresis diagnostic mask: "
            f"seed>{seed_threshold:.3g} frac={seed_frac:.4f}, "
            f"support>{support_threshold:.3g} frac={support_frac:.4f}, "
            f"final frac={final_frac:.4f}"
        )

    return final


def robust_z(patch: np.ndarray) -> np.ndarray:
    x = patch.astype(np.float32, copy=False)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-6:
        std = float(np.std(x))
        mad = std if std > 1e-6 else 1.0
    return (x - med) / mad


def residual_patch_score(patch: np.ndarray, cfg: dict) -> dict[str, float]:
    z = robust_z(patch)
    abs_z = np.abs(z)
    high = abs_z >= float(cfg["residual_score_abs_z_threshold"])
    row_high_frac = high.mean(axis=1)
    row_abs = np.percentile(
        abs_z,
        float(cfg["residual_score_row_quantile"]),
        axis=1,
    )
    col_high_frac = high.mean(axis=0)

    horizontal_score = float(np.percentile(row_abs, 99.0))
    row_occupancy_score = float(np.percentile(row_high_frac, 99.0))
    broadband_column_score = float(np.percentile(col_high_frac, 99.0))
    high_fraction = float(high.mean())

    score = (
        horizontal_score
        + 8.0 * row_occupancy_score
        + 2.0 * broadband_column_score
        + 4.0 * high_fraction
    )
    return {
        "residual_score": float(score),
        "residual_horizontal_score": horizontal_score,
        "residual_row_occupancy_score": row_occupancy_score,
        "residual_broadband_column_score": broadband_column_score,
        "residual_high_fraction": high_fraction,
        "residual_mean": float(np.mean(patch)),
        "residual_std": float(np.std(patch)),
        "residual_p01": float(np.percentile(patch, 1)),
        "residual_p50": float(np.percentile(patch, 50)),
        "residual_p99": float(np.percentile(patch, 99)),
    }


def _sample_patch_number(info: dict, patch_size: int) -> int:
    return int(info["sample_start"]) // int(patch_size)


def _select_top_residual_candidates(candidates, patch_infos, cfg: dict, n_visualize: int):
    min_gap = int(cfg.get("residual_score_min_time_patch_gap", 0))
    patch_size = int(cfg["patch_size"])
    selected = []
    for item in sorted(candidates, key=lambda x: x["residual_score"], reverse=True):
        info = patch_infos[item["patch_index"]]
        duplicate = False
        if min_gap > 0:
            time_patch = _sample_patch_number(info, patch_size)
            for prev in selected:
                prev_info = patch_infos[prev["patch_index"]]
                same_freq = int(info["channel_start"]) == int(prev_info["channel_start"])
                prev_time_patch = _sample_patch_number(prev_info, patch_size)
                if same_freq and abs(time_patch - prev_time_patch) < min_gap:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(item)
        if len(selected) >= n_visualize:
            break
    return selected


def score_residual_patches(patches: torch.Tensor, binary: torch.Tensor, cfg: dict, patch_infos):
    if patch_infos is None:
        raise ValueError("patch_infos is required for residual_score selection")

    source = str(cfg.get("residual_score_source", "cleaned")).lower()
    if source not in ("cleaned", "input"):
        raise ValueError(f"Unknown residual_score_source: {source}")

    candidates = []
    for idx in range(int(patches.shape[0])):
        patch = patches[idx, 0].detach().float().cpu().numpy()
        binary_i = binary[idx].detach().float().cpu().numpy()
        score_patch = patch * (1.0 - binary_i) if source == "cleaned" else patch
        stats = residual_patch_score(score_patch, cfg)
        stats["patch_index"] = int(idx)
        candidates.append(stats)
    return candidates


def select_patch_indices(prob, binary, cfg: dict, patches=None, patch_infos=None):
    n = prob.shape[0]
    if cfg.get("visualize_indices") is not None:
        indices = [int(i) for i in cfg["visualize_indices"]]
        for idx in indices:
            if idx < 0 or idx >= n:
                raise IndexError(f"visualize index {idx} out of range 0..{n - 1}")
        return indices, cfg.get("_visualize_index_extra", {})

    n_visualize = min(int(cfg["n_visualize"]), n)
    mode = str(cfg.get("selection_mode", "first")).lower()
    if mode == "first":
        return list(range(n_visualize)), {}
    if mode == "uniform":
        return np.linspace(0, n - 1, n_visualize, dtype=int).tolist(), {}
    if mode == "mask_fraction":
        score = binary.float().mean(dim=(1, 2))
        return torch.argsort(score, descending=True)[:n_visualize].cpu().tolist(), {}
    if mode == "max_prob":
        score = prob.amax(dim=(1, 2))
        return torch.argsort(score, descending=True)[:n_visualize].cpu().tolist(), {}
    if mode == "residual_score":
        if patches is None:
            raise ValueError("patches is required for residual_score selection")
        print("  Scoring cleaned patches for residual-RFI ranking...")
        candidates = score_residual_patches(patches, binary, cfg, patch_infos)
        selected = _select_top_residual_candidates(
            candidates,
            patch_infos,
            cfg,
            n_visualize,
        )
        score_by_index = {item["patch_index"]: item for item in selected}
        return [item["patch_index"] for item in selected], score_by_index
    raise ValueError(f"Unknown selection_mode: {mode}")


def plot_patch_diagnostic(
    patch: np.ndarray,
    prob: np.ndarray,
    pure_binary: np.ndarray,
    binary: np.ndarray,
    out_path: Path,
    threshold: float,
    info: dict,
    dpi: int,
    hys_enabled: bool,
) -> None:
    patch = patch.astype(np.float32)
    prob = prob.astype(np.float32)
    pure_binary = pure_binary.astype(np.float32)
    binary = binary.astype(np.float32)
    cleaned = patch * (1.0 - binary)

    vmin, vmax = np.percentile(patch, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(patch)), float(np.nanmax(patch))

    ncols = 5 if hys_enabled else 4
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    im0 = axes[0].imshow(
        patch,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Input (freq-time)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(prob, aspect="auto", origin="upper", cmap="hot", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted RFI Signal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        pure_binary,
        aspect="auto",
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(f"Pure Mask (RFI>{threshold})")
    plt.colorbar(im2, ax=axes[2])

    cleaned_ax_idx = 3
    if hys_enabled:
        im_hys = axes[3].imshow(
            binary,
            aspect="auto",
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[3].set_title("Hysteresis Mask")
        plt.colorbar(im_hys, ax=axes[3])
        cleaned_ax_idx = 4

    im3 = axes[cleaned_ax_idx].imshow(
        cleaned,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[cleaned_ax_idx].set_title("Cleaned (input with final mask)")
    plt.colorbar(im3, ax=axes[cleaned_ax_idx])

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

    title = (
        "Patch {patch_index} | seg {segment} | ch {channel_start}:{channel_end} | "
        "samples {sample_start}:{sample_end} | pure={pure_mask_fraction:.3f} | "
        "final={mask_fraction:.3f} | added={hys_added_fraction:.3f} | "
        "max p={max_prob:.3f}"
    )
    if "residual_score" in info:
        title += " | residual={residual_score:.3f}"
    if "source_result_index" in info:
        title += " | from result_{source_result_index}"

    fig.suptitle(title.format(**info), fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_outputs(cfg, patches, patch_infos, prob, pure_binary, binary, selected, selection_extra):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_records = []
    selected_patches = []
    selected_probs = []
    selected_pure_binary = []
    selected_binary = []
    selected_cleaned = []

    for out_i, patch_idx in enumerate(selected):
        patch = patches[patch_idx, 0].detach().cpu().numpy()
        prob_i = prob[patch_idx].detach().cpu().numpy()
        pure_binary_i = pure_binary[patch_idx].detach().cpu().numpy().astype(np.float32)
        binary_i = binary[patch_idx].detach().cpu().numpy().astype(np.float32)
        cleaned_i = patch * (1.0 - binary_i)

        info = dict(patch_infos[patch_idx])
        info.update(
            {
                "patch_index": int(patch_idx),
                "result_index": int(out_i),
                "pure_mask_fraction": float(pure_binary_i.mean()),
                "mask_fraction": float(binary_i.mean()),
                "hys_added_fraction": float(np.maximum(binary_i - pure_binary_i, 0.0).mean()),
                "max_prob": float(prob_i.max()),
                "mean_prob": float(prob_i.mean()),
            }
        )
        info.update(selection_extra.get(int(patch_idx), {}))
        out_path = out_dir / f"result_{out_i}.png"
        plot_patch_diagnostic(
            patch,
            prob_i,
            pure_binary_i,
            binary_i,
            out_path,
            float(cfg["threshold"]),
            info,
            int(cfg["dpi"]),
            bool(cfg.get("hys_enabled", False)),
        )

        selected_records.append({**info, "path": str(out_path)})
        selected_patches.append(patch)
        selected_probs.append(prob_i)
        selected_pure_binary.append(pure_binary_i)
        selected_binary.append(binary_i)
        selected_cleaned.append(cleaned_i)

    decoder_refine_stages = cfg.get(
        "decoder_horizontal_refine_stages",
        TRAIN_CONFIG.get("decoder_horizontal_refine_stages", ("up1",)),
    )
    if isinstance(decoder_refine_stages, str):
        decoder_refine_stages = [
            item.strip()
            for item in decoder_refine_stages.split(",")
            if item.strip()
        ]
    else:
        decoder_refine_stages = list(decoder_refine_stages)

    vertical_refine_stages = cfg.get(
        "decoder_vertical_refine_stages",
        TRAIN_CONFIG.get("decoder_vertical_refine_stages", ()),
    )
    if isinstance(vertical_refine_stages, str):
        vertical_refine_stages = [
            item.strip()
            for item in vertical_refine_stages.split(",")
            if item.strip()
        ]
    else:
        vertical_refine_stages = list(vertical_refine_stages)

    summary = {
        "input_fil": cfg["input_fil"],
        "checkpoint": cfg["checkpoint"],
        "model": cfg.get("model", TRAIN_CONFIG["model"]),
        "channels": list(cfg.get("channels", TRAIN_CONFIG["channels"])),
        "axis_reduction": int(cfg.get("axis_reduction", TRAIN_CONFIG.get("axis_reduction", 4))),
        "anisotropic_kernel": int(cfg.get("anisotropic_kernel", TRAIN_CONFIG.get("anisotropic_kernel", 9))),
        "bottleneck_dilation": int(cfg.get("bottleneck_dilation", TRAIN_CONFIG.get("bottleneck_dilation", 2))),
        "shape_kernel": int(cfg.get("shape_kernel", TRAIN_CONFIG.get("shape_kernel", 9))),
        "decoder_horizontal_refine_enabled": bool(
            cfg.get(
                "decoder_horizontal_refine_enabled",
                TRAIN_CONFIG.get("decoder_horizontal_refine_enabled", False),
            )
        ),
        "decoder_horizontal_refine_stages": decoder_refine_stages,
        "decoder_horizontal_refine_kernel": int(
            cfg.get(
                "decoder_horizontal_refine_kernel",
                TRAIN_CONFIG.get("decoder_horizontal_refine_kernel", 15),
            )
        ),
        "decoder_vertical_refine_enabled": bool(
            cfg.get(
                "decoder_vertical_refine_enabled",
                TRAIN_CONFIG.get("decoder_vertical_refine_enabled", False),
            )
        ),
        "decoder_vertical_refine_stages": vertical_refine_stages,
        "decoder_vertical_refine_kernel": int(
            cfg.get(
                "decoder_vertical_refine_kernel",
                TRAIN_CONFIG.get("decoder_vertical_refine_kernel", 9),
            )
        ),
        "threshold": float(cfg["threshold"]),
        "hys_enabled": bool(cfg.get("hys_enabled", False)),
        "hys_seed_threshold": cfg.get("hys_seed_threshold", None),
        "hys_support_threshold": float(cfg.get("hys_support_threshold", 0.0)),
        "selection_mode": cfg.get("selection_mode"),
        "requested_patch_indices": (
            [int(idx) for idx in cfg["visualize_indices"]]
            if cfg.get("visualize_indices") is not None
            else None
        ),
        "requested_result_indices": cfg.get("requested_result_indices"),
        "source_diagnostic_summary": cfg.get("source_diagnostic_summary"),
        "n_total_patches": int(patches.shape[0]),
        "selected": selected_records,
    }
    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if cfg.get("save_selected_arrays", True) and selected_records:
        np.save(out_dir / "selected_input_patches.npy", np.stack(selected_patches))
        np.save(out_dir / "selected_pred_prob.npy", np.stack(selected_probs))
        np.save(out_dir / "selected_pure_binary.npy", np.stack(selected_pure_binary))
        np.save(out_dir / "selected_pred_binary.npy", np.stack(selected_binary))
        np.save(out_dir / "selected_cleaned_patches.npy", np.stack(selected_cleaned))

    print(f"Saved {len(selected_records)} diagnostic figures to {out_dir}")


def build_pipeline_config(user_cfg: dict) -> dict:
    cfg = deepcopy(V6_CONFIG)
    cfg.update(user_cfg)
    cfg["tensorrt_path"] = None
    cfg["runtime_stats"] = False
    if not torch.cuda.is_available():
        cfg["use_amp"] = False
        cfg["raw_compute_dtype"] = "float32"
    return cfg


def run(cfg: dict) -> None:
    cfg = build_pipeline_config(cfg)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    print(f"\n[1/4] Loading filterbank: {cfg['input_fil']}")
    raw, header = _load_filterbank_to_gpu(cfg, device)
    print(f"  Shape: {tuple(raw.shape)}, tsamp={header.tsamp * 1e3:.3f} ms")

    print("\n[2/4] MARS GPU preprocessing and patching...")
    _sync(device)
    patches, patch_infos = _preprocess_filterbank_to_nn_patches(raw, header, cfg, device)
    del raw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _sync(device)
    print(f"  NN patches: {tuple(patches.shape)}")

    print("\n[3/4] Loading MARS and running inference...")
    model = load_mars_model(cfg, device)
    prob = infer_patch_prob(model, patches, cfg, device)
    pure_binary = prob > float(cfg["threshold"])
    if cfg.get("hys_enabled", False):
        binary = apply_hysteresis_mask(prob, cfg)
    else:
        binary = pure_binary
    del model
    _sync(device)

    print("\n[4/4] Selecting patches and writing diagnostics...")
    selected, selection_extra = select_patch_indices(
        prob,
        binary,
        cfg,
        patches=patches,
        patch_infos=patch_infos,
    )
    print(f"  Selected patch indices: {selected}")
    write_outputs(cfg, patches, patch_infos, prob, pure_binary, binary, selected, selection_extra)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-fil", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--hys-support-threshold", type=float)
    parser.add_argument(
        "--hysteresis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional appendix post-processing; disabled by default.",
    )
    parser.add_argument("--n-visualize", type=int)
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=("first", "uniform", "mask_fraction", "max_prob", "residual_score"),
    )
    parser.add_argument(
        "--visualize-indices",
        nargs="+",
        help="Original NN patch indices to visualize, e.g. 725 608 or 725,608.",
    )
    parser.add_argument(
        "--result-indices",
        nargs="+",
        help=(
            "result_N indices from an existing diagnostic_summary.json. These are "
            "translated back to original NN patch indices."
        ),
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        help=(
            "Existing result_*.png paths to revisit. The script infers result indices "
            "and, unless --result-summary is set, uses diagnostic_summary.json from "
            "the same directory."
        ),
    )
    parser.add_argument(
        "--result-summary",
        type=str,
        help="diagnostic_summary.json used with --result-indices/--result-files.",
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args) -> dict:
    cfg = deepcopy(cfg)
    cli_result_indices = _parse_index_values(args.result_indices, label="result")
    cli_result_file_indices = _parse_index_values(args.result_files, label="result file")
    cli_patch_indices = _parse_index_values(args.visualize_indices, label="patch")

    if cli_result_indices is not None and cli_result_file_indices is not None:
        raise ValueError("Use only one of --result-indices or --result-files")
    if cli_patch_indices is not None and (
        cli_result_indices is not None or cli_result_file_indices is not None
    ):
        raise ValueError(
            "Use --visualize-indices for original patch indices, or "
            "--result-indices/--result-files for previous result_N outputs, not both"
        )

    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint
    if args.input_fil is not None:
        cfg["input_fil"] = args.input_fil
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    if args.hys_support_threshold is not None:
        cfg["hys_support_threshold"] = args.hys_support_threshold
    if args.hysteresis is not None:
        cfg["hys_enabled"] = bool(args.hysteresis)
    if args.n_visualize is not None:
        cfg["n_visualize"] = args.n_visualize
    if args.selection_mode is not None:
        cfg["selection_mode"] = args.selection_mode
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    cli_selects_patches = (
        cli_patch_indices is not None
        or cli_result_indices is not None
        or cli_result_file_indices is not None
    )

    if cli_patch_indices is not None:
        cfg["visualize_indices"] = cli_patch_indices

    result_indices = cli_result_indices
    result_files = args.result_files if cli_result_file_indices is not None else None
    summary_path = args.result_summary
    if cli_result_file_indices is not None:
        result_indices = cli_result_file_indices

    if not cli_selects_patches:
        config_patch_indices = _parse_index_values(
            cfg.get("visualize_indices"),
            label="config visualize_indices",
        )
        config_result_indices = _parse_index_values(
            cfg.get("visualize_result_indices"),
            label="config visualize_result_indices",
        )
        config_result_file_indices = _parse_index_values(
            cfg.get("visualize_result_files"),
            label="config visualize_result_files",
        )

        if config_result_indices is not None and config_result_file_indices is not None:
            raise ValueError(
                "Use only one of visualize_result_indices or visualize_result_files in CONFIG"
            )
        if config_patch_indices is not None and (
            config_result_indices is not None or config_result_file_indices is not None
        ):
            raise ValueError(
                "Use only one CONFIG patch-selection mode: visualize_indices, "
                "visualize_result_indices, or visualize_result_files"
            )

        if config_patch_indices is not None:
            cfg["visualize_indices"] = config_patch_indices
        elif config_result_indices is not None:
            result_indices = config_result_indices
            summary_path = cfg.get("visualize_result_summary")
        elif config_result_file_indices is not None:
            result_indices = config_result_file_indices
            result_files = cfg.get("visualize_result_files")
            summary_path = cfg.get("visualize_result_summary")

    if result_indices is not None:
        if summary_path is None:
            if result_files:
                summary_path = _default_summary_from_result_files(result_files)
            else:
                raise ValueError(
                    "result summary is required when using result indices. Set "
                    "VISUALIZE_RESULT_SUMMARY in CONFIG or pass --result-summary."
                )
        patch_indices, extra = _patch_indices_from_result_summary(summary_path, result_indices)
        cfg["visualize_indices"] = patch_indices
        cfg["_visualize_index_extra"] = extra
        cfg["requested_result_indices"] = [int(idx) for idx in result_indices]
        cfg["source_diagnostic_summary"] = str(_repo_path(summary_path))
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(apply_overrides(CONFIG, args))


if __name__ == "__main__":
    main(sys.argv[1:])
