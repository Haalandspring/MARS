"""Inference-time input cleanup for MARS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def replace_continuous_negative_segments(
    x: torch.Tensor,
    cfg: dict,
    *,
    label: str = "negative-segment-preprocess",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace long negative-only channel segments with local Gaussian noise.

    The input is expected to be an NN-input tensor with shape ``(..., time)``.
    Only sustained negative portions are replaced; positive samples inside the
    same channel are preserved. Replacement noise is generated in the same
    domain as ``x``.
    """

    if not bool(cfg.get("negative_segment_preprocess_enabled", False)):
        return x, {"enabled": 0.0, "replaced_fraction": 0.0}

    if x.numel() == 0 or x.ndim < 2:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}

    time_len = int(x.shape[-1])
    min_len = int(cfg.get("negative_segment_min_len", 96))
    if min_len <= 1 or time_len < min_len:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}
    if min_len % 2 == 0:
        min_len += 1

    low_threshold = float(cfg.get("negative_segment_low_threshold", 0.0))
    low_fraction = float(cfg.get("negative_segment_low_fraction", 0.90))
    replace_ceiling = float(cfg.get("negative_segment_replace_ceiling", 0.0))
    context_abs_max = float(cfg.get("negative_segment_context_abs_max", 0.35))
    min_context_fraction = float(cfg.get("negative_segment_min_context_fraction", 0.20))
    std_floor = float(cfg.get("negative_segment_std_floor", 0.015))
    std_scale = float(cfg.get("negative_segment_std_scale", 1.0))
    clip = float(cfg.get("negative_segment_clip", cfg.get("input_clip", 0.999)))

    orig_shape = x.shape
    flat = x.reshape(-1, time_len)
    flat_f = flat.float()

    low = (flat_f < low_threshold).to(dtype=flat_f.dtype).unsqueeze(1)
    kernel = torch.ones((1, 1, min_len), device=x.device, dtype=flat_f.dtype)
    frac = F.conv1d(low, kernel, padding=min_len // 2) / float(min_len)
    centers = frac >= low_fraction
    support = F.max_pool1d(
        centers.to(dtype=flat_f.dtype),
        kernel_size=min_len,
        stride=1,
        padding=min_len // 2,
    ).squeeze(1) > 0
    replace = support & (flat_f < replace_ceiling)
    if not bool(replace.any()):
        return x, {
            "enabled": 1.0,
            "replaced_fraction": 0.0,
            "rows_touched": 0.0,
            "n_rows": float(flat_f.shape[0]),
            "min_len": float(min_len),
            "low_threshold": low_threshold,
            "label": label,
        }

    context = (~replace) & torch.isfinite(flat_f) & (flat_f.abs() <= context_abs_max)
    context_count = context.sum(dim=1, keepdim=True)
    min_context = max(4, int(round(time_len * min_context_fraction)))

    if bool(context.any()):
        global_vals = flat_f[context]
    else:
        global_vals = flat_f[torch.isfinite(flat_f)]
    if global_vals.numel() == 0:
        global_mean = torch.zeros((), device=x.device, dtype=flat_f.dtype)
        global_std = torch.full((), std_floor, device=x.device, dtype=flat_f.dtype)
    else:
        global_mean = global_vals.mean()
        global_std = global_vals.std(unbiased=False).clamp_min(std_floor)

    denom = context_count.clamp_min(1)
    row_mean = (flat_f * context.to(dtype=flat_f.dtype)).sum(dim=1, keepdim=True) / denom
    row_var = (
        ((flat_f - row_mean) ** 2) * context.to(dtype=flat_f.dtype)
    ).sum(dim=1, keepdim=True) / denom
    row_std = row_var.sqrt().clamp_min(std_floor) * std_scale

    enough_context = context_count >= min_context
    row_mean = torch.where(enough_context, row_mean, global_mean.expand_as(row_mean))
    row_std = torch.where(enough_context, row_std, global_std.expand_as(row_std) * std_scale)

    noise = torch.randn_like(flat_f) * row_std + row_mean
    noise = noise.clamp(-clip, clip)
    out = flat_f.clone()
    out[replace] = noise[replace]

    replaced_fraction = float(replace.sum().detach().cpu().item()) / float(replace.numel())
    rows_touched = float(replace.any(dim=1).sum().detach().cpu().item())
    stats = {
        "enabled": 1.0,
        "replaced_fraction": replaced_fraction,
        "rows_touched": rows_touched,
        "n_rows": float(replace.shape[0]),
        "min_len": float(min_len),
        "low_threshold": low_threshold,
        "label": label,
    }
    return out.reshape(orig_shape).to(dtype=x.dtype), stats


def replace_continuous_negative_segments_per_segment(
    x: torch.Tensor,
    cfg: dict,
    *,
    segment_dim: int = 1,
    label: str = "negative-segment-preprocess",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run negative-segment cleanup one segment at a time.

    This preserves the per-channel/per-segment detector semantics while keeping
    the temporary conv/pool tensors bounded to one segment.
    """

    if not bool(cfg.get("negative_segment_preprocess_enabled", False)):
        return x, {"enabled": 0.0, "replaced_fraction": 0.0}

    if x.ndim < 3:
        return replace_continuous_negative_segments(x, cfg, label=label)

    segment_dim = segment_dim % x.ndim
    n_segments = int(x.shape[segment_dim])
    if n_segments <= 0:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}

    total_replaced = 0.0
    total_pixels = 0.0
    total_rows_touched = 0.0
    total_rows = 0.0
    enabled = 0.0
    min_len = None
    low_threshold = None

    for segment_idx in range(n_segments):
        seg = x.select(segment_dim, segment_idx)
        cleaned, stats = replace_continuous_negative_segments(
            seg,
            cfg,
            label=f"{label}-{segment_idx}",
        )
        if cleaned.data_ptr() != seg.data_ptr():
            seg.copy_(cleaned.to(dtype=seg.dtype))

        n_pixels = float(seg.numel())
        n_rows = float(seg.numel() // int(seg.shape[-1]))
        total_pixels += n_pixels
        total_rows += n_rows
        total_replaced += float(stats.get("replaced_fraction", 0.0)) * n_pixels
        total_rows_touched += float(stats.get("rows_touched", 0.0))
        enabled = max(enabled, float(stats.get("enabled", 0.0)))
        min_len = stats.get("min_len", min_len)
        low_threshold = stats.get("low_threshold", low_threshold)

    replaced_fraction = total_replaced / total_pixels if total_pixels else 0.0
    out_stats = {
        "enabled": enabled,
        "replaced_fraction": replaced_fraction,
        "rows_touched": total_rows_touched,
        "n_rows": total_rows,
        "label": label,
    }
    if min_len is not None:
        out_stats["min_len"] = float(min_len)
    if low_threshold is not None:
        out_stats["low_threshold"] = float(low_threshold)
    return x, out_stats


def replace_negative_blocks_2d(
    x: torch.Tensor,
    cfg: dict,
    *,
    label: str = "negative-block-preprocess",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace locally coherent negative 2D blocks with local Gaussian noise.

    This is intentionally different from clipping all negative samples.  Normal
    z-domain noise is expected to contain many negative pixels, so this detector
    only fires when a local frequency-time window is mostly below the configured
    threshold.
    """

    if not bool(cfg.get("negative_block_preprocess_enabled", False)):
        return x, {"enabled": 0.0, "replaced_fraction": 0.0}

    if x.numel() == 0 or x.ndim != 2:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}

    h, w = int(x.shape[0]), int(x.shape[1])
    freq_window = int(cfg.get("negative_block_freq_window", 9))
    time_window = int(cfg.get("negative_block_time_window", 33))
    if freq_window <= 1 and time_window <= 1:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}
    freq_window = max(1, min(freq_window + (freq_window % 2 == 0), h))
    time_window = max(1, min(time_window + (time_window % 2 == 0), w))
    if freq_window <= 1 and time_window <= 1:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}

    low_threshold = float(cfg.get("negative_block_low_threshold", 0.0))
    low_fraction = float(cfg.get("negative_block_low_fraction", 0.75))
    replace_ceiling = float(cfg.get("negative_block_replace_ceiling", 0.0))
    context_abs_max = float(cfg.get("negative_block_context_abs_max", 0.35))
    min_context_fraction = float(cfg.get("negative_block_min_context_fraction", 0.20))
    std_floor = float(cfg.get("negative_block_std_floor", 0.015))
    std_scale = float(cfg.get("negative_block_std_scale", 1.0))
    clip = float(cfg.get("negative_block_clip", cfg.get("input_clip", 0.999)))

    x_f = x.float()
    low = (x_f < low_threshold).to(dtype=x_f.dtype).view(1, 1, h, w)
    frac = F.avg_pool2d(
        low,
        kernel_size=(freq_window, time_window),
        stride=1,
        padding=(freq_window // 2, time_window // 2),
        count_include_pad=False,
    )
    centers = frac >= low_fraction
    support = F.max_pool2d(
        centers.to(dtype=x_f.dtype),
        kernel_size=(freq_window, time_window),
        stride=1,
        padding=(freq_window // 2, time_window // 2),
    ).view(h, w) > 0
    replace = support & (x_f < replace_ceiling)

    if not bool(replace.any()):
        return x, {
            "enabled": 1.0,
            "replaced_fraction": 0.0,
            "rows_touched": 0.0,
            "n_rows": float(h),
            "windows": float(centers.sum().detach().cpu().item()),
            "freq_window": float(freq_window),
            "time_window": float(time_window),
            "low_threshold": low_threshold,
            "label": label,
        }

    context = (~replace) & torch.isfinite(x_f) & (x_f.abs() <= context_abs_max)
    context_count = context.sum(dim=1, keepdim=True)
    min_context = max(4, int(round(w * min_context_fraction)))

    if bool(context.any()):
        global_vals = x_f[context]
    else:
        global_vals = x_f[torch.isfinite(x_f)]
    if global_vals.numel() == 0:
        global_mean = torch.zeros((), device=x.device, dtype=x_f.dtype)
        global_std = torch.full((), std_floor, device=x.device, dtype=x_f.dtype)
    else:
        global_mean = global_vals.mean()
        global_std = global_vals.std(unbiased=False).clamp_min(std_floor)

    denom = context_count.clamp_min(1)
    row_mean = (x_f * context.to(dtype=x_f.dtype)).sum(dim=1, keepdim=True) / denom
    row_var = (
        ((x_f - row_mean) ** 2) * context.to(dtype=x_f.dtype)
    ).sum(dim=1, keepdim=True) / denom
    row_std = row_var.sqrt().clamp_min(std_floor) * std_scale

    enough_context = context_count >= min_context
    row_mean = torch.where(enough_context, row_mean, global_mean.expand_as(row_mean))
    row_std = torch.where(enough_context, row_std, global_std.expand_as(row_std) * std_scale)

    noise = torch.randn_like(x_f) * row_std + row_mean
    noise = noise.clamp(-clip, clip)
    out = x_f.clone()
    out[replace] = noise[replace]

    replaced_fraction = float(replace.sum().detach().cpu().item()) / float(replace.numel())
    rows_touched = float(replace.any(dim=1).sum().detach().cpu().item())
    stats = {
        "enabled": 1.0,
        "replaced_fraction": replaced_fraction,
        "rows_touched": rows_touched,
        "n_rows": float(h),
        "windows": float(centers.sum().detach().cpu().item()),
        "freq_window": float(freq_window),
        "time_window": float(time_window),
        "low_threshold": low_threshold,
        "label": label,
    }
    return out.to(dtype=x.dtype), stats


def replace_negative_blocks_2d_per_segment(
    x: torch.Tensor,
    cfg: dict,
    *,
    segment_dim: int = 1,
    label: str = "negative-block-preprocess",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run 2D negative-block cleanup one segment at a time."""

    if not bool(cfg.get("negative_block_preprocess_enabled", False)):
        return x, {"enabled": 0.0, "replaced_fraction": 0.0}

    if x.ndim < 3:
        return replace_negative_blocks_2d(x, cfg, label=label)

    segment_dim = segment_dim % x.ndim
    n_segments = int(x.shape[segment_dim])
    if n_segments <= 0:
        return x, {"enabled": 1.0, "replaced_fraction": 0.0}

    total_replaced = 0.0
    total_pixels = 0.0
    total_rows_touched = 0.0
    total_rows = 0.0
    total_windows = 0.0
    enabled = 0.0
    freq_window = None
    time_window = None
    low_threshold = None

    for segment_idx in range(n_segments):
        seg = x.select(segment_dim, segment_idx)
        cleaned, stats = replace_negative_blocks_2d(
            seg,
            cfg,
            label=f"{label}-{segment_idx}",
        )
        if cleaned.data_ptr() != seg.data_ptr():
            seg.copy_(cleaned.to(dtype=seg.dtype))

        n_pixels = float(seg.numel())
        n_rows = float(seg.shape[0])
        total_pixels += n_pixels
        total_rows += n_rows
        total_replaced += float(stats.get("replaced_fraction", 0.0)) * n_pixels
        total_rows_touched += float(stats.get("rows_touched", 0.0))
        total_windows += float(stats.get("windows", 0.0))
        enabled = max(enabled, float(stats.get("enabled", 0.0)))
        freq_window = stats.get("freq_window", freq_window)
        time_window = stats.get("time_window", time_window)
        low_threshold = stats.get("low_threshold", low_threshold)

    replaced_fraction = total_replaced / total_pixels if total_pixels else 0.0
    out_stats = {
        "enabled": enabled,
        "replaced_fraction": replaced_fraction,
        "rows_touched": total_rows_touched,
        "n_rows": total_rows,
        "windows": total_windows,
        "label": label,
    }
    if freq_window is not None:
        out_stats["freq_window"] = float(freq_window)
    if time_window is not None:
        out_stats["time_window"] = float(time_window)
    if low_threshold is not None:
        out_stats["low_threshold"] = float(low_threshold)
    return x, out_stats
