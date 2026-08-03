"""
End-to-end MARS GPU RFI mitigation pipeline
===========================================

This version keeps the pieces that are currently justified by experiments:
  - segment-level raw-domain persistent RFI gate before normalisation
  - zero replacement for combined RFI/blanked mask pixels
  - exact output-mean fill (128) on masked pixels after rescale
  - filtool-style block rescale
  - optional post-replacement standard zdot

Single-script pipeline that runs entirely on GPU:

  1. Load raw .fil -> GPU
  2. Load NN model / TensorRT engine
  3. Segment-wise compute: normalize -> patches -> mask
  4. Baseline removal after replacement
  5. Rescale to uint8 (GPU)
  6. Write output .fil

Data flow:
  raw --(median/MAD + tanh)--> patches -> NN -> mask patches
      `--(mean/std science branch)--> science_data
  science_data + mask --> replacement --> baseline --> rescale --> uint8 --> .fil

Memory: the current implementation reads the full observation and keeps its
        working arrays GPU-resident. Segmenting controls the model geometry; it
        is not yet out-of-core streaming. See docs/limitations.md.
"""

import contextlib
import math
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mars-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mars-numba"))

import torch
import torch.nn.functional as F
import numpy as np
from sigpyproc.readers import FilReader

from .config import CONFIG as PAPER_TRAINING_CONFIG
from .preprocess import (
    replace_negative_blocks_2d,
    replace_negative_blocks_2d_per_segment,
    replace_continuous_negative_segments,
    replace_continuous_negative_segments_per_segment,
)
from .provenance import (
    model_config_mismatches,
    validate_engine_metadata,
    validate_training_identity,
)

# =============================================================================
# CONFIG
# =============================================================================

MINIMUM_SUPPORTED_CHANNELS = 512
DETERMINISTIC_MEDIAN_SORT_MAX_ELEMENTS = 8 * 1024 * 1024


def validate_supported_channel_count(nchans: int, cfg: dict) -> int:
    """Enforce the curated C>=512 scope at every public filterbank entry point."""

    minimum_channels = int(
        cfg.get("minimum_supported_channels", MINIMUM_SUPPORTED_CHANNELS)
    )
    if minimum_channels < MINIMUM_SUPPORTED_CHANNELS:
        raise ValueError(
            "minimum_supported_channels cannot be lowered below the curated "
            f"runtime boundary C={MINIMUM_SUPPORTED_CHANNELS}"
        )
    if int(nchans) < minimum_channels:
        raise NotImplementedError(
            f"This MARS runtime profile supports C >= {minimum_channels} channels; "
            f"the input contains C={int(nchans)}. The C<512 paper branch is "
            "intentionally outside the curated implementation scope."
        )
    return minimum_channels

CONFIG = {
    # Artifact identity: prevents a topology-identical loss ablation from being
    # accepted as the selected paper checkpoint/engine.
    "experiment_id": PAPER_TRAINING_CONFIG["experiment_id"],
    "artifact_role": PAPER_TRAINING_CONFIG["artifact_role"],
    "training_fingerprint": PAPER_TRAINING_CONFIG["training_fingerprint"],
    "allow_unverified_artifacts": False,
    "tensorrt_verification_max_diff": 0.02,

    # --- I/O ---
    "input_fil":    None,
    "output_fil":   None,
    # Mask outputs are disabled by default so normal science runs write only
    # output_fil. Set write_mask_files=True to also write diagnostic masks.
    #   output_mask_fil         = combined mask used for replacement
    #   output_nn_mask_fil      = pure NN prediction only
    #   output_blanked_mask_fil = unusable / blanked / fully-flagged channels
    # Set a mask path to None to derive it from output_fil when enabled.
    "write_mask_files": False,
    "output_mask_fil": None,
    "output_nn_mask_fil": None,
    "output_blanked_mask_fil": None,
    "checkpoint":   "artifacts/checkpoints/mars-paper/best_f1.pt",

    # Optional stage-by-stage diagnostic filterbanks. Set to None to disable.
    # These let you run the same acceleration search after each major
    # mitigation stage and identify where the pulsar fundamental is damaged.
    "diagnostic_fil_prefix": None,
    "diagnostic_global_normalisation": True,
    "diagnostic_channel_chunk": 64,
    "diagnostic_time_chunk": 4096,
    # Runtime statistics print scalar GPU values via .item(). Keep this enabled
    # for interactive experiments; production wrapper disables it to avoid
    # non-I/O CPU/GPU synchronization.
    "runtime_stats": True,

    # --- Science branch zero-DM matched filter ---
    # PulsarX/filtool `zdot` fits and removes the per-channel component that is
    # correlated with the zero-DM profile. This is different from subtracting
    # the same zero-DM time series from every channel.
    "science_zdot": False,  # master switch for the zdot experiment
    "science_zdot_stage": "post_replacement",  # pre_replacement | post_replacement | off
    "science_zdot_mode": "standard",  # standard | safe_gate
    # Soft-zdot strength. 0.0 is no zdot; 1.0 matches the current full
    # PulsarX/filtool-style subtraction. Intermediate values are used to test
    # whether full zero-DM projection damages the pulsar fundamental.
    "science_zdot_strength": 1.0,
    "science_zdot_profile_live_only": True,
    "science_zdot_apply_live_only": True,
    "science_zdot_channel_chunk": 64,
    # Conservative common-mode gate. This is an experimental test bed for the
    # ML-gate idea and is intentionally not enabled by default.
    "science_zdot_safe_width_samples": 513,
    "science_zdot_safe_gate_sigma": 1.25,
    "science_zdot_safe_gate_softness": 0.35,
    "science_zdot_safe_profile": "lowpass",  # raw | lowpass
    # Optional standalone diagnostics for the zdot experiment. These do not
    # enable the full diagnostic_fil_prefix set.
    "science_zdot_basis_fil": None,
    "science_zdot_after_fil": None,

    # --- Model ---
    "model":         "trt_shape_unet",
    "in_channels":   1,
    "out_channels":  1,
    "channels":      (8, 16, 32, 64),
    "output_bias_prior": 0.0,
    "axis_reduction": 4,
    "anisotropic_kernel": 9,
    "shape_kernel": 9,
    "decoder_horizontal_refine_enabled": True,
    "decoder_horizontal_refine_stages": ("up2", "up1", "up0"),
    "decoder_horizontal_refine_kernel": 31,
    "decoder_vertical_refine_enabled": True,
    "decoder_vertical_refine_stages": ("up2", "up1", "up0"),
    "decoder_vertical_refine_kernel": 31,
    "expected_parameters": 270_769,
    "batch_size":    32,
    "use_amp":       True,
    "threshold":     0.5,
    # Keep the library default on the historical throughput-oriented path.
    # Reproducibility profiles opt into strict, cross-process deterministic
    # PyTorch inference explicitly; the article timing profile opts out.
    "deterministic_inference": False,
    "inference_seed": 1234,

    # Optional hysteresis post-processing for the NN mask. Seed from confident
    # pixels, then grow through connected lower-probability support. This helps
    # complete partially detected horizontal RFI ridges without lowering the
    # global threshold everywhere.
    # Disabled for every main-text result in the paper. Appendix experiments
    # can enable it explicitly through the CLI or a JSON config.
    "hys_enabled": False,
    "hys_seed_threshold": None,
    "hys_support_threshold": 0.35,
    "hys_connect_time_radius": 5,
    "hys_connect_freq_radius": 1,
    "hys_connect_iterations": 1,
    "hys_close_time_radius": 3,
    "hys_dilate_freq_radius": 0,
    "hys_dilate_time_radius": 1,
    "hys_row_min_support_fraction": 0.01,
    "hys_max_patch_mask_fraction": 0.15,
    "hys_max_added_fraction": 0.02,
    # Apply HYS in patch chunks to avoid CUDA int32 indexing overflow in
    # max_pool2d when an observation produces thousands of 512x512 patches.
    "hys_chunk_size": 64,

    # --- TensorRT (set tensorrt_path only after compiling this exact model) ---
    "tensorrt_path": None,

    # --- Normalisation ---
    "target_segment_seconds": 2.0,
    "patch_size":       512,
    # C<512 is intentionally outside the supported scope of this curated
    # runtime. Helper-level packing code is retained for development only.
    "minimum_supported_channels": MINIMUM_SUPPORTED_CHANNELS,
    "sat_sigma":        6.0,
    "sat_ratio_segment": 0.5,
    "mad_const":        1.4826,
    "mad_min_valid":    0.5,
    "tanh_scale":       6.0,
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
    # How channels marked unusable before NN inference are represented in the
    # NN input. `zero` is the old behavior. `interp` fills those rows from
    # neighboring live channels in frequency. `gaussian` fills them with
    # z-domain N(0, std) noise before tanh compression. Final replacement still
    # uses the original blank mask.
    "nn_input_blank_mode": "interp",  # zero | interp | gaussian
    "nn_input_blank_noise_std": 1.0,
    # Raw filterbank samples are copied to GPU with this dtype. uint8 minimizes
    # the persistent raw-data footprint, but normalization still converts raw
    # slices to raw_compute_dtype because median/std/mean need floating point.
    "raw_gpu_dtype":    "float16",  # uint8 | float16 | float32
    "raw_compute_dtype": "float16",  # float16 | float32

    # --- Persistent RFI detection ---
    # Raw-domain segment detector for persistent bright RFI that would be
    # flattened by per-segment normalisation. This is intentionally not SKF:
    # it uses segment-local median/spread/bright-occupancy features plus
    # frequency-density expansion, and never full-observation skew/kurtosis.
    "raw_segment_detector_enabled": True,
    "raw_segment_detector_mode": "guarded",  # guarded | legacy
    "raw_segment_detector_max_live_fraction": 0.5,
    "raw_segment_detector_high_sigma": 5.0,
    "raw_segment_detector_low_sigma": 2.5,
    "raw_segment_detector_seed_occupancy": 0.10,
    "raw_segment_detector_support_occupancy": 0.02,
    "raw_segment_detector_density_window_chans": 49,
    "raw_segment_detector_density_threshold": 0.20,
    "raw_segment_detector_support_density_threshold": 0.35,
    "raw_segment_detector_dilate_chans": 4,
    # --- Post-processing ---
    "segment_channel_flag_enabled": True,
    "segment_channel_flag_ratio": 0.5,
    "science_prebaseline_enabled": False,
    "final_baseline_enabled": True,
    "baseline_width":     1.0,   # seconds
    "baseline_channel_chunk": 256,
    # RFI replacement in the science branch. `zero` matches filtool's default
    # z-domain mean fill after equalization; `clean_median` is the previous
    # per-channel clean-median constant replacement.
    # Set to False for ablations where the NN mask is diagnosed but not used
    # for replacement. Segment/raw persistent blanking is still applied.
    "use_nn_mask_for_replacement": True,
    # When False, historical ablations still load/run the NN even if its mask is
    # not used. Enable this for true non-NN science-branch tests.
    "skip_nn_when_unused": False,
    "force_run_nn_inference": False,
    "replacement_fill_mode": "zero",  # zero | clean_median
    # With block/global rescale, z=0 is close to but not guaranteed to become
    # exactly out_mean. Keep masked output pixels exactly at 128 for PRESTO.
    "replacement_force_output_mean": True,
    # filtool rescales each processed block globally to the requested output
    # mean/std. Keep the old per-channel robust rescale available for A/B tests.
    "rescale_mode":        "filtool_block",  # filtool_block | filtool_global | per_channel_mad
    "out_mean":           128.0,
    "out_std":            6.0,
    # Optional whole-channel flags supplied by an upstream instrument or
    # observing log. Segment-local dead/saturated/persistent flags are added
    # independently during preprocessing.
    "preflag_channels": (),
    # torch.cuda.empty_cache() is useful for debugging OOM, but it is a global
    # allocator sync point and makes speed tests look slower than the pipeline
    # work itself. Keep it off unless memory pressure requires it.
    "empty_cache_between_stages": False,
}


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cuda_cache(cfg=None):
    enabled = False
    if isinstance(cfg, dict):
        enabled = bool(cfg.get("empty_cache_between_stages", False))
    elif cfg is not None:
        enabled = bool(cfg)
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _deterministic_median_values(values, *, dim, keepdim=False):
    """Compute lower medians with bounded-memory stable value sorting."""

    normalised_dim = int(dim) % values.ndim
    width = int(values.shape[normalised_dim])
    if width < 1:
        raise IndexError("median cannot be computed over an empty dimension")
    midpoint = (width - 1) // 2
    moved = values.movedim(normalised_dim, -1)
    rows = moved.reshape(-1, width)
    result_flat = torch.empty(
        rows.shape[0], dtype=values.dtype, device=values.device
    )
    rows_per_chunk = max(
        1, DETERMINISTIC_MEDIAN_SORT_MAX_ELEMENTS // width
    )
    for start in range(0, rows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, rows.shape[0])
        sorted_chunk = rows[start:stop].sort(dim=1, stable=True)
        result_flat[start:stop] = sorted_chunk.values[:, midpoint]
        del sorted_chunk
    result = result_flat.reshape(moved.shape[:-1])
    return result.unsqueeze(normalised_dim) if keepdim else result


def _median_values(values, *, dim=None, keepdim=False):
    """Return median values without CUDA's nondeterministic index output."""

    if dim is None:
        return values.median()
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        return _deterministic_median_values(values, dim=dim, keepdim=keepdim)
    return values.median(dim=dim, keepdim=keepdim).values


def resolve_inference_runtime_settings(cfg):
    """Return the explicit backend settings selected for one pipeline run."""

    deterministic = bool(cfg.get("deterministic_inference", False))
    seed = int(cfg.get("inference_seed", 1234))
    if not 0 <= seed < 2**63:
        raise ValueError("inference_seed must be in the range [0, 2**63)")
    tensorrt = bool(cfg.get("tensorrt_path"))
    return {
        "deterministic_inference": deterministic,
        "inference_seed": seed,
        "cudnn_benchmark": not deterministic,
        "cudnn_deterministic": deterministic,
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "use_amp": bool(cfg.get("use_amp", True)),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "backend": "tensorrt" if tensorrt else "pytorch",
        "determinism_scope": (
            "torch-pre-and-postprocessing-only"
            if tensorrt
            else "pytorch-end-to-end"
        ),
        "tensorrt_determinism_guaranteed": False if tensorrt else None,
    }


@contextlib.contextmanager
def inference_runtime_settings(cfg):
    """Apply per-run inference settings and restore process-global flags."""

    settings = resolve_inference_runtime_settings(cfg)
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cpu_rng = torch.random.get_rng_state()
    previous_cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    try:
        torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
        torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
        torch.manual_seed(settings["inference_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings["inference_seed"])
        print(
            "Inference runtime: "
            f"backend={settings['backend']}, "
            f"deterministic={settings['deterministic_inference']}, "
            f"seed={settings['inference_seed']}, "
            f"cudnn_benchmark={settings['cudnn_benchmark']}, "
            f"cudnn_deterministic={settings['cudnn_deterministic']}, "
            f"deterministic_algorithms={settings['deterministic_algorithms']}, "
            f"amp={settings['use_amp']}, "
            f"cudnn_allow_tf32={settings['cudnn_allow_tf32']}, "
            f"matmul_allow_tf32={settings['matmul_allow_tf32']}, "
            "cublas_workspace_config="
            f"{settings['cublas_workspace_config'] or 'unset'}"
        )
        if (
            settings["deterministic_inference"]
            and settings["cublas_workspace_config"] is None
        ):
            print(
                "  WARNING: set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching "
                "Python if this configuration adds a cuBLAS-backed operation."
            )
        if settings["backend"] == "tensorrt" and settings["deterministic_inference"]:
            print(
                "  WARNING: deterministic_inference controls PyTorch pre/post-processing "
                "but cannot guarantee determinism inside a TensorRT engine."
            )
        yield settings
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.random.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)


# =============================================================================
# GPU: raw-domain segment-level persistent RFI detector
# =============================================================================

def _positive_robust_score_by_segment(values, cfg):
    """
    Positive robust z-score over channels independently for each segment.
    values: (nchans, nsegments)
    """
    center = _median_values(values, dim=0)
    residual = values - center.unsqueeze(0)
    scale = _median_values(residual.abs(), dim=0) * cfg["mad_const"]
    score = residual / torch.clamp(scale.unsqueeze(0), min=1e-6)
    return torch.clamp(score, min=0.0)


def _rolling_channel_density(mask, window_chans):
    """
    Frequency-direction rolling density for a (nchans, nsegments) bool mask.
    Boundary channels use the actual number of covered channels as denominator.
    """
    nchans, _ = mask.shape
    window = max(1, int(window_chans))
    if window % 2 == 0:
        window += 1
    window = min(window, nchans if nchans % 2 == 1 else nchans - 1)
    window = max(1, window)

    x = mask.T.float().unsqueeze(1)  # (nsegments, 1, nchans)
    kernel = torch.ones(1, 1, window, dtype=x.dtype, device=x.device)
    padding = window // 2

    counts = F.conv1d(x, kernel, padding=padding)
    denom = F.conv1d(torch.ones_like(x), kernel, padding=padding)
    density = counts / torch.clamp(denom, min=1.0)
    return density.squeeze(1).T


def _rolling_channel_any(mask, radius_chans):
    radius = max(0, int(radius_chans))
    if radius == 0:
        return mask
    return _rolling_channel_density(mask, 2 * radius + 1) > 0.0


def detect_raw_segment_persistent_rfi_gpu(
    seg_ch_medians,
    seg_ch_spread,
    seg_bright_fraction,
    cfg,
    label="segments",
):
    """
    Segment-local detector for strong persistent/structured RFI before
    normalisation can suppress it.

    Inputs are (nchans, nsegments) feature tensors computed in raw units:
      - seg_ch_medians: per-channel segment median
      - seg_ch_spread:  per-channel segment MAD/std-like spread
      - seg_bright_fraction: fraction of samples above a robust bright threshold

    Returns a (nchans, nsegments) bool mask. This detector deliberately avoids
    full-observation skew/kurtosis so it is not just SKF with another threshold.
    """
    if not cfg.get("raw_segment_detector_enabled", False):
        return torch.zeros_like(seg_ch_medians, dtype=torch.bool)

    high_sigma = float(cfg["raw_segment_detector_high_sigma"])
    low_sigma = float(cfg["raw_segment_detector_low_sigma"])
    seed_occupancy = float(cfg["raw_segment_detector_seed_occupancy"])
    support_occupancy = float(cfg["raw_segment_detector_support_occupancy"])
    density_window = int(cfg["raw_segment_detector_density_window_chans"])
    density_threshold = float(cfg["raw_segment_detector_density_threshold"])
    support_density_threshold = float(cfg["raw_segment_detector_support_density_threshold"])
    dilate_chans = int(cfg["raw_segment_detector_dilate_chans"])

    median_score = _positive_robust_score_by_segment(seg_ch_medians, cfg)
    spread_score = _positive_robust_score_by_segment(
        torch.log(torch.clamp(seg_ch_spread, min=1e-6)),
        cfg,
    )

    high_seed = (
        (median_score > high_sigma)
        | (spread_score > high_sigma)
        | (seg_bright_fraction > seed_occupancy)
    )
    low_evidence = (
        (median_score > low_sigma)
        | (spread_score > low_sigma)
        | (seg_bright_fraction > support_occupancy)
    )

    high_density = _rolling_channel_density(high_seed, density_window)
    low_density = _rolling_channel_density(low_evidence, density_window)
    dense_core = high_density >= density_threshold
    dense_support = low_density >= support_density_threshold

    expanded_core = _rolling_channel_any(dense_core, dilate_chans)
    dense_band = dense_core | (expanded_core & dense_support)
    near_band = _rolling_channel_any(dense_band, dilate_chans)

    flags = high_seed | dense_band | (near_band & low_evidence)

    n_segments = flags.shape[1]
    if cfg.get("runtime_stats", True) and n_segments > 0:
        n_flags = int(flags.sum())
        n_high_seed = int(high_seed.sum())
        n_dense_band = int(dense_band.sum())
        per_seg = flags.sum(dim=0).float()
        seg_with_flags = int((per_seg > 0).sum())
        print(
            f"  Raw segment detector ({label}): {n_flags} channel-segments, "
            f"high_seed={n_high_seed}, dense_band={n_dense_band}, "
            f"segments={seg_with_flags}/{n_segments}, "
            f"median/max per segment={per_seg.median().item():.0f}/{per_seg.max().item():.0f}"
        )

    return flags


def _guard_raw_detector_flags(raw_flags, dead_channels, cfg, label):
    """
    Suppress raw-detector output for segments where it tries to blank an
    implausibly large fraction of live channels.

    raw_flags/dead_channels: (nchans, nsegments) bool tensors.
    """
    max_live_fraction = cfg.get("raw_segment_detector_max_live_fraction", None)
    if max_live_fraction is None:
        return raw_flags

    max_live_fraction = float(max_live_fraction)
    if max_live_fraction <= 0.0 or max_live_fraction >= 1.0:
        return raw_flags & ~dead_channels

    live = ~dead_channels
    guarded = raw_flags & live
    live_count = live.sum(dim=0).clamp(min=1)
    flagged_count = guarded.sum(dim=0)
    suppress = flagged_count.float() > (max_live_fraction * live_count.float())
    if suppress.any():
        guarded[:, suppress] = False
        if cfg.get("runtime_stats", True):
            n_suppressed = int(suppress.sum().item())
            n_segments = raw_flags.shape[1]
            print(
                f"  Raw detector guard ({label}): suppressed "
                f"{n_suppressed}/{n_segments} segments "
                f"(max_live_fraction={max_live_fraction:.2f})"
            )
    return guarded


def _apply_raw_detector_mode(raw_flags, dead_channels, cfg, label):
    mode = str(cfg.get("raw_segment_detector_mode", "guarded")).lower()
    if mode in ("legacy", "old", "unprotected"):
        return raw_flags
    if mode in ("guarded", "safe"):
        return _guard_raw_detector_flags(raw_flags, dead_channels, cfg, label)
    raise ValueError(
        "raw_segment_detector_mode must be 'guarded' or 'legacy' "
        f"(got {mode!r})"
    )


def apply_nn_input_blank_mode(z_segs, degenerate, cfg):
    """
    Represent pre-NN unusable channels without changing final replacement masks.

    z_segs: (nchans, nsegments, seg_len)
    degenerate: (nchans, nsegments) bool mask that should not be trusted as
        actual signal-bearing input.

    `zero` keeps the historical behavior. `gaussian` fills degenerate rows
    with z-domain N(0, nn_input_blank_noise_std) noise before tanh compression.
    `interp` fills degenerate rows from nearest live frequency neighbors within
    the same segment. Edge rows fall back to the nearest available live channel;
    all-degenerate segments become zero.
    """
    mode = str(cfg.get("nn_input_blank_mode", "zero")).lower()
    if mode in ("zero", "zeros", "blank"):
        z_segs[degenerate, :] = 0.0
        return z_segs
    if mode in ("gaussian", "gauss", "noise", "random"):
        target = z_segs[degenerate, :]
        if target.numel() > 0:
            std = float(cfg.get("nn_input_blank_noise_std", 1.0))
            noise = torch.randn(
                target.shape,
                device=z_segs.device,
                dtype=torch.float32,
            ) * std
            z_segs[degenerate, :] = noise.to(dtype=z_segs.dtype)
        return z_segs
    if mode not in ("interp", "interpolate", "linear_interp"):
        raise ValueError(
            "nn_input_blank_mode must be one of: zero, interp, gaussian "
            f"(got {mode!r})"
        )

    nchans, nsegments, _ = z_segs.shape
    chan_idx = torch.arange(nchans, device=z_segs.device)
    for seg_idx in range(nsegments):
        bad = degenerate[:, seg_idx]
        if not bool(bad.any()):
            continue
        live = ~bad
        if not bool(live.any()):
            z_segs[:, seg_idx, :] = 0.0
            continue

        prev_idx = torch.where(live, chan_idx, torch.full_like(chan_idx, -1))
        prev_idx = torch.cummax(prev_idx, dim=0).values
        next_idx = torch.where(live, chan_idx, torch.full_like(chan_idx, nchans))
        next_idx = torch.cummin(torch.flip(next_idx, dims=(0,)), dim=0).values
        next_idx = torch.flip(next_idx, dims=(0,))

        has_prev = prev_idx >= 0
        has_next = next_idx < nchans
        left_idx = prev_idx.clamp(0, nchans - 1)
        right_idx = next_idx.clamp(0, nchans - 1)

        seg = z_segs[:, seg_idx, :]
        left = seg.index_select(0, left_idx)
        right = seg.index_select(0, right_idx)

        denom = (right_idx - left_idx).clamp(min=1).to(torch.float32)
        weight = ((chan_idx - left_idx).to(torch.float32) / denom).to(seg.dtype)
        weight = weight.unsqueeze(1)

        both = (has_prev & has_next).unsqueeze(1)
        only_prev = (has_prev & ~has_next).unsqueeze(1)
        only_next = (~has_prev & has_next).unsqueeze(1)
        fill = torch.where(
            both,
            left * (1.0 - weight) + right * weight,
            torch.where(only_prev, left, torch.where(only_next, right, torch.zeros_like(seg))),
        )
        seg[bad, :] = fill[bad, :]

    return z_segs


# =============================================================================
# GPU: split segment into 512x512 patches
# =============================================================================

def split_segment_to_patches(seg, patch_size=512):
    """
    Split (nchans, seg_len) into (N, 1, 512, 512) patches for NN.
    Handles non-512-divisible nchans via overlap.
    Returns: patches (N, 1, 512, 512), metadata for reconstruction
    """
    nchans, seg_len = seg.shape
    if nchans <= 0:
        raise ValueError("A segment must contain at least one frequency channel")
    n_time_patches = seg_len // patch_size
    full_chan_blocks = nchans // patch_size
    has_overlap = nchans % patch_size != 0
    overlap_chans = nchans - full_chan_blocks * patch_size

    # For C < patch_size, pack consecutive time blocks along the frequency
    # dimension and pad only unused rows. This is reversible and implements the
    # small-C path described in the paper.
    if nchans < patch_size:
        time_blocks_per_patch = max(1, patch_size // nchans)
        patch_count = (
            n_time_patches + time_blocks_per_patch - 1
        ) // time_blocks_per_patch
        packed = torch.zeros(
            patch_count,
            patch_size,
            patch_size,
            dtype=seg.dtype,
            device=seg.device,
        )
        for time_idx in range(n_time_patches):
            patch_idx, slot = divmod(time_idx, time_blocks_per_patch)
            row_start = slot * nchans
            time_start = time_idx * patch_size
            packed[
                patch_idx,
                row_start:row_start + nchans,
                :,
            ] = seg[:, time_start:time_start + patch_size]
        meta = {
            "nchans": nchans,
            "n_time_patches": n_time_patches,
            "full_chan_blocks": 0,
            "has_overlap": False,
            "overlap_chans": 0,
            "small_channel_packing": True,
            "time_blocks_per_patch": time_blocks_per_patch,
            "patch_count": patch_count,
        }
        return packed.unsqueeze(1), meta

    patches = []

    # Full channel blocks
    for i in range(full_chan_blocks):
        block = seg[i * patch_size:(i + 1) * patch_size, :n_time_patches * patch_size]
        # (512, n_time_patches * 512) -> (n_time_patches, 512, 512)
        block = block.reshape(patch_size, n_time_patches, patch_size).permute(1, 0, 2)
        patches.append(block)

    # Overlap block (last 512 channels)
    if has_overlap:
        block = seg[nchans - patch_size:nchans, :n_time_patches * patch_size]
        block = block.reshape(patch_size, n_time_patches, patch_size).permute(1, 0, 2)
        patches.append(block)

    if patches:
        patch_tensor = torch.cat(patches, dim=0).unsqueeze(1)
    else:
        patch_tensor = torch.empty(
            0, 1, patch_size, patch_size, dtype=seg.dtype, device=seg.device
        )

    meta = {
        "nchans": nchans,
        "n_time_patches": n_time_patches,
        "full_chan_blocks": full_chan_blocks,
        "has_overlap": has_overlap,
        "overlap_chans": overlap_chans,
        "small_channel_packing": False,
        "time_blocks_per_patch": 1,
        "patch_count": int(patch_tensor.shape[0]),
    }
    return patch_tensor, meta


# =============================================================================
# GPU: reconstruct mask from patches for one segment
# =============================================================================

def reconstruct_segment_mask(mask_patches, meta, patch_size=512):
    """
    Reconstruct (nchans, seg_time) mask from (N, 512, 512) patches.
    Inverse of split_segment_to_patches.
    """
    nchans = meta["nchans"]
    n_tp = meta["n_time_patches"]
    full_cb = meta["full_chan_blocks"]
    seg_time = n_tp * patch_size

    mask = torch.zeros(nchans, seg_time, dtype=torch.bool,
                       device=mask_patches.device)

    if meta.get("small_channel_packing", False):
        time_blocks_per_patch = int(meta["time_blocks_per_patch"])
        for time_idx in range(n_tp):
            patch_idx, slot = divmod(time_idx, time_blocks_per_patch)
            row_start = slot * nchans
            time_start = time_idx * patch_size
            mask[:, time_start:time_start + patch_size] = mask_patches[
                patch_idx,
                row_start:row_start + nchans,
                :,
            ]
        return mask

    idx = 0
    for i in range(full_cb):
        # (n_tp, 512, 512) -> (512, n_tp, 512) -> (512, seg_time)
        block = mask_patches[idx:idx + n_tp]  # (n_tp, 512, 512)
        block = block.permute(1, 0, 2).reshape(patch_size, seg_time)
        mask[i * patch_size:(i + 1) * patch_size, :] = block
        idx += n_tp

    if meta["has_overlap"]:
        block = mask_patches[idx:idx + n_tp]
        block = block.permute(1, 0, 2).reshape(patch_size, seg_time)
        # Only fill the non-overlapping part
        overlap_start = full_cb * patch_size
        mask[overlap_start:nchans, :] = block[patch_size - meta["overlap_chans"]:, :]

    return mask


# =============================================================================
# GPU: zero-DM matched filter (PulsarX/filtool-style zdot)
# =============================================================================

def _smooth_1d_reflect(x, width):
    """Moving-average smooth a 1D GPU tensor with reflect padding."""
    width = int(width)
    if width <= 1:
        return x
    if width % 2 == 0:
        width += 1
    pad = width // 2
    y = F.pad(x.view(1, 1, -1).float(), (pad, pad), mode="reflect")
    kernel = torch.ones(1, 1, width, dtype=torch.float32, device=x.device) / float(width)
    return F.conv1d(y, kernel).view(-1).to(x.dtype)


def build_safe_zdot_profile_gpu(s, cfg, label="zdot", verbose=True):
    """
    Build a conservative zero-DM profile for gated zdot.

    The gate is intentionally simple for this first test: it uses the local
    average absolute robust z-score of the zero-DM profile. Isolated/narrow
    periodic samples receive a small gate; broad common-mode excursions receive
    a larger gate. This tests the safe-gating surface before replacing it with
    a learned gate.
    """
    width = max(3, int(cfg.get("science_zdot_safe_width_samples", 513)))
    if width % 2 == 0:
        width += 1
    gate_sigma = float(cfg.get("science_zdot_safe_gate_sigma", 1.25))
    softness = max(1e-3, float(cfg.get("science_zdot_safe_gate_softness", 0.35)))
    profile_mode = cfg.get("science_zdot_safe_profile", "lowpass")

    s32 = s.float()
    med = s32.median()
    mad = (s32 - med).abs().median() * float(cfg.get("mad_const", 1.4826))
    mad = torch.clamp(mad, min=1e-6)
    absz = (s32 - med).abs() / mad
    local_absz = _smooth_1d_reflect(absz, width)
    gate = torch.sigmoid((local_absz - gate_sigma) / softness)
    gate = _smooth_1d_reflect(gate, width)

    if profile_mode == "lowpass":
        profile = _smooth_1d_reflect(s32 - med, width) * gate
    elif profile_mode == "raw":
        profile = (s32 - med) * gate
    else:
        raise ValueError(f"Unsupported science_zdot_safe_profile: {profile_mode}")

    if verbose:
        print(
            f"  {label}: safe gate width={width}, sigma={gate_sigma:.3g}, "
            f"softness={softness:.3g}, profile={profile_mode}, "
            f"gate mean={gate.mean().item():.3g}, "
            f"p95={torch.quantile(gate, 0.95).item():.3g}, "
            f"max={gate.max().item():.3g}"
        )
    return profile.to(s.dtype)


def apply_zdot_gpu(z_data, live_mask, cfg, label="zdot", verbose=True, return_basis=False):
    """
    Zero-DM matched filter, matching the core PulsarX/XLibs RFI::zdot logic.

    For each time bin, compute the zero-DM profile s(t), then for every channel
    fit x_ch(t) ~= alpha_ch * s(t) + beta_ch and subtract only that fitted
    correlated component. This is intentionally not the same as subtracting
    the same zero-DM time series from every channel.

    z_data: (nchans, ntime) float32 on GPU, modified in-place.
    live_mask: (nchans,) bool on GPU; fully flagged channels are left at 0.
    """
    nchans, ntime = z_data.shape
    device = z_data.device
    channel_chunk = max(1, int(cfg.get("science_zdot_channel_chunk", 64)))
    strength = float(cfg.get("science_zdot_strength", 1.0))

    if cfg.get("science_zdot_profile_live_only", True):
        profile_mask = live_mask
    else:
        profile_mask = torch.ones(nchans, dtype=torch.bool, device=device)

    if cfg.get("science_zdot_apply_live_only", True):
        apply_mask = live_mask
    else:
        apply_mask = torch.ones(nchans, dtype=torch.bool, device=device)

    profile_indices = torch.where(profile_mask)[0]
    apply_indices = torch.where(apply_mask)[0]
    n_profile = int(profile_indices.numel())
    n_apply = int(apply_indices.numel())

    if n_profile <= 0 or n_apply <= 0:
        if verbose:
            print(f"  {label}: skipped; no live channels")
        return

    mode = cfg.get("science_zdot_mode", "standard")
    if verbose:
        print(
            f"  {label}: profile channels={n_profile}/{nchans}, "
            f"corrected channels={n_apply}/{nchans}, chunk={channel_chunk}, "
            f"strength={strength:.3g}, mode={mode}"
        )

    if abs(strength) <= 1e-12:
        if verbose:
            print(f"  {label}: strength is 0; leaving data unchanged")
        z_data[~live_mask, :] = 0.0
        return

    s = torch.zeros(ntime, dtype=torch.float64, device=device)
    for c0 in range(0, n_profile, channel_chunk):
        idx = profile_indices[c0:c0 + channel_chunk]
        rows = z_data[idx].to(torch.float64)
        s += rows.sum(dim=0)
        del rows
    s /= float(n_profile)
    if mode == "safe_gate":
        s = build_safe_zdot_profile_gpu(s, cfg, label=label, verbose=verbose)
    elif mode != "standard":
        raise ValueError(f"Unsupported science_zdot_mode: {mode}")

    se = s.sum()
    ss = (s * s).sum()
    n = float(ntime)
    denom = se * se - ss * n
    if abs(denom.item()) <= 1e-10:
        if verbose:
            print(f"  {label}: skipped; zero-DM profile has near-zero variance")
        del s
        return

    alpha_min = None
    alpha_max = None
    beta_abs_max = None
    basis_out = torch.zeros_like(z_data) if return_basis else None
    for c0 in range(0, n_apply, channel_chunk):
        idx = apply_indices[c0:c0 + channel_chunk]
        rows = z_data[idx].to(torch.float64)

        xe = rows.sum(dim=1)
        xs = (rows * s.unsqueeze(0)).sum(dim=1)
        alpha = (xe * se - xs * n) / denom
        beta = (xs * se - xe * ss) / denom

        basis = alpha.unsqueeze(1) * s.unsqueeze(0) + beta.unsqueeze(1)
        if basis_out is not None:
            basis_out[idx] = (strength * basis).float()
        rows -= strength * basis
        z_data[idx] = rows.float()

        cur_alpha_min = alpha.min()
        cur_alpha_max = alpha.max()
        cur_beta_abs_max = beta.abs().max()
        alpha_min = cur_alpha_min if alpha_min is None else torch.minimum(alpha_min, cur_alpha_min)
        alpha_max = cur_alpha_max if alpha_max is None else torch.maximum(alpha_max, cur_alpha_max)
        beta_abs_max = cur_beta_abs_max if beta_abs_max is None else torch.maximum(beta_abs_max, cur_beta_abs_max)
        del rows, xe, xs, alpha, beta, basis

    z_data[~live_mask, :] = 0.0
    if verbose:
        print(
            f"  {label}: alpha range [{alpha_min.item():.3g}, {alpha_max.item():.3g}], "
            f"max |beta| {beta_abs_max.item():.3g}"
        )
    del s
    return basis_out


# =============================================================================
# GPU: baseline removal
# =============================================================================

def running_median_1d_gpu(x, window):
    """
    Exact 1D running median on GPU using reflect padding.

    The baseline profile is only one time series, so the unfolded window tensor
    is modest compared with the full filterbank and avoids the old CPU scipy
    median_filter round trip.
    """
    n = x.numel()
    if n <= 1 or window <= 1:
        return x

    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= 1:
        return x

    pad = window // 2
    y = F.pad(x.view(1, 1, -1), (pad, pad), mode="reflect")
    windows = y.unfold(2, window, 1)
    return _median_values(windows, dim=-1).view(-1)


def remove_baseline_gpu(z_data, live_mask, fully_flagged, cfg, verbose=True):
    """
    Running median + per-channel linear regression baseline removal.
    Equivalent to filtool --baseline <width>.
    """
    nchans, ntime = z_data.shape
    tsamp = cfg["_tsamp"]

    baseline_window = max(3, int(cfg["baseline_width"] / tsamp))
    if baseline_window % 2 == 0:
        baseline_window += 1
    if verbose:
        print(f"  Baseline width: {cfg['baseline_width']}s = {baseline_window} samples")

    channel_chunk = max(1, int(cfg.get("baseline_channel_chunk", 256)))
    live_count = live_mask.to(torch.float32).sum().clamp(min=1.0)

    # Mean of live channels per time bin, accumulated on GPU in channel chunks.
    s_raw = torch.zeros(ntime, dtype=z_data.dtype, device=z_data.device)
    for c0 in range(0, nchans, channel_chunk):
        c1 = min(c0 + channel_chunk, nchans)
        rows = z_data[c0:c1]
        weights = live_mask[c0:c1].to(rows.dtype).unsqueeze(1)
        s_raw += (rows * weights).sum(dim=0)
    s_raw /= live_count

    s = running_median_1d_gpu(s_raw, baseline_window).to(torch.float64)

    # Per-channel linear regression (vectorized)
    se = s.sum()
    ss = (s * s).sum()
    n = len(s)
    denom = se * se - ss * n

    safe_denom = torch.where(denom.abs() > 1e-10, denom, torch.ones_like(denom))
    fit_scale = (denom.abs() > 1e-10).to(torch.float64)

    # Chunked regression: process channels in blocks to avoid OOM.
    chunk_size = max(1, int(0.5e9 / (ntime * 8)))  # ~0.5GB per chunk (float64)
    for c0 in range(0, nchans, chunk_size):
        c1 = min(c0 + chunk_size, nchans)
        rows = z_data[c0:c1].to(torch.float64)

        xe = rows.sum(dim=1)
        xs = (rows * s.unsqueeze(0)).sum(dim=1)

        alpha = (xe * se - xs * n) / safe_denom
        beta = (xs * se - xe * ss) / safe_denom

        correction = alpha.unsqueeze(1) * s.unsqueeze(0) + beta.unsqueeze(1)
        row_live = live_mask[c0:c1].to(torch.float64).unsqueeze(1)
        rows -= fit_scale * row_live * correction
        z_data[c0:c1] = rows.float()
        del rows, xe, xs, alpha, beta, correction, row_live

    z_data[~live_mask, :] = 0.0

    if verbose:
        print(f"  Baseline removed from {int(live_count.item())} live channels")


# =============================================================================
# GPU: rescale to uint8
# =============================================================================

def rescale_to_uint8_gpu(z_data, fully_flagged, cfg):
    """
    Per-channel median/MAD rescale to mean=128, std=6, chunked by channel.
    Returns uint8 tensor on GPU.
    """
    out_mean = cfg["out_mean"]
    out_std = cfg["out_std"]
    nchans, ntime = z_data.shape
    channel_chunk = int(cfg.get("diagnostic_channel_chunk", 64))
    out = torch.empty(nchans, ntime, dtype=torch.uint8, device=z_data.device)

    for c0 in range(0, nchans, channel_chunk):
        c1 = min(c0 + channel_chunk, nchans)
        chunk = z_data[c0:c1]
        med = _median_values(chunk, dim=1).unsqueeze(1)
        mad = _median_values((chunk - med).abs(), dim=1).unsqueeze(1) * cfg["mad_const"]

        valid = (mad.squeeze(1) > 1e-6) & ~fully_flagged[c0:c1]
        safe_mad = torch.where(mad > 1e-6, mad, torch.ones_like(mad))
        scaled = ((chunk - med) / safe_mad * out_std + out_mean)
        scaled[~valid, :] = out_mean
        out[c0:c1] = scaled.clamp(0, 255).to(torch.uint8)
        del chunk, med, mad, safe_mad, scaled

    return out


def rescale_to_uint8_filtool_gpu(z_data, cfg, block_len=None):
    """
    Match PulsarX/filtool FilterbankWriter: for each output block, compute
    global mean/std over all pixels and map to cfg out_mean/out_std.
    """
    nchans, ntime = z_data.shape
    if block_len is None or block_len <= 0:
        block_len = ntime

    out = torch.empty(nchans, ntime, dtype=torch.uint8, device=z_data.device)
    for start in range(0, ntime, block_len):
        end = min(start + block_len, ntime)
        block = z_data[:, start:end]
        tmpmean = block.mean()
        tmpstd = torch.sqrt(torch.clamp((block * block).mean() - tmpmean * tmpmean, min=0.0))

        safe_tmpstd = torch.clamp(tmpstd, min=1e-6)
        scl = cfg["out_std"] / safe_tmpstd
        offs = cfg["out_mean"] - scl * tmpmean
        scaled = scl * block + offs
        scaled = torch.where(
            tmpstd > 1e-6,
            scaled,
            torch.full_like(block, cfg["out_mean"]),
        )

        out[:, start:end] = scaled.round().clamp(0, 255).to(torch.uint8)
        del scaled, safe_tmpstd

    return out


def rescale_for_output_gpu(z_data, fully_flagged, cfg):
    mode = cfg.get("rescale_mode", "per_channel_mad")
    if mode == "filtool_block":
        return rescale_to_uint8_filtool_gpu(
            z_data,
            cfg,
            block_len=cfg.get("_rescale_block_len", z_data.shape[1]),
        )
    if mode == "filtool_global":
        return rescale_to_uint8_filtool_gpu(z_data, cfg, block_len=z_data.shape[1])
    if mode == "per_channel_mad":
        return rescale_to_uint8_gpu(z_data, fully_flagged, cfg)
    raise ValueError(f"Unknown rescale_mode: {mode}")


def write_z_filterbank_gpu(path, z_data, fully_flagged, cfg, header_bytes, label):
    """
    Write z-domain diagnostics with per-channel median/MAD rescale.

    This is intentionally chunked and streaming. The old full-tensor path
    allocated another nchans x ntime float32 tensor while diagnostics were
    active, which is enough to OOM on full GMRT files.
    """
    if not path:
        return
    print(f"  Writing diagnostic {label}: {path}")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nchans, ntime = z_data.shape
    channel_chunk = int(cfg.get("diagnostic_channel_chunk", 64))
    time_chunk = int(cfg.get("diagnostic_time_chunk", 4096))
    out_mean = cfg["out_mean"]
    out_std = cfg["out_std"]

    ch_medians = torch.empty(nchans, 1, dtype=z_data.dtype, device=z_data.device)
    ch_mads = torch.empty(nchans, 1, dtype=z_data.dtype, device=z_data.device)
    for c0 in range(0, nchans, channel_chunk):
        c1 = min(c0 + channel_chunk, nchans)
        chunk = z_data[c0:c1]
        med = _median_values(chunk, dim=1).unsqueeze(1)
        mad = _median_values((chunk - med).abs(), dim=1).unsqueeze(1) * cfg["mad_const"]
        ch_medians[c0:c1] = med
        ch_mads[c0:c1] = mad
        del chunk, med, mad

    valid = (ch_mads.squeeze(1) > 1e-6) & ~fully_flagged
    safe_mads = torch.where(ch_mads > 1e-6, ch_mads, torch.ones_like(ch_mads))

    with open(path, 'wb') as f:
        f.write(header_bytes)
        for t0 in range(0, ntime, time_chunk):
            t1 = min(t0 + time_chunk, ntime)
            block = z_data[:, t0:t1]
            scaled = ((block - ch_medians) / safe_mads * out_std + out_mean)
            scaled[~valid, :] = out_mean
            data_u8 = scaled.clamp(0, 255).to(torch.uint8)
            f.write(data_u8.cpu().numpy().T.ravel().tobytes())
            del block, scaled, data_u8
    del ch_medians, ch_mads, safe_mads


def write_z_filtool_scale_filterbank_gpu(path, z_data, cfg, header_bytes, label):
    """Write z-domain diagnostics with the same global block rescale as filtool."""
    if not path:
        return
    print(f"  Writing diagnostic {label}: {path}")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ntime = z_data.shape[1]
    block_len = cfg.get("_rescale_block_len", ntime)
    if block_len is None or block_len <= 0:
        block_len = ntime

    with open(path, 'wb') as f:
        f.write(header_bytes)
        for start in range(0, ntime, block_len):
            end = min(start + block_len, ntime)
            block = z_data[:, start:end]
            tmpmean = block.mean()
            tmpstd = torch.sqrt(torch.clamp((block * block).mean() - tmpmean * tmpmean, min=0.0))

            safe_tmpstd = torch.clamp(tmpstd, min=1e-6)
            scl = cfg["out_std"] / safe_tmpstd
            offs = cfg["out_mean"] - scl * tmpmean
            scaled = scl * block + offs
            scaled = torch.where(
                tmpstd > 1e-6,
                scaled,
                torch.full_like(block, cfg["out_mean"]),
            )

            data_u8 = scaled.round().clamp(0, 255).to(torch.uint8)
            f.write(data_u8.cpu().numpy().T.ravel().tobytes())
            del block, scaled, safe_tmpstd, data_u8


def write_z_fixed_scale_filterbank_gpu(path, z_data, fully_flagged, cfg, header_bytes, label):
    """
    Write z-domain data using a fixed affine map instead of per-channel
    median/MAD rescaling: uint8 = z * out_std + out_mean.

    This separates damage from the normalization itself versus damage from
    the diagnostic write-back rescale.
    """
    if not path:
        return
    print(f"  Writing diagnostic {label}: {path}")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ntime = z_data.shape[1]
    time_chunk = int(cfg.get("diagnostic_time_chunk", 4096))

    with open(path, 'wb') as f:
        f.write(header_bytes)
        for t0 in range(0, ntime, time_chunk):
            t1 = min(t0 + time_chunk, ntime)
            data = z_data[:, t0:t1] * cfg["out_std"] + cfg["out_mean"]
            data[fully_flagged, :] = cfg["out_mean"]
            data_u8 = data.clamp(0, 255).to(torch.uint8)
            f.write(data_u8.cpu().numpy().T.ravel().tobytes())
            del data, data_u8


def write_raw_filterbank_gpu(path, raw_data, header_bytes, label):
    """Write a raw-domain diagnostic .fil without normalization or mitigation."""
    if not path:
        return
    print(f"  Writing diagnostic {label}: {path}")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data_u8 = raw_data.clamp(0, 255).to(torch.uint8)
    data_to_write = data_u8.cpu().numpy().T.ravel()
    del data_u8

    with open(path, 'wb') as f:
        f.write(header_bytes)
        f.write(data_to_write.tobytes())


def resolve_mask_output_paths(cfg):
    """Return mask paths only when mask output is explicitly enabled."""
    if not cfg.get("write_mask_files", False):
        return {"combined": None, "nn": None, "blanked": None}

    output_fil = cfg.get("output_fil")
    if not output_fil:
        raise ValueError("output_fil must be set before deriving mask paths")
    root, ext = os.path.splitext(output_fil)
    if not ext:
        ext = ".fil"

    return {
        "combined": cfg.get("output_mask_fil") or f"{root}_mask{ext}",
        "nn": cfg.get("output_nn_mask_fil") or f"{root}_nn_mask{ext}",
        "blanked": cfg.get("output_blanked_mask_fil") or f"{root}_blanked_mask{ext}",
    }


# =============================================================================
# Model loading
# =============================================================================

def _extract_model_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def _validate_engine_metadata(engine_path, cfg):
    warnings = validate_engine_metadata(
        engine_path,
        cfg,
        allow_unverified=bool(cfg.get("allow_unverified_artifacts", False)),
    )
    for warning in warnings:
        print(f"  WARNING: {warning}")


def load_model(cfg, device):
    """Load a MARS checkpoint or an explicitly provided TensorRT engine."""
    trt_path = cfg.get("tensorrt_path")

    if trt_path and not os.path.exists(trt_path):
        raise FileNotFoundError(
            f"Requested TensorRT engine does not exist; refusing PyTorch fallback: {trt_path}"
        )
    if trt_path:
        if device.type != "cuda":
            raise RuntimeError("TensorRT inference requires a CUDA device")
        _validate_engine_metadata(trt_path, cfg)
        # Load pre-compiled TensorRT engine (.engine file)
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {trt_path}")
        ctx = TRTInferenceContext(engine, cfg["batch_size"], device)
        print(f"  TensorRT engine loaded: {trt_path}")
        return ctx

    from .model import build_model, count_parameters
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    try:
        validate_training_identity(ckpt_cfg, source=f"checkpoint {cfg['checkpoint']}")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    mismatches = model_config_mismatches(cfg, ckpt_cfg)
    if mismatches:
        raise RuntimeError(
            "Checkpoint model config does not match the requested model config: "
            + "; ".join(mismatches)
        )
    model_cfg = dict(cfg)
    model_cfg.update(ckpt_cfg)
    model = build_model(model_cfg).to(device)
    parameter_count = count_parameters(model)
    expected_parameters = model_cfg.get("expected_parameters")
    if expected_parameters is not None and parameter_count != int(expected_parameters):
        raise RuntimeError(
            f"Model has {parameter_count:,} trainable parameters; expected "
            f"{int(expected_parameters):,} for the paper architecture."
        )

    state_dict = _extract_model_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match mars_rfi.model. "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(
        f"  {model_cfg.get('model', 'trt_shape_unet')} loaded: {cfg['checkpoint']} "
        f"(epoch {epoch}, params {parameter_count:,})"
    )

    return model


# =============================================================================
# GPU: TensorRT inference context (reusable)
# =============================================================================

class TRTInferenceContext:
    """Persistent TensorRT context with pre-allocated buffers. Created once."""
    def __init__(self, engine, batch_size, device):
        self.context = engine.create_execution_context()
        self.batch_size = batch_size
        self.device = device
        self.input_dtype = self._torch_dtype(engine.get_tensor_dtype("input"))
        self.output_dtype = self._torch_dtype(engine.get_tensor_dtype("output"))
        # Pre-allocate input/output buffers (reused every call)
        self.input_buf = torch.zeros(batch_size, 1, 512, 512,
                                     dtype=self.input_dtype, device=device)
        self.output_buf = torch.zeros(batch_size, 1, 512, 512,
                                      dtype=self.output_dtype, device=device)
        # Set shape and addresses once
        self.context.set_input_shape("input", (batch_size, 1, 512, 512))
        self.context.set_tensor_address("input", self.input_buf.data_ptr())
        self.context.set_tensor_address("output", self.output_buf.data_ptr())
        self.stream = torch.cuda.Stream(device=device)
        print(
            "  TensorRT I/O dtype: "
            f"input={self.input_dtype}, output={self.output_dtype}"
        )

    @staticmethod
    def _torch_dtype(trt_dtype):
        name = str(trt_dtype).split(".")[-1].lower()
        if name in ("half", "float16", "fp16"):
            return torch.float16
        if name in ("float", "float32", "fp32"):
            return torch.float32
        raise TypeError(f"Unsupported TensorRT tensor dtype for torch binding: {trt_dtype}")


# =============================================================================
# GPU: batched NN inference on patches
# =============================================================================

def _dilate_bool(mask, freq_radius, time_radius):
    freq_radius = max(0, int(freq_radius))
    time_radius = max(0, int(time_radius))
    if freq_radius == 0 and time_radius == 0:
        return mask
    kernel = (2 * freq_radius + 1, 2 * time_radius + 1)
    padding = (freq_radius, time_radius)
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=padding)
    return y.squeeze(1) > 0.0


def _erode_bool(mask, freq_radius, time_radius):
    return ~_dilate_bool(~mask, freq_radius, time_radius)


def _close_bool(mask, freq_radius, time_radius):
    return _erode_bool(
        _dilate_bool(mask, freq_radius, time_radius),
        freq_radius,
        time_radius,
    )


def apply_hysteresis_mask(prob, cfg):
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

    return final


def apply_hysteresis_mask_chunked(prob, cfg):
    chunk_size = int(cfg.get("hys_chunk_size", 0) or 0)
    if chunk_size <= 0 or prob.shape[0] <= chunk_size:
        return apply_hysteresis_mask(prob, cfg)

    out = torch.empty(prob.shape, dtype=torch.bool, device=prob.device)
    for start in range(0, prob.shape[0], chunk_size):
        stop = min(start + chunk_size, prob.shape[0])
        out[start:stop] = apply_hysteresis_mask(prob[start:stop], cfg)
    return out


@torch.inference_mode()
def infer_patches(model, patches, cfg, device):
    """
    Run NN inference on (N, 1, 512, 512) patches.
    Supports both PyTorch nn.Module and TRTInferenceContext.
    Returns binary mask (N, 512, 512) bool.
    """
    N = patches.shape[0]
    batch_size = cfg["batch_size"]
    threshold = cfg["threshold"]
    patch_h, patch_w = patches.shape[-2:]
    hys_enabled = bool(cfg.get("hys_enabled", False))

    if N == 0:
        return torch.empty(0, patch_h, patch_w, dtype=torch.bool, device=device)

    if not hys_enabled:
        threshold = float(threshold)
        if threshold <= 0.0:
            return torch.ones(N, patch_h, patch_w, dtype=torch.bool, device=device)
        if threshold >= 1.0:
            return torch.zeros(N, patch_h, patch_w, dtype=torch.bool, device=device)
        logit_threshold = math.log(threshold / (1.0 - threshold))

    # TensorRT path
    if isinstance(model, TRTInferenceContext):
        ctx = model
        trt_stream = ctx.stream
        if cfg.get("runtime_stats", True):
            print(
                "  NN inference backend: TensorRT "
                f"(patches={N}, batch={batch_size}, batches={math.ceil(N / batch_size)}, "
                f"input={ctx.input_dtype}, output={ctx.output_dtype})"
            )
        logit_patches = torch.empty(
            N,
            1,
            patch_h,
            patch_w,
            dtype=ctx.output_dtype,
            device=device,
        )
        current_stream = torch.cuda.current_stream(device)
        if patches.dtype != ctx.input_dtype:
            patches = patches.to(dtype=ctx.input_dtype)
        if not patches.is_contiguous():
            patches = patches.contiguous()

        trt_stream.wait_stream(current_stream)
        for i in range(0, N, batch_size):
            batch = patches[i:i + batch_size]
            actual_bs = batch.shape[0]

            if actual_bs == batch_size:
                out_view = logit_patches[i:i + actual_bs]
                ctx.context.set_tensor_address("input", batch.data_ptr())
                ctx.context.set_tensor_address("output", out_view.data_ptr())
                executed = ctx.context.execute_async_v3(trt_stream.cuda_stream)
                if not executed:
                    raise RuntimeError("TensorRT execution failed for a full inference batch")
            else:
                with torch.cuda.stream(trt_stream):
                    ctx.input_buf[:actual_bs].copy_(batch)
                    ctx.input_buf[actual_bs:].zero_()
                ctx.context.set_tensor_address("input", ctx.input_buf.data_ptr())
                ctx.context.set_tensor_address("output", ctx.output_buf.data_ptr())
                executed = ctx.context.execute_async_v3(trt_stream.cuda_stream)
                if not executed:
                    raise RuntimeError("TensorRT execution failed for a padded inference batch")
                with torch.cuda.stream(trt_stream):
                    logit_patches[i:i + actual_bs].copy_(ctx.output_buf[:actual_bs])

        current_stream.wait_stream(trt_stream)

        if hys_enabled:
            logits_2d = logit_patches.squeeze(1)
            chunk_size = int(cfg.get("hys_chunk_size", 0) or 0)
            if chunk_size > 0 and N > chunk_size:
                mask_patches = torch.empty(N, patch_h, patch_w, dtype=torch.bool, device=device)
                for start in range(0, N, chunk_size):
                    stop = min(start + chunk_size, N)
                    probs = torch.sigmoid(logits_2d[start:stop].float())
                    mask_patches[start:stop] = apply_hysteresis_mask(probs, cfg)
                return mask_patches

            probs = torch.sigmoid(logits_2d.float())
            return apply_hysteresis_mask(probs, cfg)
        return logit_patches.squeeze(1) >= logit_threshold

    # PyTorch path
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    if cfg.get("runtime_stats", True):
        print(
            "  NN inference backend: PyTorch "
            f"(patches={N}, batch={batch_size}, batches={math.ceil(N / batch_size)}, "
            f"amp={use_amp})"
        )
    mask_patches = torch.empty(N, patch_h, patch_w, dtype=torch.bool, device=device)
    for i in range(0, N, batch_size):
        batch = patches[i:i + batch_size]
        actual_bs = batch.shape[0]
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(batch)
        else:
            logits = model(batch)

        if hys_enabled:
            probs = torch.sigmoid(logits).float().squeeze(1)
            mask_patches[i:i + actual_bs] = apply_hysteresis_mask_chunked(probs, cfg)
        else:
            mask_patches[i:i + actual_bs] = logits.squeeze(1) >= logit_threshold

    return mask_patches


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@torch.inference_mode()
def _run_pipeline_impl(cfg):
    for required_key in ("input_fil", "output_fil"):
        if not cfg.get(required_key):
            raise ValueError(f"{required_key} must be provided")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if os.path.abspath(cfg["input_fil"]) == os.path.abspath(cfg["output_fil"]):
        raise ValueError("input_fil and output_fil must be different files")

    timings = {}  # step_name -> seconds

    t_total = time.perf_counter()

    # =========================================================================
    # STEP 1: Load raw data -> GPU
    # =========================================================================
    print("\n[Step 1] Loading filterbank...")
    _sync_device(device)
    t_step = time.perf_counter()

    fil = FilReader(cfg["input_fil"])
    nchans = fil.header.nchans
    ntime = fil.header.nsamples
    tsamp = fil.header.tsamp
    validate_supported_channel_count(nchans, cfg)
    block = fil.read_block(0, ntime)
    input_nbits = int(getattr(fil.header, "nbits", 8))
    if input_nbits != 8:
        raise ValueError(
            "MARS currently preserves the input SIGPROC header while writing "
            f"uint8 samples, so only 8-bit input is safe (got nbits={input_nbits})."
        )
    cfg["_tsamp"] = tsamp

    header_size = fil.header.stream_info.entries[0].hdrlen
    with open(cfg["input_fil"], 'rb') as f:
        header_bytes = f.read(header_size)

    raw_dtype_name = str(cfg.get("raw_gpu_dtype", "float32")).lower()
    if raw_dtype_name in ("uint8", "u8", "byte"):
        raw_dtype = torch.uint8
    elif raw_dtype_name in ("float16", "fp16", "half"):
        raw_dtype = torch.float16
    elif raw_dtype_name in ("float32", "fp32", "float"):
        raw_dtype = torch.float32
    else:
        raise ValueError(
            "raw_gpu_dtype must be one of: uint8, u8, byte, float16, fp16, half, "
            f"float32, fp32, float (got {raw_dtype_name!r})"
        )

    raw_compute_dtype_name = str(cfg.get("raw_compute_dtype", "float16")).lower()
    if raw_compute_dtype_name in ("float16", "fp16", "half"):
        raw_compute_dtype = torch.float16
    elif raw_compute_dtype_name in ("float32", "fp32", "float"):
        raw_compute_dtype = torch.float32
    else:
        raise ValueError(
            "raw_compute_dtype must be one of: float16, fp16, half, "
            f"float32, fp32, float (got {raw_compute_dtype_name!r})"
        )

    raw = torch.from_numpy(block.data).to(device=device, dtype=raw_dtype)
    cfg["_raw_gpu_dtype"] = raw_dtype_name
    cfg["_raw_compute_dtype"] = raw_compute_dtype_name
    cfg["_raw_gpu_element_size"] = torch.empty((), dtype=raw_dtype).element_size()
    del block

    diag_prefix = cfg.get("diagnostic_fil_prefix")
    mask_paths = resolve_mask_output_paths(cfg)
    save_mask = mask_paths["combined"] is not None
    save_nn_mask = mask_paths["nn"] is not None
    save_blanked_mask = mask_paths["blanked"] is not None
    use_nn_mask_for_replacement = bool(cfg.get("use_nn_mask_for_replacement", True))
    need_nn_inference = (
        use_nn_mask_for_replacement
        or save_nn_mask
        or bool(cfg.get("force_run_nn_inference", False))
        or not bool(cfg.get("skip_nn_when_unused", False))
    )
    if diag_prefix:
        write_raw_filterbank_gpu(
            f"{diag_prefix}_raw_copy.fil",
            raw,
            header_bytes,
            "raw-copy",
        )

    _sync_device(device)
    timings["1_load_data"] = time.perf_counter() - t_step
    print(f"  Shape: ({nchans}, {ntime}), tsamp={tsamp*1e3:.3f} ms")
    print(f"  Raw GPU dtype: {raw_dtype_name}")
    print(f"  Raw compute dtype: {raw_compute_dtype_name}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    unusable = torch.zeros(nchans, dtype=torch.bool, device=device)
    preflag_channels = tuple(int(channel) for channel in cfg.get("preflag_channels", ()))
    if preflag_channels:
        invalid = [channel for channel in preflag_channels if not 0 <= channel < nchans]
        if invalid:
            raise ValueError(
                f"preflag_channels contains indices outside [0, {nchans}): {invalid}"
            )
        unusable[list(preflag_channels)] = True
    science_global_blank = unusable.clone()
    nn_global_blank = unusable.clone()

    if diag_prefix and cfg.get("diagnostic_global_normalisation", False):
        print("  Building diagnostic global channel normalisation...")
        global_med = _median_values(raw, dim=1, keepdim=True)
        global_mad = (
            _median_values((raw - global_med).abs(), dim=1, keepdim=True)
            * cfg["mad_const"]
        )
        global_std = raw.std(dim=1, keepdim=True)
        global_scale = torch.where(global_mad < cfg["mad_min_valid"], global_std, global_mad)
        global_safe_scale = torch.where(
            global_scale > 1e-6, global_scale, torch.ones_like(global_scale)
        )
        z_global = (raw - global_med) / global_safe_scale
        no_channel_flags = torch.zeros_like(unusable)
        write_z_filterbank_gpu(
            f"{diag_prefix}_global_normalized_no_channel_blank.fil",
            z_global,
            no_channel_flags,
            cfg,
            header_bytes,
            "global normalized, no channel blank",
        )
        write_z_fixed_scale_filterbank_gpu(
            f"{diag_prefix}_global_normalized_no_channel_blank_fixed_scale.fil",
            z_global,
            no_channel_flags,
            cfg,
            header_bytes,
            "global normalized, no channel blank, fixed scale",
        )
        del global_med, global_mad, global_std, global_scale, global_safe_scale, z_global

    # =========================================================================
    # STEP 2: Load model
    # =========================================================================
    _sync_device(device)
    t_step = time.perf_counter()

    if need_nn_inference:
        print("\n[Step 2] Loading model...")
        model = load_model(cfg, device)
    else:
        print("\n[Step 2] Loading model skipped (NN inference not needed)...")
        model = None

    _sync_device(device)
    timings["3_load_model"] = time.perf_counter() - t_step

    # =========================================================================
    # STEP 3: Segment-wise normalisation + inference
    # =========================================================================
    step3_label = "Segment-wise normalisation + NN inference"
    if not need_nn_inference:
        step3_label = "Segment-wise science normalisation without NN inference"
    print(f"\n[Step 3] {step3_label}...")
    _sync_device(device)
    t_step = time.perf_counter()

    patch_size = cfg["patch_size"]
    target_samples = cfg["target_segment_seconds"] / tsamp
    n_patches_per_seg = max(1, int(round(target_samples / patch_size)))
    seg_len = n_patches_per_seg * patch_size

    n_full_segs = ntime // seg_len
    tail_len = ntime - n_full_segs * seg_len
    cfg["_rescale_block_len"] = seg_len
    print(f"  Segment: {seg_len} samples ({seg_len * tsamp:.3f}s), "
          f"{n_full_segs} segments + {tail_len} tail")
    if cfg.get("raw_segment_detector_enabled", False):
        print(f"  Raw segment detector mode: {cfg.get('raw_segment_detector_mode', 'guarded')}")

    # ---- 4a: Batched normalisation ----
    _sync_device(device)
    t_sub = time.perf_counter()

    usable_time = n_full_segs * seg_len
    raw_segs = raw[:, :usable_time].reshape(nchans, n_full_segs, seg_len)
    if raw_segs.dtype != raw_compute_dtype:
        raw_segs = raw_segs.to(raw_compute_dtype)

    MAD_CONST = 1.4826
    seg_medians = _median_values(raw_segs, dim=2, keepdim=True)
    seg_mads = (
        _median_values((raw_segs - seg_medians).abs(), dim=2, keepdim=True)
        * MAD_CONST
    )
    science_vars, science_means = torch.var_mean(
        raw_segs,
        dim=2,
        keepdim=True,
        unbiased=True,
    )
    seg_stds = torch.sqrt(torch.clamp(science_vars, min=0.0))
    seg_dead = seg_stds.squeeze(2) < 1e-6

    seg_ch_medians = seg_medians.squeeze(2)
    seg_global_median = _median_values(seg_ch_medians, dim=0)
    seg_global_mad = (
        _median_values((seg_ch_medians - seg_global_median).abs(), dim=0)
        * MAD_CONST
    )
    seg_sat_thresh = seg_global_median + cfg["sat_sigma"] * seg_global_mad
    seg_sat_fraction = (raw_segs > seg_sat_thresh.unsqueeze(0).unsqueeze(2)).float().mean(dim=2)
    seg_sat_unusable = seg_sat_fraction > cfg["sat_ratio_segment"]
    runtime_stats = cfg.get("runtime_stats", True)

    if cfg.get("raw_segment_detector_enabled", False) and runtime_stats:
        print(
            "  Segment saturation flags before raw detector: "
            f"{int(seg_sat_unusable.sum())} channel-segments "
            f"(>{cfg['sat_ratio_segment']*100:.0f}% bright occupancy)"
        )
    if cfg.get("raw_segment_detector_enabled", False):
        seg_raw_persistent_rfi = detect_raw_segment_persistent_rfi_gpu(
            seg_ch_medians,
            seg_mads.squeeze(2),
            seg_sat_fraction,
            cfg,
            label="full",
        )
        seg_raw_persistent_rfi = _apply_raw_detector_mode(
            seg_raw_persistent_rfi,
            seg_dead,
            cfg,
            label="full",
        )
    else:
        seg_raw_persistent_rfi = torch.zeros_like(seg_sat_unusable)
    seg_unusable = seg_sat_unusable | seg_raw_persistent_rfi

    combined_unusable = nn_global_blank.unsqueeze(1) | seg_unusable

    use_std = seg_mads < cfg["mad_min_valid"]
    scale = torch.where(use_std, seg_stds, seg_mads)
    safe_scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    z_segs = (raw_segs - seg_medians) / safe_scale

    # Science/output branch: filtool-style per-channel mean/std
    # normalisation. The NN branch above stays median/MAD to preserve the
    # model input distribution it was trained with.
    science_stds = seg_stds
    safe_science_stds = torch.where(
        science_stds > 1e-6, science_stds, torch.ones_like(science_stds)
    )
    science_segs = (raw_segs - science_means) / safe_science_stds
    science_degenerate = (
        (science_stds.squeeze(2) < 1e-6)
        | science_global_blank.unsqueeze(1)
        | seg_unusable
    )
    science_flagged_per_seg = science_degenerate & ~science_global_blank.unsqueeze(1)

    z_data_no_blank = None
    if diag_prefix:
        z_data_no_blank = torch.zeros(nchans, ntime, dtype=torch.float32, device=device)
        z_data_no_blank[:, :usable_time] = z_segs.reshape(nchans, usable_time)

    degenerate = (scale.squeeze(2) < 1e-6) | combined_unusable
    z_segs = apply_nn_input_blank_mode(z_segs, degenerate, cfg)

    z_data = None
    if diag_prefix:
        z_data = torch.zeros(nchans, ntime, dtype=torch.float32, device=device)
        z_data[:, :usable_time] = z_segs.reshape(nchans, usable_time)

    del raw_segs, seg_medians, seg_mads, seg_ch_medians
    del seg_global_median, seg_global_mad, seg_sat_thresh, seg_sat_fraction
    del seg_sat_unusable, seg_raw_persistent_rfi, combined_unusable
    del seg_stds, seg_dead, use_std, scale, safe_scale, degenerate
    del science_means, science_stds, safe_science_stds
    _empty_cuda_cache(cfg)

    _sync_device(device)
    timings["4a_normalise"] = time.perf_counter() - t_sub

    # ---- 4b: Tanh compress ----
    _sync_device(device)
    t_sub = time.perf_counter()
    neg_cleanup_time = 0.0
    neg_block_cleanup_time = 0.0

    _sync_device(device)
    t_neg = time.perf_counter()
    z_segs, neg_seg_stats = replace_continuous_negative_segments_per_segment(
        z_segs,
        cfg,
        segment_dim=1,
        label="pipeline-full-segments",
    )
    _sync_device(device)
    neg_cleanup_time += time.perf_counter() - t_neg

    _sync_device(device)
    t_neg_block = time.perf_counter()
    z_segs, neg_block_stats = replace_negative_blocks_2d_per_segment(
        z_segs,
        cfg,
        segment_dim=1,
        label="pipeline-full-segments",
    )
    _sync_device(device)
    neg_block_cleanup_time += time.perf_counter() - t_neg_block

    tanh_data = torch.tanh(z_segs / cfg["tanh_scale"])
    del z_segs
    _empty_cuda_cache(cfg)
    if runtime_stats and neg_seg_stats.get("enabled", 0.0):
        print(
            "  Negative-segment NN-input cleanup (pre-tanh full): "
            f"replaced={neg_seg_stats['replaced_fraction']:.4f}, "
            f"rows={int(neg_seg_stats.get('rows_touched', 0))}/"
            f"{int(neg_seg_stats.get('n_rows', 0))}"
        )
    if runtime_stats and neg_block_stats.get("enabled", 0.0):
        print(
            "  Negative-block NN-input cleanup (pre-tanh full): "
            f"replaced={neg_block_stats['replaced_fraction']:.4f}, "
            f"rows={int(neg_block_stats.get('rows_touched', 0))}/"
            f"{int(neg_block_stats.get('n_rows', 0))}, "
            f"windows={int(neg_block_stats.get('windows', 0))}"
        )

    science_tail_degen = None
    tail_start = usable_time
    usable_tail = 0
    if tail_len >= patch_size:
        tail_seg = raw[:, tail_start:ntime]
        if tail_seg.dtype != raw_compute_dtype:
            tail_seg = tail_seg.to(raw_compute_dtype)
        tail_median = _median_values(tail_seg, dim=1, keepdim=True)
        tail_mad = (
            _median_values((tail_seg - tail_median).abs(), dim=1, keepdim=True)
            * MAD_CONST
        )
        tail_std = tail_seg.std(dim=1, keepdim=True)
        tail_dead = tail_std.squeeze(1) < 1e-6
        tail_use_std = tail_mad < cfg["mad_min_valid"]
        tail_scale = torch.where(tail_use_std, tail_std, tail_mad)
        safe_tail_scale = torch.where(tail_scale > 1e-6, tail_scale, torch.ones_like(tail_scale))
        z_tail = (tail_seg - tail_median) / safe_tail_scale
        tail_ch_medians = tail_median.squeeze(1)
        tail_global_median = tail_ch_medians.median()
        tail_global_mad = (
            (tail_ch_medians - tail_global_median).abs().median() * MAD_CONST
        )
        tail_sat_thresh = tail_global_median + cfg["sat_sigma"] * tail_global_mad
        tail_sat_fraction = (tail_seg > tail_sat_thresh).float().mean(dim=1)
        tail_sat_unusable = tail_sat_fraction > cfg["sat_ratio_segment"]
        if cfg.get("raw_segment_detector_enabled", False):
            tail_raw_persistent_rfi = detect_raw_segment_persistent_rfi_gpu(
                tail_ch_medians.unsqueeze(1),
                tail_mad.squeeze(1).unsqueeze(1),
                tail_sat_fraction.unsqueeze(1),
                cfg,
                label="tail",
            )
            tail_raw_persistent_rfi = _apply_raw_detector_mode(
                tail_raw_persistent_rfi,
                tail_dead.unsqueeze(1),
                cfg,
                label="tail",
            ).squeeze(1)
        else:
            tail_raw_persistent_rfi = torch.zeros_like(tail_sat_unusable)
        tail_unusable = tail_sat_unusable | tail_raw_persistent_rfi
        tail_degen = (
            (tail_scale.squeeze(1) < 1e-6)
            | nn_global_blank
            | tail_unusable
        )

        tail_science_mean = tail_seg.mean(dim=1, keepdim=True)
        tail_science_std = tail_seg.std(dim=1, keepdim=True)
        safe_tail_science_std = torch.where(
            tail_science_std > 1e-6, tail_science_std, torch.ones_like(tail_science_std)
        )
        science_tail = (tail_seg - tail_science_mean) / safe_tail_science_std
        science_tail_degen = (
            (tail_science_std.squeeze(1) < 1e-6)
            | science_global_blank
            | tail_unusable
        )
        if z_data_no_blank is not None:
            z_data_no_blank[:, tail_start:ntime] = z_tail
        z_tail = apply_nn_input_blank_mode(
            z_tail.unsqueeze(1),
            tail_degen.unsqueeze(1),
            cfg,
        ).squeeze(1)
        if z_data is not None:
            z_data[:, tail_start:ntime] = z_tail
        usable_tail = (tail_len // patch_size) * patch_size
        _sync_device(device)
        t_neg = time.perf_counter()
        z_tail_for_nn, neg_tail_stats = replace_continuous_negative_segments(
            z_tail[:, :usable_tail],
            cfg,
            label="pipeline-tail",
        )
        _sync_device(device)
        neg_cleanup_time += time.perf_counter() - t_neg

        _sync_device(device)
        t_neg_block = time.perf_counter()
        z_tail_for_nn, neg_tail_block_stats = replace_negative_blocks_2d(
            z_tail_for_nn,
            cfg,
            label="pipeline-tail",
        )
        _sync_device(device)
        neg_block_cleanup_time += time.perf_counter() - t_neg_block

        tanh_tail = torch.tanh(z_tail_for_nn / cfg["tanh_scale"])
        del z_tail_for_nn
        if runtime_stats and neg_tail_stats.get("enabled", 0.0):
            print(
                "  Negative-segment NN-input cleanup (pre-tanh tail): "
                f"replaced={neg_tail_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_tail_stats.get('rows_touched', 0))}/"
                f"{int(neg_tail_stats.get('n_rows', 0))}"
            )
        if runtime_stats and neg_tail_block_stats.get("enabled", 0.0):
            print(
                "  Negative-block NN-input cleanup (pre-tanh tail): "
                f"replaced={neg_tail_block_stats['replaced_fraction']:.4f}, "
                f"rows={int(neg_tail_block_stats.get('rows_touched', 0))}/"
                f"{int(neg_tail_block_stats.get('n_rows', 0))}, "
                f"windows={int(neg_tail_block_stats.get('windows', 0))}"
            )
        del tail_science_mean, tail_science_std, safe_tail_science_std
        del tail_median, tail_mad, tail_std, tail_use_std, tail_scale, safe_tail_scale
        del tail_dead, z_tail
        del tail_ch_medians, tail_global_median, tail_global_mad, tail_sat_thresh
        del tail_sat_fraction, tail_sat_unusable, tail_raw_persistent_rfi, tail_unusable, tail_degen
    elif tail_len > 0:
        usable_tail = 0

    del raw
    _empty_cuda_cache(cfg)

    _sync_device(device)
    timings["4b_tanh"] = time.perf_counter() - t_sub
    timings["4n_negative_cleanup"] = neg_cleanup_time
    timings["4q_negative_block_cleanup"] = neg_block_cleanup_time

    if z_data_no_blank is not None:
        no_channel_flags = torch.zeros_like(unusable)
        write_z_filterbank_gpu(
            f"{diag_prefix}_normalized_no_channel_blank.fil",
            z_data_no_blank,
            no_channel_flags,
            cfg,
            header_bytes,
            "normalized, no channel blank",
        )
        write_z_fixed_scale_filterbank_gpu(
            f"{diag_prefix}_normalized_no_channel_blank_fixed_scale.fil",
            z_data_no_blank,
            no_channel_flags,
            cfg,
            header_bytes,
            "normalized, no channel blank, fixed scale",
        )
        write_z_filtool_scale_filterbank_gpu(
            f"{diag_prefix}_normalized_no_channel_blank_filtool_scale.fil",
            z_data_no_blank,
            cfg,
            header_bytes,
            "normalized, no channel blank, filtool scale",
        )
        del z_data_no_blank

    if diag_prefix:
        write_z_filterbank_gpu(
            f"{diag_prefix}_normalized_only.fil",
            z_data,
            nn_global_blank,
            cfg,
            header_bytes,
            "normalized-only",
        )
        write_z_fixed_scale_filterbank_gpu(
            f"{diag_prefix}_normalized_only_fixed_scale.fil",
            z_data,
            nn_global_blank,
            cfg,
            header_bytes,
            "normalized-only, fixed scale",
        )
        write_z_filtool_scale_filterbank_gpu(
            f"{diag_prefix}_normalized_only_filtool_scale.fil",
            z_data,
            cfg,
            header_bytes,
            "normalized-only, filtool scale",
        )

        science_data_no_blank = torch.zeros(nchans, ntime, dtype=torch.float32, device=device)
        science_data_no_blank[:, :usable_time] = science_segs.reshape(nchans, usable_time)
        if tail_len >= patch_size:
            science_data_no_blank[:, tail_start:ntime] = science_tail
        write_z_filtool_scale_filterbank_gpu(
            f"{diag_prefix}_science_meanstd_no_channel_blank_filtool_scale.fil",
            science_data_no_blank,
            cfg,
            header_bytes,
            "science mean/std, no channel blank, filtool scale",
        )
        del science_data_no_blank

    science_segs[science_degenerate, :] = 0.0
    science_data = torch.zeros(nchans, ntime, dtype=torch.float32, device=device)
    science_data[:, :usable_time] = science_segs.reshape(nchans, usable_time)
    if tail_len >= patch_size:
        science_tail[science_tail_degen, :] = 0.0
        science_data[:, tail_start:ntime] = science_tail
        del science_tail
    elif tail_len > 0:
        science_data[:, usable_time:] = 0.0
    del science_segs

    if diag_prefix:
        write_z_filtool_scale_filterbank_gpu(
            f"{diag_prefix}_science_meanstd_only_filtool_scale.fil",
            science_data,
            cfg,
            header_bytes,
            "science mean/std, channel blanked, filtool scale",
        )

    timings["4pre_baseline"] = 0.0
    if cfg.get("science_prebaseline_enabled", False):
        print("\n[Step 3p] Science branch baseline before zdot/replacement...")
        _sync_device(device)
        t_prebaseline = time.perf_counter()

        remove_baseline_gpu(
            science_data,
            ~science_global_blank,
            science_global_blank,
            cfg,
            verbose=runtime_stats,
        )

        _sync_device(device)
        timings["4pre_baseline"] = time.perf_counter() - t_prebaseline

        if diag_prefix:
            write_z_filtool_scale_filterbank_gpu(
                f"{diag_prefix}_science_prebaseline_meanstd_only_filtool_scale.fil",
                science_data,
                cfg,
                header_bytes,
                "science prebaseline mean/std, channel blanked, filtool scale",
            )

    timings["4z_zdot"] = 0.0
    zdot_stage = cfg.get("science_zdot_stage", "off") if cfg.get("science_zdot", False) else "off"
    if zdot_stage == "pre_replacement":
        print("\n[Step 3z] Science branch zdot before replacement...")
        _sync_device(device)
        t_zdot = time.perf_counter()

        apply_zdot_gpu(science_data, ~science_global_blank, cfg, label="science zdot pre-replacement")

        _sync_device(device)
        timings["4z_zdot"] = time.perf_counter() - t_zdot

        if diag_prefix:
            write_z_filtool_scale_filterbank_gpu(
                f"{diag_prefix}_science_zdot_meanstd_only_filtool_scale.fil",
                science_data,
                cfg,
                header_bytes,
                "science zdot mean/std, channel blanked, filtool scale",
            )
    elif zdot_stage not in ("post_replacement", "off", None, ""):
        raise ValueError(f"Unsupported science_zdot_stage: {zdot_stage}")

    if z_data is not None:
        del z_data

    # ---- 4c: Split patches ----
    _sync_device(device)
    t_sub = time.perf_counter()

    all_patches = None
    all_metas = []
    if need_nn_inference:
        all_patches_list = []
        for s in range(n_full_segs):
            patches, meta = split_segment_to_patches(tanh_data[:, s, :], patch_size)
            all_patches_list.append(patches)
            all_metas.append(meta)
        if tail_len >= patch_size:
            patches, meta = split_segment_to_patches(tanh_tail, patch_size)
            all_patches_list.append(patches)
            all_metas.append(meta)

        if all_patches_list:
            all_patches = torch.cat(all_patches_list, dim=0)
        else:
            all_patches = torch.empty(
                0,
                1,
                patch_size,
                patch_size,
                dtype=tanh_data.dtype,
                device=device,
            )
        del all_patches_list
        print(f"  Total patches: {all_patches.shape[0]}")
    else:
        print("  Skipped patch splitting; NN mask is not used or requested")

    del tanh_data
    if tail_len >= patch_size:
        del tanh_tail
    _empty_cuda_cache(cfg)

    _sync_device(device)
    timings["4c_split"] = time.perf_counter() - t_sub

    # ---- 4d: One-shot NN inference ----
    _sync_device(device)
    t_sub = time.perf_counter()

    if need_nn_inference:
        all_mask_patches = infer_patches(model, all_patches, cfg, device)
    else:
        all_mask_patches = None
        print("  Skipped NN inference")
    _sync_device(device)
    timings["4d_inference"] = time.perf_counter() - t_sub
    if all_patches is not None:
        del all_patches
    if model is not None:
        del model
    _empty_cuda_cache(cfg)

    # ---- 4e: Reconstruct masks + replace ----
    _sync_device(device)
    t_sub = time.perf_counter()

    # Full masks are optional diagnostics. Keep them disabled for normal
    # science runs unless they are needed for writing or exact output-mean fill.
    force_masked_output_mean = cfg.get("replacement_force_output_mean", False)
    track_combined_mask = save_mask or force_masked_output_mean
    if track_combined_mask:
        full_mask = torch.zeros(nchans, ntime, dtype=torch.bool, device=device)
    if save_nn_mask:
        full_nn_mask = torch.zeros(nchans, ntime, dtype=torch.bool, device=device)
    if save_blanked_mask:
        full_blanked_mask = torch.zeros(nchans, ntime, dtype=torch.bool, device=device)

    total_rfi_pixels = torch.zeros((), dtype=torch.float64, device=device)
    total_pixels = 0
    segment_channel_flag_total = torch.zeros((), dtype=torch.float64, device=device)
    segment_channel_flag_segments = torch.zeros((), dtype=torch.float64, device=device)
    patch_offset = 0
    n_segs_to_process = n_full_segs + (1 if tail_len >= patch_size else 0)
    replacement_fill_mode = cfg.get("replacement_fill_mode", "clean_median")
    if replacement_fill_mode not in ("zero", "clean_median"):
        raise ValueError(
            "replacement_fill_mode must be one of: zero, clean_median "
            f"(got {replacement_fill_mode!r})"
        )
    zero_fill = torch.zeros((), dtype=science_data.dtype, device=device)
    print(f"  Replacement fill mode: {replacement_fill_mode}")
    print(f"  Use NN mask for replacement: {use_nn_mask_for_replacement}")
    print(f"  Run NN inference: {need_nn_inference}")
    print(f"  Write mask files: {cfg.get('write_mask_files', False)}")

    for s in range(n_segs_to_process):
        if s < n_full_segs:
            start = s * seg_len
            end = start + seg_len
            seg_flagged = science_flagged_per_seg[:, s]
        else:
            start = usable_time
            end = start + usable_tail
            seg_flagged = science_tail_degen if tail_len >= patch_size else science_global_blank

        seg_time = end - start
        if need_nn_inference:
            meta = all_metas[s]
            n_patches_this = int(meta.get(
                "patch_count",
                meta["n_time_patches"] * (
                    meta["full_chan_blocks"] + (1 if meta["has_overlap"] else 0)
                ),
            ))
            nn_seg_mask = reconstruct_segment_mask(
                all_mask_patches[patch_offset:patch_offset + n_patches_this], meta, patch_size)
            patch_offset += n_patches_this
            if use_nn_mask_for_replacement:
                seg_mask = nn_seg_mask
            else:
                seg_mask = torch.zeros_like(nn_seg_mask)
        else:
            nn_seg_mask = torch.zeros(nchans, seg_time, dtype=torch.bool, device=device)
            seg_mask = nn_seg_mask

        if cfg.get("segment_channel_flag_enabled", False):
            seg_channel_flagged = seg_mask.float().mean(dim=1) > cfg["segment_channel_flag_ratio"]
            n_seg_channel = seg_channel_flagged.sum()
            segment_channel_flag_total += n_seg_channel.to(torch.float64)
            segment_channel_flag_segments += (n_seg_channel > 0).to(torch.float64)
            seg_flagged = seg_flagged | seg_channel_flagged

        blanked_seg = (
            science_global_blank.unsqueeze(1).expand(-1, seg_time)
            | seg_flagged.unsqueeze(1).expand(-1, seg_time)
        )
        combined = seg_mask | blanked_seg

        total_rfi_pixels += combined.sum().to(torch.float64)
        total_pixels += nchans * seg_time

        science_seg = science_data[:, start:end]
        if replacement_fill_mode == "zero":
            science_data[:, start:end] = torch.where(combined, zero_fill, science_seg)
        else:
            # Per-channel clean median replacement (constant, no noise).
            work = torch.where(combined, torch.tensor(float('inf'), device=device), science_seg)
            n_clean = (~combined).sum(dim=1)
            work_sorted = work.sort(dim=1).values

            few_clean = n_clean < 10
            safe_n = torch.where(few_clean, torch.tensor(seg_time, device=device), n_clean)
            mid = safe_n // 2
            ch_medians = work_sorted[torch.arange(nchans, device=device), mid]

            if few_clean.any():
                full_sorted = science_seg[few_clean].sort(dim=1).values
                ch_medians[few_clean] = full_sorted[:, seg_time // 2]

            science_data[:, start:end] = torch.where(combined, ch_medians.unsqueeze(1), science_seg)
            del work, work_sorted
        if track_combined_mask:
            full_mask[:, start:end] = combined
        if save_nn_mask:
            full_nn_mask[:, start:end] = nn_seg_mask
        if save_blanked_mask:
            full_blanked_mask[:, start:end] = blanked_seg
        del nn_seg_mask, seg_mask, combined, blanked_seg

    if all_mask_patches is not None:
        del all_mask_patches
    del science_flagged_per_seg

    unprocessed_tail_start = ntime
    if tail_len >= patch_size and tail_len > usable_tail:
        unprocessed_tail_start = usable_time + usable_tail
    elif 0 < tail_len < patch_size:
        unprocessed_tail_start = usable_time

    if unprocessed_tail_start < ntime:
        unprocessed_tail = ntime - unprocessed_tail_start
        science_data[:, unprocessed_tail_start:ntime] = 0.0
        total_rfi_pixels += float(nchans * unprocessed_tail)
        total_pixels += nchans * unprocessed_tail
        if track_combined_mask:
            full_mask[:, unprocessed_tail_start:ntime] = True
        if save_blanked_mask:
            full_blanked_mask[:, unprocessed_tail_start:ntime] = True
        print(
            f"  Unprocessed tail: {unprocessed_tail} samples set to zero and counted as masked"
        )

    if diag_prefix:
        write_z_filterbank_gpu(
            f"{diag_prefix}_combined_replaced_no_channelflag_no_baseline.fil",
            science_data,
            science_global_blank,
            cfg,
            header_bytes,
            "science combined-replaced, no channel-flag, no baseline",
        )
        write_z_filtool_scale_filterbank_gpu(
            f"{diag_prefix}_combined_replaced_no_channelflag_no_baseline_filtool_scale.fil",
            science_data,
            cfg,
            header_bytes,
            "science combined-replaced, no channel-flag, no baseline, filtool scale",
        )

    _sync_device(device)
    timings["4e_replace"] = time.perf_counter() - t_sub

    timings["4_stream_total"] = (
        timings["4a_normalise"]
        + timings["4b_tanh"]
        + timings["4pre_baseline"]
        + timings["4z_zdot"]
        + timings["4c_split"]
        + timings["4d_inference"]
        + timings["4e_replace"]
    )

    if runtime_stats:
        rfi_fraction = (100.0 * total_rfi_pixels / max(1, total_pixels)).item()
        print(f"  Total RFI fraction: {rfi_fraction:.2f}%")
    if runtime_stats and cfg.get("segment_channel_flag_enabled", False):
        print(
            "  Segment channel flags: "
            f"{int(segment_channel_flag_total.item())} channel-segments across "
            f"{int(segment_channel_flag_segments.item())}/{n_segs_to_process} segments "
            f"(>{cfg['segment_channel_flag_ratio']*100:.0f}% NN-mask occupancy)"
        )
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    fully_flagged = science_global_blank
    live_mask = ~fully_flagged

    if fully_flagged.any():
        science_data[fully_flagged, :] = 0.0

    if zdot_stage == "post_replacement":
        print("\n[Step 3z] Science branch zdot after replacement...")
        _sync_device(device)
        t_zdot_post = time.perf_counter()

        need_basis = cfg.get("science_zdot_basis_fil") is not None
        basis = apply_zdot_gpu(
            science_data,
            live_mask,
            cfg,
            label="science zdot post-replacement",
            return_basis=need_basis,
        )

        _sync_device(device)
        timings["4z_post_zdot"] = time.perf_counter() - t_zdot_post

        if basis is not None:
            write_z_filtool_scale_filterbank_gpu(
                cfg.get("science_zdot_basis_fil"),
                basis,
                cfg,
                header_bytes,
                "science post-replacement zdot removed basis, filtool scale",
            )
            del basis
            _empty_cuda_cache(cfg)

        if cfg.get("science_zdot_after_fil") is not None:
            write_z_filtool_scale_filterbank_gpu(
                cfg.get("science_zdot_after_fil"),
                science_data,
                cfg,
                header_bytes,
                "science post-replacement zdot after, filtool scale",
            )

    # =========================================================================
    # STEP 4: Baseline removal
    # =========================================================================
    print("\n[Step 4] Baseline removal...")
    _sync_device(device)
    t_step = time.perf_counter()

    if cfg.get("final_baseline_enabled", True):
        remove_baseline_gpu(science_data, live_mask, fully_flagged, cfg, verbose=runtime_stats)
    else:
        print("  Skipped final baseline removal (final_baseline_enabled=False)")

    _sync_device(device)
    timings["6_baseline_removal"] = time.perf_counter() - t_step


    # =========================================================================
    # STEP 5: Rescale to uint8
    # =========================================================================
    print("\n[Step 5] Rescaling to uint8...")
    _sync_device(device)
    t_step = time.perf_counter()

    print(f"  Rescale mode: {cfg.get('rescale_mode', 'per_channel_mad')}")
    data_u8 = rescale_for_output_gpu(science_data, fully_flagged, cfg)
    del science_data
    _empty_cuda_cache(cfg)

    if force_masked_output_mean:
        full_mask[fully_flagged, :] = True
        masked_output_value = int(round(cfg["out_mean"]))
        masked_output_value = max(0, min(255, masked_output_value))
        data_u8[full_mask] = masked_output_value
        print(f"  Masked output pixels forced to {masked_output_value}")
        if not save_mask:
            del full_mask
            _empty_cuda_cache(cfg)

    _sync_device(device)
    timings["7_rescale"] = time.perf_counter() - t_step
    if runtime_stats:
        print(f"  Range: [{data_u8.min().item()}, {data_u8.max().item()}]")
        print(f"  Mean: {data_u8.float().mean().item():.2f}")

    # =========================================================================
    # STEP 6: Write output .fil
    # =========================================================================
    print(f"\n[Step 6] Writing output: {cfg['output_fil']}")
    t_step = time.perf_counter()

    data_np = data_u8.cpu().numpy()
    del data_u8
    _empty_cuda_cache(cfg)

    data_to_write = data_np.T.ravel()

    output_dir = os.path.dirname(cfg["output_fil"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cfg["output_fil"], 'wb') as f:
        f.write(header_bytes)
        f.write(data_to_write.tobytes())

    # Write mask .fil files (same header, mask as uint8: 0=clean, 1=RFI).
    def write_mask_fil(mask_path, mask_tensor, label):
        print(f"  Writing {label}: {mask_path}")
        mask_dir = os.path.dirname(mask_path)
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)
        mask_u8 = mask_tensor.cpu().numpy().astype(np.uint8).T.ravel()
        with open(mask_path, 'wb') as f:
            f.write(header_bytes)
            f.write(mask_u8.tobytes())

    if save_mask:
        # Combined mask is the exact mask family used by replacement, plus
        # channels fully zeroed by segment/global blanking.
        full_mask[fully_flagged, :] = True
        write_mask_fil(mask_paths["combined"], full_mask, "combined mask")
        del full_mask
    if save_nn_mask:
        # Pure NN mask: do not add blanked/unusable/fully-flagged channels.
        write_mask_fil(mask_paths["nn"], full_nn_mask, "pure NN mask")
        del full_nn_mask
    if save_blanked_mask:
        # Blanked mask: all non-NN masking, including channels fully zeroed
        # by segment/global blanking.
        full_blanked_mask[fully_flagged, :] = True
        write_mask_fil(mask_paths["blanked"], full_blanked_mask, "blanked mask")
        del full_blanked_mask

    timings["8_write_fil"] = time.perf_counter() - t_step

    # =========================================================================
    # TIMING SUMMARY
    # =========================================================================
    total_time = time.perf_counter() - t_total
    timings["total"] = total_time

    gpu_compute_keys = [
        "4_stream_total",
        "6_baseline_removal",
        "7_rescale",
    ]
    if "4z_post_zdot" in timings:
        gpu_compute_keys.append("4z_post_zdot")
    gpu_compute = sum(timings.get(key, 0.0) for key in gpu_compute_keys)
    timings["gpu_compute"] = gpu_compute

    sep = "=" * 64
    print(f"\n{sep}")
    print("PIPELINE TIMING SUMMARY")
    print(sep)
    print(f"{'Step':<35} {'Time (s)':>10} {'%':>7}")
    print("-" * 54)

    step_order = [
        ("1.  Load data (I/O)",           "1_load_data"),
        ("2.  Load model (I/O)",           "3_load_model"),
        ("3.  Norm+infer+replace (total)", "4_stream_total"),
        ("  3a. Normalise (batched)",      "4a_normalise"),
        ("  3b. Tanh compress",            "4b_tanh"),
        ("  3n. Negative 1D cleanup (in 3b)", "4n_negative_cleanup"),
        ("  3q. Negative 2D cleanup (in 3b)", "4q_negative_block_cleanup"),
        ("  3c. Split patches",            "4c_split"),
        ("  3d. NN inference",             "4d_inference"),
        ("  3e. Reconstruct + replace",    "4e_replace"),
        ("4.  Baseline removal",           "6_baseline_removal"),
        ("5.  Rescale to uint8",           "7_rescale"),
        ("6.  Write .fil (I/O)",           "8_write_fil"),
    ]
    if timings.get("4pre_baseline", 0.0) > 0.0:
        step_order.insert(5, ("  3p. Science pre-baseline", "4pre_baseline"))
    if timings.get("4z_zdot", 0.0) > 0.0:
        step_order.insert(5, ("  3z. Science pre-zdot", "4z_zdot"))
    if "4z_post_zdot" in timings:
        step_order.insert(8, ("  3z. Science post-zdot", "4z_post_zdot"))

    for label, key in step_order:
        t = timings[key]
        pct = 100.0 * t / total_time
        print(f"{label:<35} {t:>10.3f} {pct:>6.1f}%")

    print("-" * 54)
    print(f"{'GPU compute':<35} {gpu_compute:>10.3f} {100*gpu_compute/total_time:>6.1f}%")
    print(f"{'I/O + setup':<35} {total_time - gpu_compute:>10.3f} {100*(total_time-gpu_compute)/total_time:>6.1f}%")
    print(f"{'TOTAL':<35} {total_time:>10.3f} {'100.0%':>7}")
    print(sep)

    # Data throughput
    data_bytes = nchans * ntime * int(cfg.get("_raw_gpu_element_size", 4))
    print(f"\nData size: {data_bytes / 1e9:.2f} GB ({nchans} x {ntime})")
    print(f"GPU compute throughput: {data_bytes / gpu_compute / 1e9:.2f} GB/s")
    print(f"End-to-end throughput: {data_bytes / total_time / 1e9:.2f} GB/s")
    print(f"Real-time factor: {ntime * tsamp / total_time:.1f}x "
          f"({ntime * tsamp:.1f}s of observation in {total_time:.1f}s)")

    input_size = os.path.getsize(cfg["input_fil"])
    output_size = os.path.getsize(cfg["output_fil"])
    print(f"\nInput size:  {input_size:,} bytes")
    print(f"Output size: {output_size:,} bytes")
    if input_size == output_size:
        print("Sizes match!")
    else:
        print(f"Size difference: {output_size - input_size} bytes")

    return timings


def run_pipeline(cfg):
    """Run mitigation with explicit, non-leaking inference backend settings."""

    with inference_runtime_settings(cfg):
        return _run_pipeline_impl(cfg)


if __name__ == "__main__":
    raise SystemExit("Use `mars-mitigate --help` to run the pipeline.")
