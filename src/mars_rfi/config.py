"""Paper-aligned defaults for the MARS training stack.

The historical development repository reused this file for whichever ablation
was running most recently.  In the public repository, ``CONFIG`` is instead the
configuration of the 270,769-parameter model described in the paper.  Ablations
live in ``configs/`` and must be selected explicitly.
"""

from __future__ import annotations

from .provenance import PAPER_ARTIFACT_ROLE, PAPER_EXPERIMENT_ID, training_fingerprint


CONFIG = {
    # Immutable identity for artifacts used by the paper-facing pipeline.
    "experiment_id": PAPER_EXPERIMENT_ID,
    "artifact_role": PAPER_ARTIFACT_ROLE,

    # Data paths. The input patches are expected to already be in tanh domain.
    "train_patch_path": "data/training_noise/train_patches.npy",
    "train_mask_path": "data/training_noise/train_masks.npy",
    "val_patch_path": "data/training_noise/val_patches.npy",
    "val_mask_path": "data/training_noise/val_masks.npy",
    "expected_train_examples": 4_800,
    "expected_val_examples": 1_200,

    # Checkpoints and logs.
    "out_dir": "artifacts/checkpoints/mars-paper",

    # Model.
    # Model choices:
    #   "light_unet_v1"  : depthwise + GroupNorm + axis context.
    #   "light_unet_v2"  : depthwise + GroupNorm + anisotropic context.
    #   "trt_fast_unet"  : TensorRT-friendly Conv-BN-ReLU model.
    #   "trt_shape_unet" : Conv-BN-ReLU plus 3x3 / 1x9 / 9x1 RFI kernels.
    "model": "trt_shape_unet",
    "in_channels": 1,
    "out_channels": 1,
    # Widths reported for MARS in the paper.
    "channels": (8, 16, 32, 64),
    "output_bias_prior": 0,
    "axis_reduction": 4,
    "anisotropic_kernel": 9,
    "bottleneck_dilation": 2,
    "shape_kernel": 9,
    # Paper horizontal decoder refinement at every reconstruction scale.
    "decoder_horizontal_refine_enabled": True,
    "decoder_horizontal_refine_stages": ("up2", "up1", "up0"),
    "decoder_horizontal_refine_kernel": 31,
    # Vertical decoder refinement for short broadband/vertical RFI.
    "decoder_vertical_refine_enabled": True,
    "decoder_vertical_refine_stages": ("up2", "up1", "up0"),
    "decoder_vertical_refine_kernel": 31,

    # Training.
    "epochs": 50,
    "batch_size": 64,
    "optimizer": "AdamW",
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "pos_weight": 3.0,
    "lambda_dice": 1.0,
    "lambda_astro": 1.0,
    "focal_gamma": 0.0,
    "clip_grad_norm": 1.0,
    "amp": True,
    "seed": 1234,
    "threshold": 0.5,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 4,
    "scheduler_min_lr": 1.0e-6,
    # Guard against silently training an ablation under the paper model name.
    "expected_parameters": 270_769,
    "log_every": 50,

    # DataLoader. The packed collate keeps each batch in one shared storage,
    # which makes multiprocessing much less fragile than returning many tensors.
    "train_num_workers": 8,
    "val_num_workers": 8,
    "pin_memory": False,
    "persistent_workers": True,
    "prefetch_factor": 1,
    "packed_collate": True,
    "sharing_strategy": "file_descriptor",
    "worker_start_method": "spawn",

    # Augmentation is intentionally compact and z-domain based:
    #   real_input ~= tanh(z / tanh_scale)
    #   synthetic = tanh((atanh(real_input) * tanh_scale + rfi_z) / tanh_scale)
    "augmentation": {
        "enabled": True,
        "augment_val": False,
        "tanh_scale": 6.0,
        "input_clip": 0.999,
        "mask_threshold": 0.5,
        "base_noise_std": 0.003,
        "background_negative_band_prob": 0.0,
        "background_negative_band": {
            "bands": (1, 3),
            "length": (360, 512),
            "height": (1, 30),
            "z_drop": (0.4, 2.2),
            "soft_edge": (2, 8),
        },
        # Clean samples keep the original patch/base mask and skip synthetic
        # RFI. This gives the model real-background hard negatives.
        "clean_sample_prob": 0.4,
        "astro_injection": {
            "enabled": True,
            "frb_prob": 0.75,
            "frb_max": 2,
            "frb_brightness_range": (0.2, 0.95),
            "frb_brightness_log_uniform": True,
            # Primary FRB width sampling is now physical: sample a width in
            # milliseconds, then convert to bins using the selected tsamp.
            # frb_pulse_width_range is kept as a bin-based fallback if the
            # physical range is set to None.
            "frb_pulse_width_ms_range": (1.0, 10.0),
            "frb_pulse_width_range": (1, 15),
            "faint_frb_boost": 0.15,
            "faint_frb_range": (0.02, 0.15),
            # Small bright-tail and low-DM pulse branches target the FRB/pulse
            # preservation failures seen in high-SNR, narrow, low-sweep tests.
            "bright_frb_boost": 0.08,
            "bright_frb_range": (0.95, 0.999),
            "bright_frb_brightness_log_uniform": True,
            "low_dm_pulse_boost": 0.12,
            "low_dm_pulse_dm_range": (10.0, 150.0),
            "low_dm_pulse_width_range": (1, 5),
            "low_dm_pulse_width_ms_range": (1.0, 5.0),
            "low_dm_pulse_brightness_range": (0.7, 0.999),
            "low_dm_pulse_brightness_log_uniform": True,
            # Targeted branch for bright, medium-width, low-DM pulses:
            # DM~50, very bright, medium-width pulses around 3-6 ms.
            "low_dm_bright_medium_pulse_boost": 0.08,
            "low_dm_bright_medium_pulse_dm_range": (30.0, 100.0),
            "low_dm_bright_medium_pulse_width_range": (3, 6),
            "low_dm_bright_medium_pulse_width_ms_range": (3.0, 6.0),
            "low_dm_bright_medium_pulse_brightness_range": (0.92, 0.999),
            "low_dm_bright_medium_pulse_brightness_log_uniform": True,
            "frb_brightness_fixed": None,
            "frb_dm_range": (10.0, 3000.0),
            "frb_dm_fixed": None,
            "frb_mask_relative_threshold": 0.01,
            "frb_mask_absolute_floor": 1.0e-3,
            "preserve_frb_prob": 0.0,
            "clean_sample_inject_frb": True,
            # Keep FRB tracks continuous across frequency. Brightness variation
            # is handled by the smooth frequency modulation below.
            "frb_spectral_occupancy_range": (1.0, 1.0),
            "frb_scintillation_prob": 0.0,
            "frb_scintillation_drop_range": (0.05, 0.4),
            "frb_freq_mod_prob": 0.6,
            "frb_freq_mod_depth_range": (0.2, 0.8),
            "frb_scattering_prob": 0.5,
            "frb_scattering_tau_range": (0.5, 15.0),
            "frb_scattering_index": -4.0,
            "frb_sub_burst_prob": 0.3,
            "frb_sub_burst_count_range": (2, 5),
            "frb_sub_burst_drift_range": (-0.15, -0.01),
            "frb_spectral_index_range": (-3.0, 1.0),
            "telescope_configs": None,
            "time_resolutions_us": [128, 256, 512, 1024, 1310],
        },
        # The paper configuration is intentionally FRB-protective. These settings restore recall
        # for complex horizontal RFI by showing more horizontal/mixed-family
        # positives without lowering the global inference threshold.
        "extra_family_prob": 0.35,
        "max_families_per_patch": 3,
        "family_weights": {
            "complex_horizontal": 5.5,
            "bright_complex_horizontal": 2.4,
            "sparse_bright_horizontal_bars": 3.0,
            "periodic_horizontal_tick_train": 1.8,
            "persistent_narrowband": 1.4,
            "diffuse_persistent_bright_band": 1.6,
            "compact_spots": 0.8,
            "rect_blocks": 1.2,
            "vertical_bursts": 1.0,
            "periodic_stripes": 0.4,
        },
        "positive_weights": {
            "complex_horizontal": 3.4,
            "bright_complex_horizontal": 4.0,
            "sparse_bright_horizontal_bars": 4.2,
            "periodic_horizontal_tick_train": 3.5,
            "persistent_narrowband": 2.2,
            "diffuse_persistent_bright_band": 2.8,
            "compact_spots": 2.0,
            "rect_blocks": 2.0,
            "vertical_bursts": 1.8,
            "periodic_stripes": 1.5,
        },
        "complex_horizontal": {
            "groups": (1, 3),
            "components_per_group": (2, 5),
            "broken_streak_prob": 0.0,
            "time_span": (70, 230),
            "freq_span": (5, 24),
            "thin_len": (30, 130),
            "thick_len": (22, 90),
            "height": (1, 6),
            "z_amp": (0.75, 3.2),
        },
        # Targeted hard family for the observed failure mode:
        # dense bright horizontal ridges, local blocky cores, shallow broken
        # sections, and optional diffuse pedestal near patch edges. The mask
        # intentionally follows the full physical horizontal structure rather
        # than only the brightest cores.
        "bright_complex_horizontal": {
            "clusters": (1, 2),
            "edge_bias_prob": 0.70,
            "edge_max_fraction": 0.35,
            "time_span": (180, 512),
            "freq_span": (45, 150),
            "ridges": (5, 12),
            "ridge_len": (70, 330),
            "ridge_height": (1, 4),
            "ridge_z_amp": (1.6, 3.4),
            "ridge_profile_low": 0.68,
            "ridge_texture": 0.14,
            "broken_prob": 0.80,
            "broken_keep_prob": 0.82,
            "broken_min_seg": 8,
            "broken_max_seg": 55,
            "broken_gap_level": (0.15, 0.55),
            "parallel_repeat_prob": 0.45,
            "blocks": (1, 4),
            "block_len": (35, 95),
            "block_height": (10, 34),
            "block_z_amp": (1.5, 3.0),
            "block_texture": 0.22,
            "pedestal_prob": 0.75,
            "pedestal_len": (180, 512),
            "pedestal_height": (14, 70),
            "pedestal_z_amp": (0.18, 0.55),
            "pedestal_profile_low": 0.52,
            "pedestal_texture": 0.18,
            "pedestal_mask_prob": 0.60,
            "mask_edge": 1,
        },
        # Targeted family for the observed residual mode: very bright sparse
        # horizontal bars whose full extent should be masked, not only the
        # terminal/core peak.
        "sparse_bright_horizontal_bars": {
            "bars": (1, 3),
            "parallel_prob": 0.45,
            "parallel_gap": (4, 20),
            "time_edge_bias_prob": 0.60,
            "length": (140, 512),
            "height": (1, 4),
            "z_amp": (1.8, 4.8),
            "line_profile_low": 0.88,
            "line_texture": 0.08,
            "core_prob": 0.92,
            "core_len": (10, 80),
            "core_height": (2, 12),
            "core_z_amp": (2.8, 6.0),
            "core_texture": 0.12,
            "core_side": "either",
            "mask_edge": 1,
        },
        # Targeted hard family for periodic tick-like RFI: short vertical
        # bursts repeating along one horizontal frequency level. The faint
        # context band is usually unmasked; the supervised target is the
        # visible periodic ticks that were missed in diagnostics.
        "periodic_horizontal_tick_train": {
            "trains": (1, 3),
            "edge_bias_prob": 0.35,
            "edge_max_fraction": 0.30,
            "time_span": (160, 512),
            "freq_span": (18, 64),
            "context_prob": 0.75,
            "context_height": (8, 28),
            "context_z_amp": (0.10, 0.30),
            "context_profile_low": 0.65,
            "context_texture": 0.22,
            "context_mask_prob": 0.0,
            "spacing": (8, 20),
            "phase_count": (1, 2),
            "time_jitter": (0.6, 2.5),
            "y_jitter": (0.4, 2.0),
            "dropout": (0.05, 0.25),
            "ticks_per_train_max": 96,
            "time_width": (1, 3),
            "freq_len": (4, 22),
            "z_amp": (0.8, 2.6),
            "strong_tick_prob": 0.18,
            "strong_z_amp": (2.0, 3.6),
            "tick_texture": 0.12,
            "mask_edge": 1,
        },
        "persistent_narrowband": {
            "bands": (1, 4),
            "length": (120, 512),
            "height": (1, 14),
            "z_amp": (0.75, 2.5),
        },
        "diffuse_persistent_bright_band": {
            "bands": (1, 3),
            "length": (360, 512),
            "height": (4, 28),
            "pedestal_z_amp": (0.25, 1.05),
            "ridge_count": (1, 3),
            "ridge_height": (1, 3),
            "ridge_z_amp": (0.65, 2.2),
            "soft_edge": (2, 8),
            "texture": (0.04, 0.18),
        },
        "compact_spots": {
            "groups": (4, 18),
            "cluster_prob": 0.35,
            "cluster_size": (2, 6),
            "cluster_time_std": (3.0, 12.0),
            "cluster_freq_std": (2.0, 10.0),
            "radius_t": (1.2, 5.5),
            "radius_f": (1.2, 4.5),
            "z_amp": (0.18, 2.6),
            "faint_prob": 0.25,
            "faint_z_amp": (0.10, 0.35),
            "mask_rel_threshold": 0.22,
        },
        "rect_blocks": {
            "blocks": (1, 5),
            "time_len": (8, 80),
            "freq_len": (4, 28),
            "z_amp": (0.9, 3.0),
        },
        "vertical_bursts": {
            "bursts": (1, 5),
            "time_width": (1, 8),
            "freq_len": (30, 220),
            "z_amp": (0.75, 2.4),
        },
        "periodic_stripes": {
            "stripes": (3, 10),
            "height": (1, 3),
            "z_amp": (0.6, 1.6),
        },
    },
}

# Stored in every checkpoint and checked again by export/inference. Recompute
# after the full nested augmentation dictionary has been defined.
CONFIG["training_fingerprint"] = training_fingerprint(CONFIG)
