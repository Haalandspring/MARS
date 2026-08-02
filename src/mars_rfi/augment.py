"""Fast, clear-boundary RFI augmentation for tanh-normalized patches.

The real pipeline feeds the network tanh-compressed normalized data. Painting
RFI directly in that compressed space changes the background distribution. This
module instead maps input patches back to an approximate z-domain, adds clear
RFI structures there, and maps the result back through tanh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray

TIME_RESOLUTIONS_US = [128, 256, 512, 1024, 1310]

TELESCOPE_CONFIGS = [
    {"fch1": 500,  "bandwidth": 200, "nchans": 512},
    {"fch1": 500,  "bandwidth": 200, "nchans": 1024},
    {"fch1": 500,  "bandwidth": 200, "nchans": 4096},
    {"fch1": 700,  "bandwidth": 200, "nchans": 512},
    {"fch1": 700,  "bandwidth": 200, "nchans": 1024},
    {"fch1": 700,  "bandwidth": 200, "nchans": 4096},
    {"fch1": 900,  "bandwidth": 200, "nchans": 512},
    {"fch1": 900,  "bandwidth": 200, "nchans": 1024},
    {"fch1": 900,  "bandwidth": 200, "nchans": 4096},
    {"fch1": 1100, "bandwidth": 200, "nchans": 512},
    {"fch1": 1100, "bandwidth": 200, "nchans": 1024},
    {"fch1": 1100, "bandwidth": 200, "nchans": 4096},
    {"fch1": 1300, "bandwidth": 200, "nchans": 512},
    {"fch1": 1300, "bandwidth": 200, "nchans": 1024},
    {"fch1": 1300, "bandwidth": 200, "nchans": 4096},
    {"fch1": 1500, "bandwidth": 200, "nchans": 512},
    {"fch1": 1500, "bandwidth": 200, "nchans": 1024},
    {"fch1": 1500, "bandwidth": 200, "nchans": 4096},
]


@dataclass(frozen=True)
class AugmentResult:
    patch: Array
    mask: Array
    weight: Array
    astro_mask: Array
    family: str


def _rand_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = int(bounds[0]), int(bounds[1])
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))


def _rand_float(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _choose_family(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = [name for name, weight in weights.items() if float(weight) > 0.0]
    if not names:
        raise ValueError("augmentation.family_weights has no positive entries")
    vals = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    vals /= vals.sum()
    return str(rng.choice(names, p=vals))


def _to_z(x: Array, scale: float, clip: float) -> Array:
    x = np.clip(x.astype(np.float32, copy=False), -clip, clip)
    return np.arctanh(x) * np.float32(scale)


def _from_z(z: Array, scale: float) -> Array:
    return np.tanh(z.astype(np.float32, copy=False) / np.float32(scale))


def _smooth_profile(rng: np.random.Generator, n: int, low: float = 0.85) -> Array:
    if n <= 1:
        return np.ones((max(1, n),), dtype=np.float32)
    anchors = rng.uniform(low, 1.0, size=max(3, min(9, n // 16 + 3))).astype(np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, n, dtype=np.float32),
        np.linspace(0.0, 1.0, anchors.size, dtype=np.float32),
        anchors,
    ).astype(np.float32)


def _smooth_vec(rng: np.random.Generator, n: int, k: int) -> Array:
    if n <= 1:
        return np.ones((max(1, n),), dtype=np.float32)
    k = max(1, min(int(k), int(n)))
    x = rng.normal(size=n).astype(np.float32)
    kernel = np.ones(k, dtype=np.float32) / np.float32(k)
    y = np.convolve(x, kernel, mode="same")
    y = np.abs(y).astype(np.float32, copy=False)
    ymax = float(y.max())
    if ymax > 0.0:
        y = y / np.float32(ymax)
    return y.astype(np.float32, copy=False)


def _choose_telescope_config(rng: np.random.Generator, configs) -> dict:
    pool = TELESCOPE_CONFIGS if configs is None else list(configs)
    if not pool:
        pool = TELESCOPE_CONFIGS
    return dict(pool[int(rng.integers(0, len(pool)))])


def _choose_time_resolution_us(rng: np.random.Generator, values) -> float:
    pool = TIME_RESOLUTIONS_US if values is None else list(values)
    if not pool:
        pool = TIME_RESOLUTIONS_US
    return float(pool[int(rng.integers(0, len(pool)))])


def _sample_pulse_width_bins(
    rng: np.random.Generator,
    *,
    pulse_width_range: tuple[float, float],
    pulse_width_ms_range: tuple[float, float] | None,
    time_resolution_us: float,
) -> tuple[float, float, str]:
    if pulse_width_ms_range is not None:
        width_ms = rng.uniform(float(pulse_width_ms_range[0]), float(pulse_width_ms_range[1]))
        width_bins = width_ms / max(float(time_resolution_us) / 1000.0, 1.0e-9)
        return max(float(width_bins), 0.5), float(width_ms), "physical_ms"

    width_bins = rng.uniform(float(pulse_width_range[0]), float(pulse_width_range[1]))
    width_ms = float(width_bins) * float(time_resolution_us) / 1000.0
    return float(width_bins), float(width_ms), "bins"


def _optional_range(value) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


def generate_synthetic_frb(
    H: int,
    W: int,
    rng: np.random.Generator,
    dm: float | None = None,
    dm_range: tuple[float, float] = (10.0, 3000.0),
    t0: float | None = None,
    pulse_width: float | None = None,
    brightness: float | None = None,
    telescope_config: dict | None = None,
    time_resolution_us: float | None = None,
    brightness_range: tuple[float, float] = (0.01, 0.8),
    brightness_log_uniform: bool = True,
    pulse_width_range: tuple[float, float] = (1, 15),
    pulse_width_ms_range: tuple[float, float] | None = None,
    mask_relative_threshold: float = 0.01,
    mask_absolute_floor: float = 1.0e-3,
    telescope_configs=None,
    time_resolutions_us=None,
    spectral_occupancy_range: tuple[float, float] = (0.3, 1.0),
    scintillation_prob: float = 0.5,
    scintillation_drop_range: tuple[float, float] = (0.05, 0.4),
    freq_mod_prob: float = 0.6,
    freq_mod_depth_range: tuple[float, float] = (0.2, 0.8),
    scattering_prob: float = 0.5,
    scattering_tau_range: tuple[float, float] = (0.5, 15.0),
    scattering_index: float = -4.0,
    sub_burst_prob: float = 0.3,
    sub_burst_count_range: tuple[int, int] = (2, 5),
    sub_burst_drift_range: tuple[float, float] = (-0.15, -0.01),
    spectral_index_range: tuple[float, float] = (-3.0, 1.0),
) -> tuple[Array, Array, dict]:
    """Generate the single-burst FRB family as a tanh-domain signal image."""

    H = int(H)
    W = int(W)
    if telescope_config is None:
        telescope_config = _choose_telescope_config(rng, telescope_configs)
    if time_resolution_us is None:
        time_resolution_us = _choose_time_resolution_us(rng, time_resolutions_us)
    time_resolution_ms = float(time_resolution_us) / 1000.0
    if pulse_width is None:
        pulse_width, pulse_width_ms, pulse_width_unit = _sample_pulse_width_bins(
            rng,
            pulse_width_range=pulse_width_range,
            pulse_width_ms_range=pulse_width_ms_range,
            time_resolution_us=float(time_resolution_us),
        )
    else:
        pulse_width = float(pulse_width)
        pulse_width_ms = pulse_width * time_resolution_ms
        pulse_width_unit = "explicit_bins"
    if brightness is None:
        br_min, br_max = float(brightness_range[0]), float(brightness_range[1])
        br_min = max(br_min, 1.0e-6)
        if brightness_log_uniform:
            brightness = float(np.exp(rng.uniform(np.log(br_min), np.log(br_max))))
        else:
            brightness = float(rng.uniform(br_min, br_max))

    fch1 = float(telescope_config["fch1"])
    bandwidth = float(telescope_config["bandwidth"])
    nchans = int(telescope_config["nchans"])
    foff = -(bandwidth / float(nchans))
    freqs = fch1 + np.arange(H, dtype=np.float32) * np.float32(foff)
    freq_max = float(freqs[0])
    freq_min = float(freqs[-1])

    for _ in range(50):
        trial_dm = float(dm) if dm is not None else float(rng.uniform(float(dm_range[0]), float(dm_range[1])))
        sweep_s = 4.149e3 * trial_dm * (freq_min**-2 - freq_max**-2)
        if (sweep_s * 1.0e3) / time_resolution_ms < 10 * W:
            break
    else:
        if dm is None:
            trial_dm = float(np.clip(800.0, float(dm_range[0]), float(dm_range[1])))
    dm = trial_dm

    max_delay_ms = 4.149e3 * dm * (freq_min**-2 - freq_max**-2) * 1.0e3
    max_delay_px = max_delay_ms / time_resolution_ms
    if t0 is None:
        t0 = float(rng.uniform(-max_delay_px, W))

    delay_px = 4.149e3 * dm * (freqs.astype(np.float64)**-2 - freq_max**-2) * 1.0e3 / time_resolution_ms
    t_centers = np.float32(t0) + delay_px.astype(np.float32)

    spec_alpha = float(rng.uniform(float(spectral_index_range[0]), float(spectral_index_range[1])))
    freq_ref = 0.5 * (freq_max + freq_min)
    spec_scaling = (freqs.astype(np.float64) / freq_ref) ** spec_alpha
    spec_scaling = spec_scaling.astype(np.float32)

    # Retain the original FRB-negative pass to clean dispersion tracks.
    # The old scattering branch can turn low-sweep, wide-pulse cases into
    # blob-like FRBs that confuse the RFI/background decision boundary.
    apply_scattering = False
    # apply_scattering = rng.random() < float(scattering_prob)
    tau_ref = float(rng.uniform(float(scattering_tau_range[0]), float(scattering_tau_range[1]))) if apply_scattering else 0.0

    # Likewise disable multi-component sub-bursts for now; they can produce
    # compact bright clumps rather than the thin pulse-like negatives we want.
    apply_sub_burst = False
    # apply_sub_burst = rng.random() < float(sub_burst_prob)
    if apply_sub_burst:
        n_sub = int(rng.integers(int(sub_burst_count_range[0]), int(sub_burst_count_range[1]) + 1))
        drift_rate = float(rng.uniform(float(sub_burst_drift_range[0]), float(sub_burst_drift_range[1])))
        sub_spacing = float(rng.uniform(pulse_width * 1.5, pulse_width * 5.0))
        sub_brightnesses = rng.uniform(0.4, 1.0, size=n_sub).astype(np.float32)
        sub_brightnesses /= max(float(sub_brightnesses.max()), 1.0e-6)
    else:
        n_sub = 1
        drift_rate = 0.0
        sub_spacing = 0.0
        sub_brightnesses = np.array([1.0], dtype=np.float32)

    frb_image = np.zeros((H, W), dtype=np.float32)
    time_axis = np.arange(W, dtype=np.float32)

    for s in range(n_sub):
        t_offset = s * sub_spacing
        for row in range(H):
            freq_drift_offset = -drift_rate * (row - H // 2) if apply_sub_burst else 0.0
            tc = float(t_centers[row]) + t_offset + freq_drift_offset
            row_step = abs(float(t_centers[min(row + 1, H - 1)]) - float(t_centers[max(row - 1, 0)])) / 2.0
            ew = max(float(pulse_width), row_step, 0.5)
            if -3.0 * ew < tc < W + 3.0 * ew:
                amp = float(brightness) * float(sub_brightnesses[s]) * float(spec_scaling[row])
                profile = amp * np.exp(-0.5 * ((time_axis - np.float32(tc)) / np.float32(ew)) ** 2)
                profile = profile.astype(np.float32, copy=False)

                if apply_scattering and tau_ref > 0.3:
                    tau_row = tau_ref * (float(freqs[row]) / freq_ref) ** float(scattering_index)
                    tau_row = max(float(tau_row), 0.3)
                    kern_len = min(int(5.0 * tau_row) + 1, W)
                    kern = np.exp(-np.arange(kern_len, dtype=np.float32) / np.float32(tau_row))
                    kern /= max(float(kern.sum()), 1.0e-12)
                    profile = np.convolve(profile, kern, mode="full")[:W].astype(np.float32, copy=False)

                frb_image[row] += profile

    freq_mod = np.ones(H, dtype=np.float32)
    occ_lo, occ_hi = float(spectral_occupancy_range[0]), float(spectral_occupancy_range[1])
    occupancy = float(rng.uniform(occ_lo, occ_hi))
    if occupancy < 1.0:
        band_len = max(1, int(round(occupancy * H)))
        start = int(rng.integers(0, H - band_len + 1))
        occ_mask = np.zeros(H, dtype=np.float32)
        occ_mask[start:start + band_len] = 1.0
        taper = max(1, int(0.05 * band_len))
        for i in range(taper):
            alpha = np.float32((i + 1) / (taper + 1))
            if start + i < H:
                occ_mask[start + i] *= alpha
            end = start + band_len - 1 - i
            if end >= 0:
                occ_mask[end] *= alpha
        freq_mod *= occ_mask

    if rng.random() < float(scintillation_prob):
        drop_frac = float(rng.uniform(float(scintillation_drop_range[0]), float(scintillation_drop_range[1])))
        freq_mod *= (rng.random(H) > drop_frac).astype(np.float32)

    if rng.random() < float(freq_mod_prob):
        depth = float(rng.uniform(float(freq_mod_depth_range[0]), float(freq_mod_depth_range[1])))
        envelope = _smooth_vec(rng, H, k=max(8, H // 16))
        freq_mod *= (1.0 - depth) + depth * envelope

    frb_image *= freq_mod[:, None]
    threshold = max(float(mask_relative_threshold) * float(brightness), float(mask_absolute_floor))
    frb_mask = (frb_image > np.float32(threshold)).astype(np.float32)
    params = {
        "type": "FRB",
        "dm": float(dm),
        "t0": float(t0),
        "pulse_width": float(pulse_width),
        "pulse_width_ms": float(pulse_width_ms),
        "pulse_width_unit": pulse_width_unit,
        "brightness": float(brightness),
        "fch1": fch1,
        "bandwidth": bandwidth,
        "foff": foff,
        "freq_max": freq_max,
        "freq_min": freq_min,
        "time_resolution_us": float(time_resolution_us),
        "sweep_pixels": float(max_delay_px),
    }
    return frb_image.astype(np.float32, copy=False), frb_mask, params


def inject_frbs(
    H: int,
    W: int,
    rng: np.random.Generator,
    max_frbs: int = 2,
    prob: float = 0.5,
    brightness_range: tuple[float, float] = (0.01, 0.8),
    brightness_log_uniform: bool = True,
    pulse_width_range: tuple[float, float] = (1, 15),
    pulse_width_ms_range: tuple[float, float] | None = None,
    faint_frb_boost: float = 0.0,
    faint_frb_range: tuple[float, float] = (0.03, 0.15),
    bright_frb_boost: float = 0.0,
    bright_frb_range: tuple[float, float] = (0.9, 0.999),
    bright_frb_brightness_log_uniform: bool = True,
    low_dm_pulse_boost: float = 0.0,
    low_dm_pulse_dm_range: tuple[float, float] = (10.0, 150.0),
    low_dm_pulse_width_range: tuple[float, float] = (1, 5),
    low_dm_pulse_width_ms_range: tuple[float, float] | None = None,
    low_dm_pulse_brightness_range: tuple[float, float] = (0.7, 0.999),
    low_dm_pulse_brightness_log_uniform: bool = True,
    low_dm_bright_medium_pulse_boost: float = 0.0,
    low_dm_bright_medium_pulse_dm_range: tuple[float, float] = (30.0, 100.0),
    low_dm_bright_medium_pulse_width_range: tuple[float, float] = (3, 6),
    low_dm_bright_medium_pulse_width_ms_range: tuple[float, float] | None = None,
    low_dm_bright_medium_pulse_brightness_range: tuple[float, float] = (0.9, 0.999),
    low_dm_bright_medium_pulse_brightness_log_uniform: bool = True,
    brightness_fixed: float | None = None,
    dm_fixed: float | None = None,
    dm_range: tuple[float, float] = (10.0, 3000.0),
    mask_relative_threshold: float = 0.01,
    mask_absolute_floor: float = 1.0e-3,
    telescope_configs=None,
    time_resolutions_us=None,
    spectral_occupancy_range: tuple[float, float] = (0.3, 1.0),
    scintillation_prob: float = 0.5,
    scintillation_drop_range: tuple[float, float] = (0.05, 0.4),
    freq_mod_prob: float = 0.6,
    freq_mod_depth_range: tuple[float, float] = (0.2, 0.8),
    scattering_prob: float = 0.5,
    scattering_tau_range: tuple[float, float] = (0.5, 15.0),
    scattering_index: float = -4.0,
    sub_burst_prob: float = 0.3,
    sub_burst_count_range: tuple[int, int] = (2, 5),
    sub_burst_drift_range: tuple[float, float] = (-0.15, -0.01),
    spectral_index_range: tuple[float, float] = (-3.0, 1.0),
) -> tuple[Array, Array, list[dict]]:
    frb_image = np.zeros((int(H), int(W)), dtype=np.float32)
    frb_mask = np.zeros((int(H), int(W)), dtype=np.float32)
    params = []
    if rng.random() > float(prob):
        return frb_image, frb_mask, params

    for _ in range(int(rng.integers(1, int(max_frbs) + 1))):
        this_dm_range = dm_range
        this_pulse_width_range = pulse_width_range
        this_pulse_width_ms_range = pulse_width_ms_range
        sample_profile = "regular_frb"
        if brightness_fixed is not None:
            this_brightness = float(brightness_fixed)
            this_range = (this_brightness, this_brightness)
            this_log = False
            sample_profile = "fixed_brightness_frb"
        else:
            medium_prob = max(0.0, float(low_dm_bright_medium_pulse_boost))
            low_prob = max(0.0, float(low_dm_pulse_boost))
            bright_prob = max(0.0, float(bright_frb_boost))
            faint_prob = max(0.0, float(faint_frb_boost))
            u = float(rng.random())
            this_brightness = None
            if u < medium_prob:
                this_range = low_dm_bright_medium_pulse_brightness_range
                this_log = low_dm_bright_medium_pulse_brightness_log_uniform
                this_dm_range = low_dm_bright_medium_pulse_dm_range
                this_pulse_width_range = low_dm_bright_medium_pulse_width_range
                this_pulse_width_ms_range = low_dm_bright_medium_pulse_width_ms_range
                sample_profile = "low_dm_bright_medium_pulse"
            elif u < medium_prob + low_prob:
                this_range = low_dm_pulse_brightness_range
                this_log = low_dm_pulse_brightness_log_uniform
                this_dm_range = low_dm_pulse_dm_range
                this_pulse_width_range = low_dm_pulse_width_range
                this_pulse_width_ms_range = low_dm_pulse_width_ms_range
                sample_profile = "low_dm_pulse"
            elif u < medium_prob + low_prob + bright_prob:
                this_range = bright_frb_range
                this_log = bright_frb_brightness_log_uniform
                sample_profile = "bright_frb"
            elif u < medium_prob + low_prob + bright_prob + faint_prob:
                this_range = faint_frb_range
                this_log = True
                sample_profile = "faint_frb"
            else:
                this_range = brightness_range
                this_log = brightness_log_uniform

        frb = np.zeros_like(frb_image)
        mask = np.zeros_like(frb_mask)
        info = {}
        for _attempt in range(5):
            frb, mask, info = generate_synthetic_frb(
                H=int(H),
                W=int(W),
                rng=rng,
                dm=dm_fixed,
                dm_range=this_dm_range,
                brightness=this_brightness,
                brightness_range=this_range,
                brightness_log_uniform=this_log,
                pulse_width_range=this_pulse_width_range,
                pulse_width_ms_range=this_pulse_width_ms_range,
                mask_relative_threshold=mask_relative_threshold,
                mask_absolute_floor=mask_absolute_floor,
                telescope_configs=telescope_configs,
                time_resolutions_us=time_resolutions_us,
                spectral_occupancy_range=spectral_occupancy_range,
                scintillation_prob=scintillation_prob,
                scintillation_drop_range=scintillation_drop_range,
                freq_mod_prob=freq_mod_prob,
                freq_mod_depth_range=freq_mod_depth_range,
                scattering_prob=scattering_prob,
                scattering_tau_range=scattering_tau_range,
                scattering_index=scattering_index,
                sub_burst_prob=sub_burst_prob,
                sub_burst_count_range=sub_burst_count_range,
                sub_burst_drift_range=sub_burst_drift_range,
                spectral_index_range=spectral_index_range,
            )
            if float(frb.sum()) > 0.0:
                break
        frb_image += frb
        frb_mask = np.maximum(frb_mask, mask)
        info = dict(info)
        info["sample_profile"] = sample_profile
        params.append(info)

    return frb_image.astype(np.float32, copy=False), frb_mask.astype(np.float32, copy=False), params


def _inject_frb_from_cfg(H: int, W: int, rng: np.random.Generator, cfg: dict):
    return inject_frbs(
        H=H,
        W=W,
        rng=rng,
        max_frbs=int(cfg.get("frb_max", 2)),
        prob=float(cfg.get("frb_prob", 0.5)),
        brightness_range=tuple(cfg.get("frb_brightness_range", (0.01, 0.8))),
        brightness_log_uniform=bool(cfg.get("frb_brightness_log_uniform", True)),
        pulse_width_range=tuple(cfg.get("frb_pulse_width_range", (1, 15))),
        pulse_width_ms_range=_optional_range(cfg.get("frb_pulse_width_ms_range", None)),
        faint_frb_boost=float(cfg.get("faint_frb_boost", 0.0)),
        faint_frb_range=tuple(cfg.get("faint_frb_range", (0.03, 0.15))),
        bright_frb_boost=float(cfg.get("bright_frb_boost", 0.0)),
        bright_frb_range=tuple(cfg.get("bright_frb_range", (0.9, 0.999))),
        bright_frb_brightness_log_uniform=bool(cfg.get("bright_frb_brightness_log_uniform", True)),
        low_dm_pulse_boost=float(cfg.get("low_dm_pulse_boost", 0.0)),
        low_dm_pulse_dm_range=tuple(cfg.get("low_dm_pulse_dm_range", (10.0, 150.0))),
        low_dm_pulse_width_range=tuple(cfg.get("low_dm_pulse_width_range", (1, 5))),
        low_dm_pulse_width_ms_range=_optional_range(cfg.get("low_dm_pulse_width_ms_range", None)),
        low_dm_pulse_brightness_range=tuple(cfg.get("low_dm_pulse_brightness_range", (0.7, 0.999))),
        low_dm_pulse_brightness_log_uniform=bool(cfg.get("low_dm_pulse_brightness_log_uniform", True)),
        low_dm_bright_medium_pulse_boost=float(cfg.get("low_dm_bright_medium_pulse_boost", 0.0)),
        low_dm_bright_medium_pulse_dm_range=tuple(cfg.get("low_dm_bright_medium_pulse_dm_range", (30.0, 100.0))),
        low_dm_bright_medium_pulse_width_range=tuple(cfg.get("low_dm_bright_medium_pulse_width_range", (3, 6))),
        low_dm_bright_medium_pulse_width_ms_range=_optional_range(
            cfg.get("low_dm_bright_medium_pulse_width_ms_range", None)
        ),
        low_dm_bright_medium_pulse_brightness_range=tuple(cfg.get("low_dm_bright_medium_pulse_brightness_range", (0.9, 0.999))),
        low_dm_bright_medium_pulse_brightness_log_uniform=bool(cfg.get("low_dm_bright_medium_pulse_brightness_log_uniform", True)),
        brightness_fixed=cfg.get("frb_brightness_fixed", None),
        dm_fixed=cfg.get("frb_dm_fixed", None),
        dm_range=tuple(cfg.get("frb_dm_range", (10.0, 3000.0))),
        mask_relative_threshold=float(cfg.get("frb_mask_relative_threshold", 0.01)),
        mask_absolute_floor=float(cfg.get("frb_mask_absolute_floor", 1.0e-3)),
        telescope_configs=cfg.get("telescope_configs", None),
        time_resolutions_us=cfg.get("time_resolutions_us", None),
        spectral_occupancy_range=tuple(cfg.get("frb_spectral_occupancy_range", (0.3, 1.0))),
        scintillation_prob=float(cfg.get("frb_scintillation_prob", 0.5)),
        scintillation_drop_range=tuple(cfg.get("frb_scintillation_drop_range", (0.05, 0.4))),
        freq_mod_prob=float(cfg.get("frb_freq_mod_prob", 0.6)),
        freq_mod_depth_range=tuple(cfg.get("frb_freq_mod_depth_range", (0.2, 0.8))),
        scattering_prob=float(cfg.get("frb_scattering_prob", 0.5)),
        scattering_tau_range=tuple(cfg.get("frb_scattering_tau_range", (0.5, 15.0))),
        scattering_index=float(cfg.get("frb_scattering_index", -4.0)),
        sub_burst_prob=float(cfg.get("frb_sub_burst_prob", 0.3)),
        sub_burst_count_range=tuple(cfg.get("frb_sub_burst_count_range", (2, 5))),
        sub_burst_drift_range=tuple(cfg.get("frb_sub_burst_drift_range", (-0.15, -0.01))),
        spectral_index_range=tuple(cfg.get("frb_spectral_index_range", (-3.0, 1.0))),
    )


def _frb_to_z_delta(frb_image: Array, scale: float, clip: float) -> Array:
    frb_image = np.clip(np.asarray(frb_image, dtype=np.float32), 0.0, clip)
    return np.arctanh(frb_image) * np.float32(scale)


def _add_frb_to_patch(patch: Array, frb_image: Array, scale: float, clip: float) -> Array:
    frb_z = _frb_to_z_delta(frb_image, scale, clip)
    return _from_z(_to_z(patch, scale, clip) + frb_z, scale)


def _bounded_start(rng: np.random.Generator, size: int, length: int) -> int:
    if length >= size:
        return 0
    return int(rng.integers(0, size - length + 1))


def _clip_box(h: int, w: int, y0: int, y1: int, x0: int, x1: int):
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    if y1 <= y0 or x1 <= x0:
        return None
    return slice(y0, y1), slice(x0, x1)


def _apply_background_negative_bands(
    patch: Array,
    rng: np.random.Generator,
    cfg: dict,
    scale: float,
    clip: float,
) -> None:
    h, w = patch.shape
    bands = _rand_int(rng, tuple(cfg.get("bands", (1, 3))))
    for _ in range(bands):
        length = min(w, _rand_int(rng, tuple(cfg.get("length", (360, w)))))
        height = min(h, _rand_int(rng, tuple(cfg.get("height", (4, 36)))))
        if length <= 0 or height <= 0:
            continue

        x0 = _bounded_start(rng, w, length)
        y0 = _bounded_start(rng, h, height)
        box = _clip_box(h, w, y0, y0 + height, x0, x0 + length)
        if box is None:
            continue

        yy, xx = box
        region = patch[yy, xx]
        y_len, x_len = region.shape
        drop = _rand_float(rng, tuple(cfg.get("z_drop", (0.4, 2.2))))

        y_profile = np.ones(y_len, dtype=np.float32)
        soft_edge = _rand_int(rng, tuple(cfg.get("soft_edge", (2, 8))))
        soft_edge = min(soft_edge, max(0, y_len // 2))
        if soft_edge > 0:
            taper = np.linspace(0.35, 1.0, soft_edge, dtype=np.float32)
            y_profile[:soft_edge] *= taper
            y_profile[-soft_edge:] *= taper[::-1]

        time_profile = _smooth_profile(rng, x_len, low=0.82)
        row_mod = rng.uniform(0.85, 1.0, size=(y_len, 1)).astype(np.float32)
        stamp_z = -np.float32(drop) * y_profile[:, None] * time_profile[None, :] * row_mod
        patch[yy, xx] = _from_z(_to_z(region, scale, clip) + stamp_z, scale)


def _paint_box(
    patch: Array,
    mask: Array,
    weight: Array,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    stamp_z: Array,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    orig_y0 = int(y0)
    orig_x0 = int(x0)
    box = _clip_box(patch.shape[0], patch.shape[1], y0, y1, x0, x1)
    if box is None:
        return
    yy, xx = box
    region = patch[yy, xx]
    support = np.ones(region.shape, dtype=bool)
    if preserve_mask is not None:
        support &= ~preserve_mask[yy, xx].astype(bool, copy=False)
    if not support.any():
        return
    stamp_z = np.asarray(stamp_z, dtype=np.float32)
    if stamp_z.shape != region.shape and stamp_z.ndim == 2:
        y_off = max(0, yy.start - orig_y0)
        x_off = max(0, xx.start - orig_x0)
        y_len, x_len = region.shape
        if stamp_z.shape[0] >= y_off + y_len and stamp_z.shape[1] >= x_off + x_len:
            stamp_z = stamp_z[y_off:y_off + y_len, x_off:x_off + x_len]
    if stamp_z.shape != region.shape:
        stamp_z = np.broadcast_to(stamp_z, region.shape)

    updated = _from_z(_to_z(region, scale, clip) + stamp_z, scale)
    if support.all():
        patch[yy, xx] = updated
    else:
        region_out = region.copy()
        region_out[support] = updated[support]
        patch[yy, xx] = region_out
    mask_view = mask[yy, xx]
    mask_view[support] = 1.0
    weight_view = weight[yy, xx]
    weight_view[support] = np.maximum(weight_view[support], np.float32(pos_weight))


def _paint_horizontal(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    length: int,
    height: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    profile = _smooth_profile(rng, length)
    row_mod = rng.uniform(0.9, 1.0, size=(height, 1)).astype(np.float32)
    stamp = (np.float32(amp_z) * row_mod * profile[None, :]).astype(np.float32)
    _paint_box(
        patch, mask, weight,
        y0, y0 + height, x0, x0 + length,
        stamp, pos_weight, scale, clip, preserve_mask,
    )


def _broken_horizontal_gate(
    rng: np.random.Generator,
    n: int,
    *,
    keep_prob: float,
    min_seg: int,
    max_seg: int,
    gap_level: tuple[float, float],
) -> Array:
    """Piecewise gate for complex horizontal RFI.

    The gate keeps most of the line bright while allowing shallow broken
    sections. The training mask still covers the whole physical streak, which
    teaches the network not to stop at only the brightest cores.
    """

    n = int(n)
    if n <= 1:
        return np.ones((max(1, n),), dtype=np.float32)
    min_seg = max(1, int(min_seg))
    max_seg = max(min_seg, int(max_seg))
    gate = np.zeros(n, dtype=np.float32)
    cursor = 0
    while cursor < n:
        seg = int(rng.integers(min_seg, max_seg + 1))
        end = min(n, cursor + seg)
        if rng.random() < float(keep_prob):
            gate[cursor:end] = np.float32(rng.uniform(0.72, 1.05))
        else:
            gate[cursor:end] = np.float32(_rand_float(rng, gap_level))
        cursor = end
    smooth = int(rng.integers(3, 9))
    kernel = np.ones(smooth, dtype=np.float32) / np.float32(smooth)
    return np.convolve(gate, kernel, mode="same").astype(np.float32, copy=False)


def _paint_complex_horizontal_line(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    length: int,
    height: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    length = int(length)
    height = int(height)
    if length <= 0 or height <= 0:
        return

    time_profile = _smooth_profile(rng, length, low=float(cfg.get("ridge_profile_low", 0.68)))
    if rng.random() < float(cfg.get("broken_prob", 0.75)):
        time_profile *= _broken_horizontal_gate(
            rng,
            length,
            keep_prob=float(cfg.get("broken_keep_prob", 0.82)),
            min_seg=int(cfg.get("broken_min_seg", 8)),
            max_seg=int(cfg.get("broken_max_seg", 55)),
            gap_level=tuple(cfg.get("broken_gap_level", (0.15, 0.55))),
        )

    yy = np.arange(height, dtype=np.float32)
    center = np.float32((height - 1) / 2.0)
    sigma = np.float32(max(0.65, float(height) / 2.15))
    freq_profile = np.exp(-0.5 * ((yy - center) / sigma) ** 2).astype(np.float32)
    stamp = np.float32(amp_z) * freq_profile[:, None] * time_profile[None, :]

    texture = float(cfg.get("ridge_texture", 0.14))
    if texture > 0.0:
        stamp *= rng.uniform(1.0 - texture, 1.0 + texture, size=stamp.shape).astype(np.float32)
    stamp = np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False)

    # Mask a small edge around thin ridges so the target follows the visually
    # obvious physical streak rather than only the exact high-z core.
    mask_edge = int(cfg.get("mask_edge", 1))
    _paint_box(
        patch, mask, weight,
        int(y0) - mask_edge, int(y0) + height + mask_edge,
        int(x0), int(x0) + length,
        np.pad(stamp, ((mask_edge, mask_edge), (0, 0)), mode="edge") if mask_edge > 0 else stamp,
        pos_weight, scale, clip, preserve_mask,
    )


def _paint_complex_horizontal_block(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    height: int,
    length: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    if int(length) <= 0 or int(height) <= 0:
        return
    row_profile = _smooth_profile(rng, int(height), low=0.55)[:, None]
    col_profile = _smooth_profile(rng, int(length), low=0.60)[None, :]
    stamp = np.float32(amp_z) * (0.65 + 0.35 * row_profile) * (0.75 + 0.25 * col_profile)
    texture = float(cfg.get("block_texture", 0.22))
    if texture > 0.0:
        stamp *= rng.uniform(1.0 - texture, 1.0 + texture, size=stamp.shape).astype(np.float32)
    _paint_box(
        patch, mask, weight,
        int(y0), int(y0) + int(height), int(x0), int(x0) + int(length),
        np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False),
        pos_weight, scale, clip, preserve_mask,
    )


def _paint_complex_horizontal_pedestal(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    height: int,
    length: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    if int(length) <= 0 or int(height) <= 0:
        return
    yy = np.arange(int(height), dtype=np.float32)
    center = np.float32((int(height) - 1) / 2.0)
    sigma = np.float32(max(1.0, float(height) / 2.5))
    freq_profile = np.exp(-0.5 * ((yy - center) / sigma) ** 2).astype(np.float32)
    time_profile = _smooth_profile(rng, int(length), low=float(cfg.get("pedestal_profile_low", 0.52)))
    stamp = np.float32(amp_z) * freq_profile[:, None] * time_profile[None, :]
    texture = float(cfg.get("pedestal_texture", 0.18))
    if texture > 0.0:
        stamp *= 1.0 + np.float32(texture) * rng.normal(0.0, 1.0, size=stamp.shape).astype(np.float32)
    stamp = np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False)

    # Pedestals are sometimes visible enough to be part of the RFI target, but
    # often act only as context. Keep this probability configurable.
    local_mask = mask if rng.random() < float(cfg.get("pedestal_mask_prob", 0.35)) else np.zeros_like(mask)
    local_weight = weight if local_mask is mask else np.ones_like(weight)
    _paint_box(
        patch, local_mask, local_weight,
        int(y0), int(y0) + int(height), int(x0), int(x0) + int(length),
        stamp, pos_weight, scale, clip, preserve_mask,
    )


def _augment_complex_horizontal(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    groups = _rand_int(rng, tuple(cfg["groups"]))
    for _ in range(groups):
        span_t = min(w, _rand_int(rng, tuple(cfg["time_span"])))
        span_f = min(h, _rand_int(rng, tuple(cfg["freq_span"])))
        x_center = int(rng.integers(0, w))
        y_center = int(rng.integers(0, h))
        group_x0 = max(0, min(w - 1, x_center - span_t // 2))
        group_y0 = max(0, min(h - 1, y_center - span_f // 2))

        component_kinds = [
            "thin_streak",
            "bright_segment",
            "thick_block",
            "parallel_pair",
            "offset_streak",
        ]
        if rng.random() < float(cfg.get("broken_streak_prob", 1.0)):
            component_kinds.append("broken_streak")
        rng.shuffle(component_kinds)
        count = _rand_int(rng, tuple(cfg["components_per_group"]))

        for kind in component_kinds[:count]:
            if kind == "thin_streak":
                length = min(span_t, _rand_int(rng, tuple(cfg["thin_len"])))
                height = 1
                amp = _rand_float(rng, tuple(cfg["z_amp"]))
            elif kind == "bright_segment":
                length = min(span_t, _rand_int(rng, tuple(cfg["thick_len"])))
                height = _rand_int(rng, (1, 3))
                amp = _rand_float(rng, (max(1.2, cfg["z_amp"][0]), cfg["z_amp"][1]))
            elif kind == "thick_block":
                length = min(span_t, _rand_int(rng, tuple(cfg["thick_len"])))
                height = _rand_int(rng, tuple(cfg["height"]))
                amp = _rand_float(rng, (max(1.0, cfg["z_amp"][0]), cfg["z_amp"][1]))
            elif kind == "parallel_pair":
                length = min(span_t, _rand_int(rng, tuple(cfg["thin_len"])))
                height = 1
                amp = _rand_float(rng, tuple(cfg["z_amp"]))
            elif kind == "offset_streak":
                length = min(span_t, _rand_int(rng, tuple(cfg["thin_len"])))
                height = 1
                amp = _rand_float(rng, (cfg["z_amp"][0], min(2.0, cfg["z_amp"][1])))
            else:
                length = min(span_t, _rand_int(rng, tuple(cfg["thin_len"])))
                height = 1
                amp = _rand_float(rng, (max(0.9, cfg["z_amp"][0]), min(2.2, cfg["z_amp"][1])))

            if length <= 0:
                continue
            x0 = group_x0 + int(rng.integers(0, max(1, span_t - length + 1)))
            y0 = group_y0 + int(rng.integers(0, max(1, span_f - height + 1)))

            if kind == "parallel_pair":
                gap = int(rng.integers(2, 6))
                _paint_horizontal(
                    patch, mask, weight, rng, y0, x0, length, height,
                    amp, pos_weight, scale, clip, preserve_mask,
                )
                _paint_horizontal(
                    patch, mask, weight, rng, y0 + gap, x0 + int(rng.integers(-8, 9)),
                    max(8, length + int(rng.integers(-20, 21))), height,
                    amp * float(rng.uniform(0.75, 1.1)), pos_weight, scale, clip, preserve_mask,
                )
            elif kind == "broken_streak":
                parts = int(rng.integers(2, 5))
                cursor = x0
                remaining = length
                for part_idx in range(parts):
                    if remaining <= 6:
                        break
                    part_len = max(6, remaining // (parts - part_idx) + int(rng.integers(-10, 11)))
                    part_len = min(part_len, remaining)
                    _paint_horizontal(
                        patch, mask, weight, rng, y0, cursor, part_len, height,
                        amp, pos_weight, scale, clip, preserve_mask,
                    )
                    gap = int(rng.integers(2, 10))
                    cursor += part_len + gap
                    remaining = x0 + length - cursor
            else:
                _paint_horizontal(
                    patch, mask, weight, rng, y0, x0, length, height,
                    amp, pos_weight, scale, clip, preserve_mask,
                )


def _augment_bright_complex_horizontal(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    """Dense bright horizontal RFI with ridges, blocky cores, and pedestals."""

    h, w = patch.shape
    clusters = _rand_int(rng, tuple(cfg.get("clusters", (1, 2))))
    edge_bias_prob = float(cfg.get("edge_bias_prob", 0.65))
    for _cluster in range(clusters):
        span_t = min(w, _rand_int(rng, tuple(cfg.get("time_span", (180, 512)))))
        span_f = min(h, _rand_int(rng, tuple(cfg.get("freq_span", (45, 150)))))
        if span_t <= 0 or span_f <= 0:
            continue

        x0_group = _bounded_start(rng, w, span_t)
        if rng.random() < edge_bias_prob:
            edge_max = max(1, min(h - span_f, int(round(h * float(cfg.get("edge_max_fraction", 0.35))))))
            y0_group = int(rng.integers(0, edge_max + 1))
        else:
            y0_group = _bounded_start(rng, h, span_f)

        if rng.random() < float(cfg.get("pedestal_prob", 0.75)):
            ped_len = min(w - x0_group, _rand_int(rng, tuple(cfg.get("pedestal_len", (180, 512)))))
            ped_h = min(h - y0_group, _rand_int(rng, tuple(cfg.get("pedestal_height", (14, 70)))))
            ped_y = y0_group + int(rng.integers(0, max(1, span_f - ped_h + 1)))
            ped_amp = _rand_float(rng, tuple(cfg.get("pedestal_z_amp", (0.18, 0.55))))
            _paint_complex_horizontal_pedestal(
                patch, mask, weight, rng, ped_y, x0_group, ped_h, ped_len,
                ped_amp, pos_weight, scale, clip, cfg, preserve_mask,
            )

        ridge_count = _rand_int(rng, tuple(cfg.get("ridges", (5, 12))))
        for _ridge in range(ridge_count):
            length = min(w - x0_group, _rand_int(rng, tuple(cfg.get("ridge_len", (70, 330)))))
            height = min(span_f, _rand_int(rng, tuple(cfg.get("ridge_height", (1, 4)))))
            if length <= 0 or height <= 0:
                continue
            x_jitter = int(rng.integers(0, max(1, span_t - length + 1)))
            y_jitter = int(rng.integers(0, max(1, span_f - height + 1)))
            x0 = x0_group + x_jitter
            y0 = y0_group + y_jitter
            amp = _rand_float(rng, tuple(cfg.get("ridge_z_amp", (1.6, 3.4))))
            _paint_complex_horizontal_line(
                patch, mask, weight, rng, y0, x0, length, height,
                amp, pos_weight, scale, clip, cfg, preserve_mask,
            )

            if rng.random() < float(cfg.get("parallel_repeat_prob", 0.45)):
                gap = int(rng.integers(2, 10))
                repeat_len = max(16, length + int(rng.integers(-45, 46)))
                repeat_x = x0 + int(rng.integers(-35, 36))
                repeat_y = y0 + gap
                _paint_complex_horizontal_line(
                    patch, mask, weight, rng,
                    repeat_y, repeat_x, repeat_len, height,
                    amp * float(rng.uniform(0.65, 1.05)),
                    pos_weight, scale, clip, cfg, preserve_mask,
                )

        block_count = _rand_int(rng, tuple(cfg.get("blocks", (1, 4))))
        for _block in range(block_count):
            block_len = _rand_int(rng, tuple(cfg.get("block_len", (35, 95))))
            block_h = _rand_int(rng, tuple(cfg.get("block_height", (10, 34))))
            x0 = x0_group + int(rng.integers(0, max(1, span_t - min(block_len, span_t) + 1)))
            y0 = y0_group + int(rng.integers(0, max(1, span_f - min(block_h, span_f) + 1)))
            block_amp = _rand_float(rng, tuple(cfg.get("block_z_amp", (1.5, 3.0))))
            _paint_complex_horizontal_block(
                patch, mask, weight, rng,
                y0, x0, min(block_h, h - y0), min(block_len, w - x0),
                block_amp, pos_weight, scale, clip, cfg, preserve_mask,
            )


def _paint_sparse_bright_bar(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    length: int,
    height: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    length = int(length)
    height = int(height)
    if length <= 0 or height <= 0:
        return

    low = float(cfg.get("line_profile_low", 0.88))
    texture = float(cfg.get("line_texture", 0.08))
    time_profile = _smooth_profile(rng, length, low=low)
    if texture > 0.0:
        time_profile *= rng.uniform(
            1.0 - texture, 1.0 + texture, size=time_profile.shape,
        ).astype(np.float32)
    time_profile = np.maximum(time_profile, np.float32(0.0))

    row_mod = rng.uniform(0.9, 1.05, size=(height, 1)).astype(np.float32)
    stamp = np.float32(amp_z) * row_mod * time_profile[None, :]
    mask_edge = int(cfg.get("mask_edge", 1))
    if mask_edge > 0:
        stamp = np.pad(stamp, ((mask_edge, mask_edge), (0, 0)), mode="edge")

    _paint_box(
        patch, mask, weight,
        int(y0) - mask_edge, int(y0) + height + mask_edge,
        int(x0), int(x0) + length,
        stamp.astype(np.float32, copy=False),
        pos_weight, scale, clip, preserve_mask,
    )


def _augment_sparse_bright_horizontal_bars(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    """Sparse long bright horizontal bars with terminal/local bright cores."""

    h, w = patch.shape
    bars = _rand_int(rng, tuple(cfg.get("bars", (1, 3))))
    base_y = int(rng.integers(0, h))
    for bar_idx in range(bars):
        length = min(w, _rand_int(rng, tuple(cfg.get("length", (120, 512)))))
        height = min(h, _rand_int(rng, tuple(cfg.get("height", (1, 4)))))
        if length <= 0 or height <= 0:
            continue

        if bar_idx == 0 or rng.random() > float(cfg.get("parallel_prob", 0.35)):
            y0 = _bounded_start(rng, h, height)
        else:
            gap = _rand_int(rng, tuple(cfg.get("parallel_gap", (4, 18))))
            sign = -1 if rng.random() < 0.5 else 1
            y0 = int(np.clip(base_y + sign * gap, 0, max(0, h - height)))
        base_y = y0

        if rng.random() < float(cfg.get("time_edge_bias_prob", 0.55)):
            x0 = 0 if rng.random() < 0.5 else max(0, w - length)
        else:
            x0 = _bounded_start(rng, w, length)

        amp = _rand_float(rng, tuple(cfg.get("z_amp", (1.8, 4.5))))
        _paint_sparse_bright_bar(
            patch, mask, weight, rng,
            y0, x0, length, height, amp,
            pos_weight, scale, clip, cfg, preserve_mask,
        )

        if rng.random() >= float(cfg.get("core_prob", 0.9)):
            continue

        core_len = min(w, _rand_int(rng, tuple(cfg.get("core_len", (8, 70)))))
        core_h = min(h, _rand_int(rng, tuple(cfg.get("core_height", (2, 10)))))
        if core_len <= 0 or core_h <= 0:
            continue

        side = str(cfg.get("core_side", "either"))
        if side == "left":
            core_x0 = x0
        elif side == "right":
            core_x0 = x0 + length - core_len
        elif rng.random() < 0.5:
            core_x0 = x0
        else:
            core_x0 = x0 + length - core_len
        core_x0 += int(rng.integers(-max(1, core_len // 3), max(2, core_len // 3 + 1)))
        core_y0 = y0 + height // 2 - core_h // 2
        core_x0 = int(np.clip(core_x0, 0, max(0, w - core_len)))
        core_y0 = int(np.clip(core_y0, 0, max(0, h - core_h)))

        core_amp = _rand_float(rng, tuple(cfg.get("core_z_amp", (2.5, 5.8))))
        row_profile = _smooth_profile(rng, core_h, low=0.78)[:, None]
        col_profile = _smooth_profile(rng, core_len, low=0.82)[None, :]
        stamp = np.float32(core_amp) * row_profile * col_profile
        texture = float(cfg.get("core_texture", 0.12))
        if texture > 0.0:
            stamp *= rng.uniform(1.0 - texture, 1.0 + texture, size=stamp.shape).astype(np.float32)

        _paint_box(
            patch, mask, weight,
            core_y0, core_y0 + core_h,
            core_x0, core_x0 + core_len,
            np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False),
            pos_weight, scale, clip, preserve_mask,
        )


def _paint_periodic_tick_context(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    height: int,
    length: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    if int(length) <= 0 or int(height) <= 0:
        return

    yy = np.arange(int(height), dtype=np.float32)
    center = np.float32((int(height) - 1) / 2.0)
    sigma = np.float32(max(1.0, float(height) / 2.8))
    freq_profile = np.exp(-0.5 * ((yy - center) / sigma) ** 2).astype(np.float32)
    time_profile = _smooth_profile(rng, int(length), low=float(cfg.get("context_profile_low", 0.65)))
    stamp = np.float32(amp_z) * freq_profile[:, None] * time_profile[None, :]

    texture = float(cfg.get("context_texture", 0.22))
    if texture > 0.0:
        stamp *= 1.0 + np.float32(texture) * rng.normal(0.0, 1.0, size=stamp.shape).astype(np.float32)
    stamp = np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False)

    local_mask = mask if rng.random() < float(cfg.get("context_mask_prob", 0.0)) else np.zeros_like(mask)
    local_weight = weight if local_mask is mask else np.ones_like(weight)
    _paint_box(
        patch, local_mask, local_weight,
        int(y0), int(y0) + int(height), int(x0), int(x0) + int(length),
        stamp, pos_weight, scale, clip, preserve_mask,
    )


def _paint_periodic_horizontal_tick(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y_center: float,
    x_center: float,
    freq_len: int,
    time_width: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    freq_len = int(freq_len)
    time_width = int(time_width)
    if freq_len <= 0 or time_width <= 0:
        return

    y0 = int(round(float(y_center) - 0.5 * float(freq_len)))
    x0 = int(round(float(x_center) - 0.5 * float(time_width)))
    yy = np.arange(freq_len, dtype=np.float32)
    center_y = np.float32((freq_len - 1) / 2.0)
    sigma_y = np.float32(max(0.75, float(freq_len) / 3.2))
    freq_profile = np.exp(-0.5 * ((yy - center_y) / sigma_y) ** 2).astype(np.float32)

    if time_width > 1:
        xx = np.arange(time_width, dtype=np.float32)
        center_x = np.float32((time_width - 1) / 2.0)
        sigma_x = np.float32(max(0.60, float(time_width) / 2.4))
        time_profile = np.exp(-0.5 * ((xx - center_x) / sigma_x) ** 2).astype(np.float32)
    else:
        time_profile = np.ones((1,), dtype=np.float32)

    stamp = np.float32(amp_z) * freq_profile[:, None] * time_profile[None, :]
    texture = float(cfg.get("tick_texture", 0.12))
    if texture > 0.0:
        stamp *= rng.uniform(1.0 - texture, 1.0 + texture, size=stamp.shape).astype(np.float32)
    stamp = np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False)

    mask_edge = int(cfg.get("mask_edge", 1))
    if mask_edge > 0:
        stamp = np.pad(stamp, ((mask_edge, mask_edge), (0, 0)), mode="edge")
    _paint_box(
        patch, mask, weight,
        y0 - mask_edge, y0 + freq_len + mask_edge, x0, x0 + time_width,
        stamp, pos_weight, scale, clip, preserve_mask,
    )


def _augment_periodic_horizontal_tick_train(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    """Periodic short tick RFI aligned along one horizontal frequency level."""

    h, w = patch.shape
    trains = _rand_int(rng, tuple(cfg.get("trains", (1, 3))))
    for _train in range(trains):
        span_t = min(w, _rand_int(rng, tuple(cfg.get("time_span", (140, 512)))))
        span_f = min(h, _rand_int(rng, tuple(cfg.get("freq_span", (16, 64)))))
        if span_t <= 0 or span_f <= 0:
            continue

        x0_group = _bounded_start(rng, w, span_t)
        if rng.random() < float(cfg.get("edge_bias_prob", 0.35)):
            edge_max = max(1, int(round(h * float(cfg.get("edge_max_fraction", 0.28)))))
            if rng.random() < 0.5:
                y_center = float(rng.uniform(0, min(h - 1, edge_max)))
            else:
                y_center = float(rng.uniform(max(0, h - edge_max - 1), h - 1))
        else:
            y0_group = _bounded_start(rng, h, span_f)
            y_center = float(y0_group + rng.uniform(0, max(1, span_f)))

        if rng.random() < float(cfg.get("context_prob", 0.75)):
            context_h = min(h, _rand_int(rng, tuple(cfg.get("context_height", (8, 28)))))
            context_y = int(round(y_center - 0.5 * float(context_h)))
            context_amp = _rand_float(rng, tuple(cfg.get("context_z_amp", (0.10, 0.28))))
            _paint_periodic_tick_context(
                patch, mask, weight, rng, context_y, x0_group, context_h, span_t,
                context_amp, pos_weight, scale, clip, cfg, preserve_mask,
            )

        spacing = max(2, _rand_int(rng, tuple(cfg.get("spacing", (8, 20)))))
        phase_count = _rand_int(rng, tuple(cfg.get("phase_count", (1, 2))))
        time_jitter = _rand_float(rng, tuple(cfg.get("time_jitter", (0.6, 2.5))))
        y_jitter = _rand_float(rng, tuple(cfg.get("y_jitter", (0.4, 2.0))))
        dropout = _rand_float(rng, tuple(cfg.get("dropout", (0.05, 0.25))))
        max_ticks = int(cfg.get("ticks_per_train_max", 96))

        for phase in range(phase_count):
            phase_offset = float(rng.uniform(0, spacing)) if phase == 0 else float((phase * spacing) / max(1, phase_count))
            xs = np.arange(
                float(x0_group) + phase_offset,
                float(x0_group + span_t),
                float(spacing),
                dtype=np.float32,
            )
            if xs.size > max_ticks:
                keep_idx = np.sort(rng.choice(xs.size, size=max_ticks, replace=False))
                xs = xs[keep_idx]

            for x_center in xs:
                if rng.random() < dropout:
                    continue
                amp = _rand_float(rng, tuple(cfg.get("z_amp", (0.8, 2.6))))
                if rng.random() < float(cfg.get("strong_tick_prob", 0.18)):
                    amp = _rand_float(rng, tuple(cfg.get("strong_z_amp", (2.0, 3.6))))
                tick_h = _rand_int(rng, tuple(cfg.get("freq_len", (4, 22))))
                tick_w = _rand_int(rng, tuple(cfg.get("time_width", (1, 3))))
                yc = y_center + float(rng.normal(0.0, y_jitter))
                xc = float(x_center) + float(rng.normal(0.0, time_jitter))
                _paint_periodic_horizontal_tick(
                    patch, mask, weight, rng, yc, xc, tick_h, tick_w,
                    amp, pos_weight, scale, clip, cfg, preserve_mask,
                )


def _augment_persistent_narrowband(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    for _ in range(_rand_int(rng, tuple(cfg["bands"]))):
        length = min(w, _rand_int(rng, tuple(cfg["length"])))
        height = min(h, _rand_int(rng, tuple(cfg["height"])))
        x0 = _bounded_start(rng, w, length)
        y0 = _bounded_start(rng, h, height)
        amp = _rand_float(rng, tuple(cfg["z_amp"]))
        _paint_horizontal(
            patch, mask, weight, rng, y0, x0, length, height,
            amp, pos_weight, scale, clip, preserve_mask,
        )


def _augment_diffuse_persistent_bright_band(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    for _ in range(_rand_int(rng, tuple(cfg["bands"]))):
        length = min(w, _rand_int(rng, tuple(cfg["length"])))
        height = min(h, _rand_int(rng, tuple(cfg["height"])))
        if length <= 0 or height <= 0:
            continue

        x0 = _bounded_start(rng, w, length)
        y0 = _bounded_start(rng, h, height)
        pedestal = _rand_float(rng, tuple(cfg["pedestal_z_amp"]))

        y_profile = np.ones(height, dtype=np.float32)
        soft_edge = min(_rand_int(rng, tuple(cfg.get("soft_edge", (2, 8)))), max(0, height // 2))
        if soft_edge > 0:
            taper = np.linspace(0.35, 1.0, soft_edge, dtype=np.float32)
            y_profile[:soft_edge] *= taper
            y_profile[-soft_edge:] *= taper[::-1]

        time_profile = _smooth_profile(rng, length, low=0.72)
        row_mod = rng.uniform(0.82, 1.0, size=(height, 1)).astype(np.float32)
        stamp = np.float32(pedestal) * y_profile[:, None] * time_profile[None, :] * row_mod

        tex_lo, tex_hi = tuple(cfg.get("texture", (0.04, 0.18)))
        texture = _rand_float(rng, (tex_lo, tex_hi))
        if texture > 0.0:
            tex = rng.normal(0.0, 1.0, size=stamp.shape).astype(np.float32)
            stamp *= 1.0 + np.float32(texture) * tex
            stamp = np.maximum(stamp, np.float32(0.0))

        ridge_count = _rand_int(rng, tuple(cfg.get("ridge_count", (1, 3))))
        for _ridge in range(ridge_count):
            ridge_h = min(height, _rand_int(rng, tuple(cfg.get("ridge_height", (1, 3)))))
            ridge_y = int(rng.integers(0, max(1, height - ridge_h + 1)))
            ridge_amp = _rand_float(rng, tuple(cfg.get("ridge_z_amp", (0.65, 2.2))))
            ridge_profile = _smooth_profile(rng, length, low=0.80)
            stamp[ridge_y:ridge_y + ridge_h, :] += np.float32(ridge_amp) * ridge_profile[None, :]

        _paint_box(
            patch, mask, weight,
            y0, y0 + height, x0, x0 + length,
            stamp.astype(np.float32, copy=False), pos_weight, scale, clip, preserve_mask,
        )


def _paint_compact_spot(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cy: float,
    cx: float,
    radius_t: float,
    radius_f: float,
    amp_z: float,
    mask_rel_threshold: float,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    pad_t = int(np.ceil(float(radius_t) * 4.0)) + 2
    pad_f = int(np.ceil(float(radius_f) * 4.0)) + 2
    y0 = max(0, int(np.floor(float(cy) - pad_f)))
    y1 = min(h, int(np.ceil(float(cy) + pad_f + 1)))
    x0 = max(0, int(np.floor(float(cx) - pad_t)))
    x1 = min(w, int(np.ceil(float(cx) + pad_t + 1)))
    box = _clip_box(h, w, y0, y1, x0, x1)
    if box is None:
        return

    yy, xx = box
    region = patch[yy, xx]
    y_axis = np.arange(yy.start, yy.stop, dtype=np.float32)[:, None]
    x_axis = np.arange(xx.start, xx.stop, dtype=np.float32)[None, :]
    stamp = np.float32(amp_z) * np.exp(
        -0.5 * (
            ((x_axis - np.float32(cx)) / np.float32(max(radius_t, 1.0e-3))) ** 2
            + ((y_axis - np.float32(cy)) / np.float32(max(radius_f, 1.0e-3))) ** 2
        )
    ).astype(np.float32)
    stamp *= rng.uniform(0.88, 1.08, size=stamp.shape).astype(np.float32)

    support = stamp > np.float32(float(amp_z) * float(mask_rel_threshold))
    if preserve_mask is not None:
        support &= ~preserve_mask[yy, xx].astype(bool, copy=False)
    if not support.any():
        return

    updated = _from_z(_to_z(region, scale, clip) + stamp, scale)
    region_out = region.copy()
    region_out[support] = updated[support]
    patch[yy, xx] = region_out

    mask_view = mask[yy, xx]
    mask_view[support] = 1.0
    weight_view = weight[yy, xx]
    weight_view[support] = np.maximum(weight_view[support], np.float32(pos_weight))


def _augment_compact_spots(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    groups = _rand_int(rng, tuple(cfg["groups"]))
    for _ in range(groups):
        if rng.random() < float(cfg.get("cluster_prob", 0.35)):
            n_spots = _rand_int(rng, tuple(cfg.get("cluster_size", (2, 6))))
            base_y = float(rng.uniform(0, h))
            base_x = float(rng.uniform(0, w))
            t_std = _rand_float(rng, tuple(cfg.get("cluster_time_std", (3.0, 12.0))))
            f_std = _rand_float(rng, tuple(cfg.get("cluster_freq_std", (2.0, 10.0))))
            centers = [
                (base_y + float(rng.normal(0.0, f_std)), base_x + float(rng.normal(0.0, t_std)))
                for _spot in range(n_spots)
            ]
        else:
            centers = [(float(rng.uniform(0, h)), float(rng.uniform(0, w)))]

        for cy, cx in centers:
            if cy < -8 or cy >= h + 8 or cx < -8 or cx >= w + 8:
                continue
            radius_t = _rand_float(rng, tuple(cfg.get("radius_t", (1.2, 5.5))))
            radius_f = _rand_float(rng, tuple(cfg.get("radius_f", (1.2, 4.5))))
            if rng.random() < float(cfg.get("faint_prob", 0.25)):
                amp = _rand_float(rng, tuple(cfg.get("faint_z_amp", (0.10, 0.35))))
            else:
                amp = _rand_float(rng, tuple(cfg.get("z_amp", (0.18, 2.6))))
            _paint_compact_spot(
                patch, mask, weight,
                rng, cy, cx, radius_t, radius_f, amp,
                float(cfg.get("mask_rel_threshold", 0.22)),
                pos_weight, scale, clip, preserve_mask,
            )


def _augment_rect_blocks(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    for _ in range(_rand_int(rng, tuple(cfg["blocks"]))):
        length = min(w, _rand_int(rng, tuple(cfg["time_len"])))
        height = min(h, _rand_int(rng, tuple(cfg["freq_len"])))
        x0 = _bounded_start(rng, w, length)
        y0 = _bounded_start(rng, h, height)
        amp = _rand_float(rng, tuple(cfg["z_amp"]))
        stamp = np.float32(amp) * rng.uniform(0.9, 1.0, size=(height, length)).astype(np.float32)
        _paint_box(
            patch, mask, weight, y0, y0 + height, x0, x0 + length,
            stamp, pos_weight, scale, clip, preserve_mask,
        )


def _augment_vertical_bursts(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    for _ in range(_rand_int(rng, tuple(cfg["bursts"]))):
        width = min(w, _rand_int(rng, tuple(cfg["time_width"])))
        height = min(h, _rand_int(rng, tuple(cfg["freq_len"])))
        x0 = _bounded_start(rng, w, width)
        y0 = _bounded_start(rng, h, height)
        amp = _rand_float(rng, tuple(cfg["z_amp"]))
        profile = _smooth_profile(rng, height)[:, None]
        stamp = (np.float32(amp) * profile * rng.uniform(0.9, 1.0, size=(1, width))).astype(np.float32)
        _paint_box(
            patch, mask, weight, y0, y0 + height, x0, x0 + width,
            stamp, pos_weight, scale, clip, preserve_mask,
        )


def _sample_duration_bins(
    rng: np.random.Generator,
    bounds_ms: tuple[float, float],
    time_resolution_us: float,
    *,
    min_bins: int = 1,
    max_bins: int | None = None,
) -> int:
    duration_ms = _rand_float(rng, bounds_ms)
    bins = int(round(duration_ms / max(float(time_resolution_us) / 1000.0, 1.0e-9)))
    bins = max(int(min_bins), bins)
    if max_bins is not None:
        bins = min(int(max_bins), bins)
    return bins


def _paint_repeating_broadband_strip(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    y0: int,
    x0: int,
    height: int,
    width: int,
    amp_z: float,
    pos_weight: float,
    scale: float,
    clip: float,
    cfg: dict,
    preserve_mask: Array | None = None,
) -> None:
    if int(height) <= 0 or int(width) <= 0:
        return

    freq_profile = _smooth_profile(
        rng,
        int(height),
        low=float(cfg.get("freq_profile_low", 0.78)),
    )[:, None]
    time_profile = _smooth_profile(
        rng,
        int(width),
        low=float(cfg.get("time_profile_low", 0.88)),
    )[None, :]
    stamp = np.float32(amp_z) * freq_profile * time_profile

    texture = float(cfg.get("texture", 0.10))
    if texture > 0.0:
        stamp *= rng.uniform(1.0 - texture, 1.0 + texture, size=stamp.shape).astype(np.float32)

    _paint_box(
        patch, mask, weight,
        int(y0), int(y0) + int(height),
        int(x0), int(x0) + int(width),
        np.maximum(stamp, np.float32(0.0)).astype(np.float32, copy=False),
        pos_weight, scale, clip, preserve_mask,
    )


def _augment_repeating_broadband_bursts(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    time_resolution_us = _choose_time_resolution_us(rng, cfg.get("time_resolutions_us", None))
    amount = _rand_int(rng, tuple(cfg.get("amount", (2, 5))))
    max_bursts = int(cfg.get("max_bursts_per_train", 256))

    for _ in range(amount):
        if rng.random() < float(cfg.get("full_band_prob", 0.75)):
            height = h
        else:
            freq_fraction = _rand_float(rng, tuple(cfg.get("freq_fraction", (0.75, 1.0))))
            height = max(1, min(h, int(round(float(h) * freq_fraction))))
        y0 = _bounded_start(rng, h, height)

        on_bins = _sample_duration_bins(
            rng,
            tuple(cfg.get("onlength_ms", (0.5, 50.0))),
            time_resolution_us,
            min_bins=int(cfg.get("min_time_width_bins", 1)),
            max_bins=max(1, w),
        )
        period_bins = _sample_duration_bins(
            rng,
            tuple(cfg.get("period_ms", (2.0, 200.0))),
            time_resolution_us,
            min_bins=on_bins + int(cfg.get("min_gap_bins", 1)),
            max_bins=max(on_bins + 1, w * 2),
        )

        amp = _rand_float(rng, tuple(cfg.get("z_amp", (0.8, 3.2))))
        phase = int(rng.integers(-on_bins + 1, max(1, period_bins)))
        cursor = phase
        painted = 0
        while cursor < w and painted < max_bursts:
            if rng.random() >= float(cfg.get("dropout", 0.0)):
                local_width = max(
                    1,
                    on_bins + int(rng.integers(
                        -int(cfg.get("time_jitter_bins", 1)),
                        int(cfg.get("time_jitter_bins", 1)) + 1,
                    )),
                )
                local_amp = amp * float(rng.uniform(
                    float(cfg.get("amp_jitter", (0.85, 1.15))[0]),
                    float(cfg.get("amp_jitter", (0.85, 1.15))[1]),
                ))
                _paint_repeating_broadband_strip(
                    patch, mask, weight, rng,
                    y0, cursor, height, local_width,
                    local_amp, pos_weight, scale, clip, cfg, preserve_mask,
                )
            jitter = int(round(rng.normal(0.0, float(cfg.get("period_jitter_std_bins", 0.0)))))
            cursor += max(on_bins + 1, period_bins + jitter)
            painted += 1


def _augment_periodic_stripes(
    patch: Array,
    mask: Array,
    weight: Array,
    rng: np.random.Generator,
    cfg: dict,
    pos_weight: float,
    scale: float,
    clip: float,
    preserve_mask: Array | None = None,
) -> None:
    h, w = patch.shape
    stripes = _rand_int(rng, tuple(cfg["stripes"]))
    height = _rand_int(rng, tuple(cfg["height"]))
    amp = _rand_float(rng, tuple(cfg["z_amp"]))
    start = int(rng.integers(0, max(1, h // max(1, stripes))))
    step = max(height + 2, h // max(1, stripes))
    for y0 in range(start, h, step):
        length = int(rng.integers(max(16, w // 4), w + 1))
        x0 = _bounded_start(rng, w, length)
        _paint_horizontal(
            patch, mask, weight, rng, y0, x0, length, height,
            amp * float(rng.uniform(0.8, 1.2)), pos_weight, scale, clip, preserve_mask,
        )


FAMILY_FUNCS: dict[str, Callable[..., None]] = {
    "complex_horizontal": _augment_complex_horizontal,
    "bright_complex_horizontal": _augment_bright_complex_horizontal,
    "sparse_bright_horizontal_bars": _augment_sparse_bright_horizontal_bars,
    "periodic_horizontal_tick_train": _augment_periodic_horizontal_tick_train,
    "persistent_narrowband": _augment_persistent_narrowband,
    "diffuse_persistent_bright_band": _augment_diffuse_persistent_bright_band,
    "compact_spots": _augment_compact_spots,
    "rect_blocks": _augment_rect_blocks,
    "vertical_bursts": _augment_vertical_bursts,
    "repeating_broadband_bursts": _augment_repeating_broadband_bursts,
    "periodic_stripes": _augment_periodic_stripes,
}


def augment_patch(
    patch: Array,
    mask: Array | None,
    cfg: dict,
    rng: np.random.Generator,
) -> AugmentResult:
    """Return augmented patch, mask, and per-pixel loss weight.

    The input patch is copied. The output shapes match the input spatial shape.
    """

    out = np.asarray(patch, dtype=np.float32).copy()
    if out.ndim != 2:
        raise ValueError(f"augment_patch expects a 2D patch, got shape {out.shape}")

    threshold = float(cfg.get("mask_threshold", 0.5))
    if mask is None:
        out_mask = np.zeros_like(out, dtype=np.float32)
    else:
        out_mask = (np.asarray(mask, dtype=np.float32) > threshold).astype(np.float32)
        if out_mask.shape != out.shape:
            raise ValueError(f"mask shape {out_mask.shape} does not match patch shape {out.shape}")

    weight = np.ones_like(out, dtype=np.float32)
    astro_mask = np.zeros_like(out, dtype=np.float32)
    if not bool(cfg.get("enabled", True)):
        return AugmentResult(out, out_mask, weight, astro_mask, "none")

    scale = float(cfg.get("tanh_scale", 6.0))
    clip = float(cfg.get("input_clip", 0.999))
    noise_std = float(cfg.get("base_noise_std", 0.0))
    if noise_std > 0.0:
        out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    bg_prob = float(cfg.get("background_negative_band_prob", 0.0))
    if bg_prob > 0.0 and rng.random() < bg_prob:
        _apply_background_negative_bands(
            out, rng, dict(cfg.get("background_negative_band", {})), scale, clip,
        )

    astro_cfg = dict(cfg.get("astro_injection", {}))
    astro_enabled = bool(astro_cfg.get("enabled", False))
    frb_image = np.zeros_like(out, dtype=np.float32)
    frb_mask = np.zeros_like(out, dtype=np.float32)

    clean_prob = float(cfg.get("clean_sample_prob", 0.0))
    if clean_prob > 0.0 and rng.random() < clean_prob:
        if astro_enabled and bool(astro_cfg.get("clean_sample_inject_frb", True)):
            frb_image, frb_mask, _ = _inject_frb_from_cfg(out.shape[0], out.shape[1], rng, astro_cfg)
            if frb_mask.any():
                out = _add_frb_to_patch(out, frb_image, scale, clip)
                astro_mask = (frb_mask * (1.0 - out_mask)).astype(np.float32)
        out = np.clip(out, -clip, clip).astype(np.float32, copy=False)
        return AugmentResult(out, out_mask, weight, astro_mask, "clean_sample")

    preserve_mask = None
    if astro_enabled:
        frb_image, frb_mask, _ = _inject_frb_from_cfg(out.shape[0], out.shape[1], rng, astro_cfg)
        if frb_mask.any() and rng.random() < float(astro_cfg.get("preserve_frb_prob", 0.7)):
            preserve_mask = frb_mask > 0.5

    family_weights = dict(cfg["family_weights"])
    pos_weights = dict(cfg.get("positive_weights", {}))
    max_families = max(1, int(cfg.get("max_families_per_patch", 1)))
    extra_family_prob = float(cfg.get("extra_family_prob", 0.0))
    if not 0.0 <= extra_family_prob <= 1.0:
        raise ValueError("extra_family_prob must lie in [0, 1]")
    family_count = 1
    # Draw each additional family independently until the configured cap. The
    # development implementation performed only one Bernoulli draw, making a
    # documented maximum of three impossible to reach.
    while family_count < max_families and rng.random() < extra_family_prob:
        family_count += 1

    first_family = "none"
    for family_idx in range(family_count):
        family = _choose_family(rng, family_weights)
        if family_idx == 0:
            first_family = family
        func = FAMILY_FUNCS.get(family)
        if func is None:
            raise KeyError(f"unknown augmentation family: {family}")
        func(
            out, out_mask, weight, rng,
            cfg.get(family, {}),
            float(pos_weights.get(family, 1.0)),
            scale,
            clip,
            preserve_mask,
        )

    if astro_enabled and frb_mask.any():
        out = _add_frb_to_patch(out, frb_image, scale, clip)
        astro_mask = (frb_mask * (1.0 - out_mask)).astype(np.float32)

    out = np.clip(out, -clip, clip).astype(np.float32, copy=False)
    return AugmentResult(out, out_mask, weight, astro_mask, first_family)
