"""Tests for paper augmentation cardinality."""

from __future__ import annotations

import importlib
import pytest


np = pytest.importorskip("numpy")

augment = importlib.import_module("mars_rfi.augment")  # noqa: E402


def test_maximum_three_families_is_reachable(monkeypatch) -> None:
    calls = []

    def record_family(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(augment.FAMILY_FUNCS, "probe", record_family)
    config = {
        "enabled": True,
        "tanh_scale": 6.0,
        "input_clip": 0.999,
        "base_noise_std": 0.0,
        "clean_sample_prob": 0.0,
        "astro_injection": {"enabled": False},
        "family_weights": {"probe": 1.0},
        "positive_weights": {"probe": 1.0},
        "extra_family_prob": 1.0,
        "max_families_per_patch": 3,
        "probe": {},
    }

    augment.augment_patch(
        np.zeros((16, 16), dtype=np.float32),
        None,
        rng=np.random.default_rng(1234),
        cfg=config,
    )

    assert len(calls) == 3
