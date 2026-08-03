"""Artifact identity tests that do not import the ML runtime."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mars_rfi.config import CONFIG
from mars_rfi.provenance import (
    HISTORICAL_PAPER_ARTIFACT_ROLE,
    PAPER_ARTIFACT_ROLE,
    STRICT_TENSORRT_ARTIFACT_ROLES,
    training_fingerprint,
    validate_training_identity,
)


def test_paper_training_identity_is_self_consistent() -> None:
    validate_training_identity(CONFIG, source="paper config")


def test_loss_only_tampering_invalidates_paper_fingerprint() -> None:
    tampered = deepcopy(CONFIG)
    tampered["lambda_astro"] = 0.0

    with pytest.raises(ValueError, match="training fingerprint mismatch"):
        validate_training_identity(tampered, source="tampered checkpoint")


def test_training_data_path_substitution_invalidates_paper_fingerprint() -> None:
    tampered = deepcopy(CONFIG)
    tampered["train_patch_path"] = "somewhere/else/train_patches.npy"

    with pytest.raises(ValueError, match="training fingerprint mismatch"):
        validate_training_identity(tampered, source="substituted paper dataset")


def test_explicit_ablation_can_be_stamped_with_a_distinct_identity() -> None:
    ablation = deepcopy(CONFIG)
    ablation.update(
        {
            "experiment_id": "mars-ablation-no-astro-v1",
            "artifact_role": "ablation",
            "lambda_astro": 0.0,
            "out_dir": "artifacts/checkpoints/ablation-no-astro",
        }
    )
    ablation["training_fingerprint"] = training_fingerprint(ablation)

    validate_training_identity(ablation, source="no-astro checkpoint")
    assert ablation["training_fingerprint"] != CONFIG["training_fingerprint"]


def test_historical_paper_role_is_strict_for_tensorrt() -> None:
    assert STRICT_TENSORRT_ARTIFACT_ROLES == frozenset(
        {PAPER_ARTIFACT_ROLE, HISTORICAL_PAPER_ARTIFACT_ROLE}
    )


def test_historical_identity_requires_its_own_fingerprint() -> None:
    historical = deepcopy(CONFIG)
    historical.update(
        {
            "experiment_id": "mars-paper-historical-2026-06-17",
            "artifact_role": HISTORICAL_PAPER_ARTIFACT_ROLE,
        }
    )
    historical["augmentation"][
        "implementation_profile"
    ] = "single-extra-family-draw-max-two-v1"

    with pytest.raises(ValueError, match="training fingerprint mismatch"):
        validate_training_identity(historical, source="unstamped historical checkpoint")

    historical["training_fingerprint"] = training_fingerprint(historical)
    validate_training_identity(historical, source="historical checkpoint")
    assert historical["training_fingerprint"] == (
        "b5e7bff877cc1f160eb952b4e41e664f74ff12f5a552679f511cfaf394fbf265"
    )
