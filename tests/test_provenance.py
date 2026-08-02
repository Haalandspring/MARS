"""Artifact identity tests that do not import the ML runtime."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mars_rfi.config import CONFIG
from mars_rfi.provenance import training_fingerprint, validate_training_identity


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
