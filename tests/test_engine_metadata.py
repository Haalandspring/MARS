"""Strict TensorRT-sidecar validation without importing torch or TensorRT."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mars_rfi.config import CONFIG
from mars_rfi.provenance import (
    HISTORICAL_PAPER_ARTIFACT_ROLE,
    training_fingerprint,
    validate_engine_metadata,
)


def _paper_runtime_config() -> dict:
    config = deepcopy(CONFIG)
    config.update(
        {
            "patch_size": 512,
            "batch_size": 64,
            "tensorrt_verification_max_diff": 0.02,
        }
    )
    return config


def _historical_runtime_config() -> dict:
    config = _paper_runtime_config()
    config.update(
        {
            "experiment_id": "mars-paper-historical-2026-06-17",
            "artifact_role": HISTORICAL_PAPER_ARTIFACT_ROLE,
        }
    )
    config["augmentation"][
        "implementation_profile"
    ] = "single-extra-family-draw-max-two-v1"
    config["training_fingerprint"] = training_fingerprint(config)
    return config


def _write_sidecar(tmp_path: Path, *, model_config: dict | None = None) -> tuple[Path, dict]:
    engine_path = tmp_path / "mars.engine"
    engine_path.write_bytes(b"paper-engine-bytes")
    metadata = {
        "schema_version": 1,
        "checkpoint_sha256": "0" * 64,
        "onnx_sha256": "1" * 64,
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "model_config": deepcopy(model_config or CONFIG),
        "trainable_parameters": 270_769,
        "verification": {
            "status": "passed",
            "metric": "sigmoid_absolute_difference",
            "max_abs_diff": 0.01,
            "mean_abs_diff": 0.001,
            "max_allowed_diff": 0.02,
            "batch_size": 4,
        },
        "build": {"patch_size": 512, "batch_size": 64},
    }
    Path(f"{engine_path}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return engine_path, metadata


def _rewrite_sidecar(engine_path: Path, metadata: dict) -> None:
    Path(f"{engine_path}.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_valid_paper_engine_sidecar_passes(tmp_path: Path) -> None:
    engine_path, _ = _write_sidecar(tmp_path)

    assert validate_engine_metadata(engine_path, _paper_runtime_config()) == []


def test_valid_historical_paper_engine_sidecar_passes_strict_validation(
    tmp_path: Path,
) -> None:
    runtime = _historical_runtime_config()
    engine_path, _ = _write_sidecar(tmp_path, model_config=runtime)

    assert validate_engine_metadata(engine_path, runtime) == []


def test_missing_sidecar_and_missing_engine_hash_fail_closed(tmp_path: Path) -> None:
    engine_path = tmp_path / "mars.engine"
    engine_path.write_bytes(b"engine")
    with pytest.raises(RuntimeError, match="metadata not found"):
        validate_engine_metadata(engine_path, _paper_runtime_config())

    engine_path, metadata = _write_sidecar(tmp_path)
    del metadata["engine_sha256"]
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="engine_sha256"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


def test_engine_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    engine_path, metadata = _write_sidecar(tmp_path)
    metadata["engine_sha256"] = "f" * 64
    _rewrite_sidecar(engine_path, metadata)

    with pytest.raises(RuntimeError, match="engine hash does not match"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


def test_same_topology_loss_ablation_is_rejected(tmp_path: Path) -> None:
    ablation = deepcopy(CONFIG)
    ablation.update(
        {
            "experiment_id": "mars-ablation-no-astro-v1",
            "artifact_role": "ablation",
            "lambda_astro": 0.0,
        }
    )
    ablation["training_fingerprint"] = training_fingerprint(ablation)
    engine_path, _ = _write_sidecar(tmp_path, model_config=ablation)

    with pytest.raises(RuntimeError, match="does not match the requested model config"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


def test_unverified_or_inconsistent_verification_is_rejected(tmp_path: Path) -> None:
    engine_path, metadata = _write_sidecar(tmp_path)
    metadata["verification"]["status"] = "skipped"
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="not recorded as numerically verified"):
        validate_engine_metadata(engine_path, _paper_runtime_config())
    warnings = validate_engine_metadata(
        engine_path,
        _paper_runtime_config(),
        allow_unverified=True,
    )
    assert len(warnings) == 1

    metadata["verification"].update(
        {"status": "passed", "max_abs_diff": 0.03, "max_allowed_diff": 0.02}
    )
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="verification metrics are inconsistent"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


def test_engine_build_profile_must_cover_runtime_shape(tmp_path: Path) -> None:
    engine_path, metadata = _write_sidecar(tmp_path)
    metadata["build"]["patch_size"] = 256
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="patch size"):
        validate_engine_metadata(engine_path, _paper_runtime_config())

    metadata["build"].update({"patch_size": 512, "batch_size": 32})
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="outside the engine profile"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


def test_schema_rejects_float_integers_and_wrong_metric(tmp_path: Path) -> None:
    engine_path, metadata = _write_sidecar(tmp_path)
    metadata["trainable_parameters"] = 270_769.9
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="positive integer"):
        validate_engine_metadata(engine_path, _paper_runtime_config())

    metadata["trainable_parameters"] = 270_769
    metadata["verification"]["metric"] = "some_other_metric"
    _rewrite_sidecar(engine_path, metadata)
    with pytest.raises(RuntimeError, match="metric is unsupported"):
        validate_engine_metadata(engine_path, _paper_runtime_config())


@pytest.mark.parametrize("requested_tolerance", [None, float("nan"), float("inf"), 0.03])
def test_paper_verification_tolerance_cannot_be_disabled_or_weakened(
    tmp_path: Path,
    requested_tolerance: float | None,
) -> None:
    engine_path, _ = _write_sidecar(tmp_path)
    runtime = _paper_runtime_config()
    runtime["tensorrt_verification_max_diff"] = requested_tolerance

    with pytest.raises(RuntimeError, match="tolerance|max_diff"):
        validate_engine_metadata(engine_path, runtime)
