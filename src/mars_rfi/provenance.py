"""Stable experiment identity and training-config fingerprints for MARS artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path


PAPER_EXPERIMENT_ID = "mars-paper-2026-07-31"
PAPER_ARTIFACT_ROLE = "paper"
FINGERPRINT_SCHEMA_VERSION = 1
PAPER_TENSORRT_MAX_DIFF = 0.02

# Worker counts and output locations are deliberately excluded. The four
# canonical relative data paths are included so CLI path substitution cannot
# retain paper identity; content hashes remain unavailable until data release.
TRAINING_FINGERPRINT_KEYS = (
    "train_patch_path",
    "train_mask_path",
    "val_patch_path",
    "val_mask_path",
    "model",
    "in_channels",
    "out_channels",
    "channels",
    "output_bias_prior",
    "shape_kernel",
    "decoder_horizontal_refine_enabled",
    "decoder_horizontal_refine_stages",
    "decoder_horizontal_refine_kernel",
    "decoder_vertical_refine_enabled",
    "decoder_vertical_refine_stages",
    "decoder_vertical_refine_kernel",
    "expected_parameters",
    "expected_train_examples",
    "expected_val_examples",
    "epochs",
    "batch_size",
    "optimizer",
    "lr",
    "weight_decay",
    "pos_weight",
    "lambda_dice",
    "lambda_astro",
    "focal_gamma",
    "clip_grad_norm",
    "amp",
    "scheduler",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_min_lr",
    "seed",
    "threshold",
    "augmentation",
)

ARTIFACT_IDENTITY_KEYS = (
    "experiment_id",
    "artifact_role",
    "training_fingerprint",
)

MODEL_CONFIG_KEYS = (
    *ARTIFACT_IDENTITY_KEYS,
    "model",
    "channels",
    "shape_kernel",
    "decoder_horizontal_refine_enabled",
    "decoder_horizontal_refine_stages",
    "decoder_horizontal_refine_kernel",
    "decoder_vertical_refine_enabled",
    "decoder_vertical_refine_stages",
    "decoder_vertical_refine_kernel",
    "expected_parameters",
)


def canonical_training_payload(config: Mapping) -> dict:
    """Return the versioned, JSON-serializable payload used for hashing."""

    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "training": {key: config.get(key) for key in TRAINING_FINGERPRINT_KEYS},
    }


def training_fingerprint(config: Mapping) -> str:
    payload = canonical_training_payload(config)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_training_identity(config: Mapping, *, source: str = "artifact") -> None:
    missing = [key for key in ARTIFACT_IDENTITY_KEYS if key not in config]
    if missing:
        raise ValueError(f"{source} is missing artifact identity keys: {missing}")

    experiment_id = str(config["experiment_id"]).strip()
    artifact_role = str(config["artifact_role"]).strip()
    if not experiment_id or not artifact_role:
        raise ValueError(f"{source} has an empty experiment_id or artifact_role")

    stored = str(config["training_fingerprint"])
    calculated = training_fingerprint(config)
    if stored != calculated:
        raise ValueError(
            f"{source} training fingerprint mismatch: stored={stored}, "
            f"calculated={calculated}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"TensorRT metadata field {field!r} is not a valid SHA-256")
    return text


def _require_positive_int(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"TensorRT metadata field {field!r} must be a positive integer")
    return value


def _require_nonnegative_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"TensorRT metadata field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(
            f"TensorRT metadata field {field!r} must be finite and non-negative"
        )
    return result


def _normalise_config_value(value):
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_config_value(item) for item in value)
    return value


def model_config_mismatches(requested_cfg: Mapping, artifact_cfg: Mapping) -> list[str]:
    if not isinstance(artifact_cfg, Mapping):
        return [f"artifact model_config is not an object: {type(artifact_cfg).__name__}"]

    mismatches = []
    for key in MODEL_CONFIG_KEYS:
        if key not in requested_cfg or key not in artifact_cfg:
            continue
        requested = _normalise_config_value(requested_cfg[key])
        built = _normalise_config_value(artifact_cfg[key])
        if requested != built:
            mismatches.append(f"{key}: requested={requested!r}, artifact={built!r}")
    return mismatches


def validate_engine_metadata(
    engine_path: str | Path,
    requested_cfg: Mapping,
    *,
    allow_unverified: bool = False,
) -> list[str]:
    """Validate the engine hash, identity, build profile, and numerical check."""

    engine_path = Path(engine_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")
    metadata_path = Path(f"{engine_path}.json")
    if not metadata_path.is_file():
        message = (
            f"TensorRT metadata not found: {metadata_path}. "
            "Checkpoint/config provenance cannot be verified."
        )
        if allow_unverified:
            return [message]
        raise RuntimeError(message)

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"TensorRT metadata is not a JSON object: {metadata_path}")
    if (
        isinstance(metadata.get("schema_version"), bool)
        or not isinstance(metadata.get("schema_version"), int)
        or metadata.get("schema_version") != 1
    ):
        raise RuntimeError(
            f"Unsupported TensorRT metadata schema in {metadata_path}: "
            f"{metadata.get('schema_version')!r}"
        )

    engine_hash = _require_sha256(metadata.get("engine_sha256"), field="engine_sha256")
    _require_sha256(metadata.get("checkpoint_sha256"), field="checkpoint_sha256")
    _require_sha256(metadata.get("onnx_sha256"), field="onnx_sha256")
    if _sha256_file(engine_path) != engine_hash:
        raise RuntimeError(f"TensorRT engine hash does not match {metadata_path}")

    engine_cfg = metadata.get("model_config")
    if not isinstance(engine_cfg, Mapping):
        raise RuntimeError(f"TensorRT metadata has no valid model_config: {metadata_path}")
    missing_keys = [
        key for key in MODEL_CONFIG_KEYS if key in requested_cfg and key not in engine_cfg
    ]
    if missing_keys:
        raise RuntimeError(
            f"TensorRT metadata is missing critical model keys {missing_keys}: {metadata_path}"
        )
    try:
        validate_training_identity(engine_cfg, source=f"TensorRT metadata {metadata_path}")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    mismatches = model_config_mismatches(requested_cfg, engine_cfg)
    if mismatches:
        raise RuntimeError(
            "TensorRT engine metadata does not match the requested model config: "
            + "; ".join(mismatches)
        )

    trainable_parameters = _require_positive_int(
        metadata.get("trainable_parameters"),
        field="trainable_parameters",
    )
    expected_parameters = requested_cfg.get("expected_parameters")
    if expected_parameters is not None:
        expected_parameters = _require_positive_int(
            expected_parameters,
            field="requested.expected_parameters",
        )
    if expected_parameters is not None and trainable_parameters != expected_parameters:
        raise RuntimeError(
            "TensorRT trainable-parameter metadata mismatch: "
            f"recorded={trainable_parameters:,}, requested={expected_parameters:,}"
        )

    build = metadata.get("build")
    if not isinstance(build, Mapping):
        raise RuntimeError("TensorRT metadata has no valid build profile")
    build_patch_size = _require_positive_int(build.get("patch_size"), field="build.patch_size")
    build_batch_size = _require_positive_int(build.get("batch_size"), field="build.batch_size")
    requested_patch_size = requested_cfg.get("patch_size")
    if requested_patch_size is not None:
        requested_patch_size = _require_positive_int(
            requested_patch_size,
            field="requested.patch_size",
        )
    if requested_patch_size is not None and build_patch_size != requested_patch_size:
        raise RuntimeError(
            f"TensorRT patch size {build_patch_size} does not match requested "
            f"{requested_patch_size}"
        )
    requested_batch_size = _require_positive_int(
        requested_cfg.get("batch_size", build_batch_size),
        field="requested.batch_size",
    )
    if requested_batch_size > build_batch_size:
        raise RuntimeError(
            f"Requested TensorRT batch size {requested_batch_size} is outside the "
            f"engine profile [1, {build_batch_size}]"
        )

    verification = metadata.get("verification")
    if not isinstance(verification, Mapping):
        verification = {}
    if verification.get("status") != "passed":
        message = (
            f"TensorRT engine is not recorded as numerically verified: {metadata_path} "
            f"(status={verification.get('status', 'missing')!r})"
        )
        if allow_unverified:
            return [message]
        raise RuntimeError(message)

    if verification.get("metric") != "sigmoid_absolute_difference":
        raise RuntimeError("TensorRT verification metric is unsupported or missing")
    max_diff = _require_nonnegative_number(
        verification.get("max_abs_diff"),
        field="verification.max_abs_diff",
    )
    mean_diff = _require_nonnegative_number(
        verification.get("mean_abs_diff"),
        field="verification.mean_abs_diff",
    )
    max_allowed = _require_nonnegative_number(
        verification.get("max_allowed_diff"),
        field="verification.max_allowed_diff",
    )
    verification_batch = _require_positive_int(
        verification.get("batch_size"),
        field="verification.batch_size",
    )
    if mean_diff > max_diff or max_diff > max_allowed:
        raise RuntimeError(
            "TensorRT verification metrics are inconsistent: "
            f"mean={mean_diff}, max={max_diff}, allowed={max_allowed}"
        )
    requested_tolerance = _require_nonnegative_number(
        requested_cfg.get("tensorrt_verification_max_diff"),
        field="requested.tensorrt_verification_max_diff",
    )
    if requested_tolerance > PAPER_TENSORRT_MAX_DIFF:
        raise RuntimeError(
            f"Requested TensorRT tolerance {requested_tolerance} exceeds the paper hard "
            f"limit {PAPER_TENSORRT_MAX_DIFF}"
        )
    if max_allowed > requested_tolerance or max_allowed > PAPER_TENSORRT_MAX_DIFF:
        raise RuntimeError(
            f"TensorRT verification tolerance {max_allowed} is weaker than requested "
            f"{requested_tolerance} or the paper hard limit {PAPER_TENSORRT_MAX_DIFF}"
        )
    if verification_batch < 1 or verification_batch > build_batch_size:
        raise RuntimeError(
            f"TensorRT verification batch {verification_batch} is outside build profile "
            f"[1, {build_batch_size}]"
        )
    return []
