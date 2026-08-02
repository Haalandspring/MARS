"""Repository-level checks that do not require PyTorch."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from mars_rfi.config import CONFIG as SOURCE_TRAIN_CONFIG
from mars_rfi.provenance import (
    PAPER_ARTIFACT_ROLE,
    PAPER_EXPERIMENT_ID,
    training_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def _conv_bn_parameters(in_channels: int, out_channels: int, kernel: int) -> int:
    """Count a bias-free convolution plus trainable BatchNorm scale/bias."""

    return in_channels * out_channels * kernel + 2 * out_channels


def _pipeline_source_defaults(keys: tuple[str, ...]) -> dict:
    """Read selected literal defaults without importing torch/sigpyproc."""

    source_path = ROOT / "src/mars_rfi/pipeline.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    config_node = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets)
    )
    assert isinstance(config_node, ast.Dict)

    selected = {}
    for key_node, value_node in zip(config_node.keys, config_node.values, strict=True):
        key = ast.literal_eval(key_node)
        if key not in keys:
            continue
        if (
            isinstance(value_node, ast.Subscript)
            and isinstance(value_node.value, ast.Name)
            and value_node.value.id == "PAPER_TRAINING_CONFIG"
        ):
            selected[key] = SOURCE_TRAIN_CONFIG[ast.literal_eval(value_node.slice)]
        else:
            selected[key] = ast.literal_eval(value_node)
    return selected


def _model_parameter_count(config: dict) -> int:
    c0, c1, c2, c3 = config["channels"]

    encoder = (
        _conv_bn_parameters(1, c0, 3 * 3)
        + _conv_bn_parameters(c0, c1, 3 * 3)
        + _conv_bn_parameters(c1, c2, 3 * 3)
        + _conv_bn_parameters(c2, c3, 3 * 3)
    )
    shape_context = (
        _conv_bn_parameters(c3, c3, 3 * 3)
        + _conv_bn_parameters(c3, c3, 1 * config["shape_kernel"])
        + _conv_bn_parameters(c3, c3, config["shape_kernel"] * 1)
        + _conv_bn_parameters(3 * c3, c3, 1)
    )
    decoder = sum(
        in_channels * out_channels
        + _conv_bn_parameters(out_channels, out_channels, 3 * 3)
        for in_channels, out_channels in ((c3, c2), (c2, c1), (c1, c0))
    )
    stage_channels = {"up2": c2, "up1": c1, "up0": c0}
    horizontal_refinement = 0
    if config["decoder_horizontal_refine_enabled"]:
        horizontal_refinement = sum(
            _conv_bn_parameters(
                stage_channels[stage],
                stage_channels[stage],
                config["decoder_horizontal_refine_kernel"],
            )
            + _conv_bn_parameters(stage_channels[stage], stage_channels[stage], 3 * 3)
            for stage in config["decoder_horizontal_refine_stages"]
        )
    vertical_refinement = 0
    if config["decoder_vertical_refine_enabled"]:
        vertical_refinement = sum(
            _conv_bn_parameters(
                stage_channels[stage],
                stage_channels[stage],
                config["decoder_vertical_refine_kernel"],
            )
            + _conv_bn_parameters(stage_channels[stage], stage_channels[stage], 3 * 3)
            for stage in config["decoder_vertical_refine_stages"]
        )
    head = c0 + 1  # 1x1 convolution with bias.

    return (
        encoder
        + shape_context
        + decoder
        + horizontal_refinement
        + vertical_refinement
        + head
    )


def test_paper_config_has_the_reported_parameter_count() -> None:
    config = json.loads((ROOT / "configs/train/paper.json").read_text(encoding="utf-8"))

    assert config["model"] == "trt_shape_unet"
    assert config["decoder_horizontal_refine_enabled"] is True
    assert config["decoder_vertical_refine_enabled"] is True
    assert config["decoder_horizontal_refine_stages"] == ["up2", "up1", "up0"]
    assert config["decoder_vertical_refine_stages"] == ["up2", "up1", "up0"]
    assert _model_parameter_count(config) == config["expected_parameters"] == 270_769
    assert config["experiment_id"] == PAPER_EXPERIMENT_ID
    assert config["artifact_role"] == PAPER_ARTIFACT_ROLE
    assert config["training_fingerprint"] == SOURCE_TRAIN_CONFIG["training_fingerprint"]
    assert config["training_fingerprint"] == training_fingerprint(SOURCE_TRAIN_CONFIG)


def test_repository_json_files_are_valid() -> None:
    paths = sorted((ROOT / "configs").rglob("*.json"))
    paths.append(ROOT / "reproduction/paper-manifest.json")

    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))


def test_decoder_ablation_parameter_guards_are_locked() -> None:
    config_dir = ROOT / "configs/train"
    expected = {
        "paper.json": 270_769,
        "no_horizontal_refinement.json": 216_785,
        "no_vertical_refinement.json": 216_785,
        "no_decoder_refinement.json": 162_801,
    }

    for name, parameter_count in expected.items():
        delta = json.loads((config_dir / name).read_text(encoding="utf-8"))
        config = {**SOURCE_TRAIN_CONFIG, **delta}
        assert config["expected_parameters"] == parameter_count
        assert _model_parameter_count(config) == parameter_count

    horizontal = json.loads(
        (config_dir / "no_horizontal_refinement.json").read_text(encoding="utf-8")
    )
    vertical = json.loads(
        (config_dir / "no_vertical_refinement.json").read_text(encoding="utf-8")
    )
    neither = json.loads(
        (config_dir / "no_decoder_refinement.json").read_text(encoding="utf-8")
    )
    assert horizontal["decoder_horizontal_refine_enabled"] is False
    assert vertical["decoder_vertical_refine_enabled"] is False
    assert neither["decoder_horizontal_refine_enabled"] is False
    assert neither["decoder_vertical_refine_enabled"] is False
    for delta in (horizontal, vertical, neither):
        assert delta["artifact_role"] == "ablation"
        assert delta["experiment_id"] != PAPER_EXPERIMENT_ID


def test_loss_ablation_has_a_distinct_training_identity() -> None:
    delta = json.loads(
        (ROOT / "configs/train/no_astro.json").read_text(encoding="utf-8")
    )
    config = {**SOURCE_TRAIN_CONFIG, **delta}

    assert _model_parameter_count(config) == 270_769
    assert training_fingerprint(config) != SOURCE_TRAIN_CONFIG["training_fingerprint"]
    assert config["artifact_role"] == "ablation"
    assert config["experiment_id"] != PAPER_EXPERIMENT_ID


def test_train_and_pipeline_paper_contracts_agree() -> None:
    train_json = json.loads(
        (ROOT / "configs/train/paper.json").read_text(encoding="utf-8")
    )
    pipeline_json = json.loads(
        (ROOT / "configs/pipeline/paper.json").read_text(encoding="utf-8")
    )
    shared_keys = (
        "experiment_id",
        "artifact_role",
        "training_fingerprint",
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
        "threshold",
    )
    source_pipeline = _pipeline_source_defaults(shared_keys)

    for key in shared_keys:
        source_value = SOURCE_TRAIN_CONFIG[key]
        if isinstance(source_value, tuple):
            source_value = list(source_value)
        assert train_json[key] == source_value
        assert pipeline_json[key] == source_value
        pipeline_source_value = source_pipeline[key]
        if isinstance(pipeline_source_value, tuple):
            pipeline_source_value = list(pipeline_source_value)
        assert pipeline_source_value == source_value
    pipeline_safety = _pipeline_source_defaults(
        ("allow_unverified_artifacts", "tensorrt_verification_max_diff")
    )
    assert pipeline_json["allow_unverified_artifacts"] is False
    assert pipeline_safety["allow_unverified_artifacts"] is False
    assert pipeline_json["tensorrt_verification_max_diff"] == 0.02
    assert pipeline_safety["tensorrt_verification_max_diff"] == 0.02


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    markdown_paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing: list[str] = []

    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = target.split("#", 1)[0]
            if relative_target and not (markdown_path.parent / relative_target).exists():
                missing.append(f"{markdown_path.relative_to(ROOT)} -> {target}")

    assert not missing, "Broken local links:\n" + "\n".join(missing)
