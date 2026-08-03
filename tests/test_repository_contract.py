"""Repository-level checks that do not require PyTorch."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
import tomllib

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
        elif isinstance(value_node, ast.Name):
            constant_node = next(
                node.value
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == value_node.id
                    for target in node.targets
                )
            )
            selected[key] = ast.literal_eval(constant_node)
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
    config = json.loads((ROOT / "configs/train/mars.json").read_text(encoding="utf-8"))

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

    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))


def test_train_and_pipeline_paper_contracts_agree() -> None:
    train_json = json.loads(
        (ROOT / "configs/train/mars.json").read_text(encoding="utf-8")
    )
    pipeline_json = json.loads(
        (ROOT / "configs/pipeline/mars.json").read_text(encoding="utf-8")
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
        (
            "allow_unverified_artifacts",
            "tensorrt_verification_max_diff",
            "minimum_supported_channels",
        )
    )
    assert pipeline_json["allow_unverified_artifacts"] is False
    assert pipeline_safety["allow_unverified_artifacts"] is False
    assert pipeline_json["tensorrt_verification_max_diff"] == 0.02
    assert pipeline_safety["tensorrt_verification_max_diff"] == 0.02
    assert pipeline_json["minimum_supported_channels"] == 512
    assert pipeline_safety["minimum_supported_channels"] == 512


def test_historical_pipeline_profile_has_a_separate_locked_identity() -> None:
    config = json.loads(
        (ROOT / "configs/pipeline/mitigation.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["experiment_id"] == "mars-paper-historical-2026-06-17"
    assert config["artifact_role"] == "historical-paper"
    assert config["training_fingerprint"] == (
        "b5e7bff877cc1f160eb952b4e41e664f74ff12f5a552679f511cfaf394fbf265"
    )
    assert config["minimum_supported_channels"] == 512
    assert config["allow_unverified_artifacts"] is False


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    markdown_paths = [
        ROOT / "README.md",
        ROOT / "artifacts/README.md",
        ROOT / "data/README.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
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


def test_search_namespace_is_separated_from_core() -> None:
    package_root = ROOT / "src/mars_rfi"
    search_only = {
        "presto",
        "presto_candidates",
    }
    core_modules = {path.stem for path in package_root.glob("*.py")}
    search_modules = {
        path.stem for path in (package_root / "search").glob("*.py")
    }

    assert core_modules.isdisjoint(search_only)
    assert search_only <= search_modules

    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("search" in module for module in imported), source_path


def test_console_scripts_make_the_runtime_boundary_explicit() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    scripts = project["scripts"]

    assert {
        "mars-train",
        "mars-mitigate",
        "mars-export-tensorrt",
        "mars-diagnostics",
        "mars-search-presto",
        "mars-search-match",
    } <= scripts.keys()
    for name, target in scripts.items():
        assert ".paper_validation." not in target
        assert ".migration." not in target
        if ".search." in target:
            assert name.startswith("mars-search-")

    assert not any(name.startswith("mars-validate-") for name in scripts)
    assert not any(name.startswith("mars-reproduce-") for name in scripts)
    assert not any(name.startswith("mars-migrate-") for name in scripts)
    assert "mars-presto" not in scripts


def test_private_validation_and_large_data_are_ignored() -> None:
    rules = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {
        "/paper_validation/",
        "/src/mars_rfi/paper_validation/",
        "/tests/paper_validation/",
        "/src/mars_rfi/migration/",
        "/tests/migration/",
        "data/*",
        "results/",
        "outputs/",
        "search_outputs/",
        "*.fil",
        "*.npy",
        "*.pt",
        "*.engine",
    } <= rules
