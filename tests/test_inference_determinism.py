"""Tests for explicit science/performance inference-mode routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("sigpyproc")

from mars_rfi.mitigate import parse_args, resolve_config  # noqa: E402
import mars_rfi.pipeline as pipeline_module  # noqa: E402
from mars_rfi.pipeline import (  # noqa: E402
    CONFIG,
    _deterministic_median_values,
    _median_values,
    inference_runtime_settings,
    resolve_inference_runtime_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def _pipeline_config(filename: str) -> dict:
    path = ROOT / "configs" / "pipeline" / filename
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_profiles_enable_determinism_explicitly() -> None:
    assert CONFIG["deterministic_inference"] is False
    assert _pipeline_config("mars.json")["deterministic_inference"] is True
    assert _pipeline_config("mitigation.json")["deterministic_inference"] is True


def test_mitigation_cli_can_select_mode_and_seed() -> None:
    args = parse_args(
        ["--deterministic-inference", "--inference-seed", "987654321"]
    )
    config = resolve_config(args)

    assert config["deterministic_inference"] is True
    assert config["inference_seed"] == 987654321


def test_inference_runtime_settings_switch_and_restore_process_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_rng = torch.random.get_rng_state().clone()
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        state_before = torch.random.get_rng_state().clone()

        with inference_runtime_settings(
            {"deterministic_inference": True, "inference_seed": 42}
        ) as settings:
            assert settings["cudnn_benchmark"] is False
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
            assert torch.are_deterministic_algorithms_enabled() is True
            assert torch.initial_seed() == 42

        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.equal(torch.random.get_rng_state(), state_before)

        with inference_runtime_settings(
            {"deterministic_inference": False, "inference_seed": 42}
        ) as settings:
            assert settings["cudnn_benchmark"] is True
            assert torch.backends.cudnn.benchmark is True
            assert torch.backends.cudnn.deterministic is False
            assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(
            original_algorithms,
            warn_only=original_warn_only,
        )
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cudnn.benchmark = original_benchmark
        torch.random.set_rng_state(original_rng)


def test_deterministic_median_is_chunked_and_matches_lower_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module, "DETERMINISTIC_MEDIAN_SORT_MAX_ELEMENTS", 4
    )
    values = torch.tensor(
        [[4.0, 1.0, 3.0, 2.0], [9.0, 7.0, 8.0, 6.0], [5.0, 5.0, 2.0, 1.0]]
    )

    expected = values.median(dim=1, keepdim=True).values
    actual = _deterministic_median_values(values, dim=1, keepdim=True)

    torch.testing.assert_close(actual, expected)
    assert _median_values(values.flatten()).item() == values.flatten().median().item()


def test_tensorrt_scope_is_not_overstated() -> None:
    settings = resolve_inference_runtime_settings(
        {
            "deterministic_inference": True,
            "inference_seed": 1,
            "tensorrt_path": "model.engine",
        }
    )

    assert settings["backend"] == "tensorrt"
    assert settings["determinism_scope"] == "torch-pre-and-postprocessing-only"
    assert settings["tensorrt_determinism_guaranteed"] is False
