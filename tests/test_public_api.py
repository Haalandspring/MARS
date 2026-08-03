"""Tests for the small, mitigation-focused public Python API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("sigpyproc")

import mars_rfi  # noqa: E402
import mars_rfi.mitigate as mitigate_module  # noqa: E402
from mars_rfi import load_pipeline_config, mitigate_filterbank  # noqa: E402


def test_public_namespace_exposes_mitigation_not_reproduction() -> None:
    assert "mitigate_filterbank" in mars_rfi.__all__
    assert "load_pipeline_config" in mars_rfi.__all__
    assert "generate_candidates" not in mars_rfi.__all__
    assert "presto" not in mars_rfi.__all__


def test_config_loader_isolated_defaults_and_accepts_json(tmp_path: Path) -> None:
    first = load_pipeline_config()
    first["threshold"] = 0.123
    assert load_pipeline_config()["threshold"] != 0.123

    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"threshold": 0.75}), encoding="utf-8")
    assert load_pipeline_config(path)["threshold"] == 0.75

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_pipeline_config(path)


def test_python_api_validates_paths_and_calls_shared_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_fil = tmp_path / "input.fil"
    input_fil.write_bytes(b"filterbank")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output_fil = tmp_path / "output.fil"
    captured: dict = {}

    def fake_run(config: dict) -> dict[str, float]:
        captured.update(config)
        return {"total": 1.25}

    monkeypatch.setattr(mitigate_module, "run_pipeline", fake_run)
    timings = mitigate_filterbank(
        input_fil,
        output_fil,
        config={"checkpoint": str(checkpoint), "threshold": 0.6},
    )

    assert timings == {"total": 1.25}
    assert captured["input_fil"] == str(input_fil)
    assert captured["output_fil"] == str(output_fil)
    assert captured["threshold"] == 0.6

