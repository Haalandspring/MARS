"""Contract tests for the architecture reported in the paper."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from mars_rfi.model import (  # noqa: E402
    PAPER_PARAMETER_COUNT,
    build_paper_model,
    count_parameters,
)


def test_paper_model_parameter_count_and_refinement_stages() -> None:
    model = build_paper_model()

    assert PAPER_PARAMETER_COUNT == 270_769
    assert count_parameters(model) == PAPER_PARAMETER_COUNT

    for stage in ("up2", "up1", "up0"):
        horizontal = getattr(model, f"refine_{stage}")
        vertical = getattr(model, f"vertical_refine_{stage}")
        assert not isinstance(horizontal, nn.Identity), f"horizontal {stage} is disabled"
        assert not isinstance(vertical, nn.Identity), f"vertical {stage} is disabled"


def test_paper_model_preserves_spatial_shape() -> None:
    model = build_paper_model().eval()
    inputs = torch.randn(2, 1, 64, 96)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 1, 64, 96)
