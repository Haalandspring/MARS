"""TensorRT export policy tests that never initialize CUDA or TensorRT."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mars_rfi import export_tensorrt  # noqa: E402
from mars_rfi.provenance import (  # noqa: E402
    HISTORICAL_PAPER_ARTIFACT_ROLE,
    PAPER_ARTIFACT_ROLE,
)


def test_diagnostic_verification_failure_recording_is_explicit_opt_in() -> None:
    default_args = export_tensorrt.parse_args([])
    diagnostic_args = export_tensorrt.parse_args(["--record-verification-failure"])

    assert default_args.record_verification_failure is False
    assert diagnostic_args.record_verification_failure is True


@pytest.mark.parametrize(
    "artifact_role",
    [PAPER_ARTIFACT_ROLE, HISTORICAL_PAPER_ARTIFACT_ROLE],
)
def test_strict_artifact_roles_cannot_skip_numerical_verification(
    artifact_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test GPU")
    monkeypatch.setattr(
        export_tensorrt,
        "load_checkpoint_model",
        lambda _args, _device: (object(), {"artifact_role": artifact_role}, 0),
    )

    with pytest.raises(RuntimeError, match="cannot be exported with --skip-verify"):
        export_tensorrt.main(["--skip-verify"])
