"""Tests for paper-specific loss and binary decision semantics."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mars_rfi.losses import astro_loss_from_logits, binary_metrics  # noqa: E402


def _logit(probability: "torch.Tensor") -> "torch.Tensor":
    return torch.logit(probability)


def test_astro_loss_uses_only_clean_astronomical_support() -> None:
    probabilities = torch.tensor([[[[0.2, 0.4, 0.6, 0.8]]]])
    logits = _logit(probabilities)
    target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
    astro_mask = torch.tensor([[[[1.0, 0.0, 1.0, 1.0]]]])

    loss = astro_loss_from_logits(logits, target, astro_mask)

    # The target-positive pixel is excluded even though it is in astro_mask.
    expected = torch.tensor((0.2**2 + 0.8**2) / 2)
    torch.testing.assert_close(loss, expected)


def test_astro_loss_falls_back_to_all_negative_targets_without_support() -> None:
    probabilities = torch.tensor([[[[0.2, 0.4, 0.6, 0.8]]]])
    logits = _logit(probabilities)
    target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
    empty_astro_mask = torch.zeros_like(target)

    loss = astro_loss_from_logits(logits, target, empty_astro_mask)

    expected = torch.tensor((0.2**2 + 0.4**2 + 0.8**2) / 3)
    torch.testing.assert_close(loss, expected)


def test_binary_metrics_treats_exactly_half_as_positive() -> None:
    logits = torch.tensor([[[[0.0, -0.001]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])

    metrics = binary_metrics(logits, target, threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["tn"] == 1.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 0.0
