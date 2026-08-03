"""Round-trip tests for channel/time patch splitting and reconstruction."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("sigpyproc")

from mars_rfi.pipeline import (  # noqa: E402
    CONFIG,
    MINIMUM_SUPPORTED_CHANNELS,
    reconstruct_segment_mask,
    run_pipeline,
    split_segment_to_patches,
)
import mars_rfi.pipeline as pipeline_module  # noqa: E402


PATCH_SIZE = 512


def _pattern(nchans: int, ntime: int) -> "torch.Tensor":
    channels = torch.arange(nchans).unsqueeze(1)
    times = torch.arange(ntime).unsqueeze(0)
    return ((channels * 3 + times * 5) % 11) < 4


def _round_trip(segment: "torch.Tensor") -> tuple["torch.Tensor", dict]:
    patches, meta = split_segment_to_patches(segment, PATCH_SIZE)
    reconstructed = reconstruct_segment_mask(patches.squeeze(1), meta, PATCH_SIZE)
    return reconstructed, meta


def test_split_reconstruct_exactly_512_channels() -> None:
    segment = _pattern(512, 2 * PATCH_SIZE + 17)

    reconstructed, meta = _round_trip(segment)

    assert meta == {
        "nchans": 512,
        "n_time_patches": 2,
        "full_chan_blocks": 1,
        "has_overlap": False,
        "overlap_chans": 0,
        "small_channel_packing": False,
        "time_blocks_per_patch": 1,
        "patch_count": 2,
    }
    assert reconstructed.shape == (512, 2 * PATCH_SIZE)
    torch.testing.assert_close(reconstructed, segment[:, : 2 * PATCH_SIZE])


def test_split_reconstruct_non_multiple_channels_covers_frequency_tail() -> None:
    nchans = 701
    segment = _pattern(nchans, 2 * PATCH_SIZE)
    # Make the final channel unique enough to catch a missing/incorrect tail copy.
    segment[-1] = False
    segment[-1, -1] = True

    patches, meta = split_segment_to_patches(segment, PATCH_SIZE)
    reconstructed = reconstruct_segment_mask(patches.squeeze(1), meta, PATCH_SIZE)

    assert patches.shape == (4, 1, PATCH_SIZE, PATCH_SIZE)
    assert meta["full_chan_blocks"] == 1
    assert meta["has_overlap"] is True
    assert meta["overlap_chans"] == nchans - PATCH_SIZE
    torch.testing.assert_close(reconstructed, segment)
    assert reconstructed[-1, -1]


class _Header:
    nsamples = 1
    tsamp = 0.001

    def __init__(self, nchans: int):
        self.nchans = nchans


def _reader_with_channels(nchans: int):
    class _Reader:
        def __init__(self, _path: str):
            self.header = _Header(nchans)

    return _Reader


def test_pipeline_rejects_fewer_than_512_channels_before_reading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline_module, "FilReader", _reader_with_channels(511))
    cfg = {
        "input_fil": "input.fil",
        "output_fil": "output.fil",
        "minimum_supported_channels": MINIMUM_SUPPORTED_CHANNELS,
    }

    with pytest.raises(NotImplementedError, match=r"C >= 512.*C=511"):
        run_pipeline(cfg)


def test_pipeline_channel_boundary_cannot_be_lowered_by_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline_module, "FilReader", _reader_with_channels(512))
    cfg = {
        "input_fil": "input.fil",
        "output_fil": "output.fil",
        "minimum_supported_channels": 256,
    }

    with pytest.raises(ValueError, match=r"cannot be lowered.*C=512"):
        run_pipeline(cfg)


def test_default_pipeline_contract_starts_at_512_channels() -> None:
    assert MINIMUM_SUPPORTED_CHANNELS == PATCH_SIZE
    assert CONFIG["minimum_supported_channels"] == PATCH_SIZE
