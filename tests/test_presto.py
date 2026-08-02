"""Dependency-light tests for the paper PRESTO command contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_rfi import presto


def test_integer_dm_uses_presto_two_decimal_output_name(tmp_path: Path) -> None:
    prefix = tmp_path / "candidate"
    generated = tmp_path / "candidate_DM100.00.dat"
    generated.touch()

    assert presto.presto_dm_arg(100.0) == "100"
    assert presto.presto_dm_label(100.0) == "100.00"
    assert presto.select_dat_file(prefix, 100.0) == generated


def test_presto_commands_use_basename_inside_output_directory(tmp_path: Path) -> None:
    args = SimpleNamespace(
        nobary=True,
        zerodm=False,
        dm=73.81,
        numdms=1,
        zmax=200,
        numharm=8,
    )
    fil_path = tmp_path / "input.fil"
    out_prefix = tmp_path / "nested" / "candidate"
    dat_path = out_prefix.parent / "candidate_DM73.81.dat"

    prepsubband = presto.build_prepsubband_command(args, fil_path, out_prefix)
    accelsearch = presto.build_accelsearch_command(args, dat_path)

    assert prepsubband[-3:-1] == ["-o", "candidate"]
    assert prepsubband[-1] == str(fil_path)
    assert accelsearch == [
        "accelsearch",
        "-zmax",
        "200",
        "-numharm",
        "8",
        "candidate_DM73.81.dat",
    ]


def test_dat_discovery_handles_exact_fallback_empty_and_ambiguous(tmp_path: Path) -> None:
    prefix = tmp_path / "candidate"
    exact = tmp_path / "candidate_DM100.00.dat"
    other = tmp_path / "candidate_DM101.00.dat"
    exact.touch()
    other.touch()
    assert presto.select_dat_file(prefix, 100) == exact

    exact.unlink()
    other.unlink()
    nonstandard = tmp_path / "candidate_DM100.0.dat"
    nonstandard.touch()
    assert presto.select_dat_file(prefix, 100) == nonstandard

    nonstandard.unlink()
    assert presto.select_dat_file(prefix, 100) is None

    (tmp_path / "candidate_DM99.00.dat").touch()
    (tmp_path / "candidate_DM101.00.dat").touch()
    with pytest.raises(RuntimeError, match="Multiple PRESTO .dat files"):
        presto.select_dat_file(prefix, 100)
