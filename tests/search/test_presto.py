"""Dependency-light tests for the paper PRESTO command contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_rfi.search import presto


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


def test_presto_cleanup_is_opt_in_by_default() -> None:
    args = presto.parse_args(["--fil", "input.fil", "--dm", "1", "--no-container"])

    assert args.cleanup_products is False
    assert args.overwrite is False


def test_cleanup_can_be_limited_to_products_created_by_current_run(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "candidate"
    preexisting = tmp_path / "candidate_DM1.00.dat"
    created = tmp_path / "candidate_DM1.00.fft"
    preexisting.write_bytes(b"keep")
    created.write_bytes(b"remove")

    _, removed = presto.cleanup_presto_products(
        out_dir=tmp_path,
        out_prefix=prefix,
        dm_label="1.00",
        keep_suffix="ACCEL_0",
        eligible_paths={created},
    )

    assert removed == [str(created)]
    assert preexisting.read_bytes() == b"keep"
    assert not created.exists()


def test_default_run_refuses_to_overwrite_existing_products(tmp_path: Path) -> None:
    input_fil = tmp_path / "input.fil"
    input_fil.write_bytes(b"placeholder")
    output_dir = tmp_path / "presto"
    output_dir.mkdir()
    existing = output_dir / "input_DM1.00.dat"
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        presto.main(
            [
                "--fil",
                str(input_fil),
                "--dm",
                "1",
                "--out-dir",
                str(output_dir),
                "--no-container",
            ]
        )

    assert existing.read_bytes() == b"keep"


def test_unrequested_existing_summary_is_left_unchanged(tmp_path: Path) -> None:
    missing_fil = tmp_path / "missing.fil"
    output_dir = tmp_path / "presto"
    output_dir.mkdir()
    summary = output_dir / "presto_accelsearch_summary.json"
    summary.write_text("keep-me", encoding="utf-8")

    result = presto.main(
        [
            "--fil",
            str(missing_fil),
            "--dm",
            "1",
            "--out-dir",
            str(output_dir),
            "--no-container",
            "--skip-missing",
        ]
    )

    assert result == 0
    assert summary.read_text(encoding="utf-8") == "keep-me"
