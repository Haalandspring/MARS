#!/usr/bin/env python3
"""Simple source-checkout entry point for MARS RFI mitigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "src"
DEFAULT_CONFIG = ROOT / "config.json"
ARTIFACT_PATH_KEYS = ("checkpoint", "tensorrt_path")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove RFI from one 8-bit SIGPROC filterbank with MARS.",
    )
    parser.add_argument(
        "-f",
        "--filterbank",
        required=True,
        type=Path,
        help="Input 8-bit SIGPROC .fil file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help=(
            "Output .fil file or output directory. A directory produces "
            "<input>_mars.fil."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"MARS JSON configuration (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--baseline-streaming",
        "-baseline_streaming",
        action="store_true",
        help=(
            "Compute the exact baseline running median in bounded-memory time "
            "chunks. Recommended for large or high-time-resolution filterbanks."
        ),
    )
    parser.add_argument(
        "--baseline-streaming-workspace-mb",
        "-baseline_streaming_workspace_mb",
        type=float,
        metavar="MIB",
        help=(
            "Approximate baseline streaming workspace in MiB "
            "(default: baseline_median_workspace_mb in the configuration; "
            "requires --baseline-streaming)."
        ),
    )
    return parser.parse_args(argv)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"MARS configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"MARS configuration must contain a JSON object: {path}")

    # Artifact paths in config.json are relative to the downloaded repository,
    # not to the shell's current working directory.
    for key in ARTIFACT_PATH_KEYS:
        value = config.get(key)
        if value:
            artifact = Path(value).expanduser()
            if not artifact.is_absolute():
                artifact = ROOT / artifact
            config[key] = str(artifact.resolve())
    return config


def resolve_output_path(input_path: Path, output_argument: Path) -> Path:
    output = output_argument.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.is_dir() or output.suffix.lower() != ".fil":
        output = output / f"{input_path.stem}_mars.fil"
    return output.resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.filterbank.expanduser().resolve()
    output_path = resolve_output_path(input_path, args.output)
    config_path = args.config.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input filterbank not found: {input_path}")

    config = load_config(config_path)
    if args.baseline_streaming_workspace_mb is not None:
        if not args.baseline_streaming:
            raise ValueError(
                "--baseline-streaming-workspace-mb requires --baseline-streaming"
            )
        if args.baseline_streaming_workspace_mb <= 0:
            raise ValueError("--baseline-streaming-workspace-mb must be greater than zero")
    # This feature is intentionally controlled per invocation, not by
    # config.json. Absence of the flag always selects the historical one-shot
    # running median even if an old local config still contains this key.
    config["baseline_median_streaming_enabled"] = bool(args.baseline_streaming)
    if args.baseline_streaming_workspace_mb is not None:
        config["baseline_median_workspace_mb"] = float(
            args.baseline_streaming_workspace_mb
        )
    sys.path.insert(0, str(SOURCE_DIR))
    try:
        from mars_rfi.mitigate import mitigate_filterbank
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        raise SystemExit(
            f"Missing Python dependency: {missing}. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    print(f"MARS config: {config_path}")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    timings = mitigate_filterbank(
        input_path,
        output_path,
        config=config,
    )
    total = timings.get("total")
    if total is not None:
        print(f"MARS completed in {float(total):.3f} s")
    else:
        print("MARS completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
