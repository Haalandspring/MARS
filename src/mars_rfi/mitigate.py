"""Command-line entry point for the MARS filterbank mitigation pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

from .pipeline import CONFIG as PIPELINE_DEFAULTS
from .pipeline import run_pipeline


def _json_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_pipeline_config(
    config: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an isolated pipeline config from defaults, JSON, or overrides."""

    resolved: dict[str, Any] = deepcopy(PIPELINE_DEFAULTS)
    if config is None:
        return resolved
    if isinstance(config, Mapping):
        override = dict(config)
    else:
        config_path = Path(config)
        with config_path.open(encoding="utf-8") as handle:
            override = json.load(handle)
        if not isinstance(override, dict):
            raise TypeError(f"Pipeline config must be a JSON object: {config_path}")
    resolved.update(override)
    return resolved


def mitigate_filterbank(
    input_fil: str | Path,
    output_fil: str | Path,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Mitigate one filterbank through the same validated path as the CLI.

    ``config`` may be a JSON path or a mapping of pipeline overrides. Input and
    output paths are explicit arguments and always take precedence over values
    stored in that configuration.
    """

    resolved = load_pipeline_config(config)
    resolved["input_fil"] = str(input_fil)
    resolved["output_fil"] = str(output_fil)
    validate_config(resolved)
    return run_pipeline(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the paper-aligned MARS mask-guided mitigation pipeline.",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config override.")
    parser.add_argument("--input-fil", type=Path)
    parser.add_argument("--output-fil", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tensorrt-engine", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use deterministic PyTorch/cuDNN algorithms for reproducible science "
            "outputs; disable for the historical throughput benchmark."
        ),
    )
    parser.add_argument("--inference-seed", type=int)
    parser.add_argument(
        "--hysteresis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional appendix post-processing; disabled in paper main results.",
    )
    parser.add_argument(
        "--zdot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the optional post-replacement zero-DM projection.",
    )
    parser.add_argument(
        "--zdot-stage",
        choices=("pre_replacement", "post_replacement", "off"),
    )
    parser.add_argument(
        "--write-mask-files",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Set an advanced flat pipeline key, for example --set raw_gpu_dtype=\"float32\".",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit without reading a filterbank.",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict:
    config = load_pipeline_config(args.config)

    if args.input_fil is not None:
        config["input_fil"] = str(args.input_fil)
    if args.output_fil is not None:
        config["output_fil"] = str(args.output_fil)
    if args.checkpoint is not None:
        config["checkpoint"] = str(args.checkpoint)
    if args.tensorrt_engine is not None:
        config["tensorrt_path"] = str(args.tensorrt_engine)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.threshold is not None:
        config["threshold"] = float(args.threshold)
    if args.deterministic_inference is not None:
        config["deterministic_inference"] = bool(args.deterministic_inference)
    if args.inference_seed is not None:
        config["inference_seed"] = int(args.inference_seed)
    if args.hysteresis is not None:
        config["hys_enabled"] = bool(args.hysteresis)
    if args.zdot is not None:
        config["science_zdot"] = bool(args.zdot)
    if args.zdot_stage is not None:
        config["science_zdot_stage"] = args.zdot_stage
    if args.write_mask_files is not None:
        config["write_mask_files"] = bool(args.write_mask_files)

    for setting in args.settings:
        if "=" not in setting:
            raise ValueError(f"Expected KEY=JSON for --set, got {setting!r}")
        key, value = setting.split("=", 1)
        key = key.strip()
        if not key or key.startswith("_"):
            raise ValueError(f"Invalid pipeline key for --set: {key!r}")
        config[key] = _json_value(value)

    return config


def validate_config(config: dict) -> None:
    if not config.get("input_fil") or not config.get("output_fil"):
        raise ValueError("--input-fil and --output-fil are required for mitigation.")
    input_path = Path(config["input_fil"])
    output_path = Path(config["output_fil"])
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output filterbanks must be different files.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input filterbank not found: {input_path}")

    engine = config.get("tensorrt_path")
    checkpoint = config.get("checkpoint")
    if engine:
        if not Path(engine).is_file():
            raise FileNotFoundError(
                f"Requested TensorRT engine not found; refusing PyTorch fallback: {engine}"
            )
        return
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            "Set checkpoint or tensorrt_path in config.json to an existing "
            "model artifact. Model binaries are distributed separately."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2, default=list))
        return
    validate_config(config)
    run_pipeline(config)


if __name__ == "__main__":
    main(sys.argv[1:])
