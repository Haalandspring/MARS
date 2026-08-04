#!/usr/bin/env python3
"""Compile the configured MARS checkpoint into a verified TensorRT engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "src"
DEFAULT_CONFIG = ROOT / "config.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the checkpoint and TensorRT build settings in config.json "
            "into a verified FP16 engine."
        )
    )
    return parser.parse_args(argv)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_build_plan(path: Path = DEFAULT_CONFIG) -> tuple[Path, Path, list[str]]:
    with path.open(encoding="utf-8") as handle:
        config: Any = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"MARS configuration must contain a JSON object: {path}")

    checkpoint_value = config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("config.json must define checkpoint before TensorRT compilation")
    checkpoint = _repo_path(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured MARS checkpoint not found: {checkpoint}")

    build = config.get("tensorrt_build")
    if not isinstance(build, dict):
        raise TypeError("config.json must contain a tensorrt_build object")
    engine_value = build.get("engine_path")
    if not engine_value:
        raise ValueError("tensorrt_build.engine_path must be set in config.json")
    engine = _repo_path(engine_value)

    precision = str(build.get("precision", "fp16")).lower()
    io_dtype = str(build.get("io_dtype", "fp16")).lower()
    if precision != "fp16" or io_dtype != "fp16":
        raise ValueError(
            "The production TensorRT build requires precision=fp16 and io_dtype=fp16"
        )

    export_args = [
        "--checkpoint",
        str(checkpoint),
        "--engine-path",
        str(engine),
        "--batch-size",
        str(int(config.get("batch_size", 64))),
        "--patch-size",
        str(int(config.get("patch_size", 512))),
        "--precision",
        precision,
        "--io-dtype",
        io_dtype,
        "--workspace-gb",
        str(float(build.get("workspace_gb", 8.0))),
        "--opset",
        str(int(build.get("opset", 18))),
        "--onnx-exporter",
        str(build.get("onnx_exporter", "auto")),
        "--verify-batch-size",
        str(int(build.get("verify_batch_size", 4))),
        "--verify-max-diff",
        str(float(build.get("verify_max_diff", 0.02))),
        "--keep-onnx" if bool(build.get("keep_onnx", False)) else "--no-keep-onnx",
        "--no-skip-verify",
        "--no-record-verification-failure",
    ]
    return checkpoint, engine, export_args


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    checkpoint, engine, export_args = load_build_plan()

    print(f"MARS config: {DEFAULT_CONFIG}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Engine:      {engine}")
    print("TensorRT:    FP16 compute + FP16 I/O; numerical verification is mandatory")

    sys.path.insert(0, str(SOURCE_DIR))
    try:
        from mars_rfi.export_tensorrt import main as export_tensorrt
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        raise SystemExit(
            f"Missing TensorRT build dependency: {missing}. "
            "Compile in an environment containing PyTorch, ONNX and TensorRT."
        ) from exc

    export_tensorrt(export_args)
    print("\nCompilation and verification finished.")
    print(
        "To enable TensorRT inference, set config.json tensorrt_path to: "
        f"{engine.relative_to(ROOT) if engine.is_relative_to(ROOT) else engine}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
