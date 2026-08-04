#!/usr/bin/env python3
"""Compile a trained MARS checkpoint to a TensorRT engine.

Pipeline:

    PyTorch checkpoint -> ONNX -> TensorRT .engine

The generated engine uses tensor names ``input`` and ``output`` to match
``mars_rfi.pipeline.TRTInferenceContext``.

Normal source-checkout users should run ``python compile_tensorrt.py`` from the
repository root. That wrapper reads the checkpoint, batch size, and FP16 build
settings from ``config.json``. This lower-level module retains explicit options
for implementation testing and advanced development work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from pathlib import Path

import torch

from .config import CONFIG as TRAIN_CONFIG
from .model import build_model, count_parameters
from .provenance import (
    STRICT_TENSORRT_ARTIFACT_ROLES,
    PAPER_TENSORRT_MAX_DIFF,
    validate_training_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# --- User-editable controls -------------------------------------------------
CONFIG = {
    # Source checkpoint.
    "checkpoint": "artifacts/checkpoints/mars-paper/best_f1.pt",

    # Output.  If output_dir is None, the script writes to
    # <checkpoint_dir>/tensorrt/.  If engine_path is set, it is used exactly.
    "output_dir": None,
    "engine_path": None,

    # Model overrides.  Keep these as None to use the checkpoint config.
    # "light_unet_v1" | "light_unet_v2" | "trt_fast_unet" |
    # "trt_shape_unet" | None
    "model": None,
    "channels": None,  # e.g. "16,32,48,64" or (16, 32, 48, 64)
    "axis_reduction": None,
    "anisotropic_kernel": None,
    "bottleneck_dilation": None,
    "shape_kernel": None,
    "decoder_horizontal_refine_enabled": None,
    "decoder_horizontal_refine_stages": None,
    "decoder_horizontal_refine_kernel": None,
    "decoder_vertical_refine_enabled": None,
    "decoder_vertical_refine_stages": None,
    "decoder_vertical_refine_kernel": None,

    # TensorRT build.
    "batch_size": 64,
    "patch_size": 512,
    "precision": "fp16",  # "fp32" | "fp16"
    "io_dtype": "fp16",  # "fp32" | "fp16"; fp16 avoids FP32 I/O copies in the pipeline.
    "workspace_gb": 8.0,
    "opset": 18,
    "onnx_exporter": "auto",  # "auto" | "dynamo" | "legacy"

    # Verification / debug.
    "keep_onnx": False,
    "skip_verify": False,
    # Diagnostic-only escape hatch: still execute verification and record the
    # measured failure in the sidecar instead of presenting the engine as
    # verified. Strict runtime profiles continue to reject that sidecar unless
    # their caller explicitly opts into unverified diagnostic execution.
    "record_verification_failure": False,
    "verify_batch_size": 4,
    "verify_max_diff": 0.02,
}


def _parse_channels(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _repo_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(args, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    ckpt_path = _repo_path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    validate_training_identity(ckpt_cfg, source=f"checkpoint {ckpt_path}")

    model_cfg = dict(TRAIN_CONFIG)
    model_cfg.update(ckpt_cfg)
    if args.model is not None:
        model_cfg["model"] = args.model
    if args.channels is not None:
        model_cfg["channels"] = _parse_channels(args.channels)
    if args.axis_reduction is not None:
        model_cfg["axis_reduction"] = int(args.axis_reduction)
    if args.anisotropic_kernel is not None:
        model_cfg["anisotropic_kernel"] = int(args.anisotropic_kernel)
    if args.bottleneck_dilation is not None:
        model_cfg["bottleneck_dilation"] = int(args.bottleneck_dilation)
    if args.shape_kernel is not None:
        model_cfg["shape_kernel"] = int(args.shape_kernel)
    if args.decoder_horizontal_refine_enabled is not None:
        model_cfg["decoder_horizontal_refine_enabled"] = bool(
            args.decoder_horizontal_refine_enabled
        )
    if args.decoder_horizontal_refine_stages is not None:
        model_cfg["decoder_horizontal_refine_stages"] = args.decoder_horizontal_refine_stages
    if args.decoder_horizontal_refine_kernel is not None:
        model_cfg["decoder_horizontal_refine_kernel"] = int(
            args.decoder_horizontal_refine_kernel
        )
    if args.decoder_vertical_refine_enabled is not None:
        model_cfg["decoder_vertical_refine_enabled"] = bool(
            args.decoder_vertical_refine_enabled
        )
    if args.decoder_vertical_refine_stages is not None:
        model_cfg["decoder_vertical_refine_stages"] = args.decoder_vertical_refine_stages
    if args.decoder_vertical_refine_kernel is not None:
        model_cfg["decoder_vertical_refine_kernel"] = int(
            args.decoder_vertical_refine_kernel
        )
    validate_training_identity(model_cfg, source="resolved export model config")

    model = build_model(model_cfg).to(device).eval()
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    parameter_count = count_parameters(model)
    expected_parameters = model_cfg.get("expected_parameters")
    if expected_parameters is not None and parameter_count != int(expected_parameters):
        raise RuntimeError(
            f"Model has {parameter_count:,} trainable parameters; expected "
            f"{int(expected_parameters):,}. Refusing to export a mislabeled engine."
        )
    print(
        f"Loaded {model_cfg.get('model')} from {ckpt_path} "
        f"(epoch {epoch}, params {parameter_count:,})"
    )
    if model_cfg.get("model") in ("trt_fast_unet", "trt_shape_unet"):
        print(
            "  decoder_horizontal_refine: "
            f"enabled={bool(model_cfg.get('decoder_horizontal_refine_enabled', False))}, "
            f"stages={model_cfg.get('decoder_horizontal_refine_stages', ())}, "
            f"kernel={int(model_cfg.get('decoder_horizontal_refine_kernel', 15))}"
        )
        print(
            "  decoder_vertical_refine: "
            f"enabled={bool(model_cfg.get('decoder_vertical_refine_enabled', False))}, "
            f"stages={model_cfg.get('decoder_vertical_refine_stages', ())}, "
            f"kernel={int(model_cfg.get('decoder_vertical_refine_kernel', 9))}"
        )
    return model, model_cfg, ckpt


def write_artifact_metadata(
    *,
    engine_path: Path,
    onnx_path: Path,
    checkpoint_path: Path,
    model_cfg: dict,
    model: torch.nn.Module,
    args,
    verification: dict,
) -> Path:
    """Bind an engine to its checkpoint, architecture, and build environment."""

    try:
        import tensorrt as trt

        tensorrt_version = trt.__version__
    except (ImportError, AttributeError):
        tensorrt_version = "unknown"

    metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "onnx_sha256": _sha256(onnx_path),
        "engine_sha256": _sha256(engine_path),
        "model_config": model_cfg,
        "trainable_parameters": count_parameters(model),
        "verification": verification,
        "build": {
            "batch_size": int(args.batch_size),
            "patch_size": int(args.patch_size),
            "precision": str(args.precision),
            "io_dtype": str(args.io_dtype),
            "workspace_gb": float(args.workspace_gb),
            "opset": int(args.opset),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "tensorrt": tensorrt_version,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def export_onnx(model: torch.nn.Module, onnx_path: Path, args, device: torch.device) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    io_dtype_name = str(args.io_dtype).lower()
    if io_dtype_name == "fp16":
        export_dtype = torch.float16
        model.half()
    elif io_dtype_name == "fp32":
        export_dtype = torch.float32
        model.float()
    else:
        raise ValueError("io_dtype must be fp32 or fp16")
    dummy = torch.randn(
        1,
        1,
        int(args.patch_size),
        int(args.patch_size),
        device=device,
        dtype=export_dtype,
    )

    print(f"\n[1/3] Exporting ONNX: {onnx_path}")
    print(f"  ONNX I/O dtype: {io_dtype_name}")
    exporter = str(args.onnx_exporter).lower()
    if exporter not in ("auto", "dynamo", "legacy"):
        raise ValueError("onnx_exporter must be auto, dynamo, or legacy")

    def export_with_dynamo() -> None:
        batch_dim = torch.export.Dim("batch", min=1, max=int(args.batch_size))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*",
            )
            torch.onnx.export(
                model,
                (dummy,),
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_shapes=({0: batch_dim},),
                opset_version=int(args.opset),
                do_constant_folding=True,
                dynamo=True,
                external_data=False,
                verbose=False,
            )

    def export_with_legacy() -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="You are using the legacy TorchScript-based ONNX export.*",
            )
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=int(args.opset),
                do_constant_folding=True,
                dynamo=False,
            )

    with torch.inference_mode():
        if exporter == "legacy":
            export_with_legacy()
        elif exporter == "dynamo":
            export_with_dynamo()
        else:
            try:
                export_with_dynamo()
            except Exception as exc:
                print(f"  WARNING: dynamo ONNX export failed; falling back to legacy exporter: {exc}")
                export_with_legacy()
    print(f"  ONNX size: {onnx_path.stat().st_size / 1e6:.2f} MB")


def _set_workspace_limit(config, trt, workspace_gb: float) -> None:
    bytes_limit = int(float(workspace_gb) * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_limit)
    else:
        config.max_workspace_size = bytes_limit


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    name = str(trt_dtype).split(".")[-1].lower()
    if name in ("half", "float16", "fp16"):
        return torch.float16
    if name in ("float", "float32", "fp32"):
        return torch.float32
    raise TypeError(f"Unsupported TensorRT tensor dtype for torch verification: {trt_dtype}")


def build_engine(onnx_path: Path, engine_path: Path, args) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python package is not available in this environment. "
            "Run this script inside an environment with tensorrt installed."
        ) from exc

    print(f"\n[2/3] Building TensorRT engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("TensorRT ONNX parsing failed")

    config = builder.create_builder_config()
    _set_workspace_limit(config, trt, float(args.workspace_gb))

    precision = str(args.precision).lower()
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Precision: FP16")
        else:
            print("  WARNING: platform has no fast FP16; building FP32 engine")
    elif precision == "fp32":
        print("  Precision: FP32")
    else:
        raise ValueError("precision must be fp32 or fp16")

    profile = builder.create_optimization_profile()
    patch = int(args.patch_size)
    batch = int(args.batch_size)
    profile.set_shape(
        "input",
        min=(1, 1, patch, patch),
        opt=(batch, 1, patch, patch),
        max=(batch, 1, patch, patch),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"  Engine size: {engine_path.stat().st_size / 1e6:.2f} MB")
    return engine_path


def verify_engine(engine_path: Path, model: torch.nn.Module, args, device: torch.device) -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is unavailable for engine verification") from exc

    print("\n[3/3] Verifying TensorRT output against PyTorch")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    batch = min(int(args.batch_size), int(args.verify_batch_size))
    patch = int(args.patch_size)
    context.set_input_shape("input", (batch, 1, patch, patch))

    input_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype("input"))
    output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype("output"))
    verification_seed = 1234
    generator = torch.Generator(device=device)
    generator.manual_seed(verification_seed)
    # MARS receives tanh-compressed patches, so verification should exercise
    # the bounded deployment domain deterministically instead of drawing an
    # unbounded, run-dependent Gaussian tensor.
    x = torch.empty(batch, 1, patch, patch, device=device, dtype=input_dtype)
    x.uniform_(-0.999, 0.999, generator=generator)
    y_trt = torch.empty(batch, 1, patch, patch, device=device, dtype=output_dtype)
    print(f"  TensorRT I/O dtype: input={input_dtype}, output={output_dtype}")

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address("output", y_trt.data_ptr())
    with torch.cuda.stream(stream):
        executed = context.execute_async_v3(stream.cuda_stream)
    if not executed:
        raise RuntimeError("TensorRT execution failed during numerical verification")
    torch.cuda.current_stream(device).wait_stream(stream)

    with torch.inference_mode():
        y_pt = model(x)

    prob_diff = (torch.sigmoid(y_pt.float()) - torch.sigmoid(y_trt.float())).abs()
    max_diff = float(prob_diff.max().item())
    mean_diff = float(prob_diff.mean().item())
    print(f"  sigmoid max abs diff:  {max_diff:.6f}")
    print(f"  sigmoid mean abs diff: {mean_diff:.6f}")
    verification = {
        "status": (
            "passed" if max_diff <= float(args.verify_max_diff) else "failed"
        ),
        "metric": "sigmoid_absolute_difference",
        "input_profile": "uniform_tanh_domain_-0.999_0.999",
        "input_seed": verification_seed,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_allowed_diff": float(args.verify_max_diff),
        "batch_size": batch,
    }
    if verification["status"] == "passed":
        print("  Verification PASSED")
        return verification
    if bool(args.record_verification_failure):
        print(
            "  Verification FAILED and was recorded for explicit diagnostic use; "
            "the engine is not a verified production artifact"
        )
        return verification
    raise RuntimeError(
        "TensorRT verification failed: "
        f"max sigmoid difference {max_diff:.6f} exceeds "
        f"{float(args.verify_max_diff):.6f}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--engine-path", default=CONFIG["engine_path"])
    parser.add_argument(
        "--model",
        choices=("light_unet_v1", "light_unet_v2", "trt_fast_unet", "trt_shape_unet"),
        default=CONFIG["model"],
    )
    parser.add_argument(
        "--channels",
        default=CONFIG["channels"],
        help="Comma-separated channels, e.g. 16,32,48,64",
    )
    parser.add_argument("--axis-reduction", type=int, default=CONFIG["axis_reduction"])
    parser.add_argument("--anisotropic-kernel", type=int, default=CONFIG["anisotropic_kernel"])
    parser.add_argument("--bottleneck-dilation", type=int, default=CONFIG["bottleneck_dilation"])
    parser.add_argument("--shape-kernel", type=int, default=CONFIG["shape_kernel"])
    parser.add_argument(
        "--decoder-horizontal-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["decoder_horizontal_refine_enabled"],
    )
    parser.add_argument(
        "--decoder-horizontal-refine-stages",
        default=CONFIG["decoder_horizontal_refine_stages"],
        help="Comma-separated decoder stages, e.g. up1 or up2,up1",
    )
    parser.add_argument(
        "--decoder-horizontal-refine-kernel",
        type=int,
        default=CONFIG["decoder_horizontal_refine_kernel"],
    )
    parser.add_argument(
        "--decoder-vertical-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["decoder_vertical_refine_enabled"],
    )
    parser.add_argument(
        "--decoder-vertical-refine-stages",
        default=CONFIG["decoder_vertical_refine_stages"],
        help="Comma-separated decoder stages, e.g. up2 or up2,up1",
    )
    parser.add_argument(
        "--decoder-vertical-refine-kernel",
        type=int,
        default=CONFIG["decoder_vertical_refine_kernel"],
    )
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--patch-size", type=int, default=CONFIG["patch_size"])
    parser.add_argument("--precision", choices=("fp32", "fp16"), default=CONFIG["precision"])
    parser.add_argument("--io-dtype", choices=("fp32", "fp16"), default=CONFIG["io_dtype"])
    parser.add_argument("--workspace-gb", type=float, default=CONFIG["workspace_gb"])
    parser.add_argument("--opset", type=int, default=CONFIG["opset"])
    parser.add_argument(
        "--onnx-exporter",
        choices=("auto", "dynamo", "legacy"),
        default=CONFIG["onnx_exporter"],
    )
    parser.add_argument(
        "--keep-onnx",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["keep_onnx"],
    )
    parser.add_argument(
        "--skip-verify",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["skip_verify"],
    )
    parser.add_argument(
        "--record-verification-failure",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["record_verification_failure"],
        help=(
            "Diagnostic only: write a sidecar with verification status=failed "
            "instead of aborting when the measured tolerance is exceeded"
        ),
    )
    parser.add_argument("--verify-batch-size", type=int, default=CONFIG["verify_batch_size"])
    parser.add_argument("--verify-max-diff", type=float, default=CONFIG["verify_max_diff"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT compilation requires a CUDA device")

    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    model, model_cfg, _ = load_checkpoint_model(args, device)
    strict_paper_artifact = model_cfg.get("artifact_role") in STRICT_TENSORRT_ARTIFACT_ROLES
    if args.skip_verify and strict_paper_artifact:
        raise RuntimeError(
            "Paper and historical-paper TensorRT artifacts cannot be exported "
            "with --skip-verify"
        )
    if strict_paper_artifact:
        tolerance = float(args.verify_max_diff)
        if (
            not math.isfinite(tolerance)
            or tolerance < 0.0
            or tolerance > PAPER_TENSORRT_MAX_DIFF
        ):
            raise RuntimeError(
                "Paper TensorRT verification tolerance must be finite and within "
                f"[0, {PAPER_TENSORRT_MAX_DIFF}]"
            )

    ckpt_path = _repo_path(args.checkpoint)
    model_name = str(model_cfg.get("model", "trt_shape_unet"))
    output_dir = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else ckpt_path.parent / "tensorrt"
    )
    stem = (
        f"{ckpt_path.stem}_{model_name}_{args.precision}_"
        f"io{args.io_dtype}_bs{int(args.batch_size)}"
    )
    onnx_path = output_dir / f"{stem}.onnx"
    engine_path = _repo_path(args.engine_path) if args.engine_path else output_dir / f"{stem}.engine"

    export_onnx(model, onnx_path, args, device)

    # Free activations before TensorRT build.  The model stays alive for optional
    # verification but should not hold temporary export buffers.
    torch.cuda.empty_cache()
    build_engine(onnx_path, engine_path, args)

    if args.skip_verify:
        verification = {
            "status": "skipped",
            "reason": "explicit --skip-verify",
            "max_allowed_diff": float(args.verify_max_diff),
        }
    else:
        verification = verify_engine(engine_path, model, args, device)

    metadata_path = write_artifact_metadata(
        engine_path=engine_path,
        onnx_path=onnx_path,
        checkpoint_path=ckpt_path,
        model_cfg=model_cfg,
        model=model,
        args=args,
        verification=verification,
    )

    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)

    print(f"\nDone. TensorRT engine: {engine_path}")
    print(f"Artifact metadata: {metadata_path}")
    print("Set config.json `tensorrt_path` to this engine before running mars.py.")


if __name__ == "__main__":
    main(sys.argv[1:])
