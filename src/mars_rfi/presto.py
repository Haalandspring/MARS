#!/usr/bin/env python3
"""Run the paper PRESTO search chain on one or more cleaned filterbanks.

The default search protocol is ``prepsubband -> realfft -> rednoise ->
accelsearch`` with ``zmax=200`` and ``numharm=8``. Real-GMRT searches should
override those values as documented in ``docs/paper-specification.md``.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    # Set this to True to pass -zerodm to prepsubband by default.
    # Command-line --zerodm / --no-zerodm can still override it for one run.
    # For the pipeline-internal zdot experiment, keep this False by default so
    # the PRESTO zero-DM filter does not remove the recovered fundamental again.
    "zerodm": False,
    "sif": None,
    "dm": None,
    "numdms": 1,
    "zmax": 200,
    "numharm": 8,
    "nobary": True,
    "rednoise": True,
    "cleanup_products": True,
    # None means keep the accelsearch product matching zmax, e.g. ACCEL_4 for
    # -zmax 4. Set explicitly only when cleaning a non-standard product.
    "keep_accel_suffix": None,
    # PRESTO's filename parsing can fail when the full root string exceeds
    # roughly 200 chars. Keep generated output basenames comfortably shorter
    # while preserving enough of the diagnostic name to identify the stage.
    "max_presto_prefix_name": 72,
}


def _resolve(path):
    return Path(path).expanduser().resolve()


def safe_presto_prefix_name(stem, max_len):
    if len(stem) <= max_len:
        return stem

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    budget = max_len - len(digest) - 2
    if budget <= 8:
        return f"{stem[:max_len - 9]}_{digest}"

    head_len = max(12, budget // 2)
    tail_len = max(12, budget - head_len)
    if head_len + tail_len > budget:
        tail_len = budget - head_len
    return f"{stem[:head_len]}_{digest}_{stem[-tail_len:]}"


def presto_dm_arg(dm):
    return f"{float(dm):.12g}"


def presto_dm_label(dm):
    return f"{float(dm):.2f}"


def find_dat_files(out_prefix):
    if not out_prefix.parent.is_dir():
        return []
    return sorted(out_prefix.parent.glob(f"{out_prefix.name}_DM*.dat"))


def select_dat_file(out_prefix, dm, *, dry_run=False):
    expected = out_prefix.parent / f"{out_prefix.name}_DM{presto_dm_label(dm)}.dat"
    if dry_run:
        return expected

    candidates = find_dat_files(out_prefix)
    if expected in candidates:
        return expected
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    names = ", ".join(path.name for path in candidates)
    raise RuntimeError(
        f"Multiple PRESTO .dat files match {out_prefix.name}, but none matches "
        f"DM {presto_dm_label(dm)}: {names}"
    )


def find_container_runtime(explicit):
    if explicit:
        runtime = shutil.which(explicit) or explicit
        return runtime
    for name in ("apptainer", "singularity"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "Could not find apptainer or singularity on PATH. "
        "Pass --runtime or use --no-container if already inside the PRESTO environment."
    )


def build_exec_command(args, presto_cmd):
    if args.no_container:
        return presto_cmd

    if not args.sif:
        raise ValueError("Pass --sif or use --no-container to run PRESTO from PATH")
    runtime = find_container_runtime(args.runtime)
    cmd = [runtime, "exec"]
    for bind in args.bind:
        cmd.extend(["--bind", bind])
    cmd.append(str(_resolve(args.sif)))
    cmd.extend(presto_cmd)
    return cmd


def run_command(cmd, cwd, dry_run=False, log_path=None):
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"$ {printable}")

    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w", buffering=1)
        log.write(f"$ {printable}\n\n")

    try:
        if dry_run:
            if log is not None:
                log.write("[dry-run] command not executed\n")
            return 0

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if log is not None:
                log.write(line)
        return proc.wait()
    finally:
        if log is not None:
            log.close()


def build_prepsubband_command(args, fil_path, out_prefix):
    cmd = ["prepsubband"]
    if args.nobary:
        cmd.append("-nobary")
    if args.zerodm:
        cmd.append("-zerodm")
    cmd.extend([
        "-lodm", presto_dm_arg(args.dm),
        "-numdms", str(args.numdms),
        "-o", out_prefix.name,
        str(fil_path),
    ])
    return cmd


def build_accelsearch_command(args, dat_path):
    return [
        "accelsearch",
        "-zmax", str(args.zmax),
        "-numharm", str(args.numharm),
        dat_path.name,
    ]


def build_realfft_command(dat_path):
    return ["realfft", dat_path.name]


def build_rednoise_command(fft_path):
    return ["rednoise", fft_path.name]


def accel_suffix_for_zmax(zmax):
    return f"ACCEL_{int(zmax)}"


def resolve_keep_accel_suffix(args):
    suffix = getattr(args, "keep_accel_suffix", None)
    if suffix is None or str(suffix).strip().lower() in ("", "auto"):
        return accel_suffix_for_zmax(args.zmax)
    return str(suffix)


def cleanup_presto_products(
    out_dir,
    out_prefix,
    dm_label,
    keep_suffix,
    dry_run=False,
    extra_paths=(),
):
    """
    Keep only the requested accelsearch product for one PRESTO run.

    The caller passes the suffix to keep, usually ACCEL_<zmax>, and this removes
    the generated .dat/.inf/.cand files for the same output prefix.
    Extra paths are used to remove optional command logs for the same run.
    """
    pattern = f"{out_prefix.name}_DM{dm_label}*"
    kept = []
    removed = []

    paths = list(sorted(out_dir.glob(pattern)))
    paths.extend(Path(p) for p in extra_paths)

    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        if path.name.endswith(keep_suffix):
            kept.append(str(path))
            continue

        removed.append(str(path))
        if dry_run:
            print(f"[dry-run] Would remove {path}")
        else:
            path.unlink()

    print(
        f"  Kept {len(kept)} file(s) ending with {keep_suffix}; "
        f"removed {len(removed)} other PRESTO product(s)"
    )
    if not kept and not dry_run:
        print(f"  WARNING: no kept product found for pattern {pattern}")
    return kept, removed


def run_one(args, fil_path):
    fil_path = _resolve(fil_path)
    if not fil_path.is_file():
        msg = f"Missing input .fil: {fil_path}"
        if args.skip_missing:
            print(f"WARNING: {msg}; skipping")
            return {"fil": str(fil_path), "status": "missing"}
        raise FileNotFoundError(msg)

    if args.out_prefix:
        if len(args.fil) != 1:
            raise SystemExit("--out-prefix is only valid with exactly one --fil")
        out_prefix = _resolve(args.out_prefix)
        out_dir = out_prefix.parent
    else:
        out_dir = _resolve(args.out_dir) if args.out_dir else fil_path.parent / "presto_accelsearch"
        out_prefix = out_dir / safe_presto_prefix_name(fil_path.stem, args.max_presto_prefix_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_label = presto_dm_label(args.dm)
    predicted_dat_path = select_dat_file(out_prefix, args.dm, dry_run=True)
    existing_dat_path = (
        select_dat_file(out_prefix, args.dm) if args.skip_existing else None
    )
    dat_path = existing_dat_path or predicted_dat_path
    prepsubband_log = out_dir / f"{out_prefix.name}.prepsubband.log"
    realfft_log = out_dir / f"{out_prefix.name}.realfft.log"
    rednoise_log = out_dir / f"{out_prefix.name}.rednoise.log"
    accelsearch_log = out_dir / f"{out_prefix.name}.accelsearch.log"

    result = {
        "fil": str(fil_path),
        "out_prefix": str(out_prefix),
        "dat": str(dat_path),
    }
    if args.write_logs:
        result["prepsubband_log"] = str(prepsubband_log)
        result["realfft_log"] = str(realfft_log)
        result["rednoise_log"] = str(rednoise_log)
        result["accelsearch_log"] = str(accelsearch_log)

    if args.cleanup_only:
        keep_suffix = resolve_keep_accel_suffix(args)
        kept, removed = cleanup_presto_products(
            out_dir=out_dir,
            out_prefix=out_prefix,
            dm_label=dm_label,
            keep_suffix=keep_suffix,
            dry_run=args.dry_run,
            extra_paths=(prepsubband_log, realfft_log, rednoise_log, accelsearch_log),
        )
        result["keep_accel_suffix"] = keep_suffix
        result["kept_products"] = kept
        result["removed_products"] = removed
        result["status"] = "ok"
        return result

    if existing_dat_path is not None and existing_dat_path.is_file():
        print(f"Skipping prepsubband; existing dat found: {dat_path}")
        prepsubband_rc = 0
    else:
        prepsubband_cmd = build_exec_command(
            args, build_prepsubband_command(args, fil_path, out_prefix)
        )
        prepsubband_rc = run_command(
            prepsubband_cmd,
            out_dir,
            dry_run=args.dry_run,
            log_path=prepsubband_log if args.write_logs else None,
        )
    result["prepsubband_returncode"] = prepsubband_rc
    if prepsubband_rc != 0:
        result["status"] = "prepsubband_failed"
        return result

    dat_path = select_dat_file(out_prefix, args.dm, dry_run=args.dry_run)
    if dat_path is None:
        result["status"] = "dat_missing_after_prepsubband"
        return result
    result["dat"] = str(dat_path)

    fft_path = dat_path.with_suffix(".fft")
    red_fft_path = fft_path.with_name(f"{fft_path.stem}_red.fft")

    search_input = dat_path
    if args.rednoise:
        realfft_cmd = build_exec_command(args, build_realfft_command(dat_path))
        realfft_rc = run_command(
            realfft_cmd,
            out_dir,
            dry_run=args.dry_run,
            log_path=realfft_log if args.write_logs else None,
        )
        result["realfft_returncode"] = realfft_rc
        if realfft_rc != 0:
            result["status"] = "realfft_failed"
            return result
        if not args.dry_run and not fft_path.is_file():
            result["status"] = "fft_missing_after_realfft"
            return result

        rednoise_cmd = build_exec_command(args, build_rednoise_command(fft_path))
        rednoise_rc = run_command(
            rednoise_cmd,
            out_dir,
            dry_run=args.dry_run,
            log_path=rednoise_log if args.write_logs else None,
        )
        result["rednoise_returncode"] = rednoise_rc
        if rednoise_rc != 0:
            result["status"] = "rednoise_failed"
            return result
        if not args.dry_run and not red_fft_path.is_file():
            result["status"] = "red_fft_missing_after_rednoise"
            return result
        search_input = red_fft_path

    result["accelsearch_input"] = str(search_input)
    accel_cmd = build_exec_command(args, build_accelsearch_command(args, search_input))
    accel_rc = run_command(
        accel_cmd,
        out_dir,
        dry_run=args.dry_run,
        log_path=accelsearch_log if args.write_logs else None,
    )
    result["accelsearch_returncode"] = accel_rc
    result["status"] = "ok" if accel_rc == 0 else "accelsearch_failed"

    if args.cleanup_products and (args.dry_run or accel_rc == 0):
        keep_suffix = resolve_keep_accel_suffix(args)
        kept, removed = cleanup_presto_products(
            out_dir=out_dir,
            out_prefix=out_prefix,
            dm_label=dm_label,
            keep_suffix=keep_suffix,
            dry_run=args.dry_run,
            extra_paths=(prepsubband_log, realfft_log, rednoise_log, accelsearch_log),
        )
        result["keep_accel_suffix"] = keep_suffix
        result["kept_products"] = kept
        result["removed_products"] = removed

    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run PRESTO prepsubband and accelsearch on mitigation diagnostics."
    )
    ap.add_argument(
        "--sif",
        default=CONFIG["sif"],
        help="Apptainer/Singularity image containing PRESTO.",
    )
    ap.add_argument(
        "--fil",
        action="append",
        required=True,
        help="Input .fil file. Repeat for multiple files.",
    )
    ap.add_argument(
        "--zerodm",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["zerodm"],
        help="Pass -zerodm to prepsubband during dedispersion. Default comes from CONFIG.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for PRESTO products. Default: <fil_dir>/presto_accelsearch.",
    )
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Exact prepsubband -o prefix. Only valid with one --fil.",
    )
    ap.add_argument(
        "--dm",
        type=float,
        required=True,
        help="Dispersion measure for -lodm (PRESTO output matching uses two decimals).",
    )
    ap.add_argument("--numdms", type=int, default=CONFIG["numdms"])
    ap.add_argument("--zmax", type=int, default=CONFIG["zmax"])
    ap.add_argument("--numharm", type=int, default=CONFIG["numharm"])
    ap.add_argument("--nobary", action=argparse.BooleanOptionalAction, default=CONFIG["nobary"])
    ap.add_argument(
        "--rednoise",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["rednoise"],
        help="Run realfft and rednoise before accelsearch (paper default: enabled).",
    )
    ap.add_argument("--runtime", default=None, help="Container runtime binary name/path.")
    ap.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Container bind mount. Repeat if needed.",
    )
    ap.add_argument(
        "--no-container",
        action="store_true",
        help="Run prepsubband/accelsearch directly on PATH.",
    )
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--write-logs",
        action="store_true",
        help="Write command stdout logs. Default is no log files.",
    )
    ap.add_argument(
        "--cleanup-products",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["cleanup_products"],
        help="After accelsearch, remove PRESTO products except files ending with --keep-accel-suffix.",
    )
    ap.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Do not run PRESTO; only clean existing products for the selected input files.",
    )
    ap.add_argument(
        "--keep-accel-suffix",
        default=CONFIG["keep_accel_suffix"],
        help=(
            "PRESTO product filename suffix to keep during cleanup. "
            "Default/auto keeps ACCEL_<zmax>."
        ),
    )
    ap.add_argument(
        "--max-presto-prefix-name",
        type=int,
        default=CONFIG["max_presto_prefix_name"],
        help="Maximum generated PRESTO output basename length before hashing/truncation.",
    )
    ap.add_argument(
        "--summary",
        default=None,
        help="Path to write summary JSON. Default is no summary unless --write-summary is set.",
    )
    ap.add_argument(
        "--write-summary",
        action="store_true",
        help="Write default summary JSON to <out-dir>/presto_accelsearch_summary.json.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    results = []
    for fil in args.fil:
        try:
            result = run_one(args, fil)
        except Exception as exc:
            if not args.keep_going:
                raise
            result = {"fil": str(fil), "status": "error", "error": str(exc)}
            print(f"ERROR: {fil}: {exc}")
        results.append(result)

        if result.get("status") not in ("ok", "missing") and not args.keep_going:
            break

    first_fil = _resolve(args.fil[0])
    default_out_dir = _resolve(args.out_dir) if args.out_dir else first_fil.parent / "presto_accelsearch"
    summary_path = _resolve(args.summary) if args.summary else default_out_dir / "presto_accelsearch_summary.json"

    if args.summary or args.write_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary written to {summary_path}")
    elif summary_path.is_file():
        if args.dry_run:
            print(f"[dry-run] Would remove {summary_path}")
        else:
            summary_path.unlink()

    failed = [r for r in results if r.get("status") not in ("ok", "missing")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
