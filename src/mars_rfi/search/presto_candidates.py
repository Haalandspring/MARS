"""Parse PRESTO accelsearch text and match a known target harmonically."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_token(token: str) -> float | None:
    """Parse PRESTO values such as ``384.96(2)`` and ``1.2x10^-5``."""

    match = FLOAT_RE.search(token.strip())
    if not match:
        return None
    value = float(match.group(0))
    exponent = re.search(r"x10\^([-+]?\d+)", token)
    if exponent:
        value *= 10.0 ** int(exponent.group(1))
    return value


def parse_accel_candidates(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    candidates: list[dict[str, Any]] = []
    active = False
    saw_candidate = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if all(word in line for word in ("Cand", "Sigma", "Period", "Frequency")):
                active = False
                continue
            if stripped and set(stripped) == {"-"} and not saw_candidate:
                active = True
                continue
            if active and not stripped:
                if saw_candidate:
                    break
                continue
            if not active:
                continue
            parts = stripped.split()
            if len(parts) < 7 or not parts[0].isdigit():
                continue
            period_ms = parse_float_token(parts[5])
            frequency_hz = parse_float_token(parts[6])
            if period_ms is None or frequency_hz is None:
                continue
            candidates.append(
                {
                    "cand": int(parts[0]),
                    "sigma": parse_float_token(parts[1]),
                    "summed_power": parse_float_token(parts[2]),
                    "coherent_power": parse_float_token(parts[3]),
                    "num_harm": int(parse_float_token(parts[4]) or 0),
                    "period_ms": period_ms,
                    "frequency_hz": frequency_hz,
                    "fft_r": parse_float_token(parts[7]) if len(parts) > 7 else None,
                    "frequency_derivative_hz_s": (
                        parse_float_token(parts[8]) if len(parts) > 8 else None
                    ),
                    "fft_z": parse_float_token(parts[9]) if len(parts) > 9 else None,
                    "acceleration_m_s2": (
                        parse_float_token(parts[10]) if len(parts) > 10 else None
                    ),
                    "notes": " ".join(parts[11:]) if len(parts) > 11 else "",
                    "raw_line": stripped,
                }
            )
            saw_candidate = True
    return candidates


def candidate_match(
    candidate: dict[str, Any],
    target_frequency_hz: float,
    max_harmonic: int,
) -> dict[str, Any]:
    frequency = candidate.get("frequency_hz")
    period_ms = candidate.get("period_ms")
    if frequency is None or not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        return {
            "match_rel_error": math.nan,
            "match_kind": "invalid",
            "match_harmonic": None,
        }

    targets: list[tuple[str, int, float]] = [
        ("fundamental", 1, float(target_frequency_hz))
    ]
    for harmonic in range(2, max_harmonic + 1):
        targets.append(("harmonic", harmonic, target_frequency_hz * harmonic))
        targets.append(("subharmonic", harmonic, target_frequency_hz / harmonic))
    kind, harmonic, expected = min(
        targets,
        key=lambda item: abs(float(frequency) - item[2]) / item[2],
    )
    relative_error = abs(float(frequency) - expected) / expected
    expected_period_ms = 1000.0 / expected
    return {
        "match_rel_error": relative_error,
        "match_abs_freq_error_hz": abs(float(frequency) - expected),
        "match_abs_period_error_ms": (
            abs(float(period_ms) - expected_period_ms)
            if period_ms is not None
            else math.nan
        ),
        "match_kind": kind,
        "match_harmonic": harmonic,
        "match_expected_frequency_hz": expected,
        "match_expected_period_ms": expected_period_ms,
    }


def select_target_candidate(
    candidates: list[dict[str, Any]],
    *,
    target_frequency_hz: float,
    relative_tolerance: float = 0.01,
    max_harmonic: int = 16,
) -> dict[str, Any] | None:
    enriched = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(candidate_match(row, target_frequency_hz, max_harmonic))
        row["matched_within_tolerance"] = (
            math.isfinite(float(row["match_rel_error"]))
            and float(row["match_rel_error"]) <= relative_tolerance
        )
        enriched.append(row)
    if not enriched:
        return None
    within = [row for row in enriched if row["matched_within_tolerance"]]
    if within:
        return min(
            within,
            key=lambda row: (
                -(float(row["sigma"]) if row.get("sigma") is not None else -math.inf),
                float(row["match_rel_error"]),
            ),
        )
    return min(enriched, key=lambda row: float(row["match_rel_error"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accel-output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-frequency-hz", type=float)
    target.add_argument("--target-period-ms", type=float)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--max-harmonic", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.relative_tolerance < 0.0:
        raise ValueError("--relative-tolerance must be non-negative")
    if args.max_harmonic < 1:
        raise ValueError("--max-harmonic must be positive")
    target_frequency = (
        float(args.target_frequency_hz)
        if args.target_frequency_hz is not None
        else 1000.0 / float(args.target_period_ms)
    )
    if not math.isfinite(target_frequency) or target_frequency <= 0.0:
        raise ValueError("The target frequency/period must be finite and positive")
    candidates = parse_accel_candidates(args.accel_output)
    selected = select_target_candidate(
        candidates,
        target_frequency_hz=target_frequency,
        relative_tolerance=float(args.relative_tolerance),
        max_harmonic=int(args.max_harmonic),
    )
    payload = {
        "schema_version": 1,
        "accel_output": str(args.accel_output),
        "accel_output_sha256": sha256_file(args.accel_output),
        "target_frequency_hz": target_frequency,
        "target_period_ms": 1000.0 / target_frequency,
        "relative_tolerance": float(args.relative_tolerance),
        "max_harmonic": int(args.max_harmonic),
        "candidate_count": len(candidates),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
