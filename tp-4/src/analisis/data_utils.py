#!/usr/bin/env python3
"""Shared helpers for analysis scripts that need simulation data."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_make_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Error ejecutando {' '.join(command)}", file=sys.stderr)
        raise SystemExit(exc.returncode)


def _extract_n_from_header(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "N=" in line:
                    for token in line.strip("# \n").split():
                        if token.startswith("N="):
                            try:
                                return int(token.split("=", 1)[1])
                            except ValueError:
                                return None
    except FileNotFoundError:
        return None
    return None


def find_energy_file(root: Path, n_value: int, run_index: int) -> Optional[Path]:
    root = root.resolve()
    patterns = [
        root / f"gaussian_N{n_value}_run{run_index:03d}_energy.txt",
        root / f"cluster_{n_value}_run{run_index:03d}_energy.txt",
    ]
    for path in patterns:
        if path.exists():
            return path

    matches = []
    for path in root.glob(f"*run{run_index:03d}_energy.txt"):
        header_n = _extract_n_from_header(path)
        if header_n == n_value:
            matches.append(path)
    matches.sort()
    return matches[0] if matches else None


def ensure_energy_file(
    root: Path,
    *,
    n_value: int,
    run_index: int,
    dt: float,
    dt_output: Optional[float],
    tf: float,
    speed: float,
    softening: float,
    collision: bool,
    dx: float,
    dy: float,
    cluster_size: Optional[int],
    seed: Optional[int],
) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    existing = find_energy_file(root, n_value, run_index)
    if existing:
        return existing

    effective_dt_output = dt_output if dt_output is not None else dt
    run_count = run_index + 1
    seed_value = seed if seed is not None else int(time.time() * 1000)

    args = [
        "--N", str(n_value),
        "--runs", str(run_count),
        "--dt", f"{dt}",
        "--dt-output", f"{effective_dt_output}",
        "--tf", f"{tf}",
        "--speed", f"{speed}",
        "--h", f"{softening}",
        "--output-dir", str(root.parent),
        "--seed", str(seed_value),
    ]

    if collision:
        args.append("--collision")
        args.extend(["--dx", f"{dx}", "--dy", f"{dy}"])
        if cluster_size is not None:
            args.extend(["--cluster-size", str(cluster_size)])

    run_make_command(["make", "run-galaxy", "ARGS=" + " ".join(args)])

    generated = find_energy_file(root, n_value, run_index)
    if generated:
        return generated
    raise SystemExit(f"No se pudo generar el archivo de energía para N={n_value}, run={run_index}")
