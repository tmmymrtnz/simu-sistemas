#!/usr/bin/env python3
"""Energy diagnostics for the N-body simulations (System 2)."""

import argparse
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_energy_file(path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    times: List[float] = []
    totals: List[float] = []
    dt = None

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# dt="):
                tokens = line[1:].split()
                for token in tokens:
                    if token.startswith("dt="):
                        dt = float(token.split("=", maxsplit=1)[1])
                        break
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            if len(parts) >= 4:
                t, _, _, total = map(float, parts[:4])
            else:
                t, total, _ = map(float, parts[:3])
            times.append(t)
            totals.append(total)

    if dt is None:
        raise ValueError(f"Could not infer dt from header in {path}")

    return dt, np.asarray(times), np.asarray(totals)


def compute_drift(times: np.ndarray, totals: np.ndarray) -> np.ndarray:
    e0 = totals[0]
    if math.isclose(e0, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return np.abs(totals - e0)
    return np.abs((totals - e0) / e0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study energy conservation vs dt")
    parser.add_argument("--root", default=os.path.join("..", "out", "galaxy", "energy"),
                        help="Directory where dt-specific energy folders will be stored")
    parser.add_argument("--pattern", default="*_energy.txt", help="Glob pattern for energy files")
    parser.add_argument("--output", default="plots", help="Directory to drop figures")
    parser.add_argument("--min-runs", type=int, default=5,
                        help="Número recomendado de realizaciones por dt (para alertas)")
    parser.add_argument("--N", type=int, default=500, help="Número de partículas para el estudio")
    parser.add_argument("--dts", type=float, nargs="+", default=[1e-3, 2e-3, 5e-3],
                        help="Lista de dt a evaluar")
    parser.add_argument("--runs", type=int, default=5, help="Cantidad de corridas por dt")
    parser.add_argument("--tf", type=float, default=5.0, help="Tiempo final de simulación")
    parser.add_argument("--dt-output", type=float, default=None, help="Paso de guardado (default = dt)")
    parser.add_argument("--speed", type=float, default=0.1, help="Módulo de velocidades iniciales")
    parser.add_argument("--softening", type=float, default=0.05, help="Parámetro de suavizado h")
    parser.add_argument("--seed", type=int, default=None, help="Semilla base para reproducibilidad")
    parser.add_argument("--skip-generate", action="store_true",
                        help="No generar datos nuevos si faltan corridas")
    parser.add_argument("--collision", action="store_true",
                        help="Usar escenario de colisión en lugar del gaussiano")
    parser.add_argument("--dx", type=float, default=4.0, help="Separación en x para la colisión")
    parser.add_argument("--dy", type=float, default=0.5, help="Separación en y para la colisión")
    args = parser.parse_args()

    dts = [float(dt) for dt in args.dts]
    dt_output_override = float(args.dt_output) if args.dt_output is not None else None

    energy_dir = Path(args.root).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_energy_data(energy_dir, args, dts, dt_output_override)

    grouped: Dict[float, List[Tuple[np.ndarray, np.ndarray, Path]]] = defaultdict(list)
    for path in sorted(energy_dir.rglob(args.pattern)):
        name = path.name
        if f"N{args.N}_" not in name:
            continue
        dt, times, totals = parse_energy_file(path)
        grouped[dt].append((times, totals, path))

    if not grouped:
        raise SystemExit(f"No energy files found under {energy_dir}")

    summary_rows = []

    plt.figure(figsize=(8, 5))
    for dt in sorted(grouped.keys()):
        entries = grouped[dt]
        min_length = min(len(times) for times, _, _ in entries)
        if min_length == 0:
            print(f"⚠️  dt={dt:g} contiene corridas sin datos; se omite.")
            continue

        times_ref = entries[0][0][:min_length]
        drifts = []
        for times, totals, path in entries:
            truncated_times = times[:min_length]
            truncated_totals = totals[:min_length]
            if not np.allclose(truncated_times, times_ref):
                print(f"⚠️  Ajustando serie con timeline diferente en {path}")
            drift = compute_drift(truncated_times, truncated_totals)
            drifts.append(drift)

        stacked = np.vstack(drifts)
        mean_drift = stacked.mean(axis=0)
        plt.plot(times_ref, mean_drift, linewidth=2.0, label=f"dt={dt:g} (n={len(entries)})")

        max_per_run = stacked.max(axis=1)
        rms_per_run = np.sqrt((stacked ** 2).mean(axis=1))
        final_per_run = stacked[:, -1]

        summary_rows.append({
            "dt": dt,
            "runs": len(entries),
            "max_mean": float(np.mean(max_per_run)),
            "max_std": float(np.std(max_per_run, ddof=0)),
            "max_best": float(np.min(max_per_run)),
            "max_worst": float(np.max(max_per_run)),
            "rms_mean": float(np.mean(rms_per_run)),
            "rms_std": float(np.std(rms_per_run, ddof=0)),
            "final_mean": float(np.mean(final_per_run)),
            "final_std": float(np.std(final_per_run, ddof=0))
        })
    plt.yscale("log")
    plt.xlabel("Tiempo")
    plt.ylabel("|E(t)-E(0)| / |E(0)|")
    plt.title("Deriva relativa de energía")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    energy_time_plot = output_dir / "energy_drift_vs_time.png"
    plt.tight_layout()
    plt.savefig(energy_time_plot, dpi=300)
    print(f"✅ Guardado {energy_time_plot}")

    dts_summary = [row["dt"] for row in summary_rows]
    max_means = [row["max_mean"] for row in summary_rows]
    max_stds = [row["max_std"] for row in summary_rows]

    plt.figure(figsize=(6, 5))
    plt.errorbar(dts_summary, max_means, yerr=max_stds, fmt="o-", capsize=4)
    plt.xlabel("dt")
    plt.ylabel("max |ΔE/E|")
    plt.title("Error máximo de energía vs dt")
    plt.grid(True, which="both", alpha=0.3)
    convergence_plot = output_dir / "energy_error_vs_dt.png"
    plt.tight_layout()
    plt.savefig(convergence_plot, dpi=300)
    print(f"✅ Guardado {convergence_plot}")

    print("\nResumen de deriva de energía (agregado por dt):")
    header = (
        "dt",
        "runs",
        "mean max|ΔE/E|",
        "std max|ΔE/E|",
        "best",
        "worst",
        "mean RMS",
        "std RMS",
        "mean final",
        "std final",
    )
    print("\t".join(header))
    for row in sorted(summary_rows, key=lambda r: r["dt"]):
        print(
            f"{row['dt']:.6g}\t{row['runs']}\t{row['max_mean']:.3e}\t{row['max_std']:.3e}\t"
            f"{row['max_best']:.3e}\t{row['max_worst']:.3e}\t{row['rms_mean']:.3e}\t"
            f"{row['rms_std']:.3e}\t{row['final_mean']:.3e}\t{row['final_std']:.3e}"
        )
        if row["runs"] < args.min_runs:
            print(f"⚠️  dt={row['dt']:.6g} tiene solo {row['runs']} realizacion(es); apuntá a {args.min_runs} para estadísticas robustas.")


def ensure_energy_data(energy_dir: Path, args, dts: List[float], dt_output_override: Optional[float]) -> None:
    energy_dir.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else int(time.time() * 1000)

    for idx, dt in enumerate(dts):
        dt_label = format_dt_label(dt)
        dt_root = energy_dir / f"dt_{dt_label}"
        energy_subdir = dt_root / "energy"

        available = count_matching_runs(energy_subdir, args.pattern, args.N, dt)
        if available >= args.runs or args.skip_generate:
            continue

        dt_root.mkdir(parents=True, exist_ok=True)
        dt_output = dt_output_override if dt_output_override is not None else dt
        seed = base_seed + idx * 1000

        scenario_flag = "--collision" if args.collision else ""
        java_args = (
            f"--N {args.N} --runs {args.runs} --dt {dt} --dt-output {dt_output} "
            f"--tf {args.tf} --speed {args.speed} --h {args.softening} --output-dir {dt_root} "
            f"--seed {seed} {scenario_flag.strip()}"
        ).strip()
        if args.collision:
            java_args += f" --dx {args.dx} --dy {args.dy}"

        run_make_command(["make", "run-galaxy", f"ARGS={java_args}"])

    for dt in dts:
        dt_root = energy_dir / f"dt_{format_dt_label(dt)}" / "energy"
        available = count_matching_runs(dt_root, args.pattern, args.N, dt)
        if available < args.runs:
            print(f"⚠️  dt={dt:g} cuenta con {available}/{args.runs} corridas."
                  " Revisa la generación de datos.")


def run_make_command(command: List[str]) -> None:
    project_root = Path(__file__).resolve().parents[2]
    try:
        subprocess.run(command, check=True, cwd=project_root)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Error ejecutando {' '.join(command)}", file=sys.stderr)
        sys.exit(exc.returncode)


def count_matching_runs(energy_subdir: Path, pattern: str, n_value: int, dt_target: float, tol: float = 1e-9) -> int:
    if not energy_subdir.exists():
        return 0
    total = 0
    for path in energy_subdir.glob(pattern):
        name = path.name
        if f"N{n_value}_" not in name:
            continue
        dt, *_ = parse_energy_file(path)
        if abs(dt - dt_target) <= tol:
            total += 1
    return total


def format_dt_label(dt: float) -> str:
    if dt == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(dt))))
    mantissa = dt / (10 ** exponent)
    return f"{mantissa:.2f}e{exponent:+d}"


if __name__ == "__main__":
    main()
