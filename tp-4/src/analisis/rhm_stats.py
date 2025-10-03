#!/usr/bin/env python3
"""Mass half-radius statistics for the galaxy simulations."""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATTERN = re.compile(r"N(\d+)_run(\d+)")


def load_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Unexpected energy file format in {path}")
    times = data[:, 0]
    rhm_values = data[:, 2]
    return times, rhm_values


def aggregate_runs(series: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_times = series[0][0]
    values = []
    for times, rhm in series:
        if not np.allclose(times, base_times):
            raise ValueError("All runs must share the same timeline")
        values.append(rhm)
    stacked = np.vstack(values)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return base_times, mean, std


def infer_group(path: Path) -> int:
    match = SNAPSHOT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not infer N from filename {path}")
    return int(match.group(1))


def find_runs(files: Iterable[Path]) -> Dict[int, List[Path]]:
    groups: Dict[int, List[Path]] = {}
    for path in files:
        try:
            group = infer_group(path)
        except ValueError:
            continue
        groups.setdefault(group, []).append(path)
    return {k: sorted(v) for k, v in groups.items()}


def compute_slope(times: np.ndarray, values: np.ndarray, tail_fraction: float, start_index: Optional[int] = None) -> float:
    if start_index is not None:
        start_idx = max(0, min(start_index, len(times) - 2))
    else:
        start_idx = int((1.0 - tail_fraction) * len(times))
        start_idx = max(0, min(start_idx, len(times) - 2))
    tail_times = times[start_idx:]
    tail_values = values[start_idx:]
    slope, _ = np.polyfit(tail_times, tail_values, 1)
    return float(slope)


def run_make_command(command: List[str]) -> None:
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Error ejecutando {' '.join(command)}", file=sys.stderr)
        sys.exit(exc.returncode)


def ensure_simulation_data(sim_root: Path, args) -> None:
    sim_root.mkdir(parents=True, exist_ok=True)
    output_dir = sim_root.parent
    required = set(args.Ns)
    compile_done = False

    for n in args.Ns:
        existing = list(sim_root.glob(f"gaussian_N{n}_run*_energy.txt"))
        if len(existing) >= args.runs:
            continue

        missing = args.runs - len(existing)
        print(f"⚙️ Generando simulaciones para N={n} (faltan {missing} corridas)...")
        if not compile_done:
            run_make_command(["make", "compile"])
            compile_done = True

        seed = int(time.time() * 1000)
        java_args = (
            f"--N {n} --runs {args.runs} --dt {args.dt} --dt-output {args.dt_output} "
            f"--tf {args.tf} --speed {args.speed} --h {args.softening} --output-dir {str(output_dir)} "
            f"--seed {seed}"
        )
        run_make_command(["make", "run-galaxy", f"ARGS={java_args}"])

    for n in required:
        files = list(sim_root.glob(f"gaussian_N{n}_run*_energy.txt"))
        if len(files) < args.runs:
            raise SystemExit(f"No se pudieron generar suficientes corridas para N={n}. Revisa los logs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute <r_hm(t)> and slopes vs N")
    parser.add_argument("--root", default=os.path.join("..", "out", "galaxy", "energy"),
                        help="Directory with energy outputs (containing rhm column)")
    parser.add_argument("--output", default="plots", help="Directory to save figures")
    parser.add_argument("--tail", type=float, default=0.3, help="Fraction of late-time samples to fit")
    parser.add_argument("--Ns", type=int, nargs="+", help="Valores de N a considerar (default: 100 200 400 800 1200 1600 2000)")
    parser.add_argument("--runs", type=int, default=5, help="Cantidad mínima de realizaciones por N")
    parser.add_argument("--dt", type=float, default=1e-3, help="Paso temporal dt")
    parser.add_argument("--dt-output", type=float, default=5e-2, help="Paso de guardado dt_output (default = dt si no se especifica)")
    parser.add_argument("--tf", type=float, default=20.0, help="Tiempo final de la simulación")
    parser.add_argument("--speed", type=float, default=0.1, help="Módulo de las velocidades iniciales")
    parser.add_argument("--softening", type=float, default=0.05, help="Parámetro de suavizado gravitacional h")
    parser.add_argument("--interactive", action="store_true",
                        help="Permite seleccionar manualmente el inicio de la tendencia estable")
    args = parser.parse_args()

    if args.Ns is None:
        args.Ns = [100, 500, 1000, 1500, 2000]
    if args.dt_output is None:
        args.dt_output = args.dt

    root = Path(args.root).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_simulation_data(root, args)

    files = list(root.glob("*.txt"))
    if not files:
        raise SystemExit(f"No snapshot files found in {root}")

    grouped_all = find_runs(files)
    target_set = set(args.Ns)
    grouped = {n: sorted(paths) for n, paths in grouped_all.items() if n in target_set}

    missing = [n for n in args.Ns if n not in grouped or len(grouped[n]) < args.runs]
    if missing:
        raise SystemExit(f"Datos insuficientes para N={missing}; verificá que las simulaciones se generaron correctamente.")

    slopes: List[Tuple[int, float]] = []

    for n, paths in sorted(grouped.items()):
        selected_paths = paths[:args.runs]
        series = [load_series(path) for path in selected_paths]
        times, mean, std = aggregate_runs(series)

        fig, ax = plt.subplots(figsize=(8, 5))
        for path, (_, values) in zip(selected_paths, series):
            label = path.stem.split("_run")[1]
            ax.plot(times, values, alpha=0.3, linewidth=0.8, label=f"run {label}")
        ax.plot(times, mean, color="black", linewidth=2.0, label="Promedio")
        ax.fill_between(times, mean - std, mean + std, alpha=0.2, color="gray")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("r_hm")
        ax.set_title(f"Evolución r_hm(t) N={n}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="x-small")

        selected_idx: Optional[int] = None
        if args.interactive:
            ax.set_title(f"Evolución r_hm(t) N={n}\nHaz click en el inicio de la tendencia estable")
            fig.canvas.draw()
            click = plt.ginput(1, timeout=-1)
            if click:
                selected_time = click[0][0]
                selected_idx = int(np.clip(np.searchsorted(times, selected_time), 0, len(times) - 2))
                ax.axvline(times[selected_idx], color="red", linestyle="--", label="Inicio estable")
                ax.legend(loc="upper right", fontsize="x-small")
                print(f"• N={n}: seleccionaste t≈{times[selected_idx]:.4f} como inicio estable")
            else:
                print(f"⚠️ N={n}: no se seleccionó punto; se usa la cola automática")

        out = output_dir / f"rhm_timeseries_N{n}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        if args.interactive:
            plt.show(block=False)
            plt.pause(0.5)
        plt.close(fig)
        print(f"✅ Guardado {out}")

        slope = compute_slope(times, mean, args.tail, start_index=selected_idx)
        print(f"Pendiente estimada para N={n}: {slope:.6e}")
        slopes.append((n, slope))

    ns, slope_values = zip(*sorted(slopes))
    plt.figure(figsize=(6, 5))
    plt.plot(ns, slope_values, marker="o")
    plt.xlabel("N")
    plt.ylabel("Pendiente <r_hm(t)>")
    plt.title("Estado estacionario: pendiente vs N")
    plt.grid(True, alpha=0.3)
    slope_plot = output_dir / "rhm_slope_vs_N.png"
    plt.tight_layout()
    plt.savefig(slope_plot, dpi=300)
    print(f"✅ Guardado {slope_plot}")


if __name__ == "__main__":
    main()
