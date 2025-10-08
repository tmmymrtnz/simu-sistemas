#!/usr/bin/env python3
"""Compute average crossing time t* where r_hm exceeds a threshold."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from rhm_stats import find_runs, load_series


def first_crossing_time(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    for i in range(1, len(times)):
        if values[i - 1] <= threshold < values[i]:
            t0, t1 = times[i - 1], times[i]
            v0, v1 = values[i - 1], values[i]
            if v1 == v0:
                return t1
            fraction = (threshold - v0) / (v1 - v0)
            return t0 + fraction * (t1 - t0)
    return float("nan")


def summarize_crossings(ns: Sequence[int], grouped: Dict[int, List[Path]], threshold: float,
                        runs: int) -> List[Tuple[int, float, float, int]]:
    results: List[Tuple[int, float, float, int]] = []
    print("N\tvalid_runs\t<t* >\tstd")
    for n in ns:
        paths = grouped.get(n, [])
        if not paths:
            print(f"{n}\t0\tNaN\tNaN")
            continue
        selected = paths[:runs]
        crossings: List[float] = []
        for path in selected:
            times, rhm = load_series(path)
            t_cross = first_crossing_time(times, rhm, threshold)
            if np.isnan(t_cross):
                continue
            crossings.append(t_cross)
        if not crossings:
            print(f"{n}\t0\tNaN\tNaN")
            continue
        arr = np.asarray(crossings)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        count = int(len(arr))
        results.append((n, mean, std, count))
        print(f"{n}\t{count}\t{mean:.6f}\t{std:.6f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Average crossing time for r_hm")
    parser.add_argument("--root", default=os.path.join("..", "out", "galaxy", "energy"),
                        help="Directory with energy outputs (containing r_hm column)")
    parser.add_argument("--output", default="plots", help="Directory to save the t* plot")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for r_hm")
    parser.add_argument("--Ns", type=int, nargs="+",
                        help="Valores de N a analizar (default: 100 200 400 800 1200 1600 2000)")
    parser.add_argument("--runs", type=int, default=10, help="Cantidad mínima de realizaciones por N")
    parser.add_argument("--dt", type=float, default=1e-3, help="Paso temporal dt")
    parser.add_argument("--dt-output", type=float, default=None, help="Paso de guardado dt_output")
    parser.add_argument("--tf", type=float, default=20.0, help="Tiempo final de cada simulación")
    parser.add_argument("--speed", type=float, default=0.1, help="Módulo de las velocidades iniciales")
    parser.add_argument("--softening", type=float, default=0.05, help="Parámetro de suavizado gravitacional h")
    args = parser.parse_args()

    if args.Ns is None:
        args.Ns = [100, 500, 1000, 1500, 2000]
    if args.dt_output is None:
        args.dt_output = args.dt

    root = Path(args.root).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(root.glob("*.txt"))
    if not files:
        raise SystemExit(f"No energy files found in {root}")

    grouped_all = find_runs(files)
    grouped = {n: grouped_all.get(n, []) for n in args.Ns}

    results = summarize_crossings(args.Ns, grouped, args.threshold, args.runs)
    results = [res for res in results if res[3] > 0]
    if not results:
        print("⚠️ No se encontraron cruces del umbral especificado.")
        return

    ordered = sorted(results)
    ns, means, stds, counts = zip(*ordered)
    plt.figure(figsize=(6, 5))
    error_container = plt.errorbar(ns, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel("N")
    plt.ylabel("<t*>")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    if hasattr(error_container, "lines"):
        main_line = error_container.lines[0]
    elif isinstance(error_container, (tuple, list)) and error_container:
        main_line = error_container[0]
    else:
        main_line = error_container
    color = main_line.get_color()
    info_text = "\n".join(
        f"N={n} ({count}/{args.runs})" for n, count in zip(ns, counts)
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize="x-small",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor=color),
    )
    out_path = output_dir / "rhm_tcross_vs_N.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Guardado {out_path}")


if __name__ == "__main__":
    main()
