from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
import os

from common import (
    PROJECT_ROOT,
    add_simulation_arguments,
    area_fraction,
    compute_scanning_rate,
    ensure_simulation,
    ensure_simulations_parallel,
    load_contacts,
    load_states,
    params_from_args,
    format_float,
)

MPL_CACHE_DIR = PROJECT_ROOT / "tmp_mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
XDG_CACHE_DIR = PROJECT_ROOT / "tmp_cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysis for TP5 point (a): scanning rate Q.")
    add_simulation_arguments(parser)
    parser.add_argument(
        "--agents-list",
        type=int,
        nargs="+",
        help="Agent counts (N) to process. Default uses --agents.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to average. Default uses --seed.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Optional CSV path to store the aggregated results.",
    )
    parser.add_argument(
        "--speed-output",
        type=float,
        help="When provided, analysis looks under output/v_<speed> for simulations.",
    )

    args = parser.parse_args()
    base_params = params_from_args(args)
    if args.speed_output is not None:
        speed_tag = format_float(args.speed_output)
        speed_base = base_params.output_base / Path(f"v_{speed_tag}")
        base_params = replace(base_params, output_base=speed_base)

    agent_values = args.agents_list or [base_params.agents]
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [base_params.seed + i for i in range(5)]

    print("Running analysis (a) for parameters:")
    print(f"  model={base_params.model} N_list={agent_values} seeds={seeds}")

    per_agent_rates = defaultdict(list)
    rows = []

    counts_to_contacts = {}
    run_summary: dict[tuple[int, int], tuple[float, float]] = {}

    ensure_simulations_parallel(base_params, agent_values, seeds)

    for count in agent_values:
        for seed in seeds:
            params = base_params.with_agents(count).with_seed(seed)
            output_dir = ensure_simulation(params)
            states = load_states(output_dir / "states.txt")
            contacts = load_contacts(output_dir / "contacts.txt")
            phi = area_fraction(states, params.domain)
            rate = compute_scanning_rate(contacts)
            per_agent_rates[count].append((phi, rate))
            rows.append((count, seed, phi, rate, output_dir))
            run_summary[(count, seed)] = (phi, rate)
            print(
                f"N={count:4d} seed={seed:6d} phi={phi:.4f} Q={rate:.6f} output={output_dir}"
            )
            counts_to_contacts.setdefault(count, []).append((seed, contacts, output_dir))

    print("\nAveraged results per N:")
    lines = ["N,phi_mean,phi_std,Q_mean,Q_std,count"]
    for count in agent_values:
        samples = per_agent_rates[count]
        if not samples:
            continue
        phis = [phi for phi, _ in samples]
        rates = [rate for _, rate in samples]
        phi_mean = mean(phis)
        phi_std = _std(phis, phi_mean)
        rate_mean = mean(rates)
        rate_std = _std(rates, rate_mean)
        print(
            f"N={count:4d} phi_mean={phi_mean:.4f} phi_std={phi_std:.4f} "
            f"Q_mean={rate_mean:.6f} Q_std={rate_std:.6f} "
            f"samples={len(samples)}"
        )
        lines.append(
            f"{count},{phi_mean:.6f},{phi_std:.6f},{rate_mean:.6f},{rate_std:.6f},{len(samples)}"
        )

    if args.store:
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved summary to {args.store}")

    if rows:
        _plot_phi_vs_q(rows)
        _plot_phi_q_aggregated(per_agent_rates)
        phi_means = {
            count: mean(phi for phi, _ in samples) for count, samples in per_agent_rates.items() if samples
        }
        _plot_contacts_comparison(counts_to_contacts, phi_means)
        _plot_q_error_curves(60, counts_to_contacts, run_summary)


def _std(values, mean_value) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5


def _plot_contacts_curve(output_dir: Path, contacts, count: int, seed: int) -> Path:
    if not contacts:
        return Path()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(contacts, key=lambda c: c.time)
    times = [c.time for c in ordered]
    ordinals = list(range(1, len(ordered) + 1))

    fig, ax = plt.subplots()
    ax.plot(times, ordinals, marker="o", linestyle="-", linewidth=1.2, markersize=3)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$N_{\mathrm{contactos}}$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path = plot_dir / f"contacts_curve_N{count}_seed{seed}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_contacts_comparison(
    counts_to_contacts: dict[int, list[tuple[int, list, Path]]],
    phi_means: dict[int, float],
) -> None:
    if not counts_to_contacts:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map: dict[int, str] = {}
    has_mean_curve = False
    selected_counts = {10, 70, 160, 240, 300}

    for count, entries in sorted(counts_to_contacts.items()):
        if count not in selected_counts:
            continue
        # collect per-seed time/ordinal arrays for averaging
        per_seed_times = []
        per_seed_ordinals = []
        for seed, contacts, _ in entries:
            if not contacts:
                continue
            ordered = sorted(contacts, key=lambda c: c.time)
            times = [c.time for c in ordered]
            ordinals = list(range(1, len(ordered) + 1))
            per_seed_times.append(np.array(times))
            per_seed_ordinals.append(np.array(ordinals))
            if count not in color_map:
                color_map[count] = color_cycle[len(color_map) % len(color_cycle)] if color_cycle else None

        # compute and plot mean temporal evolution across seeds for this N
        if per_seed_times:
            # choose a common grid from 0 to max final time among seeds
            max_time = max(t[-1] for t in per_seed_times)
            time_grid = np.linspace(0.0, max_time, 300)
            interp_matrix = []
            for t_arr, o_arr in zip(per_seed_times, per_seed_ordinals):
                # ensure arrays are increasing in time; np.interp requires increasing x
                if len(t_arr) == 0:
                    continue
                # use left=0 (no contacts before first), right=last ordinal
                interp = np.interp(time_grid, t_arr, o_arr, left=0.0, right=float(o_arr[-1]))
                interp_matrix.append(interp)
            if interp_matrix:
                stacked = np.vstack(interp_matrix)
                mean_curve = stacked.mean(axis=0)
                color = color_map.get(count, None)
                phi_value = phi_means.get(count, float("nan"))
                if phi_value == phi_value:
                    mean_label = fr"$\phi = {phi_value:.4f}$"
                else:
                    mean_label = f"N={count}"
                ax.plot(
                    time_grid,
                    mean_curve,
                    linewidth=3.0,
                    color=color,
                    linestyle="-",
                    label=mean_label,
                    zorder=3,
                )
                has_mean_curve = True

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$N_{\mathrm{contactos}}$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if has_mean_curve:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "contacts_vs_time.png", dpi=200)
    plt.close(fig)


def _plot_q_error_curves(
    target_count: int,
    counts_to_contacts: dict[int, list[tuple[int, list, Path]]],
    run_summary: dict[tuple[int, int], tuple[float, float]],
) -> None:
    entries = counts_to_contacts.get(target_count)
    if not entries:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    has_curve = False

    for seed, contacts, _ in entries:
        if not contacts:
            continue
        ordered = sorted(contacts, key=lambda c: c.time)
        if len(ordered) < 3:
            continue
        times = np.array([c.time for c in ordered], dtype=float)
        ordinals = np.array([c.ordinal for c in ordered], dtype=float)

        slope = compute_scanning_rate(ordered)
        if slope != slope:
            continue

        slope = float(slope)
        span = max(abs(slope) * 0.4, 1e-3)
        q_min = max(slope - 3 * span, 1e-6)
        q_max = slope + 3 * span
        q_values = np.linspace(q_min, q_max, 300)

        mse_values = _mse_curve(times, ordinals, q_values)
        mse_opt = _mse_curve(times, ordinals, np.array([slope]))[0]

        phi_value, stored_slope = run_summary.get((target_count, seed), (float("nan"), slope))
        label = f"seed={seed}"
        if phi_value == phi_value:
            label += f" | $\\phi={phi_value:.4f}$"
        label += f" | $Q={stored_slope:.4f}$"

        ax.plot(q_values, mse_values, label=label)
        ax.scatter([slope], [mse_opt], marker="o", s=30)
        has_curve = True

    if not has_curve:
        plt.close(fig)
        return

    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$\mathrm{ECM}(Q)$")
    ax.set_title(f"Hipérbolas de ECM para N={target_count}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"q_mse_hyperbolas_N{target_count}.png", dpi=200)
    plt.close(fig)


def _mse_curve(times: np.ndarray, ordinals: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    if times.size == 0:
        return np.zeros_like(slopes)
    x_mean = times.mean()
    y_mean = ordinals.mean()
    mse = []
    for slope in slopes:
        intercept = y_mean - slope * x_mean
        residuals = ordinals - (slope * times + intercept)
        mse.append(np.mean(residuals**2))
    return np.asarray(mse, dtype=float)


def _plot_phi_vs_q(rows) -> None:
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phis = [phi for _, _, phi, _, _ in rows]
    rates = [rate for _, _, _, rate, _ in rows]

    fig, ax = plt.subplots()
    ax.scatter(phis, rates, marker="o")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$Q$ [1/s]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "phi_vs_Q.png", dpi=200)
    plt.close(fig)


def _plot_phi_q_aggregated(per_agent_rates: dict[int, list[tuple[float, float]]]) -> None:
    """Plot the aggregated means per N with error bars in both x (phi) and y (Q).

    per_agent_rates: mapping N -> list of (phi, Q) samples (one per seed/run)
    """
    if not per_agent_rates:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    counts = sorted(per_agent_rates.keys())
    phi_means = []
    phi_stds = []
    q_means = []
    q_stds = []

    for count in counts:
        samples = per_agent_rates[count]
        if not samples:
            continue
        phis = [p for p, _ in samples]
        qs = [q for _, q in samples]
        phi_mean = mean(phis)
        phi_std = _std(phis, phi_mean)
        q_mean = mean(qs)
        q_std = _std(qs, q_mean)
        phi_means.append(phi_mean)
        phi_stds.append(phi_std)
        q_means.append(q_mean)
        q_stds.append(q_std)

    fig, ax = plt.subplots()
    # errorbar with xerr and yerr
    ax.errorbar(phi_means, q_means, xerr=phi_stds, yerr=q_stds, fmt='o', capsize=4)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$Q$ [1/s]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "phi_Q_aggregated.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
