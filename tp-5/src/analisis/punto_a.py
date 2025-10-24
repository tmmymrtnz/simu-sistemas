from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import os

from common import (
    PROJECT_ROOT,
    add_simulation_arguments,
    area_fraction,
    compute_scanning_rate,
    ensure_simulation,
    load_contacts,
    load_states,
    params_from_args,
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

    args = parser.parse_args()
    base_params = params_from_args(args)
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
        _plot_contacts_comparison(counts_to_contacts)


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
    ax.set_xlabel("t [s]")
    ax.set_ylabel("unique contacts")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path = plot_dir / f"contacts_curve_N{count}_seed{seed}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_contacts_comparison(counts_to_contacts: dict[int, list[tuple[int, list, Path]]]) -> None:
    if not counts_to_contacts:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map: dict[int, str] = {}
    labeled: set[int] = set()

    for count, entries in sorted(counts_to_contacts.items()):
        for seed, contacts, _ in entries:
            if not contacts:
                continue
            ordered = sorted(contacts, key=lambda c: c.time)
            times = [c.time for c in ordered]
            ordinals = list(range(1, len(ordered) + 1))
            if count not in color_map:
                color_map[count] = color_cycle[len(color_map) % len(color_cycle)] if color_cycle else None
            color = color_map[count]
            label = f"N={count}" if count not in labeled else None
            ax.plot(times, ordinals, linewidth=1.2, color=color, label=label)
            labeled.add(count)

    ax.set_xlabel("t [s]")
    ax.set_ylabel("unique contacts")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if labeled:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "contacts_vs_time.png", dpi=200)
    plt.close(fig)


def _plot_phi_vs_q(rows) -> None:
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phis = [phi for _, _, phi, _, _ in rows]
    rates = [rate for _, _, _, rate, _ in rows]

    fig, ax = plt.subplots()
    ax.scatter(phis, rates, marker="o")
    ax.set_xlabel("phi")
    ax.set_ylabel("Q [1/s]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "phi_vs_Q.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
