from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean
import os

from common import (
    PROJECT_ROOT,
    add_simulation_arguments,
    ensure_simulation,
    ensure_simulations_parallel,
    load_contacts,
    load_states,
    params_from_args,
    area_fraction,
    compute_scanning_rate,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Q_max vs desired speed v")
    add_simulation_arguments(parser)
    parser.add_argument(
        "--v-list",
        type=float,
        nargs="+",
        help="List of desired speeds v to process. Default uses --desired-speed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to average. Default uses --seed and 4 next seeds.",
    )
    parser.add_argument(
        "--agents-list",
        type=int,
        nargs="+",
        help="List of agent counts (N) to sweep. Default 10 20 ... 100.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of realizations to average together when computing Q (default 1).",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Optional CSV path to store the aggregated Qmax results.",
    )

    args = parser.parse_args()
    base_params = params_from_args(args)

    if args.v_list:
        v_values = args.v_list
    else:
        v_values = [base_params.desired_speed]

    if args.seeds:
        seeds = args.seeds
    else:
        # use 5 repetitions by default
        seeds = [base_params.seed + i for i in range(5)]

    if args.agents_list:
        agents_list = args.agents_list
    else:
        agents_list = list(range(10, 101, 10))

    group_size = args.group_size
    print(f"Running Qmax vs v analysis: v_values={v_values} seeds={seeds} group_size={group_size}")

    results = []  # list of tuples (v, qmax, best_N, phi_of_qmax, sample_count)

    for v in v_values:
        v_tag = format_float(v)
        base_for_v = replace(
            base_params,
            desired_speed=v,
            output_base=base_params.output_base / f"v_{v_tag}",
        )
        ensure_simulations_parallel(base_for_v, agents_list, seeds)

        per_N_stats: list[tuple[int, float, float]] = []  # list of (N, q_mean, phi_mean)
        for N in agents_list:
            qs = []
            phis = []
            for seed in seeds:
                params = (
                    base_for_v.with_agents(N)
                    .with_seed(seed)
                )
                output_dir = ensure_simulation(params)
                states = load_states(output_dir / "states.txt")
                contacts = load_contacts(output_dir / "contacts.txt")
                phi = area_fraction(states, params.domain)
                q = compute_scanning_rate(contacts)
                qs.append(q)
                phis.append(phi)
                print(f"v={v:.3f} N={N} seed={seed} phi={phi:.4f} Q={q:.6f} output={output_dir}")

            # compute mean Q across the seeds for this N (ignore NaN)
            valid_qs = [q for q in qs if q == q]
            q_mean = mean(valid_qs) if valid_qs else float("nan")
            phi_mean = mean(phis) if phis else float("nan")
            per_N_stats.append((N, q_mean, phi_mean))

        # choose N that gives maximum q_mean
        valid_per_N = [s for s in per_N_stats if s[1] == s[1]]
        if valid_per_N:
            best = max(valid_per_N, key=lambda s: s[1])
            best_N, qmax, phi_of_qmax = best
            print(f"v={v:.3f} best N={best_N} Q_max={qmax:.6f} phi_mean={phi_of_qmax:.4f}")
        else:
            best_N = None
            qmax = float("nan")
            phi_of_qmax = float("nan")

        results.append((v, qmax, best_N, phi_of_qmax, len(agents_list) * len(seeds)))

    # write CSV if requested
    lines = ["v,Qmax,best_N,phi_of_qmax,samples"]
    for v, qmax, best_N, phi_of_qmax, cnt in results:
        n_txt = "" if best_N is None else str(best_N)
        lines.append(f"{v:.6f},{qmax:.6f},{n_txt},{phi_of_qmax:.6f},{cnt}")
    if args.store:
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved summary to {args.store}")

    _plot_qmax_vs_v(results)


def _plot_qmax_vs_v(results) -> None:
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    vs = [r[0] for r in results]
    qmaxs = [r[1] for r in results]
    phi_means = [r[3] for r in results]

    fig, ax = plt.subplots()
    ax.plot(vs, qmaxs, marker="o", linestyle="-")
    ax.set_xlabel(r"$v$ [m/s]")
    ax.set_ylabel(r"$Q_{\max}$ [1/s]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "Qmax_vs_v.png", dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_dir / 'Qmax_vs_v.png'}")


if __name__ == "__main__":
    main()
