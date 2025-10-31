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
    ensure_simulation,
    inter_contact_times,
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
import numpy as np

try:
    import powerlaw  # type: ignore
except Exception:  # pragma: no cover
    powerlaw = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysis for TP5 point (b): inter-contact times.")
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
        "--xmin",
        type=float,
        help="Optional xmin for the power-law fit.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Optional CSV path to store exponent estimates.",
    )

    args = parser.parse_args()
    base_params = params_from_args(args)
    agent_values = args.agents_list or [base_params.agents]
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [base_params.seed + i for i in range(5)]

    if powerlaw is None:
        print("Warning: powerlaw package not available. Exponent fits will be skipped.")

    per_agent_stats = defaultdict(list)
    lines = ["N,seed,phi,count,alpha,sigma,output"]

    alpha_points = []

    for count in agent_values:
        for seed in seeds:
            params = base_params.with_agents(count).with_seed(seed)
            output_dir = ensure_simulation(params)
            states = load_states(output_dir / "states.txt")
            contacts = load_contacts(output_dir / "contacts.txt")
            phi = area_fraction(states, params.domain)
            deltas = inter_contact_times(contacts)

            alpha = float("nan")
            sigma = float("nan")
            fit = None
            if powerlaw is not None and len(deltas) >= 2:
                fit = powerlaw.Fit(deltas, xmin=args.xmin, discrete=False, verbose=False)
                alpha = float(fit.alpha)
                sigma = float(fit.sigma)

            per_agent_stats[count].append((phi, len(deltas), alpha, sigma))
            lines.append(
                f"{count},{seed},{phi:.6f},{len(deltas)},{alpha:.6f},{sigma:.6f},{output_dir}"
            )
            print(
                f"N={count:4d} seed={seed:6d} phi={phi:.4f} samples={len(deltas):5d} "
                f"alpha={alpha:.6f} sigma={sigma:.6f} output={output_dir}"
            )
            _plot_inter_contact_distribution(output_dir, count, seed, deltas, fit)
            if not _is_nan(alpha):
                alpha_points.append((phi, alpha, sigma))

    print("\nAveraged exponent per N:")
    for count in agent_values:
        samples = per_agent_stats[count]
        if not samples:
            continue
        phis = [phi for phi, _, _, _ in samples]
        alpha_values = [alpha for _, _, alpha, _ in samples if not _is_nan(alpha)]
        sigma_values = [sigma for _, _, _, sigma in samples if not _is_nan(sigma)]
        phi_mean = mean(phis)
        alpha_mean = mean(alpha_values) if alpha_values else float("nan")
        sigma_mean = mean(sigma_values) if sigma_values else float("nan")
        print(
            f"N={count:4d} phi_mean={phi_mean:.4f} alpha_mean={alpha_mean:.6f} "
            f"sigma_mean={sigma_mean:.6f} samples={len(samples)}"
        )

    if args.store:
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved detailed rows to {args.store}")

    _plot_alpha_vs_phi(alpha_points)
    _plot_alpha_phi_aggregated(per_agent_stats)


def _is_nan(value: float) -> bool:
    return value != value


def _plot_inter_contact_distribution(output_dir: Path, count: int, seed: int, deltas, fit) -> None:
    if len(deltas) < 2:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sorted_deltas = np.sort(np.asarray(deltas))
    ccdf = 1.0 - np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas)

    fig, ax = plt.subplots()
    ax.loglog(sorted_deltas, ccdf, marker="o", linestyle="none", markersize=3)

    if fit is not None:
        xmin = fit.xmin
        xmax = sorted_deltas.max()
        if xmax > xmin:
            x = np.logspace(np.log10(xmin), np.log10(xmax), 200)
            theoretical = (x / xmin) ** (1 - fit.alpha)
            mask = sorted_deltas >= xmin
            if mask.any():
                ref_x = sorted_deltas[mask][0]
                ref_y = ccdf[mask][0]
                theoretical *= ref_y / ((ref_x / xmin) ** (1 - fit.alpha))
            ax.loglog(x, theoretical, linestyle="--", color="red")

    ax.set_xlabel("tau [s]")
    ax.set_ylabel("P(T > tau)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / f"inter_contact_ccdf_N{count}_seed{seed}.png", dpi=200)
    plt.close(fig)


def _plot_alpha_vs_phi(alpha_points) -> None:
    if not alpha_points:
        return
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phis = [phi for phi, _, _ in alpha_points]
    alphas = [alpha for _, alpha, _ in alpha_points]
    sigmas = [sigma if sigma == sigma else 0.0 for _, _, sigma in alpha_points]

    fig, ax = plt.subplots()
    ax.errorbar(phis, alphas, yerr=sigmas, fmt="o", capsize=3)
    ax.set_xlabel("phi")
    ax.set_ylabel("alpha")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_vs_phi.png", dpi=200)
    plt.close(fig)


def _std(values, mean_value) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _plot_alpha_phi_aggregated(per_agent_stats: dict[int, list[tuple[float, int, float, float]]]) -> None:
    """Plot aggregated means per N with error bars in x (phi) and y (alpha).

    per_agent_stats maps N -> list of tuples (phi, count_samples, alpha, sigma)
    """
    if not per_agent_stats:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    counts = sorted(per_agent_stats.keys())
    phi_means = []
    phi_stds = []
    alpha_means = []
    alpha_stds = []

    for count in counts:
        samples = per_agent_stats[count]
        if not samples:
            continue
        phis = [phi for phi, _, _, _ in samples]
        alphas = [alpha for _, _, alpha, _ in samples if not _is_nan(alpha)]
        phi_mean = mean(phis)
        phi_std = _std(phis, phi_mean)
        if alphas:
            alpha_mean = mean(alphas)
            alpha_std = _std(alphas, alpha_mean)
        else:
            alpha_mean = float("nan")
            alpha_std = 0.0

        phi_means.append(phi_mean)
        phi_stds.append(phi_std)
        alpha_means.append(alpha_mean)
        alpha_stds.append(alpha_std)

    fig, ax = plt.subplots()
    ax.errorbar(phi_means, alpha_means, xerr=phi_stds, yerr=alpha_stds, fmt='o', capsize=4)
    for i, count in enumerate(counts):
        if i < len(phi_means):
            ax.annotate(str(count), (phi_means[i], alpha_means[i]), textcoords='offset points', xytext=(6, 4))

    ax.set_xlabel("phi")
    ax.set_ylabel("alpha")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_phi_aggregated.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
