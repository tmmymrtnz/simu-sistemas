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
    counts_to_deltas = defaultdict(list)
    # we'll store pooled results later if requested
    lines = ["N,phi_mean,total_samples,alpha_pooled,sigma_pooled,xmin_pooled"]

    for count in agent_values:
        for seed in seeds:
            params = base_params.with_agents(count).with_seed(seed)
            output_dir = ensure_simulation(params)
            states = load_states(output_dir / "states.txt")
            contacts = load_contacts(output_dir / "contacts.txt")
            phi = area_fraction(states, params.domain)
            deltas = inter_contact_times(contacts)

            # store deltas per N for pooled CCDF plotting
            counts_to_deltas[count].append(list(deltas))

            # keep per-agent stats shape compatible (alpha,sigma as NaN)
            per_agent_stats[count].append((phi, len(deltas), float("nan"), float("nan")))
            # do not perform per-seed fits or per-seed plots — we'll work on pooled data only

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

    # First compute pooled fits and plots for CCDF; get pooled alpha per N
    pooled_fits = _plot_ccdf_aggregated(counts_to_deltas)

    # Print and optionally store pooled fit parameters per N
    pooled_lines = ["N,phi_mean,total_samples,alpha_pooled,sigma_pooled,xmin_pooled"]
    for count in sorted(per_agent_stats.keys()):
        phis = [phi for phi, _, _, _ in per_agent_stats[count]]
        if not phis:
            continue
        phi_mean = mean(phis)
        pooled_alpha = float("nan")
        pooled_sigma = float("nan")
        pooled_xmin = float("nan")
        total_samples = 0
        if count in counts_to_deltas:
            pooled = np.hstack([np.asarray(d) for d in counts_to_deltas[count] if len(d) > 0])
            total_samples = pooled.size
        if count in pooled_fits:
            pooled_alpha, pooled_sigma, pooled_xmin = pooled_fits[count]
        print(f"N={count:4d} phi_mean={phi_mean:.4f} pooled_samples={total_samples:6d} alpha={pooled_alpha:.6f} sigma={pooled_sigma:.6f} xmin={pooled_xmin}")
        pooled_lines.append(f"{count},{phi_mean:.6f},{total_samples},{pooled_alpha:.6f},{pooled_sigma:.6f},{pooled_xmin}")

    if args.store:
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text("\n".join(pooled_lines) + "\n", encoding="utf-8")
        print(f"\nSaved pooled fit rows to {args.store}")

    # Use pooled fits for alpha-vs-phi plots
    pooled_alpha_points = []
    for count in sorted(per_agent_stats.keys()):
        if count not in pooled_fits:
            continue
        phis = [phi for phi, _, _, _ in per_agent_stats[count]]
        if not phis:
            continue
        phi_mean = mean(phis)
        alpha_p, sigma_p, xmin_p = pooled_fits[count]
        pooled_alpha_points.append((phi_mean, alpha_p, sigma_p))

    _plot_alpha_vs_phi(pooled_alpha_points)
    _plot_alpha_phi_aggregated(per_agent_stats, pooled_fits)
    return


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

    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"$P(T > \tau)$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / f"inter_contact_ccdf_N{count}_seed{seed}.png", dpi=200)
    plt.close(fig)


def _plot_inter_contact_evolution(output_dir: Path, count: int, seed: int, deltas) -> None:
    """Plot the temporal evolution of inter-contact times for a single run.

    Top: inter-contact times Δ_i vs event index i
    Bottom: successive difference Δ_{i+1} - Δ_i vs index i
    """
    if len(deltas) < 2:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    deltas_arr = np.asarray(deltas)
    indices = np.arange(1, len(deltas_arr) + 1)

    diffs = deltas_arr[1:] - deltas_arr[:-1]
    diff_indices = indices[:-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(indices, deltas_arr, marker='o', linestyle='-', markersize=3)
    ax1.set_ylabel(r'Δ t [s]')
    ax1.set_title(f'Inter-contact times evolution N={count} seed={seed}')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    ax2.plot(diff_indices, diffs, marker='o', linestyle='-', markersize=3, color='tab:orange')
    ax2.axhline(0.0, color='k', linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_xlabel('event index i')
    ax2.set_ylabel(r'Δ_{i+1}-Δ_i [s]')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(plot_dir / f"inter_contact_evolution_N{count}_seed{seed}.png", dpi=200)
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
    ax.set_xlabel("φ")
    ax.set_ylabel("α") 
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_vs_phi.png", dpi=200)
    plt.close(fig)


def _std(values, mean_value) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _plot_alpha_phi_aggregated(per_agent_stats: dict[int, list[tuple[float, int, float, float]]], pooled_fits: dict[int, tuple[float, float, float]] | None = None) -> None:
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
        phi_mean = mean(phis)
        phi_std = _std(phis, phi_mean)

        # If pooled fit info provided, use it for the aggregated alpha
        if pooled_fits and count in pooled_fits:
            alpha_mean, alpha_std, _ = pooled_fits[count]
        else:
            alphas = [alpha for _, _, alpha, _ in samples if not _is_nan(alpha)]
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

    ax.set_xlabel("φ")
    ax.set_ylabel("α")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_phi_aggregated.png", dpi=200)
    plt.close(fig)


def _plot_ccdf_aggregated(counts_to_deltas: dict[int, list[list[float]]]) -> dict[int, tuple[float, float, float]]:
    """Compute and plot aggregated CCDF per N.

    For each N, interpolates per-run CCDFs onto a common log-spaced grid,
    computes median and 16/84 percentiles and plots median with shaded band.
    """
    if not counts_to_deltas:
        return

    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pooled_fits: dict[int, tuple[float, float, float]] = {}
    for count, runs in sorted(counts_to_deltas.items()):
        # collect only runs with at least one delta
        arrays = [np.asarray(d) for d in runs if len(d) > 0]
        if not arrays:
            continue

        # Interpolate all per-run CCDFs onto a common log-spaced grid and
        # produce a single log-log plot. If the `powerlaw` package is
        # available, fit a power-law to the pooled deltas for this N and
        # draw the fitted theoretical line scaled to the median.
        xgrid_log = np.logspace(np.log10(1e-3), np.log10(15.0), 300)

        ccdf_matrix_log = []
        for arr in arrays:
            # ensure strictly positive values for log interpolation
            arr_pos = np.maximum(arr, 1e-12)
            sorted_arr = np.sort(arr_pos)
            ccdf = 1.0 - np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
            interp = np.interp(xgrid_log, sorted_arr, ccdf, left=1.0, right=0.0)
            ccdf_matrix_log.append(interp)

        stacked_log = np.vstack(ccdf_matrix_log)
        median_log = np.percentile(stacked_log, 50, axis=0)
        lower_log = np.percentile(stacked_log, 16, axis=0)
        upper_log = np.percentile(stacked_log, 84, axis=0)

        fig, ax = plt.subplots()
        # Compute pooled empirical CCDF (concatenate all deltas) and plot only that line
        pooled_alpha = float("nan")
        pooled_sigma = float("nan")
        pooled_xmin = float("nan")
        try:
            pooled = np.hstack(arrays)
            if pooled.size > 0:
                pooled_pos = np.maximum(pooled, 1e-12)
                sorted_pooled = np.sort(pooled_pos)
                ccdf_pooled = 1.0 - np.arange(1, len(sorted_pooled) + 1) / len(sorted_pooled)
                ax.loglog(sorted_pooled, ccdf_pooled, color='k', linewidth=1.8)
            # compute pooled fit parameters but do NOT draw the theoretical line
            if powerlaw is not None and pooled.size >= 2:
                fit = powerlaw.Fit(pooled, xmin=None, discrete=False, verbose=False)
                pooled_alpha = float(fit.alpha)
                pooled_sigma = float(getattr(fit, "sigma", float("nan")))
                pooled_xmin = float(fit.xmin)
        except Exception:
            # if pooling/fitting fails, continue silently
            pass

        # If possible, fit power-law to pooled deltas and draw theoretical line
        # pooled fit defaults
        pooled_alpha = float("nan")
        pooled_sigma = float("nan")
        pooled_xmin = float("nan")
        try:
            if powerlaw is not None:
                pooled = np.hstack(arrays)
                if pooled.size >= 2:
                    fit = powerlaw.Fit(pooled, xmin=None, discrete=False, verbose=False)
                    pooled_alpha = float(fit.alpha)
                    pooled_sigma = float(getattr(fit, "sigma", float("nan")))
                    pooled_xmin = float(fit.xmin)
                    # construct theoretical line and scale to median at xmin_fit
                    xmin_fit = pooled_xmin
                    x_theo = xgrid_log[xgrid_log >= xmin_fit]
                    if x_theo.size > 0:
                        theo = (x_theo / xmin_fit) ** (1.0 - pooled_alpha)
                        # scale so that theo(ref_x) == median(ref_x)
                        ref_idx = np.searchsorted(xgrid_log, xmin_fit)
                        if ref_idx >= len(median_log):
                            ref_idx = len(median_log) - 1
                        ref_x = xgrid_log[ref_idx]
                        ref_y = median_log[ref_idx]
                        scale = ref_y / ((ref_x / xmin_fit) ** (1.0 - pooled_alpha))
                        theo *= scale
                        ax.loglog(x_theo, theo, linestyle='--', color='red', linewidth=1.8, label=f'power-law fit alpha={pooled_alpha:.3f}')
        except Exception:
            # any failure in fitting/drawing should not break plotting
            pass

        pooled_fits[count] = (pooled_alpha, pooled_sigma, pooled_xmin)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-3, 15.0)
        # label axes using Greek letters: tau on x and probability on y
        ax.set_xlabel('τ [s]')
        ax.set_ylabel('P(T > τ)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(plot_dir / f'ccdf_aggregated_N{count}_loglog.png', dpi=200)
        plt.close(fig)

    return pooled_fits


if __name__ == "__main__":
    main()
