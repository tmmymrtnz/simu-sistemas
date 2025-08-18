#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib

OUT_DIR = pathlib.Path("out/bench_points_summary")

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def fit_slope_loglog(x_vals, y_vals):
    """
    Ajusta log10(y) = m * log10(x) + b con numpy.polyfit.
    Filtra no positivos y NaN. Devuelve (m, b) o (nan, nan).
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
    x = np.log10(x_vals[mask])
    y = np.log10(y_vals[mask])
    if x.size < 2:
        return np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    return m, b

def main():
    ap = argparse.ArgumentParser(description="Tiempos CIM vs BRUTE en log–log, con pendientes.")
    ap.add_argument("--fit-min-N", type=int, default=None, help="Usar solo N >= este valor para el ajuste.")
    ap.add_argument("--fit-max-N", type=int, default=None, help="Usar solo N <= este valor para el ajuste.")
    args = ap.parse_args()

    mean_path = OUT_DIR / "summary_mean.csv"
    std_path  = OUT_DIR / "summary_std.csv"
    if not mean_path.exists() or not std_path.exists():
        raise SystemExit("No se encuentran summary_mean.csv / summary_std.csv. Corré primero analisis/analyze.py.")

    mean_df = pd.read_csv(mean_path)
    std_df  = pd.read_csv(std_path)

    # Chequeos mínimos
    need_mean = {"N","M","CIM_ms","BRUTE_ms"}
    if not need_mean.issubset(mean_df.columns):
        raise SystemExit(f"CSV mean sin columnas {need_mean}. Tiene: {list(mean_df.columns)}")
    if not need_mean.issubset(std_df.columns):
        raise SystemExit(f"CSV std sin columnas {need_mean}. Tiene: {list(std_df.columns)}")

    # Filtros de N para ajuste
    def apply_fit_range(df):
        if args.fit_min_N is not None:
            df = df[df["N"] >= args.fit_min_N]
        if args.fit_max_N is not None:
            df = df[df["N"] <= args.fit_max_N]
        return df

    mean_df_fit = apply_fit_range(mean_df)
    std_df_fit  = apply_fit_range(std_df)

    # Conjuntos ordenados
    Ms = sorted(mean_df["M"].unique())

    # ===== BRUTE (M=1) — agregamos por N (por si viene repetido por M) =====
    brute_mean_by_N = mean_df.groupby("N", as_index=False)["BRUTE_ms"].mean().sort_values("N")
    brute_std_by_N  = std_df.groupby("N", as_index=False)["BRUTE_ms"].mean().sort_values("N")

    brute_mean_by_N_fit = apply_fit_range(brute_mean_by_N)
    brute_std_by_N_fit  = apply_fit_range(brute_std_by_N)

    brute_Ns   = brute_mean_by_N["N"].values
    brute_mean = brute_mean_by_N["BRUTE_ms"].values
    brute_std  = brute_std_by_N["BRUTE_ms"].values

    brute_Ns_fit   = brute_mean_by_N_fit["N"].values
    brute_mean_fit = brute_mean_by_N_fit["BRUTE_ms"].values
    brute_slope_N, _ = fit_slope_loglog(brute_Ns_fit, brute_mean_fit)

    # ===== CIM por cada M =====
    colors = {'CIM': 'tab:blue', 'BRUTE': 'tab:red'}
    styles = ['-', '--', ':', '-.', (0,(1,1))]
    linestyles = {M: styles[i % len(styles)] for i, M in enumerate(Ms)}

    # ---------- FIGURA 1: tiempo vs N (log–log) ----------
    plt.figure(figsize=(10, 6))

    cim_slopes = []
    for M in Ms:
        mrow = mean_df[mean_df["M"] == M].sort_values("N")
        srow = std_df[std_df["M"] == M].sort_values("N")

        # Para el ajuste, aplicar filtro de N
        mrow_fit = apply_fit_range(mrow)

        xN = mrow["N"].values
        y_mean = mrow["CIM_ms"].values
        y_std  = srow["CIM_ms"].values

        # Pendiente vs N (log–log)
        slope, _ = fit_slope_loglog(mrow_fit["N"].values, mrow_fit["CIM_ms"].values)
        cim_slopes.append((M, slope))

        plt.errorbar(
            xN, y_mean, yerr=y_std,
            label=f'CIM (M={M}, m≈{slope:.2f})',
            color=colors['CIM'], linestyle=linestyles[M], marker='o', capsize=3, alpha=0.9
        )

    plt.errorbar(
        brute_Ns, brute_mean, yerr=brute_std,
        label=f'BRUTE (M=1, m≈{brute_slope_N:.2f})',
        color=colors['BRUTE'], linestyle='-', marker='s', capsize=3, alpha=0.9
    )

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('N (número de partículas)')
    plt.ylabel('Tiempo de ejecución (ms)')
    ttl_extra = []
    if args.fit_min_N: ttl_extra.append(f"N≥{args.fit_min_N}")
    if args.fit_max_N: ttl_extra.append(f"N≤{args.fit_max_N}")
    plt.title('Tiempos CIM vs BRUTE (log–log) ' + (f"[fit: {', '.join(ttl_extra)}]" if ttl_extra else ""))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    ensure_dir(OUT_DIR)
    out_png1 = OUT_DIR / 'times_loglog_cim_brute_with_slopes.png'
    plt.savefig(out_png1, dpi=150)
    plt.close()

    # ---------- FIGURA 2: tiempo vs comparaciones (log–log) ----------
    # Si están disponibles las columnas de comparaciones, hacemos el segundo gráfico
    have_comps = {"CIM_comps","BRUTE_comps"}.issubset(mean_df.columns)
    if have_comps:
        # Agregar BRUTE comps por N
        brute_comps_by_N = mean_df.groupby("N", as_index=False)["BRUTE_comps"].mean().sort_values("N")
        brute_comps_by_N_fit = apply_fit_range(brute_comps_by_N)

        # Pendiente esperada ~1 (tiempo ∝ comparaciones)
        brute_slope_vsC, _ = fit_slope_loglog(brute_comps_by_N_fit["BRUTE_comps"].values,
                                              brute_mean_by_N_fit["BRUTE_ms"].values)

        plt.figure(figsize=(10,6))
        # CIM por M
        cim_vsC_slopes = []
        for M in Ms:
            mrow = mean_df[mean_df["M"] == M].sort_values("N")
            srow = std_df[std_df["M"] == M].sort_values("N")
            mrow_fit = apply_fit_range(mrow)

            xC = mrow["CIM_comps"].values
            y = mrow["CIM_ms"].values
            yerr = srow["CIM_ms"].values

            slopeC, _ = fit_slope_loglog(mrow_fit["CIM_comps"].values, mrow_fit["CIM_ms"].values)
            cim_vsC_slopes.append((M, slopeC))

            plt.errorbar(
                xC, y, yerr=yerr,
                label=f'CIM (M={M}, m≈{slopeC:.2f})',
                color=colors['CIM'], linestyle=linestyles[M], marker='o', capsize=3, alpha=0.9
            )

        # BRUTE (una curva)
        plt.errorbar(
            brute_comps_by_N["BRUTE_comps"].values,
            brute_mean_by_N["BRUTE_ms"].values,
            yerr=brute_std_by_N["BRUTE_ms"].values,
            label=f'BRUTE (M=1, m≈{brute_slope_vsC:.2f})',
            color=colors['BRUTE'], linestyle='-', marker='s', capsize=3, alpha=0.9
        )

        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Comparaciones')
        plt.ylabel('Tiempo de ejecución (ms)')
        plt.title('Tiempo vs Comparaciones (log–log)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_png2 = OUT_DIR / 'times_vs_comparisons_loglog_with_slopes.png'
        plt.savefig(out_png2, dpi=150)
        plt.close()
    else:
        out_png2 = None

    # ---------- Reporte en consola ----------
    print("Pendientes (log t vs log N):")
    print(f"BRUTE (M=1): m = {brute_slope_N:.2f}")
    for M, slope in cim_slopes:
        print(f"CIM (M={M}): m = {slope:.2f}")

    if have_comps:
        print("\nPendientes (log t vs log comparaciones):")
        print(f"BRUTE (M=1): m = {brute_slope_vsC:.2f}")
        for M, slopeC in cim_vsC_slopes:
            print(f"CIM (M={M}): m = {slopeC:.2f}")

        # ns por comparación (en el mayor N) para BRUTE (diagnóstico)
        lastN = brute_Ns[-1]
        last_ms = brute_mean[-1]
        last_comps = brute_comps_by_N["BRUTE_comps"].values[-1]
        ns_per_comp = (last_ms * 1e6) / last_comps  # ms -> ns
        print(f"\nBRUTE @ N={int(lastN)}: {ns_per_comp:.2f} ns por comparación")

    print(f"\nGráfico guardado en: {out_png1}")
    if out_png2:
        print(f"Gráfico guardado en: {out_png2}")

if __name__ == "__main__":
    main()
message.txt
9 KB