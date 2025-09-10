#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_fit_pressure.py

Corre la simulación Java múltiples veces para un N fijo y una lista de L,
promedia la presión (en estado estacionario o, opcionalmente, en los últimos N puntos)
y ajusta el modelo teórico P = k * A^{-1} (sin intercepto).
Grafica con barras de error y compara P·A vs L con la línea teórica horizontal.

Nuevo:
  --last-n N   -> usa SOLO los últimos N puntos de cada corrida (saltea heurística de estacionario)

Actualizado:
  - Imprime en la terminal todos los puntos y sus errores de ambos gráficos.
  - Para el primer gráfico (P vs A⁻¹) también reporta, para cada L,
    cuánto se aparta P·A del k ajustado (abs y relativo).
"""

from __future__ import annotations
import argparse
import time
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# --- Geometría del sistema (de tu Java) ---
L_FIXED_DEFAULT = 0.09  # metros

# ----------------- Utils sistema / Java -----------------

def sh(cmd: List[str], cwd: Path):
    print(f"[cmd] ({cwd}) $ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=str(cwd))
    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Comando falló ({ret}): {' '.join(cmd)}")

def compile_java(src_dir: Path):
    sim_dir = src_dir / "simulation"
    if not sim_dir.exists():
        raise FileNotFoundError(f"No existe {sim_dir}")
    java_files = sorted(sim_dir.glob("*.java"), key=lambda p: p.name)
    if not java_files:
        raise FileNotFoundError(f"No hay .java en {sim_dir}")
    rels = [str(p.relative_to(src_dir)) for p in java_files]
    sh(["javac", "-d", "."] + rels, cwd=src_dir)
    sh(["jar", "cfe", "sim.jar", "simulation.Main", "-C", ".", "simulation"], cwd=src_dir)
    if not (src_dir / "sim.jar").exists():
        raise FileNotFoundError("No se generó sim.jar")

def run_java_once(src_dir: Path, N: int, L: float) -> Tuple[Path, Path]:
    project_root = src_dir.parent
    jar_path_rel = (src_dir / "sim.jar").relative_to(project_root)
    jar = project_root / jar_path_rel
    if not jar.exists():
        raise FileNotFoundError(f"No existe {jar}.")
    N_str = str(N); L_str = f"{L:.3f}"
    sh(["java", "-jar", str(jar_path_rel), N_str, L_str], cwd=project_root)
    out_dir = project_root / "out"
    events = out_dir / f"events_L={L_str}_N={N_str}.txt"
    press  = out_dir / f"pressure_L={L_str}_N={N_str}.txt"
    if not press.exists():
        raise FileNotFoundError(f"No se encontró {press}")
    return events, press

# ----------------- Lectura presión & helpers -----------------

def read_pressure_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee archivo: t  P_left  P_right -> (t, P_left, P_right)."""
    t, p1, p2 = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                ti = float(parts[0]); p1i = float(parts[1]); p2i = float(parts[2])
            except ValueError:
                continue
            t.append(ti); p1.append(p1i); p2.append(p2i)
    if not t:
        raise ValueError(f"Archivo vacío: {path}")
    t = np.asarray(t, float)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    idx = np.argsort(t)
    return t[idx], p1[idx], p2[idx]

def find_steady_start(t: np.ndarray,
                      pL: np.ndarray,
                      pR: np.ndarray,
                      win_sec: float = 2.0,
                      tol_rel: float = 0.03,
                      persist_sec: float = 2.0) -> int:
    """
    Índice i0 del primer instante desde el cual, por 'persist_sec',
    <|pL - pR|>/< (pL+pR)/2 > (promedios móviles con ventana win_sec) < tol_rel.
    Si no encuentra, -1.
    """
    n = len(t)
    if n < 2: return -1
    absdiff = np.abs(pL - pR)
    meanp   = 0.5*(pL + pR)
    eps = 1e-12

    def roll_avg(arr: np.ndarray, i: int, half: float) -> float:
        ti = t[i]
        j0 = i
        while j0 > 0 and t[j0-1] >= ti - half:
            j0 -= 1
        return float(arr[j0:i+1].mean()) if i >= j0 else float(arr[i])

    half = 0.5*win_sec
    for i in range(n):
        r = roll_avg(absdiff, i, half) / max(eps, roll_avg(meanp, i, half))
        if r < tol_rel:
            tend = t[i] + persist_sec
            k = i; ok = True
            while k < n and t[k] <= tend:
                rk = roll_avg(absdiff, k, half) / max(eps, roll_avg(meanp, k, half))
                if rk >= tol_rel: ok=False; break
                k += 1
            if ok: return i
    return -1

def steady_average(t: np.ndarray, y: np.ndarray, i0: int) -> Tuple[float, float]:
    """Promedio temporal (trapezoidal) y SD simple a partir de i0."""
    if i0 < 0 or i0 >= len(t)-1:
        return float(np.mean(y)), float(np.std(y, ddof=1) if len(y)>1 else 0.0)
    ts = t[i0:]; ys = y[i0:]
    if len(ts) < 2:
        return float(np.mean(ys)), 0.0
    dt = np.diff(ts); ymid = 0.5*(ys[1:]+ys[:-1])
    integ = float(np.sum(ymid*dt))
    Tspan = float(ts[-1]-ts[0]) if ts[-1]>ts[0] else 1.0
    ybar = integ/Tspan
    sd = float(np.std(ys, ddof=1) if len(ys)>1 else 0.0)
    return ybar, sd

def fit_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Ajuste P = k x (sin intercepto). Devuelve (k, se_k, R2_origen)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2: raise ValueError("Se requieren ≥ 2 puntos")
    sx2 = float(np.sum(x*x))
    if sx2 <= 0: raise ValueError("sum(x^2)=0")
    sxy = float(np.sum(x*y))
    k = sxy/sx2
    resid = y - k*x
    sse = float(np.sum(resid*resid))
    dof = max(1, len(x)-1)
    sigma2 = sse/dof
    se_k = float(np.sqrt(sigma2/sx2))
    sst0 = float(np.sum(y*y))
    R2 = 1.0 - (sse/sst0 if sst0>0 else np.nan)
    return k, se_k, R2

# ----------------- Lógica principal -----------------

def main():
    ap = argparse.ArgumentParser(description="Corridas múltiples por L, promedio en steady-state o últimos N puntos, ajuste P ~ A^{-1}.")
    ap.add_argument("--N", type=int, required=True, help="Cantidad de partículas (fijo para todas las L).")
    ap.add_argument("--Ls", type=float, nargs="+", required=True, help="Lista de L (m).")
    ap.add_argument("--reps", type=int, default=5, help="Corridas por cada L (default=5).")
    ap.add_argument("--L-fixed", type=float, default=L_FIXED_DEFAULT, help="L_fixed (m).")
    ap.add_argument("--no-compile", action="store_true", help="No recompilar Java (usa sim.jar existente).")
    # Estacionario / alternativas
    ap.add_argument("--steady-tol", type=float, default=0.03)
    ap.add_argument("--steady-win", type=float, default=2.0)
    ap.add_argument("--steady-persist", type=float, default=2.0)
    ap.add_argument("--fallback-last-frac", type=float, default=0.30)
    ap.add_argument("--last-n", type=int, default=None,
                    help="Usar SOLO los últimos N puntos de cada corrida (ignora detección de estacionario).")
    # Salidas
    ap.add_argument("--csv-out", type=str, default="out/batch_steady_pressure.csv")
    ap.add_argument("--plot-out", type=str, default="out/batch_fit_P_vs_Ainv.png")
    ap.add_argument("--plot2-out", type=str, default="out/batch_PA_vs_L.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    src_dir = script_path.parents[1]  # .../src
    if not (src_dir / "simulation").exists():
        print(f"[error] No encuentro {src_dir/'simulation'}", file=sys.stderr)
        sys.exit(1)
    if (not args.no_compile) or (not (src_dir/"sim.jar").exists()):
        compile_java(src_dir)

    N = args.N
    Ls = [float(L) for L in args.Ls]
    Lf = float(args.L_fixed)

    # Limpiar viejos _rep*.txt para este N y estas L (opcional)
    for L in Ls:
        for old in src_dir.glob(f"pressure_L={L:.3f}_N={N}_rep*.txt"):
            try: old.unlink()
            except: pass

    per_rep_rows = []  # una fila por corrida individual
    for L in Ls:
        print(f"\n=== L = {L:.3f}  (reps = {args.reps}) ===")
        for r in range(1, args.reps+1):
            # Ejecutar Java
            _, pr_path = run_java_once(src_dir, N, L)
            # Copiar a nombre único
            dst = pr_path.parent / f"pressure_L={L:.3f}_N={N}_rep{r}.txt"
            shutil.copy2(pr_path, dst)
            print(f"[ok] {dst.name}")
            time.sleep(0.2)  # desincronizar seeds

            # Parseo + promedio
            try:
                t, pL, pR = read_pressure_file(dst)
            except Exception as e:
                print(f"[warn] {dst.name}: {e}", file=sys.stderr)
                continue

            if args.last_n is not None and args.last_n > 0:
                i0 = max(0, len(t) - args.last_n)
            else:
                i0 = find_steady_start(t, pL, pR,
                                       win_sec=args.steady_win,
                                       tol_rel=args.steady_tol,
                                       persist_sec=args.steady_persist)
                if i0 < 0:
                    i0 = int(max(0, len(t) - int(len(t)*args.fallback_last_frac)))

            p_mean = 0.5*(pL + pR)
            P_ss, sd_ss = steady_average(t, p_mean, i0=i0)
            A  = Lf*(Lf + L)
            Ainv = 1.0/A if A>0 else np.nan
            per_rep_rows.append({
                "L": L, "rep": r, "N": N,
                "t0_ss": float(t[i0]),
                "P_ss": P_ss, "sd": sd_ss,
                "A": A, "Ainv": Ainv, "PA": P_ss*A
            })

    if not per_rep_rows:
        print("[error] No se obtuvieron datos.", file=sys.stderr)
        sys.exit(1)

    # Agregado por L
    L_to_vals: Dict[float, List[dict]] = {}
    for row in per_rep_rows:
        L_to_vals.setdefault(row["L"], []).append(row)

    agg_rows = []
    for L in sorted(L_to_vals.keys()):
        rows = L_to_vals[L]
        P_vals = np.array([r["P_ss"] for r in rows], float)
        P_mean = float(np.mean(P_vals))
        P_sd   = float(np.std(P_vals, ddof=1) if len(P_vals)>1 else 0.0)
        P_sem  = float(P_sd/np.sqrt(len(P_vals))) if len(P_vals)>1 else 0.0
        A  = rows[0]["A"]; Ainv = rows[0]["Ainv"]
        agg_rows.append({
            "L": L, "N": N, "reps": len(rows),
            "P_mean": P_mean, "P_sd": P_sd, "P_sem": P_sem,
            "A": A, "Ainv": Ainv, "PA_mean": P_mean*A
        })

    # Ajuste con promedios por L
    agg_rows_sorted = sorted(agg_rows, key=lambda r: r["L"])
    x = np.array([r["Ainv"] for r in agg_rows_sorted], float)
    y = np.array([r["P_mean"] for r in agg_rows_sorted], float)

    # (re)def local por si movés el archivo
    def fit_through_origin_local(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        sx2 = float(np.sum(x*x)); sxy = float(np.sum(x*y))
        k = sxy/sx2
        resid = y - k*x
        sse = float(np.sum(resid*resid))
        dof = max(1, len(x)-1)
        sigma2 = sse/dof
        se_k = float(np.sqrt(sigma2/sx2))
        sst0 = float(np.sum(y*y))
        R2 = 1.0 - (sse/sst0 if sst0>0 else np.nan)
        return k, se_k, R2

    k, se_k, R2 = fit_through_origin_local(x, y)

    print("\n=== Ajuste teórico sobre promedios por L: P = k * A^{-1} ===")
    print(f"k = {k:.6g}  (± {se_k:.6g})")
    print(f"R^2 (origen) = {R2:.5f}")
    print("Puntos usados:", len(x))

    # --------- NUEVO: imprimir puntos y errores de ambos gráficos ---------
    print("\n--- Puntos para GRÁFICO 1: P vs A^{-1} (con SEM) ---")
    for r in agg_rows_sorted:
        print(f"L={r['L']:.3f} | Ainv={r['Ainv']:.6g} | P={r['P_mean']:.6g} ± {r['P_sem']:.6g}")

    print("\n--- Comparación P·A vs k (usando el k del ajuste de P vs A^{-1}) ---")
    diffs_abs = []
    diffs_rel = []
    for r in agg_rows_sorted:
        PA = r["PA_mean"]
        # error de PA (SEM propagado): A * P_sem
        PA_sem = r["A"] * r["P_sem"]
        delta = PA - k
        rel = (delta / k * 100.0) if k != 0 else float("nan")
        diffs_abs.append(abs(delta))
        if np.isfinite(rel):
            diffs_rel.append(abs(rel))
        print(f"L={r['L']:.3f} | A={r['A']:.6g} | P·A={PA:.6g} ± {PA_sem:.6g} | Δ={delta:.6g} ({rel:.3g}%)")

    if diffs_abs:
        mad = float(np.mean(diffs_abs))
        maxd = float(np.max(diffs_abs))
        if diffs_rel:
            mrd = float(np.mean(diffs_rel))
            maxrd = float(np.max(diffs_rel))
            print(f"\nResumen PA vs k:  ⟨|Δ|⟩={mad:.6g}  (⟨|Δ|/k⟩={mrd:.3g}%),  max|Δ|={maxd:.6g}  (max%={maxrd:.3g}%)")
        else:
            print(f"\nResumen PA vs k:  ⟨|Δ|⟩={mad:.6g}  ,  max|Δ|={maxd:.6g}")

    print("\n--- Puntos para GRÁFICO 2: P·A vs L (con SEM) ---")
    for r in agg_rows_sorted:
        PA = r["PA_mean"]
        PA_sem = r["A"] * r["P_sem"]
        print(f"L={r['L']:.3f} | P·A={PA:.6g} ± {PA_sem:.6g}")

    # Guardar CSV (individual + agregado)
    import csv
    csv_path = Path(args.csv_out)
    with csv_path.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["type","L","N","rep_or_reps","t0_ss","P_ss_or_mean","sd","sem","A","Ainv","PA_or_PAmean"])
        for r in per_rep_rows:
            w.writerow(["rep", f"{r['L']:.6f}", r["N"], r["rep"],
                        f"{r['t0_ss']:.6f}", f"{r['P_ss']:.9g}", f"{r['sd']:.9g}", "",
                        f"{r['A']:.9g}", f"{r['Ainv']:.9g}", f"{r['PA']:.9g}"])
        for r in agg_rows_sorted:
            w.writerow(["agg", f"{r['L']:.6f}", r["N"], r["reps"],
                        "", f"{r['P_mean']:.9g}", f"{r['P_sd']:.9g}", f"{r['P_sem']:.9g}",
                        f"{r['A']:.9g}", f"{r['Ainv']:.9g}", f"{r['PA_mean']:.9g}"])
    print(f"\n[ok] CSV: {csv_path.resolve()}")

    # --- Figura 1: P vs 1/A con barras de error (SEM) y recta ajustada ---
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    yerr = np.array([r["P_sem"] for r in agg_rows_sorted], float)
    ax1.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Promedios por L (± SEM)")
    xs = np.linspace(min(x), max(x), 200)
    ys = k * xs
    ax1.plot(xs, ys, label=f"Ajuste P = k·A⁻¹\nk={k:.4g} ± {se_k:.2g}\nR²={R2:.4f}")
    ax1.set_xlabel("A⁻¹ (1/m²)")
    ax1.set_ylabel("P (N/m)")
    ax1.set_title(f"P vs A⁻¹ (N={N}, reps={len(per_rep_rows)//len(agg_rows_sorted)})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    fig1.tight_layout()
    out1 = Path(args.plot_out)
    fig1.savefig(out1, dpi=140)
    print(f"[ok] Figura: {out1.resolve()}")

    # --- Figura 2: P·A vs L con barras de error + línea teórica ---
    Larr  = np.array([r["L"] for r in agg_rows_sorted], float)
    PA    = np.array([r["PA_mean"] for r in agg_rows_sorted], float)
    PA_sem = np.array([r["A"]*r["P_sem"] for r in agg_rows_sorted], float)
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    ax2.errorbar(Larr, PA, yerr=PA_sem, fmt="o", capsize=4, label="P·A promedio (± SEM)")
    ax2.axhline(k, color="C1", lw=1.8, label="Teórico: P·A = k")
    ax2.axhspan(k - se_k, k + se_k, color="C1", alpha=0.15, label="± se(k)")
    ax2.set_xlabel("L (m)")
    ax2.set_ylabel("P·A (unid. consistentes)")
    ax2.set_title(f"P·A vs L (N={N}, reps={len(per_rep_rows)//len(agg_rows_sorted)})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    out2 = Path(args.plot2_out)
    fig2.savefig(out2, dpi=140)
    print(f"[ok] Figura: {out2.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig2)


if __name__ == "__main__":
    main()
