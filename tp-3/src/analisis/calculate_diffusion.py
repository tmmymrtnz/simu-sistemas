#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_diffusion.py

Calcula el coeficiente de difusión a partir de un archivo de eventos.
1. Lee las posiciones de las partículas de los snapshots.
2. Calcula el Desplazamiento Cuadrático Medio (MSD) en función del tiempo.
3. Realiza un ajuste lineal MSD(t) = 4Dt en el régimen estacionario.
4. Grafica MSD vs t y reporta el valor de D.

Uso:
  python calculate_diffusion.py --file out/events_L=0.090_N=300.txt --t-start 30.0 --show
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Re-usar parser de snapshots de animate_sim.py
from dataclasses import dataclass

@dataclass
class Snapshot:
    t: float
    # id -> (x, y, r)
    state: Dict[int, Tuple[float, float, float]]

VALS_RE = re.compile(
    r"id=(?P<id>\d+)\s+"
    r"x=(?P<x>-?\d+(?:\.\d+)?)\s+"
    r"y=(?P<y>-?\d+(?:\.\d+)?)\s+"
    r"vx=(?P<vx>-?\d+(?:\.\d+)?)\s+"
    r"vy=(?P<vy>-?\d+(?:\.\d+)?)\s+"
    r"r=(?P<r>-?\d+(?:\.\d+)?)"
)

def parse_events_file(path: Path) -> List[Snapshot]:
    snaps: List[Snapshot] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("SNAPSHOT"):
                continue
            m_t = re.search(r"t=([0-9]+\.[0-9]+)", line)
            if not m_t:
                continue
            t = float(m_t.group(1))
            parts = line.strip().split(" | ")
            state: Dict[int, Tuple[float, float, float]] = {}
            for p in parts[1:]:
                m = VALS_RE.search(p)
                if not m:
                    continue
                i = int(m.group("id"))
                x = float(m.group("x"))
                y = float(m.group("y"))
                r = float(m.group("r"))
                state[i] = (x, y, r)
            if state:
                snaps.append(Snapshot(t=t, state=state))
    if not snaps:
        raise ValueError(f"No se encontraron SNAPSHOTs en {path}")
    return snaps

# Re-usar ajuste lineal de fit_pressure_vs_area.py
def fit_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Ajuste y = k*x. Devuelve (k, se_k, R2_origen)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2: raise ValueError("Se requieren >= 2 puntos para el ajuste")
    sx2 = float(np.sum(x*x))
    if sx2 <= 1e-12: raise ValueError("sum(x^2) es cero, no se puede ajustar.")
    sxy = float(np.sum(x*y))
    k = sxy/sx2
    resid = y - k*x
    sse = float(np.sum(resid*resid))
    dof = max(1, len(x)-1)
    sigma2 = sse/dof
    se_k = float(np.sqrt(sigma2/sx2))
    sst0 = float(np.sum(y*y))
    R2 = 1.0 - (sse/sst0 if sst0 > 0 else np.nan)
    return k, se_k, R2

def calculate_msd_absolute(snaps: List[Snapshot]) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula el MSD para toda la simulación desde el inicio."""
    if not snaps:
        return np.array([]), np.array([])
    
    initial_snap = snaps[0]
    t0 = initial_snap.t
    initial_pos = {pid: (s[0], s[1]) for pid, s in initial_snap.state.items()}
    particle_ids = sorted(initial_pos.keys())

    absolute_times = []
    msd_values = []

    for current_snap in snaps:
        sq_displacements = []
        for pid in particle_ids:
            if pid in current_snap.state and pid in initial_pos:
                x0, y0 = initial_pos[pid]
                x, y, _ = current_snap.state[pid]
                sq_disp = (x - x0)**2 + (y - y0)**2
                sq_displacements.append(sq_disp)
        
        if sq_displacements:
            msd = np.mean(sq_displacements)
            absolute_times.append(current_snap.t)
            msd_values.append(msd)

    return np.array(absolute_times), np.array(msd_values)

def interactive_plot_subset(snaps: List[Snapshot], main_title: str):
    """
    Plots the full data and allows the user to select a subset for a new linear fit.
    """
    # 1. First, calculate and plot the full MSD from t=0.0
    full_absolute_times, full_msd_values = calculate_msd_absolute(snaps)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(full_absolute_times, full_msd_values, 'o', markersize=4)
    
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Desplazamiento Cuadrático Medio (m²)")
    ax.set_title(f"{main_title}\n(Haz clic en 2 puntos para seleccionar un rango)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show(block=False)  # Show the plot but don't block the script
    
    # 2. Get user clicks
    print("Por favor, haz clic en el gráfico para seleccionar los dos extremos del nuevo rango de tiempo.")
    try:
        points = plt.ginput(2, timeout=60, show_clicks=True)
        if len(points) < 2:
            print("Selección cancelada o no se hicieron suficientes clics.")
            plt.close(fig) # Close the first figure
            return
    except RuntimeError:
        print("El gráfico se cerró antes de la selección. No se generará un nuevo gráfico.")
        plt.close(fig)
        return

    plt.close(fig) # Close the original plot

    # 3. Extract x-axis range from clicks and filter the pre-calculated data
    t_min = min(points[0][0], points[1][0])
    t_max = max(points[0][0], points[1][0])
    print(f"Rango de tiempo seleccionado: [{t_min:.2f}s, {t_max:.2f}s]")
    
    indices = np.where((full_absolute_times >= t_min) & (full_absolute_times <= t_max))
    
    selected_absolute_times = full_absolute_times[indices]
    selected_msd_values = full_msd_values[indices]
    selected_relative_times = selected_absolute_times - selected_absolute_times[0]
        
    if len(selected_relative_times) < 2:
        print("El rango seleccionado no contiene suficientes datos para realizar un ajuste.")
        return

    # 4. Perform a new fit on the subset data
    print("Realizando nuevo ajuste lineal para el subconjunto de datos...")
    try:
        slope_subset, se_slope_subset, r2_subset = fit_through_origin(selected_relative_times, selected_msd_values - selected_msd_values[0])
        D_subset = slope_subset / 4.0
        se_D_subset = se_slope_subset / 4.0
    except ValueError as e:
        print(f"[error] No se pudo realizar el ajuste para el subconjunto: {e}", file=sys.stderr)
        return

    print("\n--- Resultados del Subconjunto ---")
    print(f"  Pendiente (k): {slope_subset:.6g} m²/s")
    print(f"  Error estándar (SE) de k: {se_slope_subset:.6g}")
    print(f"  R² (respecto al origen): {r2_subset:.5f}")
    print(f"Coeficiente de Difusión (D = k/4): {D_subset:.6g} m²/s (± {se_D_subset:.2g})")
    
    # 5. Plot the subset
    fig_subset, ax_subset = plt.subplots(figsize=(8, 6))
    ax_subset.plot(selected_absolute_times, selected_msd_values, 'o', markersize=4, label="Datos de simulación")
    fit_line_subset = slope_subset * (selected_absolute_times - selected_absolute_times[0]) + selected_msd_values[0]
    ax_subset.plot(selected_absolute_times, fit_line_subset, 'r-', lw=2, label=f"Ajuste lineal")
    
    ax_subset.set_xlabel("Tiempo (s)")
    ax_subset.set_ylabel("Desplazamiento Cuadrático Medio (m²)")
    ax_subset.set_title(f"Ajuste para el rango [{t_min:.2f}s, {t_max:.2f}s]")
    ax_subset.grid(True, alpha=0.4)
    ax_subset.legend()
    fig_subset.tight_layout()
    plt.show() # This plot will block, keeping the window open until you close it.


def main():
    ap = argparse.ArgumentParser(description="Calcula y grafica el coeficiente de difusión a partir de un archivo de eventos.")
    ap.add_argument("--file", type=str, required=True, help="Ruta al archivo events_*.txt.")
    ap.add_argument("--t-start", type=float, default=0.0, help="Tiempo de inicio (s) para el cálculo de MSD y el ajuste lineal (para usar el régimen estacionario).")
    ap.add_argument("--out", type=str, default="diffusion_plot.png", help="Archivo de salida para el gráfico.")
    ap.add_argument("--show", action="store_true", help="Mostrar el gráfico al finalizar.")
    args = ap.parse_args()

    events_path = Path(args.file)
    if not events_path.exists():
        print(f"[error] No se encuentra el archivo: {events_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parseando {events_path.name}...")
    snaps = parse_events_file(events_path)
    
    if args.show:
        # Run the interactive plotting mode
        interactive_plot_subset(snaps, events_path.name)
    else:
        # Run the non-interactive plotting and saving mode
        print(f"Calculando MSD a partir de t={args.t_start:.2f}s...")
        full_absolute_times, full_msd_values = calculate_msd_absolute(snaps)
        
        # Filter data based on --t-start parameter
        indices = np.where(full_absolute_times >= args.t_start)
        absolute_times = full_absolute_times[indices]
        msd_values = full_msd_values[indices]
        relative_times = absolute_times - absolute_times[0]

        if len(relative_times) < 2:
            print("[error] No hay suficientes datos para realizar el ajuste.", file=sys.stderr)
            sys.exit(1)

        print("Realizando ajuste lineal MSD(t) = 4Dt...")
        # Fit through origin on the shifted data
        slope, se_slope, r2 = fit_through_origin(relative_times, msd_values - msd_values[0])
        
        D = slope / 4.0
        se_D = se_slope / 4.0

        print("\n--- Resultados ---")
        print(f"Ajuste lineal: MSD(t) = k * t")
        print(f"  Pendiente (k): {slope:.6g} m²/s")
        print(f"  Error estándar (SE) de k: {se_slope:.6g}")
        print(f"  R² (respecto al origen): {r2:.5f}")
        print(f"Coeficiente de Difusión (D = k/4): {D:.6g} m²/s (± {se_D:.2g})")

        # Gráfico
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(absolute_times, msd_values, 'o', markersize=4, label="MSD calculado (simulación)")
        
        # Plot the fit line
        fit_line = slope * relative_times + msd_values[0]
        ax.plot(absolute_times, fit_line, 'r-', lw=2, label=f"Ajuste lineal (D = {D:.4g} m²/s)")
        
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Desplazamiento Cuadrático Medio (m²)")
        ax.grid(True, alpha=0.4)
        ax.legend()
        fig.tight_layout()

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"\n[ok] Gráfico guardado en: {out_path.resolve()}")
        plt.close(fig)

if __name__ == "__main__":
    main()
