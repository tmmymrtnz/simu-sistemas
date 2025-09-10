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

def calculate_msd(snaps: List[Snapshot], t_start: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula el MSD a partir de t_start."""
    
    # Encontrar el snapshot inicial (o el más cercano a t_start)
    times = np.array([s.t for s in snaps])
    start_idx = np.searchsorted(times, t_start, side='left')
    if start_idx >= len(snaps):
        raise ValueError(f"t_start={t_start}s está más allá del final de la simulación ({times[-1]:.2f}s).")

    initial_snap = snaps[start_idx]
    t0 = initial_snap.t
    initial_pos = {pid: (s[0], s[1]) for pid, s in initial_snap.state.items()}
    particle_ids = sorted(initial_pos.keys())
    N = len(particle_ids)

    msd_times = []
    msd_values = []

    for i in range(start_idx, len(snaps)):
        current_snap = snaps[i]
        if current_snap.t < t0: continue

        sq_displacements = []
        for pid in particle_ids:
            if pid in current_snap.state and pid in initial_pos:
                x0, y0 = initial_pos[pid]
                x, y, _ = current_snap.state[pid]
                sq_disp = (x - x0)**2 + (y - y0)**2
                sq_displacements.append(sq_disp)
        
        if sq_displacements:
            msd = np.mean(sq_displacements)
            msd_times.append(current_snap.t - t0)
            msd_values.append(msd)

    return np.array(msd_times), np.array(msd_values)


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
    
    print(f"Calculando MSD a partir de t={args.t_start:.2f}s...")
    try:
        time_axis, msd_values = calculate_msd(snaps, args.t_start)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if len(time_axis) < 2:
        print("[error] No hay suficientes datos para realizar el ajuste.", file=sys.stderr)
        sys.exit(1)

    print("Realizando ajuste lineal MSD(t) = 4Dt...")
    slope, se_slope, r2 = fit_through_origin(time_axis, msd_values)
    
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
    ax.plot(time_axis, msd_values, 'o', markersize=4, label="Datos de simulación (MSD)")
    
    fit_line = slope * time_axis
    ax.plot(time_axis, fit_line, 'r-', lw=2, label=f"Ajuste lineal (D = {D:.4g} m²/s)")
    
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Desplazamiento Cuadrático Medio (m²)")
    ax.set_title(f"Análisis de Difusión (N={len(snaps[0].state)}, L={re.search(r'L=([0-9.]+)', events_path.name).group(1) if re.search(r'L=([0-9.]+)', events_path.name) else '?'})")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n[ok] Gráfico guardado en: {out_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
