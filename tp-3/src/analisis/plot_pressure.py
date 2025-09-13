#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pressure.py
Compila el Java, ejecuta la simulación para todas las combinaciones de N y L,
y grafica las presiones de cada corrida.

Ejemplos:
    python plot_pressure.py --Ns 12 24 --Ls 0.030 0.050 0.070 --show
    python plot_pressure.py --Ns 12 --Ls 0.050 --out presiones.png --moving-avg 5 --diff --rel-diff
    python plot_pressure.py --Ls 0.050 0.070 --Ns 8 16 32 --no-compile
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- utilidades sistema ----------------------

def sh(cmd, cwd: Path):
    """Ejecuta un comando mostrando stdout/stderr en vivo."""
    print(f"[cmd] ({cwd}) $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Comando falló con código {ret}: {' '.join(cmd)}")


def compile_java(src_dir: Path):
    """Compila el Java en src/ y arma el JAR (expandiendo *.java correctamente)."""
    sim_dir = src_dir / "simulation"
    if not sim_dir.exists():
        raise FileNotFoundError(f"No existe {sim_dir}")

    # Expandir fuentes .java explícitamente
    java_files = sorted((p for p in sim_dir.glob("*.java")), key=lambda p: p.name)
    if not java_files:
        raise FileNotFoundError(f"No hay fuentes .java en {sim_dir}")

    # Compilar con paths explícitos (sin globs)
    rel_java_files = [str(p.relative_to(src_dir)) for p in java_files]
    sh(["javac", "-d", "."] + rel_java_files, cwd=src_dir)

    # Empaquetar todas las clases bajo ./simulation/ en el jar
    # -C . simulation  -> cambia a '.' e incluye 'simulation' recursivamente
    sh(["jar", "cfe", "sim.jar", "simulation.Main", "-C", ".", "simulation"], cwd=src_dir)

    jar_path = src_dir / "sim.jar"
    if not jar_path.exists():
        raise FileNotFoundError("No se generó sim.jar")
    return jar_path


def run_java_pair(src_dir: Path, N: int, L: float):
    """Corre una simulación para el par (N,L) desde el root del proyecto."""
    project_root = src_dir.parent
    jar_path_rel = (src_dir / "sim.jar").relative_to(project_root)
    jar = project_root / jar_path_rel
    if not jar.exists():
        raise FileNotFoundError(f"No se generó sim.jar en {src_dir}")

    N_str = str(N)
    L_str = f"{L:.3f}"
    sh(["java", "-jar", str(jar_path_rel), N_str, L_str], cwd=project_root)
    
    out_dir = project_root / "out"
    events = out_dir / f"events_L={L_str}_N={N_str}.txt"
    press  = out_dir / f"pressure_L={L_str}_N={N_str}.txt"
    if not press.exists():
        raise FileNotFoundError(f"No se encontró {press} tras ejecutar la simulación.")
    return events, press


# ---------------------- utilidades datos/grafico ----------------------

def read_pressure_file(path: Path):
    """Lee archivo de presión: filas = t  P_left  P_right (ignora líneas con #)."""
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
        raise ValueError(f"Archivo vacío o sin datos válidos: {path}")
    return np.array(t), np.array(p1), np.array(p2)


def moving_average(y: np.ndarray, w: int | None):
    if w is None or w <= 1:
        return y
    w = int(w)
    if w > len(y):
        w = len(y)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(y, kernel, mode="same")


def infer_LN_from_name(name: str):
    """Extrae L y N de 'pressure_L=0.050_N=12.txt'."""
    mL = re.search(r"L=([0-9]*\.?[0-9]+)", name)
    mN = re.search(r"N=([0-9]+)", name)
    L = mL.group(1) if mL else None
    N = mN.group(1) if mN else None
    return L, N


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Compila, corre y grafica presiones para múltiples N y L.")
    ap.add_argument("--Ns", type=int, nargs="+", required=True, help="Lista de Ns (cantidad de partículas).")
    ap.add_argument("--Ls", type=float, nargs="+", required=True, help="Lista de Ls (apertura del pasillo, en m).")
    ap.add_argument("--use-file", type=str, help="Archivo de presión existente para graficar (no corre simulación).")
    ap.add_argument("--no-compile", action="store_true", help="No recompilar Java (usa sim.jar existente).")
    ap.add_argument("--out", default="out/pressure_plot.png", help="Imagen de salida del gráfico.")
    ap.add_argument("--show", action="store_true", help="Mostrar gráfico en pantalla.")
    ap.add_argument("--moving-avg", type=int, default=1, help="Ventana de media móvil (muestras).")
    ap.add_argument("--diff", action="store_true", help="Graficar P_left - P_right.")
    ap.add_argument("--rel-diff", action="store_true", help="Graficar |P_left - P_right| / max(P_left, P_right).")
    
    args = ap.parse_args()

    # Este script está en src/analisis/
    script_path = Path(__file__).resolve()
    src_dir = script_path.parents[1]   # .../src
    java_dir = src_dir / "simulation"
    if not java_dir.exists():
        print(f"[error] No encuentro {java_dir}", file=sys.stderr)
        sys.exit(1)

    # Compilar si corresponde
    jar_path = src_dir / "sim.jar"
    if not args.no_compile or not jar_path.exists():
        try:
            jar_path = compile_java(src_dir)
        except Exception as e:
            print(f"[error] Falló la compilación: {e}", file=sys.stderr)
            sys.exit(1)

    # Ejecutar todas las combinaciones
    pressure_files = [] 
    if args.use_file:
        # Buscar el archivo en la carpeta 'out' relativa al root del proyecto
        project_root = src_dir.parent
        pressure_path = project_root  / args.use_file
        if not pressure_path.exists():
            print(f"[error] No se encontró {pressure_path}", file=sys.stderr)
            sys.exit(1)
        pressure_files = [pressure_path]
    else:
        if not args.Ns or not args.Ls:
            print("[error] Debes especificar --Ns y --Ls si no usas --use-file.", file=sys.stderr)
            sys.exit(1)
        for N in args.Ns:
            for L in args.Ls:
                try:
                    events, press = run_java_pair(src_dir, N=N, L=L)
                    pressure_files.append(press)
                except Exception as e:
                    print(f"[warn] Falló N={N}, L={L:.3f}: {e}", file=sys.stderr)

    if not pressure_files:
        print("[error] No hay archivos de presión para graficar.", file=sys.stderr)
        sys.exit(1)

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False

    for p in sorted(pressure_files):
        try:
            t, pL, pR = read_pressure_file(p)
        except Exception as e:
            print(f"[warn] {p.name}: {e}", file=sys.stderr)
            continue

        pL_s = moving_average(pL, args.moving_avg)
        pR_s = moving_average(pR, args.moving_avg)

        Ltag, Ntag = infer_LN_from_name(p.name)
        labelL = f"P_left (L={Ltag}, N={Ntag})" if (Ltag and Ntag) else f"P_left [{p.name}]"
        labelR = f"P_right (L={Ltag}, N={Ntag})" if (Ltag and Ntag) else f"P_right [{p.name}]"

        ax.plot(t, pL_s, label=labelL)
        ax.plot(t, pR_s, label=labelR)

        if args.diff:
            ax.plot(
                t, moving_average(pL - pR, args.moving_avg),
                linestyle="--",
                label=(f"P_left-P_right (L={Ltag}, N={Ntag})" if (Ltag and Ntag) else f"P_left-P_right [{p.name}]")
            )

        if args.rel_diff:
            denom = np.maximum(np.maximum(pL, pR), 1e-12)
            rel = np.abs(pL - pR) / denom
            ax.plot(
                t, moving_average(rel, args.moving_avg),
                linestyle=":",
                label=(f"rel diff (L={Ltag}, N={Ntag})" if (Ltag and Ntag) else f"rel diff [{p.name}]")
            )

        any_plotted = True

    if not any_plotted:
        print("[error] No se pudo graficar: no hay datos válidos.", file=sys.stderr)
        sys.exit(1)

    ax.set_xlabel("t (s)")
    ax.set_ylabel("Presión (N/m)")
    ax.set_title("Presión por recinto vs tiempo (múltiples N, L)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    print(f"[ok] Gráfico guardado en: {out_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
