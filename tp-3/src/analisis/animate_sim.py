#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
animate_sim.py  (tiempo real)
Compila y ejecuta la simulación Java para un solo par (N, L) y crea una animación
a partir del archivo de eventos 'events_L=..._N=... .txt'.

Ahora el video se genera en TIEMPO REAL:
- Se re-muestrean los snapshots en una grilla de tiempo uniforme con dt = 1/fps
- Duración del video ≈ (t_final - t_inicial)

Uso típico:
  python animate_sim.py --N 12 --L 0.050 --show
  python animate_sim.py --N 300 --L 0.050 -o anim.mp4 --fps 30
  python animate_sim.py --use-file ../events_L=0.050_N=300.txt --show
  # Si querés acelerar igual, podés mantener --skip (aplicado tras el re-muestreo)
"""

from __future__ import annotations
import argparse
import bisect
import re
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

# --------- parámetros geométricos del mapa (de tu Java) ----------
L_FIXED = 0.09  # L_fixed en metros (cuadrado izquierdo de 0..L_FIXED, 0..L_FIXED)

# --------- utils de sistema / build & run Java ----------

def sh(cmd, cwd: Path):
    print(f"[cmd] ({cwd}) $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Comando falló ({ret}): {' '.join(cmd)}")

def compile_java(src_dir: Path) -> Path:
    sim_dir = src_dir / "simulation"
    if not sim_dir.exists():
        raise FileNotFoundError(f"No existe {sim_dir}")
    java_files = sorted(sim_dir.glob("*.java"), key=lambda p: p.name)
    if not java_files:
        raise FileNotFoundError(f"No hay fuentes .java en {sim_dir}")

    rel_java = [str(p.relative_to(src_dir)) for p in java_files]
    sh(["javac", "-d", "."] + rel_java, cwd=src_dir)
    # Incluir recursivamente todo el package simulation en el jar
    sh(["jar", "cfe", "sim.jar", "simulation.Main", "-C", ".", "simulation"], cwd=src_dir)
    jar_path = src_dir / "sim.jar"
    if not jar_path.exists():
        raise FileNotFoundError("No se generó sim.jar")
    return jar_path

def run_java_once(src_dir: Path, N: int, L: float) -> Tuple[Path, Path]:
    project_root = src_dir.parent
    jar_path_rel = (src_dir / "sim.jar").relative_to(project_root)
    jar = project_root / jar_path_rel
    if not jar.exists():
        raise FileNotFoundError(f"No existe {jar}; compilá primero o no uses --no-compile")

    N_str = str(N)
    L_str = f"{L:.3f}"
    sh(["java", "-jar", str(jar_path_rel), N_str, L_str], cwd=project_root)

    out_dir = project_root / "out"
    events = out_dir / f"events_L={L_str}_N={N_str}.txt"
    press  = out_dir / f"pressure_L={L_str}_N={N_str}.txt"
    if not events.exists():
        raise FileNotFoundError(f"No se encontró {events}")
    return events, press

# --------- parseo del archivo de eventos ----------

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
            # Ejemplo: SNAPSHOT POST t=0.338808899 | id=0 x=... ...
            # 1) tiempo
            m_t = re.search(r"t=([0-9]+\.[0-9]+)", line)
            if not m_t:
                continue
            t = float(m_t.group(1))
            # 2) por cada " | id=... x=... y=... ... r=..."
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

# --------- paredes (mismo layout que en Java/Map) ----------

def build_walls(L: float):
    """Devuelve lista de segmentos [(x1,y1,x2,y2), ...]"""
    segs = []
    Lf = L_FIXED
    # Caja izquierda
    segs += [
        (0, 0, Lf, 0),
        (0, 0, 0, Lf),
        (0, Lf, Lf, Lf),
        # pared imaginaria NO se dibuja; sólo para presión
    ]
    # Puertas verticales en x=Lf
    y_low = (Lf - L)/2.0
    y_high = (Lf + L)/2.0
    segs += [
        (Lf, 0,     Lf, y_low),
        (Lf, Lf,    Lf, y_high),
        # Pasillo
        (Lf, y_low, 2*Lf, y_low),
        (Lf, y_high,2*Lf, y_high),
        # Pared derecha
        (2*Lf, y_high, 2*Lf, y_low),
    ]
    return segs

# --------- util: re-muestreo temporal a FPS constante ---------

def resample_to_uniform_time(snaps: List[Snapshot], fps: int, extra_skip: int = 1):
    """
    Dada la secuencia de snapshots (no uniformes en tiempo),
    elige índices cercanos a la grilla uniforme t_k = t0 + k*(1/fps).
    Retorna:
      idxs      -> lista de índices de 'snaps' a usar
      times_out -> tiempos uniformes (t_k) para mostrar en la animación
    'extra_skip' se aplica después, para acelerar aún más si se desea.
    """
    if fps <= 0:
        fps = 30
    dt = 1.0 / fps
    times = [s.t for s in snaps]
    t0, tf = times[0], times[-1]
    K = max(1, int((tf - t0) / dt) + 1)
    times_uniform = [t0 + k*dt for k in range(K)]

    # para cada t_k, buscar snapshot más cercano
    idxs = []
    for tk in times_uniform:
        j = bisect.bisect_left(times, tk)
        if j == 0:
            idxs.append(0)
        elif j >= len(times):
            idxs.append(len(times) - 1)
        else:
            # elegir el más cercano entre j-1 y j
            if (tk - times[j-1]) <= (times[j] - tk):
                idxs.append(j-1)
            else:
                idxs.append(j)

    # aplicar extra_skip (opcional, e.g. 2 toma 1 de cada 2)
    s = max(1, extra_skip)
    idxs = idxs[::s]
    times_uniform = times_uniform[::s]
    return idxs, times_uniform

# --------- animación ----------

def animate_snapshots(snaps: List[Snapshot], L: float, out_path: Path,
                      fps: int = 30, skip: int = 1, show: bool = False):
    """
    Crea animación en TIEMPO REAL:
    - re-muestrea a grilla uniforme dt=1/fps -> duración del video ≈ duración de la simulación
    - 'skip' se aplica luego del re-muestreo para acelerar si querés
    """
    # Orden estable por id
    ids_sorted = sorted(snaps[0].state.keys())

    # Re-muestreo temporal a FPS constante
    idxs, times_uniform = resample_to_uniform_time(snaps, fps=fps, extra_skip=skip)

    # Pre-armar frames de posiciones/radios siguiendo idxs seleccionados
    frames_xy: List[List[Tuple[float,float,float]]] = []
    for j in idxs:
        s = snaps[j]
        frames_xy.append([s.state[i] for i in ids_sorted])

    # Figura
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 2*L_FIXED)
    ax.set_ylim(0, L_FIXED)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Simulación N={len(ids_sorted)}, L={L:.3f} m")

    # Dibujar paredes
    for (x1,y1,x2,y2) in build_walls(L):
        ax.plot([x1,x2], [y1,y2], lw=2, color="black")

    # Partículas como círculos (patches)
    patches = []
    xs0, ys0, rs0 = zip(*frames_xy[0])
    for (x, y, r) in frames_xy[0]:
        c = Circle((x, y), radius=r, fill=False)
        patches.append(c)
        ax.add_patch(c)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        time_text.set_text("")
        return patches + [time_text]

    def update(frame_idx):
        xs, ys, rs = zip(*frames_xy[frame_idx])
        for k, c in enumerate(patches):
            c.center = (xs[k], ys[k])
            c.radius = rs[k]
        time_text.set_text(f"t = {times_uniform[frame_idx]:.3f} s")
        return patches + [time_text]

    # interval_ms solo afecta la vista interactiva (no el archivo final)
    interval_ms = 1000.0 / max(1, fps)
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(frames_xy), interval=interval_ms, blit=True)

    # Guardar (mp4 si hay ffmpeg, sino gif con Pillow), usando el fps solicitado
    out_path = out_path.with_suffix(out_path.suffix or ".mp4")
    if out_path.suffix.lower() == ".mp4":
        if shutil.which("ffmpeg"):
            Writer = animation.FFMpegWriter
            # preset ultrafast para acelerar la codificación
            writer = Writer(fps=fps, codec='libx264',
                            extra_args=['-preset','ultrafast','-crf','23'])
            anim.save(out_path, writer=writer)
        else:
            print("[warn] ffmpeg no disponible; guardando como GIF.")
            out_path = out_path.with_suffix(".gif")
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))

    print(f"[ok] Animación guardada en: {out_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# --------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Corre Java una vez (N,L) y anima la simulación desde events_*.txt (tiempo real).")
    ap.add_argument("--N", type=int, help="Cantidad de partículas")
    ap.add_argument("--L", type=float, help="Apertura del pasillo (m)")
    ap.add_argument("--use-file", type=str, help="Ruta a un events_*.txt existente (saltea ejecutar Java)")
    ap.add_argument("--no-compile", action="store_true", help="No recompilar Java (usa sim.jar existente)")
    ap.add_argument("--out", type=str, default="out/anim.mp4", help="Archivo de salida (mp4 o gif)")
    ap.add_argument("--fps", type=int, default=30, help="FPS objetivo del video (define la grilla temporal uniforme)")
    ap.add_argument("--skip", type=int, default=1, help="Tomar 1 de cada 'skip' frames DESPUÉS del re-muestreo")
    ap.add_argument("--show", action="store_true", help="Mostrar la animación al finalizar")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    src_dir = script_path.parents[1]  # .../src
    java_dir = src_dir / "simulation"
    if not java_dir.exists():
        print(f"[error] No encuentro {java_dir}", file=sys.stderr)
        sys.exit(1)

    events_path: Path
    if args.use_file:
        events_path = Path(args.use_file).resolve()
        if not events_path.exists():
            print(f"[error] No existe {events_path}", file=sys.stderr)
            sys.exit(1)
        # Intentar inferir L y N
        if args.L is None:
            m = re.search(r"L=([0-9]*\.?[0-9]+)", events_path.name)
            if m:
                args.L = float(m.group(1))
        if args.N is None:
            m = re.search(r"N=(\d+)", events_path.name)
            if m:
                args.N = int(m.group(1))
    else:
        if args.N is None or args.L is None:
            print("[error] Especificá --N y --L, o usa --use-file", file=sys.stderr)
            sys.exit(1)
        # Compilar si hace falta
        jar = src_dir / "sim.jar"
        if (not args.no_compile) or (not jar.exists()):
            compile_java(src_dir)
        # Correr Java
        events_path, _ = run_java_once(src_dir, args.N, args.L)

    # Parsear y animar
    snaps = parse_events_file(events_path)
    L_for_walls = args.L if args.L is not None else L_FIXED
    out_path = Path(args.out)
    animate_snapshots(snaps, L=L_for_walls, out_path=out_path, fps=max(1, args.fps), skip=max(1, args.skip), show=args.show)


if __name__ == "__main__":
    main()
rgs.show


if __name__ == "__main__":
    main()
