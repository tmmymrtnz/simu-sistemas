#!/usr/bin/env python3
import argparse, pathlib, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def load_static(dirp):
    P={}
    with open(dirp/'static.txt') as fh:
        for ln in fh:
            parts = ln.strip().split(maxsplit=1)
            if len(parts) == 2:
                k, v = parts
                P[k] = v
    P["L"] = float(P.get("L", 20.0))
    if "R" in P:
        try: P["R"] = float(P["R"])
        except: P["R"] = None
    else:
        P["R"] = None
    return P

def list_frames(dirp):
    return sorted(dirp.glob('t*.txt'),
                  key=lambda p: int(re.findall(r't(\d+)\.txt', p.name)[0]))

def load_frame(path):
    # columnas: id x y vx vy theta
    return np.loadtxt(path, skiprows=1)

# mantener viva la animación
anim = None

def main():
    global anim
    ap = argparse.ArgumentParser(description="Animación: puntos + flecha chica de dirección.")
    ap.add_argument("outBase", help="carpeta dentro de ./out")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--save", type=pathlib.Path, help="ruta de salida (.mp4 o .gif)")
    ap.add_argument("--dot-size", type=float, default=70.0, help="tamaño de partícula en puntos^2")
    ap.add_argument("--no-arrows", action="store_true", help="no dibujar flechas de dirección")
    ap.add_argument("--arrow-len", type=float, default=None, help="longitud de flecha (unidades del sistema)")
    ap.add_argument("--arrow-width", type=float, default=0.006, help="ancho de flecha (coords de ejes)")
    args = ap.parse_args()

    base = pathlib.Path("out")/args.outBase
    P = load_static(base)
    L = P["L"]; R = P.get("R", None)

    frames = list_frames(base)
    if not frames:
        raise SystemExit(f"No hay frames t####.txt en {base}. Corré la simulación primero.")

    d0 = load_frame(frames[0])

    # longitud por defecto de la flecha: ~0.6*R si se conoce, si no ~L/40
    default_arrow_len = (0.6*R) if (R is not None and R > 0) else (L/40.0)
    arrow_len = args.arrow_len if (args.arrow_len is not None and args.arrow_len > 0) else default_arrow_len

    fig, ax = plt.subplots(figsize=(6.8,6.8))
    ax.set_aspect('equal'); ax.set_xlim(0,L); ax.set_ylim(0,L)
    title = ax.set_title("t=0")

    # Puntos visibles
    S = ax.scatter(d0[:,1], d0[:,2],
                   s=args.dot_size,
                   c="tab:blue",
                   edgecolors='k', linewidths=0.3, alpha=0.95)

    # Flechas chiquitas de dirección (longitud fija), color neutro
    if not args.no_arrows:
        u0 = np.cos(d0[:,5]) * arrow_len
        v0 = np.sin(d0[:,5]) * arrow_len
        Q = ax.quiver(d0[:,1], d0[:,2], u0, v0,
                      angles='xy', scale_units='xy', scale=1,
                      width=args.arrow_width, pivot='tail', color="black")
    else:
        Q = None

    def update(i):
        d = load_frame(frames[i])
        S.set_offsets(d[:,1:3])
        if Q is not None:
            u = np.cos(d[:,5]) * arrow_len
            v = np.sin(d[:,5]) * arrow_len
            Q.set_offsets(d[:,1:3]); Q.set_UVC(u, v)
        title.set_text(f"t={i}")
        return (S, Q, title) if Q is not None else (S, title)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000/args.fps, blit=False, repeat=False)

    if args.save:
        ext = args.save.suffix.lower()
        if ext == ".gif":
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps)
        anim.save(str(args.save), writer=writer)
        print(f"💾 Animación guardada en {args.save}")
    else:
        plt.show()

if __name__=="__main__":
    main()
