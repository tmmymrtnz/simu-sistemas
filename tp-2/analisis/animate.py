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

def angle_to_color(theta):
    cmap = plt.cm.hsv
    u = (theta % (2*np.pi)) / (2*np.pi)
    return cmap(u)

# mantener viva la animación
anim = None

def main():
    global anim
    ap = argparse.ArgumentParser(description="Animación: puntos + flecha chica de dirección (opcional color por ángulo).")
    ap.add_argument("outBase", help="carpeta dentro de ./out")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--save", type=pathlib.Path, help="ruta de salida (.mp4 o .gif)")
    ap.add_argument("--dot-size", type=float, default=70.0, help="tamaño de partícula en puntos^2")
    ap.add_argument("--no-dots", action="store_true", help="no dibujar puntos")
    ap.add_argument("--no-arrows", action="store_true", help="no dibujar flechas de dirección")
    ap.add_argument("--arrow-len", type=float, default=None, help="longitud de flecha (unidades del sistema)")
    ap.add_argument("--arrow-width", type=float, default=0.006, help="ancho de flecha (coords de ejes)")
    ap.add_argument("--color-by-angle", action="store_true", help="colorear vectores (y puntos) según el ángulo")
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

    # Colores (opcional por ángulo)
    cols0 = angle_to_color(d0[:,5]) if args.color_by_angle else None

    # Puntos visibles (si no se desactivan)
    if not args.no-dots if False else False:  # guardrail; will be replaced below
        pass
    # puntos
    S = None
    if not args.no_dots:
        S = ax.scatter(d0[:,1], d0[:,2],
                       s=args.dot_size,
                       c=cols0 if args.color_by_angle else "tab:blue",
                       edgecolors='k', linewidths=0.3, alpha=0.95)

    # Flechas chiquitas de dirección (longitud fija)
    Q = None
    if not args.no_arrows:
        u0 = np.cos(d0[:,5]) * arrow_len
        v0 = np.sin(d0[:,5]) * arrow_len
        Q = ax.quiver(d0[:,1], d0[:,2], u0, v0,
                      angles='xy', scale_units='xy', scale=1,
                      width=args.arrow_width, pivot='tail',
                      color=cols0 if args.color_by_angle else "black")

    def update(i):
        d = load_frame(frames[i])
        if S is not None:
            S.set_offsets(d[:,1:3])
            if args.color_by_angle:
                S.set_facecolors(angle_to_color(d[:,5]))
        if Q is not None:
            u = np.cos(d[:,5]) * arrow_len
            v = np.sin(d[:,5]) * arrow_len
            Q.set_offsets(d[:,1:3]); Q.set_UVC(u, v)
            if args.color_by_angle:
                Q.set_color(angle_to_color(d[:,5]))
        title.set_text(f"t={i}")
        return (tuple(x for x in (S,Q,title) if x is not None))

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
