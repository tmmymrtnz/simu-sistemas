#!/usr/bin/env python3
import argparse, re, pathlib, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_neigh(path):
    d={}
    with open(path) as fh:
        for ln in fh:
            pid, arr = ln.split(':',1)
            arr = arr.strip()
            if arr.startswith('[') and arr.endswith(']'):
                items = [s.strip() for s in arr[1:-1].split(',') if s.strip()]
                d[int(pid)] = {int(x) for x in items}
            else:
                d[int(pid)] = set()
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outBase", help="carpeta bajo ./out/")
    ap.add_argument("--id", type=int, default=0, help="id de partícula a resaltar")
    ap.add_argument("--save", help="ruta para guardar (mp4/gif). Si falta, se muestra en pantalla")
    args = ap.parse_args()

    base = pathlib.Path("out")/args.outBase
    frames = sorted(base.glob("particles_t*.csv"),
                    key=lambda p: int(re.search(r't(\d+)', p.stem).group(1)))
    if not frames:
        raise SystemExit(f"❌ No hay frames en {base}. Corré la simulación con frames>1.")

    neigh_files = {int(re.search(r't(\d+)', p.stem).group(1)).__int__(): p
                   for p in base.glob("neighbours_t*.txt")}

    fig, ax = plt.subplots(figsize=(6,6))
    scat_all = ax.scatter([], [], s=20, alpha=.3, label='otras')
    scat_nb  = ax.scatter([], [], s=60, marker='s', label='vecinas')
    scat_id  = ax.scatter([], [], s=120, marker='*', label='partícula')
    ax.set_aspect('equal'); ax.legend()

    # init with frame 0 for limits
    f0 = pd.read_csv(frames[0])
    L = max(f0['x'].max(), f0['y'].max())
    ax.set_xlim(0, L); ax.set_ylim(0, L)

    def update(k):
        csv = frames[k]
        tnum = int(re.search(r't(\d+)', csv.stem).group(1))
        pos = pd.read_csv(csv)
        neigh = load_neigh(neigh_files[tnum])
        if args.id not in set(pos['id']):
            return scat_all, scat_nb, scat_id
        p0 = pos[pos['id']==args.id].iloc[0]
        nb = pos[pos['id'].isin(neigh.get(args.id, set()))]
        scat_all.set_offsets(pos[['x','y']].values)
        scat_nb.set_offsets(nb[['x','y']].values)
        scat_id.set_offsets([[p0.x, p0.y]])
        ax.set_title(f"Frame {tnum} — id={args.id} (vecinos={len(nb)})")
        return scat_all, scat_nb, scat_id

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False, repeat=True)

    if args.save:
        ext = pathlib.Path(args.save).suffix.lower()
        if ext == '.gif':
            anim.save(args.save, writer='pillow', fps=10)
        else:
            anim.save(args.save, writer='ffmpeg', fps=10)
    else:
        plt.show()

if __name__ == "__main__":
    main()
