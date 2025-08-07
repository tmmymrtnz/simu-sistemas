#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt, pathlib

def load_neigh(path):
    d = {}
    with open(path) as fh:
        for ln in fh:
            pid, arr = ln.split(':')
            d[int(pid)] = {int(x) for x in arr.strip()[1:-1].split(',') if x}
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outBase", help="nombre de carpeta bajo ./out/")
    ap.add_argument("particle", type=int, help="id a resaltar")
    args = ap.parse_args()
    base = pathlib.Path("out", args.outBase)
    pos  = pd.read_csv(base/"particles.csv")
    neigh = load_neigh(base/"neighbours.txt")

    p0 = pos[pos.id==args.particle].iloc[0]
    friends = pos[pos.id.isin(neigh[args.particle])]

    plt.figure(figsize=(6,6))
    plt.gca().set_aspect('equal')
    plt.title(f"Vecinos de {args.particle}")
    plt.scatter(pos.x,pos.y,s=20,alpha=.3,label='otras')
    plt.scatter(friends.x,friends.y,s=60,marker='s',label='vecinas')
    plt.scatter([p0.x],[p0.y],s=120,marker='*',label='partícula')
    plt.legend(); plt.xlim(0,pos.x.max()); plt.ylim(0,pos.y.max()); plt.show()

if __name__ == "__main__":
    main()
