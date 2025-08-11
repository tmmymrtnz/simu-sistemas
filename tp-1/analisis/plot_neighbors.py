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
    import sys
    base = pathlib.Path("out/run")
    if not base.exists() or not (base/"particles.csv").exists() or not (base/"neighbours.txt").exists():
        print("No se encontró la carpeta de resultados 'out/run'. Compila y corre la simulación primero con 'make all'.")
        sys.exit(1)
    pos  = pd.read_csv(base/"particles.csv")
    neigh = load_neigh(base/"neighbours.txt")

    # Pedir id de partícula por input interactivo, mostrando solo el rango
    ids = sorted(pos.id.unique())
    if len(ids) > 0:
        print(f"IDs de partículas disponibles: {ids[0]} a {ids[-1]}")
    else:
        print("No hay partículas disponibles en el archivo.")
        sys.exit(1)
    while True:
        try:
            particle = int(input("Ingrese el id de la partícula a resaltar: "))
            if particle in pos.id.values:
                break
            else:
                print(f"ID no válido. Debe estar entre {ids[0]} y {ids[-1]}.")
        except Exception:
            print("Entrada inválida. Intente nuevamente.")

    p0 = pos[pos.id==particle].iloc[0]
    friends = pos[pos.id.isin(neigh[particle])]

    plt.figure(figsize=(6,6))
    plt.gca().set_aspect('equal')
    plt.title(f"Vecinos de {particle}")
    plt.scatter(pos.x,pos.y,s=20,alpha=.3,label='otras')
    plt.scatter(friends.x,friends.y,s=60,marker='s',label='vecinas')
    plt.scatter([p0.x],[p0.y],s=120,marker='*',label='partícula')
    plt.legend(); plt.xlim(0,pos.x.max()); plt.ylim(0,pos.y.max()); plt.show()

if __name__ == "__main__":
    main()
