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
    if not base.exists() or not (base/"particles_t0000.csv").exists() or not (base/"neighbours_t0000.txt").exists():
        print("No se encontró la carpeta de resultados 'out/run'. Compila y corre la simulación primero con 'make all'.")
        sys.exit(1)
    pos  = pd.read_csv(base/"particles_t0000.csv")
    neigh = load_neigh(base/"neighbours_t0000.txt")

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

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    plt.title(f"Vecinos de {particle}")
    # Dibujar círculos para todas las partículas (solo cambia el color)
    mask_others = ~pos.id.isin(friends.id) & (pos.id != p0.id)
    for _, row in pos[mask_others].iterrows():
        circ = plt.Circle((row.x, row.y), row.r, color='gray', alpha=0.7, lw=1, fill=True)
        ax.add_patch(circ)
    for _, row in friends[friends.id != p0.id].iterrows():
        circ = plt.Circle((row.x, row.y), row.r, color='orange', alpha=0.7, lw=1, fill=True)
        ax.add_patch(circ)
    circ = plt.Circle((p0.x, p0.y), p0.r, color='deepskyblue', alpha=0.7, lw=1, fill=True)
    ax.add_patch(circ)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='otras', markerfacecolor='gray', markersize=10, alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', label='vecinas', markerfacecolor='orange', markersize=10, alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', label='partícula', markerfacecolor='deepskyblue', markersize=10, alpha=0.7)
    ])
    plt.xlim(0,pos.x.max()); plt.ylim(0,pos.y.max()); plt.show()

if __name__ == "__main__":
    main()
