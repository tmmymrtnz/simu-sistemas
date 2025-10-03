#!/usr/bin/env python3

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rhm(path: Path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("El archivo no tiene la columna de r_hm (se esperan al menos 3 columnas)")
    return data[:, 0], data[:, 2]


def main():
    parser = argparse.ArgumentParser(description="Graficar r_hm(t) a partir de un archivo de energía")
    parser.add_argument("archivo", help="Ruta al archivo *_energy.txt")
    parser.add_argument("--output", help="Ruta opcional para guardar la figura", default=None)
    args = parser.parse_args()

    archivo = Path(args.archivo).resolve()
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}")

    tiempos, rhm = load_rhm(archivo)

    plt.figure(figsize=(8, 4.5))
    plt.plot(tiempos, rhm, linewidth=1.2)
    plt.xlabel("Tiempo")
    plt.ylabel("r_hm")
    plt.title(f"r_hm(t) para {archivo.name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.output:
        salida = Path(args.output)
    else:
        salida = archivo.parent / (archivo.stem + "_rhm.png")
    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida, dpi=300)
    print(f"✅ Guardado {salida}")


if __name__ == "__main__":
    main()
