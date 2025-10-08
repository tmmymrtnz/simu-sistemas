#!/usr/bin/env python3
"""Plot r_hm(t) for a single simulation run, generating data if needed."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from data_utils import ensure_energy_file


def load_rhm_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"El archivo {path} no contiene la columna r_hm")
    return data[:, 0], data[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficar r_hm(t) para una corrida individual")
    parser.add_argument("--root", default="../out/galaxy/energy", help="Directorio con archivos *_energy.txt")
    parser.add_argument("--output", default="plots", help="Directorio donde guardar la figura")
    parser.add_argument("--N", type=int, default=500, help="Número de partículas")
    parser.add_argument("--run", type=int, default=0, help="Índice de corrida a utilizar (runXXX)")
    parser.add_argument("--dt", type=float, default=1e-3, help="Paso temporal dt")
    parser.add_argument("--dt-output", type=float, default=None, help="Paso de guardado dt_output (default = dt)")
    parser.add_argument("--tf", type=float, default=20.0, help="Tiempo final de simulación")
    parser.add_argument("--speed", type=float, default=0.1, help="Módulo de las velocidades iniciales")
    parser.add_argument("--softening", type=float, default=0.05, help="Parámetro de suavizado gravitacional h")
    parser.add_argument("--collision", action="store_true", help="Usar condiciones iniciales de colisión")
    parser.add_argument("--dx", type=float, default=4.0, help="Separación en x para colisión")
    parser.add_argument("--dy", type=float, default=0.5, help="Separación en y para colisión")
    parser.add_argument("--cluster-size", type=int, default=None,
                        help="Partículas por cúmulo cuando se usa el escenario de colisión")
    parser.add_argument("--seed", type=int, default=None, help="Semilla base para reproducibilidad")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    energy_path = ensure_energy_file(
        root,
        n_value=args.N,
        run_index=args.run,
        dt=args.dt,
        dt_output=args.dt_output,
        tf=args.tf,
        speed=args.speed,
        softening=args.softening,
        collision=args.collision,
        dx=args.dx,
        dy=args.dy,
        cluster_size=args.cluster_size,
        seed=args.seed,
    )

    times, rhm = load_rhm_series(energy_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"rhm_timeseries_N{args.N}_run{args.run:03d}.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, rhm, linewidth=1.5, label=energy_path.name)
    plt.xlabel("Tiempo")
    plt.ylabel("r_hm")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"✅ Guardado {fig_path}")


if __name__ == "__main__":
    main()
