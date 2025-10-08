#!/usr/bin/env python3
"""Generate an energy vs time plot for a single galaxy simulation run."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from data_utils import ensure_energy_file, find_energy_file


def format_scientific(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    formatted = f"{value:.{digits}e}"
    mantissa, exponent = formatted.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = ""
    if exponent.startswith("-"):
        sign = "-"
        exponent = exponent[1:]
    exponent = exponent.lstrip("+").lstrip("0")
    if exponent == "":
        exponent = "0"
    return f"{mantissa}e{sign}{exponent}"


def format_dt_label(dt: float) -> str:
    if dt == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(dt))))
    mantissa = dt / (10 ** exponent)
    return f"{mantissa:.2f}e{exponent:+d}"


def load_energy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"El archivo {path} no contiene la columna de energía total")
    return data[:, 0], data[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficar energía total vs tiempo para una corrida")
    parser.add_argument("--root", default="../out/galaxy/energy", help="Directorio con los archivos *_energy.txt")
    parser.add_argument("--output", default="plots", help="Directorio donde guardar la figura")
    parser.add_argument("--N", type=int, default=500, help="Número de partículas de la simulación")
    parser.add_argument("--run", type=int, default=0, help="Índice de corrida a utilizar (runXXX)")
    parser.add_argument("--dt", type=float, default=1e-3, help="Paso temporal dt")
    parser.add_argument("--dt-output", type=float, default=None, help="Paso de guardado dt_output (default=dt)")
    parser.add_argument("--tf", type=float, default=20.0, help="Tiempo final de simulación")
    parser.add_argument("--speed", type=float, default=0.1, help="Módulo de las velocidades iniciales")
    parser.add_argument("--softening", type=float, default=0.05, help="Parámetro de suavizado gravitacional h")
    parser.add_argument("--collision", action="store_true", help="Usar condiciones iniciales de colisión")
    parser.add_argument("--dx", type=float, default=4.0, help="Separación en x para el escenario de colisión")
    parser.add_argument("--dy", type=float, default=0.5, help="Separación en y para el escenario de colisión")
    parser.add_argument("--cluster-size", type=int, default=None,
                        help="Cantidad de partículas por cúmulo para el escenario de colisión")
    parser.add_argument("--seed", type=int, default=None, help="Semilla base para reproducibilidad")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidate_dirs = []
    if args.dt is not None:
        dt_label = format_dt_label(args.dt)
        dt_dir = root / f"dt_{dt_label}" / "energy"
        if dt_dir.exists():
            candidate_dirs.append(dt_dir)
    candidate_dirs.append(root)

    energy_path = None
    for energy_dir in candidate_dirs:
        energy_path = find_energy_file(energy_dir, args.N, args.run)
        if energy_path:
            break

    if energy_path is None:
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
    else:
        print(f"ℹ️ Usando datos existentes en {energy_path.parent}")

    times, total_energy = load_energy(energy_path)
    mean_energy = float(np.mean(total_energy))
    std_energy = float(np.std(total_energy))
    percent_error = (std_energy / abs(mean_energy) * 100.0) if mean_energy != 0.0 else float('nan')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"energy_timeseries_N{args.N}_run{args.run:03d}.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, total_energy, linewidth=1.5)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía total")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_scientific(value)))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"✅ Guardado {fig_path}")
    print(
        "Resumen energía: media={:.6e}, desvío={:.6e}, %std/|media|={:.6f}%".format(
            mean_energy,
            std_energy,
            percent_error,
        )
    )


if __name__ == "__main__":
    main()
