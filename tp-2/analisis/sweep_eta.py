#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import tempfile
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

JAR = pathlib.Path(__file__).resolve().parents[1]/"simulacion/flocking.jar"

def run_java(props: dict, eta, base="sweep_eta"):
    props = props.copy()
    props.setdefault("outputBase", f"{base}_{eta}")
    with tempfile.NamedTemporaryFile("w+", suffix=".properties", delete=False) as tf:
        for k,v in props.items(): tf.write(f"{k}={v}\n")
        tf.flush()
        subprocess.run(["java", "-jar", str(JAR), tf.name], check=True, stdout=subprocess.DEVNULL)
    return pathlib.Path("out")/props["outputBase"]

def read_va_mean(out_dir: pathlib.Path, discard_time=None):
    """
    Reads va data and calculates the mean and std, discarding a specified time.
    """
    obs = pd.read_csv(out_dir/"observables.csv")
    
    if discard_time is not None and discard_time > 0:
        cut = obs[obs["t"] >= discard_time].index[0]
        sta = obs.iloc[cut:]
    else:
        sta = obs
        
    return sta["va"].mean(), sta["va"].std()

def run_simulations(args):
    """
    Runs the simulations and saves the raw and summary data to a file.
    """
    N = int(round(args.rho * args.L * args.L))
    created = []
    rows = []
    print("Running simulations...")
    for eta in args.etas:
        out = run_java(
            dict(N=N, L=args.L, R=args.R, v0=args.v0, dt=args.dt,
                steps=args.steps, eta=eta, rule=args.rule,
                periodic=True, seed=10_000 + int(eta)),
            eta
        )
        created.append(out)
        # Pass the discard time to the function
        m, s = read_va_mean(out, args.discard_time)
        rows.append((eta, m, s, out.name))
            
    df = pd.DataFrame(rows, columns=["eta", "va_mean", "va_std", "run"])
    g = df.groupby("eta")["va_mean"]
    summary = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])

    out_dir = pathlib.Path("out") / "sweep_eta_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw_runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    
    # Save the arguments used for this run
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    print("\nSimulations finished.")
    print(f"Data saved to: {out_dir/'raw_runs.csv'} and {out_dir/'summary.csv'}")


# ...existing code...

def plot_data(args):
    """
    Para cada corrida sweep_eta_{eta}, grafica va(t), pide cutoff y calcula el promedio estacionario.
    Luego grafica <va> vs eta.
    """
    base = pathlib.Path("out")
    runs = sorted(base.glob("sweep_eta_*"))
    if not runs:
        print("No se encontraron carpetas sweep_eta_* en out/")
        return

    results = []
    for run in runs:
        try:
            obs = pd.read_csv(run/"observables.csv")
            eta = float(run.name.split("_")[-1])
        except Exception as e:
            print(f"Error leyendo {run}: {e}")
            continue

        plt.figure(figsize=(7,4))
        plt.plot(obs["t"], obs["va"], lw=1.5, label="va(t)")
        plt.xlabel("t"); plt.ylabel("va")
        plt.title(f"va(t) — {run.name}")
        plt.tight_layout()
        plt.show(block=False)

        max_time = obs["t"].max()
        while True:
            try:
                user_input = input(f"[{run.name}] Ingrese t de corte para promedio estacionario (ej: {max_time*0.5:.2f}), Enter para no descartar: ")
                if user_input == "":
                    discard_time = 0.0
                    break
                discard_time = float(user_input)
                if discard_time >= 0:
                    break
                else:
                    print("Ingrese un valor no negativo.")
            except ValueError:
                print("Ingrese un número válido.")

        if discard_time > 0:
            cut = obs[obs["t"] >= discard_time].index[0]
        else:
            cut = 0
        sta = obs.iloc[cut:]
        va_mean, va_std = sta["va"].mean(), sta["va"].std()
        print(f"  Ventana estacionaria: t∈[{obs['t'].iloc[cut]:.2f},{obs['t'].iloc[-1]:.2f}]  |  ⟨va⟩={va_mean:.4f}  σ={va_std:.4f}")
        results.append((eta, va_mean, va_std, discard_time))

        plt.close()  # Cierra el gráfico anterior

    # Ordena por eta
    results.sort()
    etas = [r[0] for r in results]
    va_means = [r[1] for r in results]
    va_stds = [r[2] for r in results]

    plt.figure()
    plt.errorbar(etas, va_means, yerr=va_stds, marker='o')
    plt.xlabel("η")
    plt.ylabel("⟨va⟩ (estacionario)")
    plt.title("⟨va⟩ vs η (ventanas estacionarias elegidas manualmente)")
    plt.tight_layout()
    plt.show()





def plot_all_va_vs_time(out_dir="out"):
    """
    Grafica va(t) para cada corrida sweep_eta_{eta} en la carpeta out/
    """
    base = pathlib.Path(out_dir)
    runs = sorted(base.glob("sweep_eta_*"))
    if not runs:
        print("No se encontraron carpetas sweep_eta_* en", out_dir)
        return

    plt.figure(figsize=(8,5))
    for run in runs:
        try:
            obs = pd.read_csv(run/"observables.csv")
            eta = run.name.split("_")[-1]
            plt.plot(obs["t"], obs["va"], label=f"η={eta}")
        except Exception as e:
            print(f"Error leyendo {run}: {e}")

    plt.xlabel("t")
    plt.ylabel("va")
    plt.title("va(t) para distintas η")
    plt.legend()
    plt.tight_layout()
    plt.show()




def main():
    ap = argparse.ArgumentParser(description="Sweep eta and plot ⟨va⟩ vs η.")
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--v0", type=float, default=0.03)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--rule", choices=["vicsek", "voter"], default="vicsek")
    ap.add_argument("--discard-time", type=float, default=0.0, help="time (in seconds) to discard from the beginning of each run")
    ap.add_argument("--cleanup", action="store_true", help="delete generated runs at the end")
    
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="Run the simulations and save the data")
    group.add_argument("--plot", action="store_true", help="Plot the data from a saved file")

    args = ap.parse_args()

    if not args.run and not args.plot:
        args.run = True

    if args.run:
        run_simulations(args)
    
    if args.plot:
        plot_data(args)

if __name__ == "__main__":
    main()
