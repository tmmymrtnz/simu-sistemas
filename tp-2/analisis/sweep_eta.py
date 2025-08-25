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

def run_java(props: dict, base="sweep_eta"):
    props = props.copy()
    props.setdefault("outputBase", base + f"_{int(time.time()*1e6)%1_000_000}")
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
        for rep in range(args.reps):
            out = run_java(dict(N=N, L=args.L, R=args.R, v0=args.v0, dt=args.dt,
                                steps=args.steps, eta=eta, rule=args.rule,
                                periodic=True, seed=10_000+rep))
            created.append(out)
            # Pass the discard time to the function
            m, s = read_va_mean(out, args.discard_time)
            rows.append((eta, rep, m, s, out.name))
            
    df = pd.DataFrame(rows, columns=["eta", "rep", "va_mean", "va_std", "run"])
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

    if args.cleanup:
        for d in created:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        print(f"Cleanup: removed {len(created)} directories from out/")

def plot_data(args):
    out_dir = pathlib.Path("out") / "sweep_eta_summary"
    if not out_dir.exists():
        print("Error: No data found. Please run the simulations first.")
        return

    try:
        summary = pd.read_csv(out_dir / "summary.csv", index_col="eta")
        with open(out_dir / "args.json", "r") as f:
            saved_args = json.load(f)
    except FileNotFoundError:
        print("Error: Data files not found. Please run the simulations first.")
        return

    print("\nLoading data and plotting...")
    print("\nResumen ⟨va⟩ vs η:\n", summary)

    plt.figure()
    plt.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker='o')
    plt.xlabel("η")
    plt.ylabel("⟨va⟩")
    plt.title(f"{saved_args['rule'].capitalize()}: ⟨va⟩ vs η  |  ρ={saved_args['rho']}, L={saved_args['L']}, R={saved_args['R']}")
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
    ap.add_argument("--reps", type=int, default=5)
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
