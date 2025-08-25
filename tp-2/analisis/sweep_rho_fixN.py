#!/usr/bin/env python3
import argparse, pathlib, subprocess, tempfile, time, math, shutil, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

JAR = pathlib.Path(__file__).resolve().parents[1]/"simulacion/flocking.jar"

def run_java(props: dict, base="sweep_rho_fixN"):
    props = props.copy()
    props.setdefault("outputBase", base + f"_{int(time.time()*1e6)%1_000_000}")
    with tempfile.NamedTemporaryFile("w+", suffix=".properties", delete=False) as tf:
        for k,v in props.items(): tf.write(f"{k}={v}\n")
        tf.flush()
        subprocess.run(["java","-jar",str(JAR),tf.name], check=True, stdout=subprocess.DEVNULL)
    return pathlib.Path("out")/props["outputBase"]

def read_va_mean(out_dir: pathlib.Path, discard=0.5):
    obs = pd.read_csv(out_dir/"observables.csv")
    cut = int(discard*(len(obs)-1))
    sta = obs.iloc[cut:]
    return sta["va"].mean(), sta["va"].std()

def run_simulations(args):
    """
    Runs the simulations and saves the raw and summary data to a file.
    """
    created = []
    rows = []
    print("Running simulations...")
    for rho in args.rhos:
        L = math.sqrt(args.N / rho)
        for rep in range(args.reps):
            out = run_java(dict(N=args.N, L=L, R=args.R, v0=args.v0, dt=args.dt,
                                steps=args.steps, eta=args.eta, rule=args.rule,
                                periodic=True, seed=20_000+rep))
            created.append(out)
            m, s = read_va_mean(out, args.discard)
            rows.append((rho, L, args.N, args.eta, rep, m, s, out.name))

    df = pd.DataFrame(rows, columns=["rho","L","N","eta","rep","va_mean","va_std","run"])
    g = df.groupby(["rho","L","N","eta"])["va_mean"]
    summary = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()}).reset_index()
    summary["sem"] = summary["std"]/np.sqrt(summary["n"])
    
    out_dir = pathlib.Path("out")/"sweep_rho_fixN_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"raw_runs.csv", index=False)
    summary.to_csv(out_dir/"summary.csv", index=False)
    
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    print("\nSimulations finished.")
    print(f"Data saved to: {out_dir/'raw_runs.csv'} and {out_dir/'summary.csv'}")

    if args.cleanup:
        for d in created:
            try:
                shutil.rmtree(d)
            except Exception: pass
        print(f"Cleanup: removed {len(created)} directories from out/")

def plot_data(args):
    """
    Loads data from a file and plots the graph.
    """
    out_dir = pathlib.Path("out") / "sweep_rho_fixN_summary"
    if not out_dir.exists():
        print("Error: No data found. Please run the simulations first.")
        return

    try:
        summary = pd.read_csv(out_dir / "summary.csv")
        with open(out_dir / "args.json", "r") as f:
            saved_args = json.load(f)
    except FileNotFoundError:
        print("Error: Data files not found. Please run the simulations first.")
        return

    print("\nLoading data and plotting...")
    print("\nResumen ⟨va⟩ vs ρ (N fijo):\n", summary[["rho","L","N","eta","mean","sem","n"]].sort_values("rho"))

    plt.figure()
    sub = summary.sort_values("rho")
    plt.errorbar(sub["rho"], sub["mean"], yerr=sub["sem"], marker='o')
    plt.xlabel("ρ = N/L² (N fijo, variar L)")
    plt.ylabel("⟨va⟩ (estacionario)")
    plt.title(f"{saved_args['rule'].capitalize()}: ⟨va⟩ vs ρ  |  N={saved_args['N']}, R={saved_args['R']}, η={saved_args['eta']}")
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="⟨va⟩ vs ρ con N fijo (variar L=√(N/ρ)).")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.2,0.5,1.0,2.0])
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--v0", type=float, default=0.03)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rule", choices=["vicsek","voter"], default="vicsek")
    ap.add_argument("--discard", type=float, default=0.5)
    ap.add_argument("--cleanup", action="store_true", help="borrar corridas generadas al final")
    
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="Run the simulations and save the data")
    group.add_argument("--plot", action="store_true", help="Plot the data from a saved file")

    args = ap.parse_args()

    if not args.run and not args.plot:
        ap.print_help()
        return

    if args.run:
        run_simulations(args)
    
    if args.plot:
        plot_data(args)

if __name__=="__main__":
    main()
