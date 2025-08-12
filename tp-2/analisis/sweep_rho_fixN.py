#!/usr/bin/env python3
import argparse, pathlib, subprocess, tempfile, time, math
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
    args = ap.parse_args()

    created = []
    rows=[]
    for rho in args.rhos:
        L = math.sqrt(args.N / rho)
        for rep in range(args.reps):
            out = run_java(dict(N=args.N, L=L, R=args.R, v0=args.v0, dt=args.dt,
                                steps=args.steps, eta=args.eta, rule=args.rule,
                                periodic=True, seed=20_000+rep))
            created.append(out)
            m,s = read_va_mean(out, args.discard)
            rows.append((rho, L, args.N, args.eta, rep, m, s, out.name))

    df = pd.DataFrame(rows, columns=["rho","L","N","eta","rep","va_mean","va_std","run"])
    g = df.groupby(["rho","L","N","eta"])["va_mean"]
    summary = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()}).reset_index()
    summary["sem"] = summary["std"]/np.sqrt(summary["n"])
    print("\nResumen ⟨va⟩ vs ρ (N fijo):\n", summary[["rho","L","N","eta","mean","sem","n"]].sort_values("rho"))

    plt.figure()
    sub = summary.sort_values("rho")
    plt.errorbar(sub["rho"], sub["mean"], yerr=sub["sem"], marker='o')
    plt.xlabel("ρ = N/L² (N fijo, variar L)")
    plt.ylabel("⟨va⟩ (estacionario)")
    plt.title(f"{args.rule.capitalize()}: ⟨va⟩ vs ρ  |  N={args.N}, R={args.R}, η={args.eta}")
    plt.tight_layout()
    plt.show()

    out_dir = pathlib.Path("out")/"sweep_rho_fixN_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"raw_runs.csv", index=False)
    summary.to_csv(out_dir/"summary.csv", index=False)
    print(f"📄 Guardados: {out_dir/'raw_runs.csv'}, {out_dir/'summary.csv'}")

    if args.cleanup:
        for d in created:
            try:
                for p in d.glob("*"): p.unlink()
                d.rmdir()
            except Exception: pass
        print(f"🧹 Limpieza: eliminadas {len(created)} carpetas de out/")

if __name__=="__main__":
    main()
