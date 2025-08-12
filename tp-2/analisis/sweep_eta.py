#!/usr/bin/env python3
import argparse, pathlib, subprocess, tempfile, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

JAR = pathlib.Path(__file__).resolve().parents[1]/"simulacion/flocking.jar"

def run_java(props: dict, base="sweep_eta"):
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
    ap = argparse.ArgumentParser(description="⟨va⟩ vs η (ρ fija).")
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--v0", type=float, default=0.03)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0,0.1,0.2,0.3,0.4,0.5])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rule", choices=["vicsek","voter"], default="vicsek")
    ap.add_argument("--discard", type=float, default=0.5)
    ap.add_argument("--cleanup", action="store_true", help="borrar corridas generadas al final")
    args = ap.parse_args()

    N = int(round(args.rho * args.L * args.L))
    created = []
    rows=[]
    for eta in args.etas:
        for rep in range(args.reps):
            out = run_java(dict(N=N, L=args.L, R=args.R, v0=args.v0, dt=args.dt,
                                steps=args.steps, eta=eta, rule=args.rule,
                                periodic=True, seed=10_000+rep))
            created.append(out)
            m,s = read_va_mean(out, args.discard)
            rows.append((eta, rep, m, s, out.name))
    df = pd.DataFrame(rows, columns=["eta","rep","va_mean","va_std","run"])
    g = df.groupby("eta")["va_mean"]
    summary = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    summary["sem"] = summary["std"]/np.sqrt(summary["n"])
    print("\nResumen ⟨va⟩ vs η:\n", summary)

    plt.figure()
    plt.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker='o')
    plt.xlabel("η"); plt.ylabel("⟨va⟩")
    plt.title(f"{args.rule.capitalize()}: ⟨va⟩ vs η  |  ρ={args.rho}, L={args.L}, R={args.R}")
    plt.tight_layout()
    plt.show()

    out_dir = pathlib.Path("out")/"sweep_eta_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"raw_runs.csv", index=False)
    summary.to_csv(out_dir/"summary.csv")
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
