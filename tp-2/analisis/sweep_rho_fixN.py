#!/usr/bin/env python3
import argparse, pathlib, subprocess, tempfile, time, math, shutil, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

JAR = pathlib.Path(__file__).resolve().parents[1]/"simulacion/flocking.jar"

def run_java(props: dict, rho):
    props = props.copy()
    out_dir = pathlib.Path("out") / "sweep_rho" / f"sweep_rho_fixN_{rho}"
    out_dir.mkdir(parents=True, exist_ok=True)
    props["outputBase"] = str(out_dir.relative_to("out"))  # <--- CAMBIO AQUÍ
    with tempfile.NamedTemporaryFile("w+", suffix=".properties", delete=False) as tf:
        for k, v in props.items():
            tf.write(f"{k}={v}\n")
        tf.flush()
        subprocess.run(["java", "-jar", str(JAR), tf.name], check=True, stdout=subprocess.DEVNULL)
    return out_dir


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
        N = int(round(rho * args.L * args.L))
        out = run_java(dict(N=N, L=args.L, R=args.R, v0=args.v0, dt=args.dt,
                            steps=args.steps, eta=args.eta, rule=args.rule,
                            periodic=True, seed=20_000+int(rho)), rho)
        created.append(out)
        m, s = read_va_mean(out, args.discard)
        rows.append((rho, args.L, N, args.eta, m, s, out.name))

    df = pd.DataFrame(rows, columns=["rho","L","N","eta","va_mean","va_std","run"])
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


def plot_data(args):
    """
    Para cada corrida sweep_rho_fixN_*, grafica va(t), pide cutoff y calcula el promedio estacionario.
    Luego grafica <va> vs rho.
    """
    base = pathlib.Path("out")/ "sweep_rho"
    runs = sorted(base.glob("sweep_rho_fixN_*"))
    if not runs:
        print("No se encontraron carpetas sweep_rho_fixN_* en out/")
        return

    results = []
    for run in runs:
        try:
            obs = pd.read_csv(run/"observables.csv")
            # Extraer rho del nombre de la carpeta (último valor después del último "_")
            rho_str = run.name.split("_")[-1]
            try:
                rho = float(rho_str)
            except ValueError:
                # Si el nombre tiene timestamp, buscar el valor de rho en static.txt
                with open(run/"static.txt") as f:
                    for line in f:
                        if line.startswith("N="): N = float(line.split("=")[1])
                        if line.startswith("L="): L = float(line.split("=")[1])
                rho = N / (L*L)
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
        results.append((rho, va_mean, va_std, discard_time))

        plt.close()  # Cierra el gráfico anterior

    # Ordena por rho
    results.sort()
    rhos = [r[0] for r in results]
    va_means = [r[1] for r in results]
    va_stds = [r[2] for r in results]

    # quiero que escriba esta data calculada en un archivo csv en la carpeta out
    df = pd.DataFrame({
        "rho": rhos,
        "va_mean": va_means,
        "va_std": va_stds,
        "discard_time": [r[3] for r in results]
    })
    df.to_csv("out/sweep_rho/sweep_rho_fixN_results.csv", index=False)

    plt.figure()
    plt.errorbar(rhos, va_means, yerr=va_stds, marker='o')
    plt.xlabel("ρ = N/L² (L fijo)")
    plt.ylabel("⟨va⟩ (estacionario)")
    plt.title("⟨va⟩ vs ρ (ventanas estacionarias elegidas manualmente)")
    plt.tight_layout()
    plt.show()


def plot_analysis(rhos, va_means, va_stds):
    plt.figure()
    plt.errorbar(rhos, va_means, yerr=va_stds, marker='o')
    plt.xlabel("ρ = N/L² (L fijo)")
    plt.ylabel("⟨va⟩ (estacionario)")
    plt.title("⟨va⟩ vs ρ (ventanas estacionarias elegidas manualmente)")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="⟨va⟩ vs ρ con L fijo (variar N).")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.2,0.5,1.0,2.0])
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--v0", type=float, default=0.03)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--rule", choices=["vicsek","voter"], default="vicsek")
    ap.add_argument("--discard", type=float, default=0.5)
    ap.add_argument("--analyze", action="store_true", help="Perform analysis on saved data")
    
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="Run the simulations and save the data")
    group.add_argument("--plot", action="store_true", help="Plot the data from a saved file")

    args = ap.parse_args()

    if not args.run and not args.plot and not args.analyze:
        ap.print_help()
        return

    if args.run:
        run_simulations(args)
    
    if args.plot:
        plot_data(args)

    if args.analyze:
        df = pd.read_csv("out/sweep_rho/sweep_rho_fixN_results.csv")
        plot_analysis(df["rho"], df["va_mean"], df["va_std"])


if __name__ == "__main__":
    main()
