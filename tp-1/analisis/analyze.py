#!/usr/bin/env python3
import subprocess, tempfile, pathlib, time, math, os, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

JAR = pathlib.Path(__file__).resolve().parents[1] / "simulacion/cim.jar"

def run_java(props: dict):
    """Corre el JAR y devuelve: out_dir, pos, neigh, ms y comparaciones (del CIM interno)."""
    props = props.copy()
    props.setdefault("frames", 1)
    base = props.get("outputBase", "bench_points")
    uniq = f"{base}_N{props['N']}_M{props['M']}_seed{props.get('seed','-')}_{int(time.time()*1e6)%1_000_000}"
    props["outputBase"] = uniq

    with tempfile.NamedTemporaryFile("w+", suffix=".properties", delete=False) as tf:
        for k, v in props.items():
            tf.write(f"{k}={v}\n")
        tf.flush()
        subprocess.run(["java", "-jar", str(JAR), tf.name],
                       stdout=subprocess.DEVNULL, check=True)

    out_dir = pathlib.Path("out") / uniq
    metr = pd.read_csv(out_dir / "metrics.csv")
    row = metr[metr["method"] == "CIM"].iloc[0]
    ms = float(row["ms"])
    comps = int(row["comparisons"])

    pos = pd.read_csv(out_dir / "particles.csv")
    neigh = {}
    with open(out_dir / "neighbours.txt") as fh:
        for ln in fh:
            pid, arr = ln.split(":", 1)
            arr = arr.strip()
            if arr.startswith("[") and arr.endswith("]"):
                items = [s.strip() for s in arr[1:-1].split(",") if s.strip()]
                neigh[int(pid)] = {int(x) for x in items}
            else:
                neigh[int(pid)] = set()
    return out_dir, pos, neigh, ms, comps

def diff_count(A: dict, B: dict) -> int:
    d=0
    for k in set(A)|set(B):
        a=A.get(k,set()); b=B.get(k,set())
        d += len(a-b) + len(b-a)
    return d

def ensure_dir(p: pathlib.Path):
    os.makedirs(p, exist_ok=True)

def select_three_M_punctual(L: float, rc: float):
    """Devuelve [M_bajo, M_medio, M_max] respetando M <= floor(L/rc)."""
    M_geom = max(1, int(math.floor(L / rc)))
    Ms = sorted(set([
        max(1, int(math.ceil(0.25 * M_geom))),
        max(1, int(math.ceil(0.50 * M_geom))),
        M_geom
    ]))
    return M_geom, Ms

def main():
    # --- Parámetros Parte 2: puntuales ---
    L, rc = 20.0, 1.0
    PERIODIC = True
    USE_RADII = False   # puntuales (centro-centro)
    REPS = 10

    M_geom, Ms = select_three_M_punctual(L, rc)  # tres Ms: bajo, medio, máximo
    Ns = [10, 200, 500, 1000, 2000]

    print(f"Criterio geométrico (puntuales): M ≤ {M_geom} (ℓ ≥ rc)")
    print(f"Usando M = {Ms} (bajo, medio, máximo)\n")

    all_rows = []
    created_dirs = set()

    for N in Ns:
        for M in Ms:
            for rep in range(REPS):
                seed = 4242 + rep  # misma semilla para CIM y BRUTE=Java(M=1)

                # --- CIM con M elegido ---
                out_cim, pos_cim, neigh_cim, cim_ms, cim_comps = run_java(dict(
                    N=N, L=L, M=M, rc=rc, useRadii=USE_RADII, periodic=PERIODIC,
                    outputBase="bench_points", seed=seed, frames=1
                ))
                created_dirs.add(out_cim)

                # --- BRUTE = correr Java con M=1 (mismo snapshot por seed) ---
                out_br, pos_br, neigh_br, brute_ms, brute_comps = run_java(dict(
                    N=N, L=L, M=1, rc=rc, useRadii=USE_RADII, periodic=PERIODIC,
                    outputBase="bench_points_brute", seed=seed, frames=1
                ))
                created_dirs.add(out_br)

                # chequeo: seeds iguales ⇒ posiciones deberían coincidir en ids
                mism = diff_count(neigh_cim, neigh_br)
                speed = brute_ms / cim_ms if cim_ms > 0 else np.nan
                all_rows.append((N, M, rep, cim_ms, brute_ms, cim_comps, brute_comps, speed, mism))

    df = pd.DataFrame(all_rows, columns=[
        "N","M","rep","CIM_ms","BRUTE_ms","CIM_comps","BRUTE_comps","speedup","mismatches"
    ])

    # ---- Agregación: promedio y desviación estándar por (N,M) ----
    agg_mean = df.groupby(["N","M"]).mean(numeric_only=True)
    agg_std  = df.groupby(["N","M"]).std(numeric_only=True).fillna(0.0)

    # Mostrar tablas
    print("== Parte 2: Puntuales (centro-centro) — Promedios ==")
    print("\nTiempos (ms, mean):")
    print(agg_mean.reset_index().pivot(index="N", columns="M", values=["CIM_ms","BRUTE_ms"]))
    print("\nTiempos (ms, std):")
    print(agg_std.reset_index().pivot(index="N", columns="M", values=["CIM_ms","BRUTE_ms"]))

    print("\nSpeedup (mean):")
    print(agg_mean.reset_index().pivot(index="N", columns="M", values="speedup"))
    print("\nSpeedup (std):")
    print(agg_std.reset_index().pivot(index="N", columns="M", values="speedup"))

    print("\nMismatches (mean):")
    print(agg_mean.reset_index().pivot(index="N", columns="M", values="mismatches"))

    # ---- Guardar CSVs de resumen ----
    out_summary = pathlib.Path("out/bench_points_summary")
    ensure_dir(out_summary)
    agg_mean.to_csv(out_summary / "summary_mean.csv")
    agg_std.to_csv(out_summary / "summary_std.csv")
    df.to_csv(out_summary / "raw_runs.csv", index=False)

    # ---- Gráficos con eje horizontal N (una línea por M) ----
    Ns_sorted = sorted(df["N"].unique())
    Ms_sorted = sorted(df["M"].unique())

    def plot_metric(metric: str, ylabel: str, fname: str):
        plt.figure()
        for M in Ms_sorted:
            y = [agg_mean.loc[(N,M), metric] for N in Ns_sorted]
            yerr = [agg_std.loc[(N,M), metric] for N in Ns_sorted]
            plt.errorbar(Ns_sorted, y, yerr=yerr, marker='o', label=f"M={M}")
        plt.xlabel("N (partículas)")
        plt.ylabel(ylabel)
        plt.title(f"{metric} vs N (puntuales, periodic={PERIODIC}, {REPS} reps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_summary / fname, dpi=150)

    plot_metric("CIM_ms", "Tiempo CIM (ms)", "cim_ms_vs_N.png")
    plot_metric("BRUTE_ms", "Tiempo BRUTE=Java(M=1) (ms)", "brute_ms_vs_N.png")
    plot_metric("speedup", "Speedup (BRUTE/CIM)", "speedup_vs_N.png")
    plot_metric("mismatches", "Mismatches (promedio)", "mismatches_vs_N.png")

    print(f"\n📈 Guardados gráficos y CSVs en: {out_summary}")

    # ---- Limpieza: borrar TODAS las carpetas out/ creadas por este script ----
    for d in sorted(created_dirs):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    print(f"🧹 Limpieza hecha: eliminadas {len(created_dirs)} carpetas temporales bajo out/")

if __name__ == "__main__":
    main()
