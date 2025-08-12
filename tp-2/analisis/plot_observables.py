#!/usr/bin/env python3
import argparse, pathlib, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot va(t) y promedio estacionario.")
    ap.add_argument("outBase")
    ap.add_argument("--discard", type=float, default=0.5, help="fracción inicial a descartar")
    args = ap.parse_args()

    base = pathlib.Path("out")/args.outBase
    obs = pd.read_csv(base/"observables.csv")
    T = len(obs); cut = int(args.discard*(T-1))
    sta = obs.iloc[cut:]
    va_mean, va_std = sta["va"].mean(), sta["va"].std()

    print(f"Ventana estacionaria: t∈[{cut},{T-1}]  |  ⟨va⟩={va_mean:.4f}  σ={va_std:.4f}")

    plt.figure(figsize=(7,4))
    plt.plot(obs["t"], obs["va"], lw=1.5, label="va(t)")
    plt.axvspan(cut, obs["t"].iloc[-1], color='orange', alpha=0.2, label='ventana promedio')
    plt.xlabel("t"); plt.ylabel("va")
    plt.title(f"va(t) — ⟨va⟩={va_mean:.3f} ± {va_std:.3f}")
    plt.legend(); plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
