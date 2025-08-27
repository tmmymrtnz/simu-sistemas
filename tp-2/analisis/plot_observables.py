#!/usr/bin/env python3
import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot va(t) y promedio estacionario para uno o más archivos.")
    ap.add_argument("outBases", nargs="+", help="Nombre(s) de carpeta dentro de out/")
    args = ap.parse_args()

    plt.figure(figsize=(7,4))
    for outBase in args.outBases:
        base = pathlib.Path("out")/outBase
        obs = pd.read_csv(base/"observables.csv")
        
        discard_time = -1.0
        max_time = obs["t"].max()

        while True:
            try:
                user_input = input(f"[{outBase}] Enter the time (in seconds) to discard (e.g., {max_time*0.5:.2f}), or press Enter for no discard: ")
                if user_input == "":
                    discard_time = 0.0
                    print("Using a discard time of 0 seconds.")
                    break
                discard_time = float(user_input)
                if discard_time >= 0:
                    print(f"Using a discard time of {discard_time} seconds.")
                    break
                else:
                    print("Invalid input. Please enter a non-negative value.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                
        if discard_time > 0:
            cut = obs[obs["t"] >= discard_time].index[0]
        else:
            cut = 0

        sta = obs.iloc[cut:]
        va_mean, va_std = sta["va"].mean(), sta["va"].std()
        cut_time = obs["t"].iloc[cut]

        print(f"[{outBase}] Ventana estacionaria: t∈[{cut_time:.2f},{obs['t'].iloc[-1]:.2f}]  |  ⟨va⟩={va_mean:.4f}  σ={va_std:.4f}")

        plt.plot(obs["t"], obs["va"], lw=1.5, label=outBase)
        if cut > 0: 
            plt.axvspan(cut_time, obs["t"].iloc[-1], color='orange', alpha=0.2)
            plt.axvline(x=cut_time, color='red', linestyle='--', linewidth=2)
            plt.text(cut_time, 0.05, f'{cut_time:.0f}', 
                     ha='center', va='bottom', color='red', 
                     bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.8))

    plt.xlabel("t(s)")
    plt.ylabel("va")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()