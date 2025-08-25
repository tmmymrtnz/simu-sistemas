#!/usr/bin/env python3
import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot va(t) y promedio estacionario.")
    ap.add_argument("outBase")
    args = ap.parse_args()

    discard_value = 0.0  
    while True:
        try:
            user_input = input("Enter the fraction to discard (e.g., 0.5), or enter if not desired: ")
            if user_input == "":
                print("Using default discard fraction of 0.0")
                break
            
            discard_value = float(user_input)

            if 0.0 < float(discard_value) < 1:
                print(f"Using a discard fraction of {discard_value}")
                break
            else:
                print("Invalid input. Please enter a value between greater than 0.0 and less than 1.0.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    base = pathlib.Path("out")/args.outBase
    obs = pd.read_csv(base/"observables.csv")
    T = len(obs)
    
    # Calculate the cut-off index based on the user's input
    aux = int(discard_value * (T - 1))
    
    if aux != 0:
        cut = aux
        sta = obs.iloc[cut:]
    else:
        sta = obs
        cut = 0

    va_mean, va_std = sta["va"].mean(), sta["va"].std()

    print(f"Ventana estacionaria: t∈[{obs['t'].iloc[cut]:.2f},{obs['t'].iloc[-1]:.2f}]  |  ⟨va⟩={va_mean:.4f}  σ={va_std:.4f}")

    plt.figure(figsize=(7,4))
    plt.plot(obs["t"], obs["va"], lw=1.5, label="va(t)")
    
    if aux > 0: 
        plt.axvspan(obs["t"].iloc[aux], obs["t"].iloc[-1], color='orange', alpha=0.2, label='ventana promedio')

    plt.xlabel("t"); plt.ylabel("va")
    plt.title(f"va(t) — ⟨va⟩={va_mean:.3f} ± {va_std:.3f}")
    plt.legend(); plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
