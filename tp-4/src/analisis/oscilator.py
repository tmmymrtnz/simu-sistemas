#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "out", "simulation", "out_full.txt")
zoomed_data_path = os.path.join(script_dir, "..", "out", "simulation", "out_1e-06.txt")
base_dir = os.path.join(script_dir, "..", "out", "simulation")


output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Function to read data and calculate MSE
def calculate_mse_and_positions(file_path):
    tiempos, pos_real, pos_beeman, pos_verlet, pos_gear = [], [], [], [], []
    try:
        with open(file_path, "r") as f:
            for line in f:
                s = line.strip()
                if s == "" or s.startswith("#") or s.startswith("t"):
                    continue
                parts = s.split()
                if len(parts) < 5:
                    continue
                try:
                    t, x_beeman, x_verlet, x_real, x_gear = map(float, parts[:5])
                    tiempos.append(t)
                    pos_real.append(x_real)
                    pos_beeman.append(x_beeman)
                    pos_verlet.append(x_verlet)
                    pos_gear.append(x_gear)
                except ValueError:
                    continue
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None

    if not pos_real:
        return None, None, None, None, None, None, None, None

    pos_real_np = np.array(pos_real)
    pos_beeman_np = np.array(pos_beeman)
    pos_verlet_np = np.array(pos_verlet)
    pos_gear_np = np.array(pos_gear)
    
    mse_beeman = np.mean((pos_real_np - pos_beeman_np)**2)
    mse_verlet = np.mean((pos_real_np - pos_verlet_np)**2)
    mse_gear = np.mean((pos_real_np - pos_gear_np)**2)
    
    return mse_beeman, mse_verlet, mse_gear, np.array(tiempos), pos_real_np, pos_beeman_np, pos_verlet_np, pos_gear_np

results = calculate_mse_and_positions(zoomed_data_path)

if results is not None:
    mse_beeman, mse_verlet, mse_gear, times, pos_real_np, pos_beeman_np, pos_verlet_np, pos_gear_np = results
    
    print("Error Cuadrático Medio para el archivo principal:")
    print(f"Beeman: {mse_beeman:.6e}")
    print(f"Verlet Original: {mse_verlet:.6e}")
    print(f"Gear Predictor-Corrector:   {mse_gear:.6e}")

    # ---- Full time plot ----
    fig, ax_main = plt.subplots(figsize=(14, 6))
    ax_main.plot(times, pos_real_np, label='Analytic', linewidth=2, linestyle='dotted', color='black') 
    ax_main.plot(times, pos_beeman_np, label='Beeman', linewidth=1.1, linestyle='--', color='steelblue')
    ax_main.plot(times, pos_verlet_np, label='Verlet Original', linewidth=1.1, linestyle='--', color='chocolate')
    ax_main.plot(times, pos_gear_np, label='Gear Predictor-Corrector', linewidth=0.8, linestyle='--', color='olivedrab')
    ax_main.set_xlabel("Tiempo (s)")
    ax_main.set_ylabel("Posición (m)")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    out_file = os.path.join(output_dir, "comparing_methods.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Saved figure to {out_file}")

    # ---- Zoomed-in plot ----
    fig2, ax2 = plt.subplots(figsize=(8,4))
    start_time = 3.1545
    end_time = 3.1545 + 10e-5
    left_extrem_idx = np.searchsorted(times, start_time)
    right_extrem_idx = np.searchsorted(times, end_time, side='right')
    times_zoomed = times[left_extrem_idx:right_extrem_idx]
    analytic_ds = pos_real_np[left_extrem_idx:right_extrem_idx]
    beeman_ds = pos_beeman_np[left_extrem_idx:right_extrem_idx]
    verlet_ds = pos_verlet_np[left_extrem_idx:right_extrem_idx]
    gear_ds = pos_gear_np[left_extrem_idx:right_extrem_idx]

    if len(times_zoomed) > 0:
        ax2.plot(times_zoomed, analytic_ds, color='black', linestyle='--', label="Analytic", linewidth=2)
        ax2.plot(times_zoomed, beeman_ds, color='steelblue', linestyle='--', label="Beeman")
        ax2.plot(times_zoomed, verlet_ds, color='chocolate', linestyle='--', label="Verlet Original")
        ax2.plot(times_zoomed, gear_ds, color='olivedrab', linestyle='--', label="Gear Predictor-Corrector")
        ax2.set_xlim(start_time, end_time)
        ax2.set_ylim(0.104847, 0.104853)
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel("Posición (m)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax2.xaxis.offsetText.set_visible(False)
        ax2.text(1.0, -0.1, f'+{start_time}', transform=ax2.transAxes, ha='right')
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.yaxis.offsetText.set_text(f'1e-6+1.048e-1')
        plt.tight_layout()
        out_file = os.path.join(output_dir, "zoomed_in_errors.png")
        plt.savefig(out_file, dpi=300)
        print(f"✅ Saved figure to {out_file}")
    else:
        print(f"❌ No data found in the time range from {start_time} to {end_time}. The sliced arrays are empty.")

# ---- New Plot: MSE vs dt (log-log scale) ----

# Data reading logic to calculate MSE for multiple dt values
dt_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
base_dir = os.path.join(script_dir, "..", "out", "simulation")

mse_beeman_dt = []
mse_verlet_dt = []
mse_gear_dt = []

for dt in dt_values:
    # Use f-string to get the correct scientific notation for the filename
    file_name = f"out_{dt:.0e}.txt"
    file_path = os.path.join(base_dir, file_name)
    
    # Use the existing function to calculate MSE
    results_dt = calculate_mse_and_positions(file_path)
    if results_dt is not None:
        mse_b, mse_v, mse_g, _, _, _, _, _ = results_dt
        mse_beeman_dt.append(mse_b)
        mse_verlet_dt.append(mse_v)
        mse_gear_dt.append(mse_g)
    else:
        print(f"⚠️ Warning: File not found or empty for dt={dt}. Skipping...")

if mse_beeman_dt:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.loglog(dt_values[:len(mse_beeman_dt)], mse_beeman_dt, marker='o', label='Beeman', color='steelblue')
    ax3.loglog(dt_values[:len(mse_verlet_dt)], mse_verlet_dt, marker='o', label='Verlet', color='chocolate')
    ax3.loglog(dt_values[:len(mse_gear_dt)], mse_gear_dt, marker='o', label='Gear', color='olivedrab')
    ax3.set_xlabel("dt (s)")
    ax3.set_ylabel("Error Cuadrático Medio")
    ax3.set_title("Precisión de los Algoritmos vs. Tamaño del Paso de Tiempo")
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    ax3.legend(title="Algoritmo")
    plt.tight_layout()
    out_file_mse_dt = os.path.join(output_dir, "mse_vs_dt_loglog.png")
    plt.savefig(out_file_mse_dt, dpi=300)
    print(f"✅ Saved figure to {out_file_mse_dt}")
else:
    print("❌ Not enough data to generate the MSE vs. dt plot. Please check your data files.")