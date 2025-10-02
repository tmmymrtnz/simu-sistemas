#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "out", "simulation", "out.txt")

output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# ---- Read data ----
tiempos = []
pos_real = []
pos_beeman = []
pos_verlet = []
pos_gear = []

with open(data_path, "r") as f:
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

times = np.array(tiempos)
pos_real_np = np.array(pos_real)
pos_beeman_np = np.array(pos_beeman)
pos_verlet_np = np.array(pos_verlet)
pos_gear_np = np.array(pos_gear)

# ---- MSE ----
mse_beeman = np.mean((pos_real_np - pos_beeman_np)**2)
mse_verlet = np.mean((pos_real_np - pos_verlet_np)**2)
mse_gear = np.mean((pos_real_np - pos_gear_np)**2)

print("Error Cuadrático Medio:")
print(f"Beeman: {mse_beeman:.6e}")
print(f"Verlet Original: {mse_verlet:.6e}")
print(f"Gear Predictor-Corrector:   {mse_gear:.6e}")

# ---- Full time plot
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

# --- Zoomed-in plot
fig2, ax2 = plt.subplots(figsize=(8,4))
start_time = 3.1545
end_time = 3.1545 + 10e-5

# Find the indices for this specific interval
left_extrem_idx = np.searchsorted(times, start_time)
right_extrem_idx = np.searchsorted(times, end_time, side='right')
times_zoomed = times[left_extrem_idx:right_extrem_idx]
analytic_ds = pos_real_np[left_extrem_idx:right_extrem_idx]
beeman_ds = pos_beeman_np[left_extrem_idx:right_extrem_idx]
verlet_ds = pos_verlet_np[left_extrem_idx:right_extrem_idx]
gear_ds = pos_gear_np[left_extrem_idx:right_extrem_idx]

# Check if data was found
if len(times_zoomed) > 0:
    # Plotting the data
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
