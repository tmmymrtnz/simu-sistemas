import pandas as pd
import numpy as np

# Leer y limpiar columnas
csv = "results.csv"
df = pd.read_csv(csv)
df.columns = df.columns.str.strip()
df = df.sort_values(by="N")
df = df.groupby("N").mean().reset_index()

N = df["N"].values
brute = df["Brute Force(ns)"].values
cim = df["CIM (ns)"].values

print("Pendientes entre puntos contiguos (escala log-log):\n")
print("Entre N[i] y N[i+1]:")
for i in range(len(N)-1):
    x1, x2 = N[i], N[i+1]
    y1b, y2b = brute[i], brute[i+1]
    y1c, y2c = cim[i], cim[i+1]
    slope_brute = (np.log(y2b) - np.log(y1b)) / (np.log(x2) - np.log(x1))
    slope_cim = (np.log(y2c) - np.log(y1c)) / (np.log(x2) - np.log(x1))
    print(f"N: {x1} -> {x2} | Brute: {slope_brute:.2f} | CIM: {slope_cim:.2f}")
