import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv("results.csv")

df.columns = df.columns.str.strip()  # Limpiar espacios en blanco en los nombres de columnas


# Agrupar por N y calcular media y desvío estándar
grouped = df.groupby("N")
df_mean = grouped.mean().reset_index()
df_std = grouped.std().reset_index()
# Ordenar por N (por si el archivo no está ordenado)
df_mean = df_mean.sort_values(by="N")
df_std = df_std.sort_values(by="N")




# Mostrar promedios y desvíos estándar
print("Promedios y desvíos estándar por N:")
for i, row in df_mean.iterrows():
	n = int(row["N"])
	brute_mean = row["Brute Force(ns)"]
	cim_mean = row["CIM (ns)"]
	brute_std = df_std.loc[df_std["N"] == n, "Brute Force(ns)"].values[0]
	cim_std = df_std.loc[df_std["N"] == n, "CIM (ns)"].values[0]
	print(f"N={n}: Brute={brute_mean:.0f} ± {brute_std:.0f}, CIM={cim_mean:.0f} ± {cim_std:.0f}")

# Transformación logarítmica para ajuste global
logN = np.log10(df_mean["N"])
logBF = np.log10(df_mean["Brute Force(ns)"])
logCIM = np.log10(df_mean["CIM (ns)"])

# Ajuste lineal en escala log-log -> pendiente ~ complejidad
slope_BF, intercept_BF = np.polyfit(logN, logBF, 1)
slope_CIM, intercept_CIM = np.polyfit(logN, logCIM, 1)

print(f"\nPendiente global Brute Force ≈ {slope_BF:.2f}")
print(f"Pendiente global CIM         ≈ {slope_CIM:.2f}")

# Pendientes entre puntos contiguos
print("\nPendientes log-log entre puntos contiguos:")
Nvals = df_mean["N"].values
brute = df_mean["Brute Force(ns)"].values
cim = df_mean["CIM (ns)"].values
for i in range(len(Nvals)-1):
	x1, x2 = Nvals[i], Nvals[i+1]
	y1b, y2b = brute[i], brute[i+1]
	y1c, y2c = cim[i], cim[i+1]
	slope_brute = (np.log(y2b) - np.log(y1b)) / (np.log(x2) - np.log(x1))
	slope_cim = (np.log(y2c) - np.log(y1c)) / (np.log(x2) - np.log(x1))
	print(f"N: {x1} -> {x2} | Brute: {slope_brute:.2f} | CIM: {slope_cim:.2f}")

# Graficar
plt.figure(figsize=(8,6))
plt.loglog(df_mean["N"], df_mean["Brute Force(ns)"], "o-", label=f"Brute Force (pendiente ≈ {slope_BF:.2f})")
plt.loglog(df_mean["N"], df_mean["CIM (ns)"], "s-", label=f"CIM (pendiente ≈ {slope_CIM:.2f})")

# Etiquetas
plt.xlabel("N (tamaño del sistema)")
plt.ylabel("Tiempo (ns)")
plt.title("Comparación Brute Force vs CIM (escala log-log)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.7)

plt.show()
