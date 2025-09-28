import matplotlib.pyplot as plt

# Cambia el nombre del archivo si es necesario
filename = 'out.txt'

# Listas para almacenar los datos
tiempos = []
pos_real = []
pos_beeman = []
pos_verlet = []

# Leer el archivo
with open(filename, 'r') as f:
    for line in f:
        if line.strip() == "" or line.strip().startswith('#') or line.strip().startswith('t'):
            continue
        partes = line.strip().split()
        if len(partes) < 4:
            continue
        try:
            t, x_beeman, x_verlet, x_real = map(float, partes[:4])
            tiempos.append(t)
            pos_real.append(x_real)
            pos_beeman.append(x_beeman)
            pos_verlet.append(x_verlet)
        except ValueError:
            continue

# Calcular error cuadrático medio
import numpy as np

pos_real_np = np.array(pos_real)
pos_beeman_np = np.array(pos_beeman)
pos_verlet_np = np.array(pos_verlet)

# Error cuadrático medio (MSE)
mse_beeman = np.mean((pos_real_np - pos_beeman_np)**2)
mse_verlet = np.mean((pos_real_np - pos_verlet_np)**2)

print(f"Error Cuadrático Medio:")
print(f"Beeman: {mse_beeman:.6e}")
print(f"Verlet: {mse_verlet:.6e}")

# Graficar
plt.plot(tiempos, pos_real, label='Real')
plt.plot(tiempos, pos_beeman, label=f'Beeman (MSE: {mse_beeman:.2e})')
plt.plot(tiempos, pos_verlet, label=f'Verlet (MSE: {mse_verlet:.2e})')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Posiciones vs Tiempo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()