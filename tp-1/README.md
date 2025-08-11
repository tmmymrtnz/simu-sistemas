
# TP-1 — Guía rápida de uso y automatización
### Integrantes
- Matias Ignacio Luchetti
- Sofia Paula Altman Vogl
- Tomas Martinez

## Requisitos
- Java JDK 17+
- Python 3.9+ y pip (instalar: pandas, matplotlib, pillow)
- Opcional: ffmpeg si quieres guardar animaciones en MP4

## Configuración
- Edita `config.properties` en la raíz del repo para definir N, L, M, rc, r/rMin-rMax, periodic, seed, outputBase, frames, dt, vmax, compareBrute, etc.


## Compilar y correr todo (flujo recomendado)

### Opción rápida: todo en un solo comando
```sh
make all
```
Esto compila, corre la simulación, analiza benchmarks y deja todo listo para graficar vecinos o animar.

---

### Pasos individuales (si querés control manual):

1. **Compilar el simulador:**
   ```sh
   make build
   ```
   (o manualmente: `chmod +x simulacion/build.sh && ./simulacion/build.sh`)

2. **Correr la simulación:**
   ```sh
   make run
   ```
   Esto ejecuta el simulador con la configuración actual y deja los resultados en `out/<outputBase>/`.

3. **Analizar benchmarks (vs M y N):**
   ```sh
   make analyze
   ```
   Imprime tablas y genera gráficos comparativos en `out/bench_points_summary/`.

4. **Graficar vecinos de una partícula:**
   ```sh
   make plot_neighbors
   ```
   El script pedirá el id de la partícula por consola y usará el último output generado.
   - El gráfico dibuja todos los círculos con el radio real de cada partícula.
   - Solo cambia el color: gris (otras), naranja (vecinas), celeste (seleccionada).
   - Si todos los radios son iguales, todos los círculos se ven iguales.
   - Si hay radios distintos, se advierte por consola.

5. **Animar vecinos a lo largo de los frames:**
   ```sh
   make animate_neighbors
   ```
   (o manualmente: `python analisis/animate_neighbors.py <outputBase> --id <id> [--save out/<outputBase>/vecinos_id<id>.mp4|gif]`)
   - Requiere que `frames > 1` en config.properties.

## Notas y tips
- Todos los comandos importantes están en el Makefile: `make build`, `make run`, `make analyze`, `make plot_neighbors`, `make animate_neighbors`.
- Puedes editar Ns/Ms dentro de `analisis/analyze.py` para cambiar el barrido de parámetros.
- Corre primero la simulación para que existan los archivos en `out/<outputBase>/`.
- Para paredes no periódicas, pon `periodic=false` en config.properties.
- Si tu archivo .properties tiene comentarios en línea (N=500 # ...), elimínalos o usa el Main.java parcheado que los ignora.
