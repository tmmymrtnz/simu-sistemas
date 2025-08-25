Proyecto: Autómata Off-Lattice (Vicsek / Voter)
================================================

Requisitos
----------
- Java 11+ (recomendado 17+)
- Python 3.9+ con: numpy, pandas, matplotlib
  Sugerencia rápida:  python -m pip install numpy pandas matplotlib

Estructura relevante
--------------------
simulacion/
  build.sh                 -> script de compilación
  flock.properties         -> parámetros de ejemplo
  src/flock/*.java         -> código Java (Vicsek/Voter + grilla)
analisis/
  animate.py               -> animación (quiver) desde archivos de texto
  plot_observables.py      -> va(t) y promedio estacionario
  sweep_eta.py             -> barrido de η con ρ fija (N = ρ L²)
  sweep_rho_fixN.py        -> barrido de ρ con N fijo (L = √(N/ρ))
out/
  <outputBase>/            -> resultados de cada corrida (se crea al ejecutar)

Cómo compilar la simulación (Java)
----------------------------------
1) Entrar a la carpeta:
   cd simulacion

2) Compilar:
   ./build.sh

   Esto genera: simulacion/flocking.jar

Cómo correr una simulación
--------------------------
Opción A) Usando el archivo de propiedades por defecto:
   java -jar simulacion/flocking.jar simulacion/flock.properties

Opción B) Indicando otro .properties (ruta propia):
   java -jar simulacion/flocking.jar /ruta/a/tu.properties

El archivo .properties acepta estas claves:
  N           (int)   cantidad de agentes
  L           (double) lado del cuadrado
  R           (double) radio de interacción (centro-centro)
  v0          (double) módulo de la velocidad
  dt          (double) paso de tiempo
  steps       (int)   cantidad de pasos a simular
  eta         (double) amplitud de ruido
  periodic    (true|false) contorno periódico
  M           (int, opcional) celdas por lado; si se omite usa floor(L/R)
  includeSelf (true|false) incluirse a sí mismo en el promedio (Vicsek)
  rule        ("vicsek" | "voter") regla de actualización
  seed        (long) semilla RNG
  outputBase  (string) nombre de carpeta bajo ./out/

Salidas de la simulación
------------------------
Se escriben en: out/<outputBase>/
  static.txt            parámetros de la corrida
  observables.csv       columnas: t,va
  t0000.txt, t0001.txt, ... (por cada paso)
    Formato por fila: id x y vx vy theta
  last.txt              alias del último frame

Animación (Python)
------------------
Animar con vectores velocidad (opcional color por ángulo):
  ./analisis/animate.py <outputBase> --fps 25 --color-by-angle

Ejemplo:
  ./analisis/animate.py ejemplo_vicsek --fps 25 --color-by-angle

Evolución temporal de va (y promedio estacionario)
--------------------------------------------------
  ./analisis/plot_observables.py <outputBase> --discard 0.5

El flag --discard indica la fracción inicial (transiente) a descartar antes de promediar.

Barrido de η (ruido) a densidad fija ρ
--------------------------------------
ρ = N / L²  (se fija L y se calcula N redondeando)

Ejemplo:
  ./analisis/sweep_eta.py --L 20 --rho 1.0 --R 1.0 --etas 0 0.1 0.2 0.3 0.4 0.5 \
                          --steps 800 --reps 5 --rule vicsek --cleanup

- Genera múltiples corridas en out/, calcula ⟨va⟩ (promedio estacionario)
  y muestra ⟨va⟩ vs η con barras de error (SEM).
- --cleanup borra las carpetas de corridas creadas al terminar.

Barrido de ρ con N fijo (variando L)
------------------------------------
Se fija N y se usa L = √(N/ρ) para cada ρ.

Ejemplo:
  ./analisis/sweep_rho_fixN.py --N 1000 --rhos 0.2 0.5 1.0 2.0 \
                               --eta 0.2 --R 1.0 --steps 800 --reps 5 --rule vicsek --cleanup

- Guarda raw_runs.csv y summary.csv en out/sweep_rho_fixN_summary/

Notas
-----
- La animación lee archivos de texto generados por la simulación, así que su velocidad es independiente de la simulación (requisito de la consigna).
- El motor de vecindad usa una grilla de celdas M×M y contorno periódico. Si no indicás M, se selecciona automáticamente M = floor(L/R).
- Si querés experimentar con partículas con radio (borde-a-borde), Agent y Grid ya tienen soporte, pero el Main por defecto usa puntuales (centro-centro).


CÓMO CORRER (TP-2 – Bandadas de agentes autopropulsados)

[PREVIO] Compilar
./simulacion/build.sh

Configuración
Editar simulacion/flock.properties (parámetros clave: rule=vicsek|voter, N, L, R, v0, dt, steps, eta, periodic, outputBase, M opcional).

a) Animaciones (vectores con origen en la posición)

Correr una simulación (ejemplo Vicsek):
java -jar simulacion/flocking.jar simulacion/flock.properties

Animación básica (puntos + flecha corta):
./analisis/animate.py <outBase> --dot-size 90 --arrow-len 0.3 --save video.mp4

Animación coloreada por ángulo:
./analisis/animate.py <outBase> --color-by-angle --dot-size 90 --arrow-len 0.3 --save video_color.mp4

Voter: cambiar rule=voter y outputBase en .properties, volver a correr la simulación y animar:
java -jar simulacion/flocking.jar simulacion/flock.properties
./analisis/animate.py <outBase_voter> --dot-size 90 --arrow-len 0.3 --save voter.mp4
./analisis/animate.py <outBase_voter> --color-by-angle --dot-size 90 --arrow-len 0.3 --save voter_color.mp4

b) Evolución temporal del observable (polarización va(t))
Muestra va(t) y promedia en ventana estacionaria (por ej., último 50%):
./analisis/plot_observables.py <outBase> --discard 0.5
(usar <outBase_voter> para la regla voter si se desea)

c) Curvas Input vs Observable (⟨va⟩ con barras de error)
c1) ⟨va⟩ vs η a densidad fija ρ (N=ρL^2):
./analisis/sweep_eta.py --L 20 --rho 1.0 --R 1.0 --etas 0 1.0 2.0 3.0 4.0 5.0 --steps 800 --reps 10 --rule vicsek --cleanup
Resultados: out/sweep_eta_summary/
c2) ⟨va⟩ vs ρ con N fijo (varía L=√(N/ρ)):
./analisis/sweep_rho_fixN.py --N 1000 --rhos  0.2 2.0 4.0 6.0 8.0 10.0 --eta 0.2 --R 1.0 --steps 800 --reps 10 --rule vicsek --cleanup
Resultados: out/sweep_rho_fixN_summary/

d) Interacción tipo Voter (repetir c) con voter
Barrido en η (ρ fija):
./analisis/sweep_eta.py --L 20 --rho 1.0 --R 1.0 --etas 0 1.0 2.0 3.0 4.0 5.0 --steps 800 --reps 10 --rule voter --cleanup
Barrido en ρ (N fijo):
./analisis/sweep_rho_fixN.py --N 1000 --rhos 0.2 2.0 4.0 6.0 8.0 10.0 --eta 0.2 --R 1.0 --steps 800 --reps 10 --rule voter --cleanup

e) (Opcional) Tiempo a consenso (umbral va ≥ 0.95, una corrida)
python - <<'PY'
import pandas as pd
obs = pd.read_csv('out/<outBase>/observables.csv')
thr=0.95
r=obs.loc[obs['va']>=thr]
print("t_consenso =", int(r['t'].iloc[0]) if len(r) else "no alcanzado")
PY

Notas
– <outBase> es el nombre de carpeta que definiste en outputBase dentro de simulacion/flock.properties.
– Si “ffmpeg no encontrado” al guardar MP4, usá --save *.gif o instalá ffmpeg (macOS: brew install ffmpeg).
– Flags útiles en animate.py: --no-dots (solo flechas), --no-arrows (solo puntos), --arrow-width para ajustar grosor de flecha.