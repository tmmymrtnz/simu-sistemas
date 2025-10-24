TP5 - Dinamica peatonal
=======================

Este directorio contiene el motor de simulacion y las herramientas de post-procesado para el TP5.

Estructura de codigo
--------------------

```
tp-5/
|-- Makefile                    # Atajos para compilar, simular y lanzar analisis/animacion
|-- src/
|   |-- tp5/simulacion/         # Codigo Java de la simulacion
|   |   |-- Main.java           # Punto de entrada con CLI parametrizable
|   |   |-- core/               # Agent, Vector2, SimulationState, ...
|   |   |-- cim/                # Cell Index Method adaptado
|   |   |-- engine/             # Integracion, deteccion de vecinos y contactos
|   |   |-- model/              # Interfaz comun + SFM2000 + AACPM completos
|   |   |-- output/             # Escritura de states.txt y contacts.txt
|   |   `-- setup/              # Generacion de condiciones iniciales
|   `-- analisis/               # Scripts de Python para cada punto del TP
|       |-- common.py           # Utilidades compartidas (IO, ensure_simulation, etc)
|       |-- punto_a.py          # Analisis de tasa de escaneo Q versus phi
|       |-- punto_b.py          # Analisis de tiempos entre contactos y ajuste de ley de potencias
|       `-- animacion.py        # Generacion de animaciones desde states.txt
`-- README.md                   # Este archivo
```

Formato de archivos de salida
-----------------------------

El motor escribe dos archivos de texto (valores separados por espacios):

* `states.txt`: `step time agentId x y vx vy ax ay radius`
* `contacts.txt`: `ordinal time agentId`

Las cabeceras (lineas con `#`) listan parametros de simulacion y metadata del modelo.
Los scripts de `src/analisis` consumen directamente estos archivos.

Flujo de trabajo con Makefile
-----------------------------

```
# Compilar la simulacion
make build

# Correr una simulacion (variables personalizables entre comillas si hay espacios)
make run N=60 SEED=42 STEPS=15000 DT=0.04 OUTPUT=output/experimento

# Analisis punto (a): barrer varios N y semillas
make analysis_a AGENTS_LIST="20 40 60" SEEDS_LIST="1 2 3" RESULTS_A=resultados/punto_a.csv

# Analisis punto (b): idem, con ajuste de ley de potencias
make analysis_b AGENTS_LIST="40 80" XMIN=0.1 RESULTS_B=resultados/punto_b.csv

# Generar animacion desde una simulacion existente (stride saltea frames)
make animate N=40 SEED=1 STRIDE=2 ANIMATION_OUTPUT=videos/caso40.mp4
```

Los scripts de Python verifican si existen `states.txt` y `contacts.txt` para los parametros solicitados.
Si faltan, lanzan automaticamente `make simulate` con los valores correspondientes antes de procesar.

Notas y pasos siguientes
------------------------

* Ajustar `SimulationFactory` para inicializar configuraciones con fraccion de area objetivo.
* Extender los scripts de analisis con graficos (matplotlib / seaborn) y reportes listos para la presentacion.
