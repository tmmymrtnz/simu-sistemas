# Makefile para TP1 - Cell Index Method

# Variables
JAR=simulacion/flocking.jar
SRC=simulacion/src/flock
ANALYZE=analisis/animate.py ejemplo --fps 25  #default for all
ANALYZE_ANGLECOLOR=analisis/animate.py ejemplo --fps 25 --color-by-angle
OUTDIR=out
OBSERVABLES=analisis/plot_observables.py ejemplo --discard 0.5

.PHONY: all build run analyze clean

all: build run analyze

build:
	@echo "Compilando Java..."
	cd simulacion && javac -d . src/flock/*.java && jar cfe flocking.jar flock.Main flock/*.class

run:
	@echo "Ejecutando simulación (Java JAR)..."
	java -jar $(JAR) ./simulacion/flock.properties

analyze:
	@echo "Ejecutando análisis Python..."
	python3 $(ANALYZE)

analyze_anglecolor:
	@echo "Ejecutando análisis Python..."
	python3 $(ANALYZE_ANGLECOLOR)

plot:
	@echo "Graficando vecinos (neighbors)..."
	python3 $(OBSERVABLES)
	cd simulacion && rm -f flock/*.class flocking.jar