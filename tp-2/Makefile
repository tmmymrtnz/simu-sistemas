# Makefile para TP1 - Cell Index Method

# Variables
JAR=simulacion/flocking.jar
SRC=simulacion/src/flock
ANALYZE=analisis/animate.py ejemplo_vicsek --fps 25 
OUTDIR=out

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

plot:
	@echo "Graficando vecinos (neighbors)..."
	python3 analisis/plot_observables.py

clean:
	@echo "Limpiando archivos generados..."
	rm -rf $(OUTDIR)
	cd simulacion && rm -f flock/*.class flocking.jar