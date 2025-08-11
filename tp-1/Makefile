# Makefile para TP1 - Cell Index Method

# Variables
JAR=simulacion/cim.jar
SRC=simulacion/src
ANALYZE=analisis/analyze.py
OUTDIR=out

.PHONY: all build run analyze clean

all: build run analyze

build:
	@echo "Compilando Java..."
	cd simulacion && javac -d . src/cim/*.java && jar cfe cim.jar cim.Main cim/*.class

run:
	@echo "Ejecutando simulación (Java JAR)..."
	java -jar $(JAR) config.properties

analyze:
	@echo "Ejecutando análisis Python..."
	python3 $(ANALYZE)

plot:
	@echo "Graficando vecinos (neighbors)..."
	python3 analisis/plot_neighbors.py

clean:
	@echo "Limpiando archivos generados..."
	rm -rf $(OUTDIR)
	cd simulacion && rm -f cim/*.class cim.jar

# Ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  make build    - Compila el código Java y genera el JAR"
	@echo "  make run      - Ejecuta el JAR con config.properties"
	@echo "  make analyze  - Ejecuta el análisis en Python"
	@echo "  make all      - Compila, corre y analiza todo"
	@echo "  make clean    - Borra archivos generados y temporales"
