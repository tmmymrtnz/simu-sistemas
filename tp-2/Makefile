# Makefile para TP1 - Cell Index Method

# Variables
JAR=simulacion/flocking.jar
SRC=simulacion/src/flock
ANALYZE=analisis/animate.py ejemplo --fps 25  #default for all
ANALYZE_ANGLECOLOR=analisis/animate.py ejemplo --fps 25 --color-by-angle
OUTDIR=out
PLOT=analisis/plot_observables.py ejemplo
SWEEP_ETA=analisis/sweep_eta.py --L 20 --rho 1.0 --R 1.0 --etas 0 0.1 0.2 0.3 0.4 0.5 --steps 800 --reps 5 --rule vicsek --cleanup
SWEEP_RHO=analisis/sweep_rho_fixN.py --N 1000 --rhos 0.2 0.5 1.0 2.0 --eta 0.2 --R 1.0 --steps 800 --reps 5 --rule vicsek --cleanup


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
	python3 $(PLOT)
	cd simulacion && rm -f flock/*.class flocking.jar

sweep_eta:
	@echo "Ejecutando barrido de eta..."
	python3 $(SWEEP_ETA)

sweep_rho:
	@echo "Ejecutando barrido de rho..."
	python3 $(SWEEP_RHO)

clean:
	@echo "Limpiando archivos generados..."
	rm -rf $(OUTDIR)
