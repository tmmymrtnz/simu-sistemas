# Makefile para TP1 - Cell Index Method

# Variables
JAR=simulacion/flocking.jar
SRC=simulacion/src/flock
ANALYZE=analisis/animate.py ejemplo1 --fps 25  #default for all
ANALYZE_ANGLECOLOR=analisis/animate.py ejemplo1 --fps 25 --color-by-angle
OUTDIR=out
ETAOUT=out/sweep_eta
RHOOUT=out/sweep_rho

PLOT=analisis/plot_observables.py ejemplo1 ejemplo2
SWEEP_ETA_RUN=analisis/sweep_eta.py --run --L 31.6 --rho 1.0 --R 1.0 --etas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 --steps 2000 --rule vicsek --discard-time 700 --cleanup
SWEEP_ETA_PLOT=analisis/sweep_eta.py --plot --L 31.6 --rho 1.0 --R 1.0 --etas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 --steps 2000  --rule vicsek  --discard-time 700 --cleanup
SWEEP_ETA_ANALYZE=analisis/sweep_eta.py --analyze --L 31.6 --rho 1.0 --R 1.0 --etas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 --steps 2000  --rule vicsek  --discard-time 700 --cleanup
SWEEP_RHO_RUN=analisis/sweep_rho_fixN.py --run --L 31.6 --rhos 0.2 0.4 0.6 0.8 1.0 1.2 1.5 2 3 4 5 6 7 8 9 10 --eta 2 --R 1.0 --steps 3000 --rule vicsek
SWEEP_RHO_PLOT=analisis/sweep_rho_fixN.py --plot --L 31.6 --rhos 0.2 0.4 0.6 0.8 1.0 1.2 1.5 2 3 4 5 6 7 8 9 10 --eta 2 --R 1.0 --steps 3000 --rule vicsek
SWEEP_RHO_ANALYZE=analisis/sweep_rho_fixN.py --analyze --L 31.6 --rhos 0.2 0.4 0.6 0.8 1.0 1.2 1.5 2 3 4 5 6 7 8 9 10 --eta 2 --R 1.0 --steps 3000 --rule vicsek

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

sweep_eta_run:
	@echo "Ejecutando barrido de eta..."
	python3 $(SWEEP_ETA_RUN)

sweep_eta_plot:
	python3 $(SWEEP_ETA_PLOT)

sweep_eta_analyze:
	python3 $(SWEEP_ETA_ANALYZE)

sweep_rho_run:
	@echo "Ejecutando barrido de rho..."
	python3 $(SWEEP_RHO_RUN)
	
sweep_rho_plot:
	python3 $(SWEEP_RHO_PLOT)

sweep_rho_analyze:
	python3 $(SWEEP_RHO_ANALYZE)

clean:
	@echo "Limpiando archivos generados..."
	rm -rf $(OUTDIR)
cleanrho:
	@echo "Limpiando archivos generados de sweep_rho..."
	rm -rf $(RHOOUT)

cleaneta:
	@echo "Limpiando archivos generados de sweep_eta..."
	rm -rf $(ETAOUT)