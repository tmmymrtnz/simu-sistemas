# Makefile for tp-4

JC = javac
JAVA = java

SRC_DIR := src
OUT_DIR := $(SRC_DIR)/out
SIM_DIR := $(SRC_DIR)/simulation
ANALYSIS_DIR := $(SRC_DIR)/analisis

THRESHOLD ?= 1.0
PROJECTION ?= xy
FPS ?= 30
OUTPUT ?=
ARGS ?=

ENERGY_N ?= 500
ENERGY_DTS ?= 1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2
ENERGY_RUNS ?= 5
ENERGY_TF ?= 5
ENERGY_DT_OUTPUT ?=
ENERGY_SPEED ?= 0.1
ENERGY_SOFTENING ?= 0.05
ENERGY_EXTRA_ARGS :=$(if $(strip $(ENERGY_DT_OUTPUT)), --dt-output $(ENERGY_DT_OUTPUT),)

JAVA_SOURCES := $(wildcard $(SIM_DIR)/*.java)

.PHONY: all compile run run-galaxy analysis analysis-oscilator analysis-energy \
	analysis-rhm analysis-rhm-interactive analysis-tcross analysis-animation plot-rhm clean

all: compile run

compile: $(JAVA_SOURCES)
	@echo "Compiling Java code..."
	@mkdir -p $(OUT_DIR)
	$(JC) -d $(OUT_DIR) $^

run: compile
	@echo "Running oscillator simulation..."
	$(JAVA) -cp $(OUT_DIR) simulation.Main

run-galaxy: compile
	@echo "Running galaxy simulation (System 2)..."
	$(JAVA) -cp $(OUT_DIR) simulation.GalaxyMain $(ARGS)

analysis: analysis-oscilator analysis-energy analysis-rhm analysis-tcross

analysis-oscilator:
	@echo "Postprocesando oscilador..."
	python3 $(ANALYSIS_DIR)/oscilator.py

analysis-energy:
	@echo "Estudiando conservación de energía..."
	python3 $(ANALYSIS_DIR)/energy_convergence.py \
		--root $(OUT_DIR)/galaxy/energy \
		--output $(ANALYSIS_DIR)/plots \
		--N $(ENERGY_N) \
		--dts $(ENERGY_DTS) \
		--runs $(ENERGY_RUNS) \
		--tf $(ENERGY_TF) \
		--speed $(ENERGY_SPEED) \
		--softening $(ENERGY_SOFTENING)$(ENERGY_EXTRA_ARGS)

analysis-rhm:
	@echo "Calculando estadísticas de r_hm..."
	python3 $(ANALYSIS_DIR)/rhm_stats.py --root $(OUT_DIR)/galaxy/energy --output $(ANALYSIS_DIR)/plots

analysis-rhm-interactive:
	@echo "Calculando estadísticas de r_hm (selección interactiva)..."
	python3 $(ANALYSIS_DIR)/rhm_stats.py --root $(OUT_DIR)/galaxy/energy --output $(ANALYSIS_DIR)/plots --interactive

analysis-tcross:
	@echo "Calculando tiempos de cruce t*..."
	python3 $(ANALYSIS_DIR)/t_crossing.py --root $(OUT_DIR)/galaxy/simulation --threshold $(THRESHOLD)

analysis-animation:
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: especificá el archivo con FILE=..."; \
		exit 1; \
	fi
	python3 $(ANALYSIS_DIR)/cluster_animation.py $(FILE) --projection $(PROJECTION) --fps $(FPS) $(if $(strip $(OUTPUT)),--output $(OUTPUT),)

plot-rhm:
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: especificá el archivo de energía con FILE=..."; \
		exit 1; \
	fi
	@mkdir -p tmp_mpl tmp_cache
	MPLCONFIGDIR=./tmp_mpl XDG_CACHE_HOME=./tmp_cache python3 $(ANALYSIS_DIR)/plot_rhm_from_energy.py $(FILE) $(if $(strip $(OUTPUT)),--output $(OUTPUT),)

clean:
	@echo "Cleaning up..."
	rm -rf $(OUT_DIR)
