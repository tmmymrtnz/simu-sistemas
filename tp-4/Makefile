# Makefile for tp-4

JC = javac
JAVA = java

SRC_DIR := src
OUT_DIR := $(SRC_DIR)/out
SIM_DIR := $(SRC_DIR)/simulation
ANALYSIS_DIR := $(SRC_DIR)/analisis

THRESHOLD ?= 1.0
PROJECTION ?= xy
FPS = 60 
SINGLE_CLUSTER_N = 2000
DUAL_CLUSTER_N = 100
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
	analysis-rhm analysis-rhm-interactive analysis-tcross analysis-animation plot-rhm clean bodies-animation

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

# New Rule: Runs a single, Gaussian cluster simulation (N=2000)
run-gaussian-2000: compile
	@echo "Running single Gaussian cluster simulation (N=$(SINGLE_CLUSTER_N))..."
	$(JAVA) -cp $(OUT_DIR) simulation.GalaxyMain \
	--N $(SINGLE_CLUSTER_N) \
	--dt 0.002 \
	--tf 10.0 \
	--h 0.05 \
	--speed 0.1 \
	--dt-output 0.02
# Output filename example: src/out/galaxy/particles/gaussian_N100_run000_particles.txt

# New Rule: Runs the large 2-cluster collision simulation (N=2000)
run-collision-100: compile
	@echo "Running large 2-cluster collision simulation (N=$(DUAL_CLUSTER_N), 50 each)..."
	$(JAVA) -cp $(OUT_DIR) simulation.GalaxyMain \
	--dt 0.002 \
	--tf 10.0 \
	--h 0.05 \
	--speed 0.1 \
	--dt-output 0.02 \
	--collision \
	--cluster-size 50 \
	--dx 8.0 \
	--dy 1.0 


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


## 1-cluster, N=2000
animate-gaussian-2000:
	@FILE=$(SRC_DIR)/out/galaxy/particles/gaussian_N$(SINGLE_CLUSTER_N)_run000_particles.txt; \
	if [ ! -f "$$FILE" ]; then \
		echo "ERROR: Run 'make run-gaussian-2000' first to generate $$FILE"; \
		exit 1; \
	fi; \
	echo "Creating single-cluster animation from $$FILE..."; \
	python3 src/analisis/animation.py \
		$$FILE \
		--output $(SRC_DIR)/analisis/plots/animation_gaussian_2000.mp4 \
		--fps $(FPS) \
		--cluster-size $(SINGLE_CLUSTER_N) # Cluster size = Total N ensures single group plot

# 2-cluster, N=100
animate-collision-100:
	@FILE=$(SRC_DIR)/out/galaxy/particles/cluster_$(DUAL_CLUSTER_N)_run000_particles.txt; \
	if [ ! -f "$$FILE" ]; then \
		echo "ERROR: Run 'make run-collision-100' first to generate $$FILE"; \
		exit 1; \
	fi; \
	echo "Creating two-cluster animation from $$FILE..."; \
	python3 src/analisis/animation.py \
		$$FILE \
		--output $(SRC_DIR)/analisis/plots/animation_collision_100.mp4 \
		--fps $(FPS) \
		--cluster-size 50 # Cluster size = N/2 ensures two distinct groups plot

clean:
	@echo "Cleaning up..."
	rm -rf $(OUT_DIR)
