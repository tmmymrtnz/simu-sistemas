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
AXIS_PERCENTILE ?=
SPEED_PERCENTILE ?=
ARGS ?=

ENERGY_N ?= 500
ENERGY_DTS ?= 1e-4 5e-4 1e-3 2e-3 3e-3
ENERGY_RUNS ?= 5
ENERGY_TF ?= 5
ENERGY_DT_OUTPUT ?=
ENERGY_SPEED ?= 0.1
ENERGY_SOFTENING ?= 0.05
ENERGY_EXTRA_ARGS :=$(if $(strip $(ENERGY_DT_OUTPUT)), --dt-output $(ENERGY_DT_OUTPUT),)

ENERGY_TS_N ?= 500
ENERGY_TS_RUN ?= 0
ENERGY_TS_DT ?= 1e-3
ENERGY_TS_DT_OUTPUT ?=
ENERGY_TS_TF ?= 5
ENERGY_TS_SPEED ?= 0.1
ENERGY_TS_SOFTENING ?= 0.05
ENERGY_TS_COLLISION ?=
ENERGY_TS_DX ?= 4.0
ENERGY_TS_DY ?= 0.5
ENERGY_TS_CLUSTER_SIZE ?=
ENERGY_TS_SEED ?=
ENERGY_TS_DT_ARG :=$(if $(strip $(ENERGY_TS_DT)), --dt $(ENERGY_TS_DT),)
ENERGY_TS_DT_OUTPUT_ARG :=$(if $(strip $(ENERGY_TS_DT_OUTPUT)), --dt-output $(ENERGY_TS_DT_OUTPUT),)
ENERGY_TS_COLLISION_ARG :=$(if $(strip $(ENERGY_TS_COLLISION)), --collision --dx $(ENERGY_TS_DX) --dy $(ENERGY_TS_DY),)
ENERGY_TS_CLUSTER_ARG :=$(if $(strip $(ENERGY_TS_CLUSTER_SIZE)), --cluster-size $(ENERGY_TS_CLUSTER_SIZE),)
ENERGY_TS_SEED_ARG :=$(if $(strip $(ENERGY_TS_SEED)), --seed $(ENERGY_TS_SEED),)
ENERGY_TS_EXTRA_ARGS :=$(strip $(ENERGY_TS_DT_ARG) $(ENERGY_TS_DT_OUTPUT_ARG) $(ENERGY_TS_COLLISION_ARG) $(ENERGY_TS_CLUSTER_ARG) $(ENERGY_TS_SEED_ARG))

RHM_TS_N ?= 500
RHM_TS_RUN ?= 0
RHM_TS_DT ?= 1e-3
RHM_TS_DT_OUTPUT ?=
RHM_TS_TF ?= 20
RHM_TS_SPEED ?= 0.1
RHM_TS_SOFTENING ?= 0.05
RHM_TS_COLLISION ?=
RHM_TS_DX ?= 4.0
RHM_TS_DY ?= 0.5
RHM_TS_CLUSTER_SIZE ?=
RHM_TS_SEED ?=
RHM_TS_DT_ARG :=$(if $(strip $(RHM_TS_DT)), --dt $(RHM_TS_DT),)
RHM_TS_DT_OUTPUT_ARG :=$(if $(strip $(RHM_TS_DT_OUTPUT)), --dt-output $(RHM_TS_DT_OUTPUT),)
RHM_TS_COLLISION_ARG :=$(if $(strip $(RHM_TS_COLLISION)), --collision --dx $(RHM_TS_DX) --dy $(RHM_TS_DY),)
RHM_TS_CLUSTER_ARG :=$(if $(strip $(RHM_TS_CLUSTER_SIZE)), --cluster-size $(RHM_TS_CLUSTER_SIZE),)
RHM_TS_SEED_ARG :=$(if $(strip $(RHM_TS_SEED)), --seed $(RHM_TS_SEED),)
RHM_TS_EXTRA_ARGS :=$(strip $(RHM_TS_DT_ARG) $(RHM_TS_DT_OUTPUT_ARG) $(RHM_TS_COLLISION_ARG) $(RHM_TS_CLUSTER_ARG) $(RHM_TS_SEED_ARG))

TCROSS_NS ?= 100 500 1000 1500 2000
TCROSS_RUNS ?= 10
TCROSS_DT ?= 1e-3
TCROSS_DT_OUTPUT ?=
TCROSS_TF ?= 20
TCROSS_SPEED ?= 0.1
TCROSS_SOFTENING ?= 0.05
TCROSS_EXTRA_ARGS :=$(if $(strip $(TCROSS_DT_OUTPUT)), --dt-output $(TCROSS_DT_OUTPUT),)

JAVA_SOURCES := $(wildcard $(SIM_DIR)/*.java)

.PHONY: all compile run run-galaxy analysis analysis-oscilator analysis-energy \
	analysis-rhm analysis-rhm-interactive analysis-tcross analysis-animation plot-rhm \
	analysis-energy-timeseries analysis-rhm-timeseries clean

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

analysis-energy-timeseries:
	@echo "Graficando energía vs tiempo para una corrida..."
	python3 $(ANALYSIS_DIR)/energy_timeseries.py \
		--root $(OUT_DIR)/galaxy/energy \
		--output $(ANALYSIS_DIR)/plots \
		--N $(ENERGY_TS_N) \
		--run $(ENERGY_TS_RUN) \
		--tf $(ENERGY_TS_TF) \
		--speed $(ENERGY_TS_SPEED) \
		--softening $(ENERGY_TS_SOFTENING) $(ENERGY_TS_EXTRA_ARGS)

analysis-rhm:
	@echo "Calculando estadísticas de r_hm..."
	python3 $(ANALYSIS_DIR)/rhm_stats.py --root $(OUT_DIR)/galaxy/energy --output $(ANALYSIS_DIR)/plots

analysis-rhm-interactive:
	@echo "Calculando estadísticas de r_hm (selección interactiva)..."
	python3 $(ANALYSIS_DIR)/rhm_stats.py --root $(OUT_DIR)/galaxy/energy --output $(ANALYSIS_DIR)/plots --interactive

analysis-rhm-timeseries:
	@echo "Graficando r_hm(t) para una corrida..."
	python3 $(ANALYSIS_DIR)/rhm_timeseries_single.py \
		--root $(OUT_DIR)/galaxy/energy \
		--output $(ANALYSIS_DIR)/plots \
		--N $(RHM_TS_N) \
		--run $(RHM_TS_RUN) \
		--tf $(RHM_TS_TF) \
		--speed $(RHM_TS_SPEED) \
		--softening $(RHM_TS_SOFTENING) $(RHM_TS_EXTRA_ARGS)

analysis-tcross:
	@echo "Calculando tiempos de cruce t*..."
	python3 $(ANALYSIS_DIR)/t_crossing.py \
		--root $(OUT_DIR)/galaxy/energy \
		--output $(ANALYSIS_DIR)/plots \
		--threshold $(THRESHOLD) \
		--Ns $(TCROSS_NS) \
		--runs $(TCROSS_RUNS)

analysis-animation:
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: especificá el archivo con FILE=..."; \
		exit 1; \
	fi
	python3 $(ANALYSIS_DIR)/cluster_animation.py $(FILE) --projection $(PROJECTION) $(if $(strip $(FPS)),--fps $(FPS),) $(if $(strip $(OUTPUT)),--output $(OUTPUT),) $(if $(strip $(AXIS_PERCENTILE)),--axis-percentile $(AXIS_PERCENTILE),) $(if $(strip $(SPEED_PERCENTILE)),--speed-percentile $(SPEED_PERCENTILE),)

plot-rhm:
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: especificá el archivo de energía con FILE=..."; \
		exit 1; \
	fi
	@mkdir -p tmp_mpl tmp_cache
	MPLCONFIGDIR=./tmp_mpl XDG_CACHE_HOME=./tmp_cache python3 $(ANALYSIS_DIR)/plot_rhm_from_energy.py $(FILE) $(if $(strip $(OUTPUT)),--output $(OUTPUT),)

animate-cm-radius:
	@if [ -z "$(SNAP)" ] || [ -z "$(ENERGY)" ]; then \
		echo "ERROR: usá SNAP=<archivo_state> y ENERGY=<archivo_energy>"; \
		exit 1; \
	fi
	@if [ ! -f "$(SNAP)" ]; then \
		echo "ERROR: no existe $(SNAP)"; \
		exit 1; \
	fi
	@if [ ! -f "$(ENERGY)" ]; then \
		echo "ERROR: no existe $(ENERGY)"; \
		exit 1; \
	fi
	@echo "Generando animación con radio de media masa..."
	MPLCONFIGDIR=./tmp_mpl XDG_CACHE_HOME=./tmp_cache python3 $(ANALYSIS_DIR)/animation_cm_radius.py \
		$(SNAP) \
		--energy-file $(ENERGY) \
		$(if $(strip $(OUTPUT)),--output $(OUTPUT),)


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
