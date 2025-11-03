SHELL := /bin/bash

JAVA_SRCDIR := src/tp5/simulacion
BUILD_DIR := build
PY := python3

MODEL := sfm
L := 12.0
RC := 0.5
DT := 0.0001
DT2 := 0.02
STEPS := 1500000
DURATION :=
MIN_RADIUS := 0.18
MAX_RADIUS := 0.21
VD := 1.7
RELAX := 0.5
SFM_A := 2000.0
SFM_B := 0.08
SFM_K := 120000.0
SFM_KAPPA := 240000.0
SFM_TAU := 0.5

AGENTS_LIST := 10 40 70 100 130 160 190 210 240 270 300 
SEEDS_LIST := 1 2 3 4 5 6 7 8 9 10

N := $(word 2,$(AGENTS_LIST))
SEED := $(word 1,$(SEEDS_LIST))

OUTPUT_DIR := output

ifeq ($(strip $(DURATION)),)
OUTPUT ?= $(OUTPUT_DIR)/$(MODEL)_n$(N)_seed$(SEED)_steps$(STEPS)
JAVA_TIME_ARG := --steps=$(STEPS)
else
OUTPUT ?= $(OUTPUT_DIR)/$(MODEL)_n$(N)_seed$(SEED)_dur$(DURATION)
JAVA_TIME_ARG := --duration=$(DURATION)
endif

JAVA_ARGS := \
	--model=$(MODEL) \
	--agents=$(N) \
	--domain=$(L) \
	--rc=$(RC) \
	--dt=$(DT) \
	--outputInterval=$(DT2) \
	$(JAVA_TIME_ARG) \
	--seed=$(SEED) \
	--minRadius=$(MIN_RADIUS) \
	--maxRadius=$(MAX_RADIUS) \
	--desiredSpeed=$(VD) \
	--relaxation=$(RELAX) \
	--sfmA=$(SFM_A) \
	--sfmB=$(SFM_B) \
	--sfmK=$(SFM_K) \
	--sfmKappa=$(SFM_KAPPA) \
	--sfmTau=$(SFM_TAU) \
	--aacpmKc=$(AACPM_KC) \
	--aacpmBeta=$(AACPM_BETA) \
	--aacpmK=$(AACPM_K) \
	--aacpmKappa=$(AACPM_KAPPA) \
	--aacpmKavo=$(AACPM_KAVO) \
	--aacpmTau=$(AACPM_TAU) \
	--aacpmDelta=$(AACPM_DELTA) \
	--aacpmOmega=$(AACPM_OMEGA) \
	--aacpmVdes=$(AACPM_VDES) \
	--aacpmVmax=$(AACPM_VMAX) \
	--aacpmAlpha=$(AACPM_ALPHA) \
	--output=$(OUTPUT)

JAVA_SOURCES := $(shell find $(JAVA_SRCDIR) -name '*.java')

XMIN ?=
STRIDE ?= 10
ANIMATION_OUTPUT ?=
RESULTS_A ?=
RESULTS_B ?=
SPEED_OUTPUT ?=

A_STORE_ARG :=
ifneq ($(strip $(RESULTS_A)),)
A_STORE_ARG := --store $(RESULTS_A)
endif

B_STORE_ARG :=
ifneq ($(strip $(RESULTS_B)),)
B_STORE_ARG := --store $(RESULTS_B)
endif

XMIN_ARG :=
ifneq ($(strip $(XMIN)),)
XMIN_ARG := --xmin $(XMIN)
endif

ANIMATION_SAVE_ARG :=
ifneq ($(strip $(ANIMATION_OUTPUT)),)
ANIMATION_SAVE_ARG := --save $(ANIMATION_OUTPUT)
endif

SPEED_OUTPUT_ARG :=
ifneq ($(strip $(SPEED_OUTPUT)),)
SPEED_OUTPUT_ARG := --speed-output $(SPEED_OUTPUT)
endif

ANALYSIS_COMMON_ARGS := \
	--model $(MODEL) \
	--agents $(N) \
	--domain $(L) \
	--rc $(RC) \
	--dt $(DT) \
	--output-interval $(DT2) \
	--seed $(SEED) \
	--min-radius $(MIN_RADIUS) \
	--max-radius $(MAX_RADIUS) \
	--desired-speed $(VD) \
	--relaxation $(RELAX) \
	--sfm-a $(SFM_A) \
	--sfm-b $(SFM_B) \
	--sfm-k $(SFM_K) \
	--sfm-kappa $(SFM_KAPPA) \
	--sfm-tau $(SFM_TAU) \

ifeq ($(strip $(DURATION)),)
ANALYSIS_COMMON_ARGS += --steps $(STEPS)
else
ANALYSIS_COMMON_ARGS += --duration $(DURATION)
endif

.PHONY: all build simulate run clean analysis_a analysis_b animate

all: build

build:
	@mkdir -p $(BUILD_DIR)
	javac -d $(BUILD_DIR) $(JAVA_SOURCES)

simulate: build
	@mkdir -p $(OUTPUT)
	java -cp $(BUILD_DIR) tp5.simulacion.Main $(JAVA_ARGS)

run: simulate

analysis_a:
	$(PY) src/analisis/punto_a.py \
		$(ANALYSIS_COMMON_ARGS) \
		--agents-list $(AGENTS_LIST) \
		--seeds $(SEEDS_LIST) $(SPEED_OUTPUT_ARG) $(A_STORE_ARG)

analysis_b:
	$(PY) src/analisis/punto_b.py \
		$(ANALYSIS_COMMON_ARGS) \
		--agents-list $(AGENTS_LIST) \
		--seeds $(SEEDS_LIST) $(SPEED_OUTPUT_ARG) $(XMIN_ARG) $(B_STORE_ARG)

animate:
	$(PY) src/analisis/animacion.py \
		$(ANALYSIS_COMMON_ARGS) \
		--stride $(STRIDE) $(ANIMATION_SAVE_ARG)

qmax_vs_v:
	$(PY) src/analisis/qmax_vs_v.py \
		$(ANALYSIS_COMMON_ARGS) \
		--v-list 1 2 3 4 5 6 7 8 9 10 \
		--seeds 1 2 3 4 5 \
		--agents-list 10 20 30 40 50 60 70 80 90 100 \
		--store $(OUTPUT_DIR)/qmax_vs_v.csv

clean:
	rm -rf $(BUILD_DIR)
