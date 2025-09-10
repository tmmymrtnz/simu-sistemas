# Makefile for tp-3

JC = javac
JAR = jar
JAVA = java

all: compile run

compile:
	@echo "Compiling Java code..."
	@mkdir -p src/out
	$(JC) -d src/out src/simulation/*.java

run: compile
	@echo "Running Java simulation..."
	$(JAVA) -cp src/out simulation.Main 12 0.05 # Example N=12, L=0.05

animate:
	@echo "Animating simulation..."
	python3 src/analisis/animate_sim.py --N 300 --L 0.09 --fps 100 --out anim.mp4

pressure_plot:
	@echo "Plotting pressure data..."
	python3 src/analisis/plot_pressure.py --Ns 300 --Ls 0.030 0.050 0.070 0.090 --show

pressure_fit:
	@echo "Fitting pressure data..."
	python3 src/analisis/fit_pressure_vs_area.py --N 300 --Ls 0.030 0.050 0.070 0.090 --reps 3 --last-n 5 --show

clean:
	@echo "Cleaning generated files..."
	rm -rf src/out
	rm -rf out
	rm -rf src/events_L*
	rm -rf src/pressure_L*
	rm -rf anim.mp4

clean_all: 
	make clean
	rm -rf src/simulation/*.class
