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
	$(JAVA) -cp src/out simulation.Main 201 0.09 

animate:
	@echo "Animating simulation..."
	python3 src/analisis/animate_sim.py --N 201 --L 0.03 --fps 100 --use-file out/events_L=0.090_N=201.txt 

total_pressure_plot:
	@echo "Plotting total pressure over time..."
	python3 src/analisis/total_pressure_over_time.py --use-files out/pressure_L=0.030_N=201.txt out/pressure_L=0.090_N=201.txt --show

pressure_plot:
	@echo "Plotting pressure data..."
	python3 src/analisis/plot_pressure.py --Ns 201 --Ls 0.090 --show --use-file out/pressure_L=0.090_N=201.txt

pressure_fit:
	@echo "Fitting pressure data..."
	python3 src/analisis/fit_pressure_vs_area.py --N 201 --Ls 0.030 0.050 0.070 0.090 --show

diffusion_plot:
	@echo "Plotting diffusion data (requires events_L=0.090_N=201.txt)..."
	python3 src/analisis/calculate_diffusion.py --file out/events_L=0.090_N=201.txt --show --t-start 0.0

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
	rm -rf src/sim.jar

only-plot:
	@echo "Only plotting pressure data from existing files..."
	python3 src/analisis/plot_pressure.py --Ns 300 --Ls 0.030 0.050 0.070 0.090 --use-file out/pressure_L=0.090_N=200.txt --show