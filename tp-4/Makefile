# Makefile for tp-4

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
	$(JAVA) -cp src/out simulation.Main 

analysis:
	@echo "Running analysis scripts..."
	python3 src/analisis/oscilator.py

clean:
	@echo "Cleaning up..."
	rm -rf src/out
