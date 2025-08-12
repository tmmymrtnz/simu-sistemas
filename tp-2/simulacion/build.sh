#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OUT=flocking.jar

# Compilar
rm -rf target
mkdir -p target
find src -name "*.java" > sources.txt
javac -d target @sources.txt
rm sources.txt

# Empaquetar con clase principal
jar --create --file "$OUT" --main-class flock.Main -C target .

echo "✅ compilado -> simulacion/$OUT"
