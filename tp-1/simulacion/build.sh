#!/usr/bin/env bash
set -e
SRC=simulacion/src
OUT=simulacion/out
mkdir -p "$OUT"
javac -d "$OUT" $(find "$SRC" -name '*.java')
jar cfe simulacion/cim.jar cim.Main -C "$OUT" .
echo "✅ compilado -> simulacion/cim.jar"
