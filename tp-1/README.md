TP-1 — HOW TO RUN EACH FILE (plain text)

Prerequisites:
- Java JDK 17+
- Python 3.9+ and pip (install: pandas, matplotlib, pillow if saving GIF)
- Optional: ffmpeg if saving MP4

Config file:
- Edit config.properties at repo root (N, L, M, rc, r/rMin-rMax, periodic, seed, outputBase, frames, dt, vmax, compareBrute)

----------------------------------------------------------------
Build the simulator (creates simulacion/cim.jar)
----------------------------------------------------------------
chmod +x simulacion/build.sh
./simulacion/build.sh

----------------------------------------------------------------
Run the simulator (reads config.properties by default)
----------------------------------------------------------------
java -jar simulacion/cim.jar
# or pass another .properties file:
java -jar simulacion/cim.jar path/to/other.properties

# Outputs are written to: out/<outputBase>/
# - particles.csv / neighbours.txt (last frame)
# - particles_t####.csv / neighbours_t####.txt (all frames)
# - metrics.csv (timings and comparisons)

----------------------------------------------------------------
Plot neighbors for a single snapshot
----------------------------------------------------------------
# Basic usage (two args):
python analisis/plot_neighbors.py <outputBase> <particleId>
# Example:
python analisis/plot_neighbors.py run 17

# If you’re using the variant that reads defaults from config:
python analisis/plot_neighbors.py                # uses outputBase from config.properties, particleId=0
python analisis/plot_neighbors.py --out run --id 17

----------------------------------------------------------------
Benchmark vs M and N (prints timing table)
----------------------------------------------------------------
python analisis/analyze.py

# (Optionally edit Ns/Ms inside the script to change the sweep)

----------------------------------------------------------------
Make an animation across frames
----------------------------------------------------------------
# Requires frames>1 and (optionally) vmax>0 in config.properties
python analisis/animate_neighbors.py <outputBase> --id 17

# Save to file:
python analisis/animate_neighbors.py <outputBase> --id 17 --save out/<outputBase>/vecinos_id17.mp4
python analisis/animate_neighbors.py <outputBase> --id 17 --save out/<outputBase>/vecinos_id17.gif

Notes:
- Ensure you run the simulator first so the out/<outputBase>/ files exist.
- For non-periodic walls set periodic=false in config.properties.
- If your .properties file had inline comments (e.g., N=500 # ...), remove them or use the patched Main.java that strips inline comments.
