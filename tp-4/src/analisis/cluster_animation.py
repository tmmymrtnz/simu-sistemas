#!/usr/bin/env python3
"""Animate galaxy simulations from *_state.txt files."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm


def parse_header_dt(path: Path) -> Optional[float]:
    dt_header: Optional[float] = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.startswith("# dt="):
                    tokens = line[1:].split()
                    for token in tokens:
                        if token.startswith("snapshot_dt="):
                            try:
                                dt_header = float(token.split("=", 1)[1])
                            except ValueError:
                                pass
                        elif token.startswith("dt="):
                            try:
                                dt_header = float(token.split("=", 1)[1])
                            except ValueError:
                                pass
    except FileNotFoundError:
        return None
    return dt_header


def read_frames(path: Path) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Optional[float]]:
    dt_header = parse_header_dt(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    times = np.unique(data[:, 0])
    times.sort()

    position_frames: List[np.ndarray] = []
    speed_frames: List[np.ndarray] = []
    for t in times:
        mask = data[:, 0] == t
        coords = data[mask, 2:5]
        velocities = data[mask, 5:8] if data.shape[1] >= 8 else np.zeros_like(coords)
        position_frames.append(coords)
        speed_frames.append(np.linalg.norm(velocities, axis=1))

    return times, position_frames, speed_frames, dt_header


def estimate_dt(times: np.ndarray, fallback: float = 0.04) -> float:
    if times.size <= 1:
        return fallback
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return fallback
    return float(np.median(positive))


def animate_collision(times: np.ndarray,
                      positions: List[np.ndarray],
                      speeds: List[np.ndarray],
                      *,
                      projection: str = "xy",
                      fps_override: Optional[float] = None,
                      dt_reference: Optional[float] = None,
                      axis_percentile: float = 95.0,
                      speed_percentile: float = 99.0,
                      output: Optional[Path] = None,
                      figsize: Tuple[float, float] = (6, 6),
                      cmap_name: str = "viridis") -> None:
    fig = plt.figure(figsize=figsize)
    if projection == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    all_positions = np.vstack(positions)
    radii = np.linalg.norm(all_positions, axis=1)
    if radii.size == 0:
        span = 1.0
    else:
        percentile = np.clip(axis_percentile, 0.0, 100.0)
        target = float(np.percentile(radii, percentile))
        span = target * 1.2 + 1e-6
        if span <= 0.0:
            span = float(np.max(radii)) * 1.2 + 1e-6 if radii.size else 1.0
    limits = (-span, span)

    all_speeds = np.concatenate(speeds)
    if all_speeds.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        speed_pct = np.clip(speed_percentile, 0.0, 100.0)
        vmax_candidate = float(np.percentile(all_speeds, speed_pct)) if speed_pct < 100.0 else float(np.max(all_speeds))
        vmax_exact = float(np.max(all_speeds))
        vmax = max(vmax_candidate, 1e-6)
        vmin = 0.0
        if vmax < vmax_exact and speed_pct < 100.0:
            vmax = max(vmax, vmax_exact * 0.9)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    if projection == "3d":
        scat = ax.scatter([], [], [], c=[], cmap=cmap, norm=norm, s=8)
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")
    else:
        scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=8)
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")
    fig.subplots_adjust(right=0.82)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    if projection == "3d":
        ax.set_zlim(*limits)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.84, 0.2, 0.04, 0.6])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("|v|")

    dt = dt_reference if dt_reference and dt_reference > 0 else estimate_dt(times)
    fps = fps_override if fps_override is not None else (1.0 / dt if dt > 0 else 25.0)
    interval_ms = dt * 1000.0

    def update(frame_idx: int):
        t = times[frame_idx]
        pts = positions[frame_idx]
        frame_speed = speeds[frame_idx]
        if projection == "3d":
            scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            colors = cmap(norm(frame_speed))
            scat.set_facecolor(colors)
            scat.set_edgecolor(colors)
        else:
            if projection == "xy":
                axes = (0, 1)
            elif projection == "xz":
                axes = (0, 2)
            else:
                axes = (1, 2)
            scat.set_offsets(pts[:, axes])
        scat.set_array(frame_speed)
        scat.set_norm(norm)
        scat.set_clim(vmin, vmax)
        time_text.set_text(f"t = {t:.3f}")
        return scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=interval_ms, blit=False)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg writer no disponible; instalalo o exportá como GIF cambiando --output")
        writer_cls = animation.writers["ffmpeg"]
        writer = writer_cls(fps=fps)
        ani.save(output, writer=writer)
        print(f"✅ Guardado {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar animaciones a partir de archivos *_state.txt")
    parser.add_argument("input", help="Archivo *_state.txt de la simulación")
    parser.add_argument("--projection", choices=["xy", "xz", "yz", "3d"], default="xy")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS del video (por defecto se usa 1/dt_output para tiempo real)")
    parser.add_argument("--output", help="Ruta del archivo MP4/GIF a guardar")
    parser.add_argument("--cmap", default="viridis", help="Mapa de colores para el módulo de la velocidad")
    parser.add_argument("--axis-percentile", type=float, default=95.0,
                        help="Percentil radial para fijar los ejes (0-100, default 95)")
    parser.add_argument("--speed-percentile", type=float, default=99.0,
                        help="Percentil para acotar la escala de velocidades (0-100, default 99)")
    args = parser.parse_args()

    path = Path(args.input).resolve()
    if not path.exists():
        raise SystemExit(f"No existe {path}")

    times, positions, speeds, dt_header = read_frames(path)
    inferred_dt = dt_header if dt_header is not None else estimate_dt(times)
    fps_override = args.fps if args.fps is not None else (1.0 / inferred_dt if inferred_dt > 0 else None)

    output_path = Path(args.output) if args.output else None
    animate_collision(
        times,
        positions,
        speeds,
        projection=args.projection,
        fps_override=fps_override,
        dt_reference=inferred_dt,
        axis_percentile=args.axis_percentile,
        speed_percentile=args.speed_percentile,
        output=output_path,
        cmap_name=args.cmap,
    )


if __name__ == "__main__":
    main()
