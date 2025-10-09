#!/usr/bin/env python3
"""Animate snapshots highlighting whether particles lie inside/outside r_hm and
rendering the half-mass radius sphere around the centre of mass."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


@dataclass
class FrameState:
    time: float
    positions: np.ndarray
    center: np.ndarray
    distances: np.ndarray
    half_mass_radius: float


def load_energy_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} does not contain the r_hm column")
    times = data[:, 0]
    radii = data[:, 2]
    sort_idx = np.argsort(times)
    return times[sort_idx], radii[sort_idx]


def load_snapshot_rows(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(f"{path} does not contain particle state columns")
    return data


def build_frames(snapshot_rows: np.ndarray,
                 energy_times: np.ndarray,
                 half_mass_series: np.ndarray) -> List[FrameState]:
    unique_times = np.unique(snapshot_rows[:, 0])
    frames: List[FrameState] = []

    for time in unique_times:
        mask = np.isclose(snapshot_rows[:, 0], time)
        positions = snapshot_rows[mask, 2:5]
        center = positions.mean(axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        radius = float(np.interp(time, energy_times, half_mass_series))

        frames.append(FrameState(time=float(time),
                                 positions=positions,
                                 center=center,
                                 distances=distances,
                                 half_mass_radius=radius))

    if not frames:
        raise ValueError("No frames found in snapshot data")

    return frames


def compute_axis_limits(frames: List[FrameState],
                        radius_percentile: float,
                        axis_padding: float) -> Tuple[np.ndarray, np.ndarray]:
    all_distances = np.concatenate([frame.distances for frame in frames])
    if all_distances.size == 0:
        radius_limit = 1.0
    else:
        pct = float(np.clip(radius_percentile, 0.0, 100.0))
        radius_limit = float(np.percentile(all_distances, pct)) if pct > 0.0 else float(np.min(all_distances))
        max_radius = float(np.max(all_distances))
        if radius_limit <= 0.0:
            radius_limit = max_radius
        radius_limit = max(radius_limit, max_radius * 0.5, 1e-6)

    centers = np.stack([frame.center for frame in frames])
    mins = centers - radius_limit
    maxs = centers + radius_limit
    global_min = mins.min(axis=0)
    global_max = maxs.max(axis=0)
    span = global_max - global_min
    span[span <= 0.0] = 1.0
    padding = axis_padding * span
    return global_min - padding, global_max + padding


def clamp_axis_limits(axis_min: np.ndarray,
                      axis_max: np.ndarray,
                      axis_abs_limit: float) -> Tuple[np.ndarray, np.ndarray]:
    if axis_abs_limit > 0.0:
        axis_min = np.maximum(axis_min, -axis_abs_limit)
        axis_max = np.minimum(axis_max, axis_abs_limit)
        for idx in range(3):
            if axis_min[idx] >= axis_max[idx]:
                axis_min[idx] = -axis_abs_limit
                axis_max[idx] = axis_abs_limit
    return axis_min, axis_max


def setup_axes(ax,
               frames: List[FrameState],
               radius_percentile: float,
               axis_padding: float,
               axis_abs_limit: float) -> None:
    axis_min, axis_max = compute_axis_limits(frames, radius_percentile, axis_padding)
    axis_min, axis_max = clamp_axis_limits(axis_min, axis_max, axis_abs_limit)
    ax.set_xlim(axis_min[0], axis_max[0])
    ax.set_ylim(axis_min[1], axis_max[1])
    ax.set_zlim(axis_min[2], axis_max[2])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def create_unit_sphere(resolution: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, resolution)
    theta = np.linspace(0.0, np.pi, resolution // 2)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return x, y, z


def animate(frames: List[FrameState],
            *,
            output: Optional[Path],
            fps: int,
            inside_color: str,
            outside_color: str,
            sphere_color: str,
            sphere_alpha: float,
            radius_tolerance: float,
            radius_percentile: float,
            axis_padding: float,
            axis_abs_limit: float) -> None:
    inside_rgba = mcolors.to_rgba(inside_color)
    outside_rgba = mcolors.to_rgba(outside_color)
    sphere_rgba = mcolors.to_rgba(sphere_color)

    def apply_colors(scatter_obj, rgba: np.ndarray) -> None:
        scatter_obj.set_facecolor(rgba)
        scatter_obj.set_edgecolor(rgba)
        scatter_obj._facecolor3d = rgba
        scatter_obj._edgecolor3d = rgba
        scatter_obj._facecolors2d = rgba
        scatter_obj._edgecolors2d = rgba

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax, frames, radius_percentile, axis_padding, axis_abs_limit)

    sphere_x, sphere_y, sphere_z = create_unit_sphere()

    initial = frames[0]
    initial_mask = initial.distances <= initial.half_mass_radius + radius_tolerance
    initial_colors = np.empty((initial_mask.size, 4))
    initial_colors[initial_mask] = inside_rgba
    initial_colors[~initial_mask] = outside_rgba
    scatter = ax.scatter(initial.positions[:, 0],
                         initial.positions[:, 1],
                         initial.positions[:, 2],
                         c=initial_colors,
                         s=35,
                         depthshade=False)
    apply_colors(scatter, initial_colors)

    sphere_artist = ax.plot_surface(initial.center[0] + initial.half_mass_radius * sphere_x,
                                    initial.center[1] + initial.half_mass_radius * sphere_y,
                                    initial.center[2] + initial.half_mass_radius * sphere_z,
                                    color=sphere_rgba,
                                    alpha=sphere_alpha,
                                    linewidth=0.0,
                                    shade=False)

    title = ax.set_title(f"t = {initial.time:.3f} | r_hm = {initial.half_mass_radius:.3f}")

    legend_handles = [
        ax.scatter([], [], [], color=inside_color, label="r ≤ r_hm"),
        ax.scatter([], [], [], color=outside_color, label="r > r_hm"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    def update(frame_index: int):
        frame = frames[frame_index]
        positions = frame.positions
        mask = frame.distances <= frame.half_mass_radius + radius_tolerance
        colors_rgba = np.empty((mask.size, 4))
        colors_rgba[mask] = inside_rgba
        colors_rgba[~mask] = outside_rgba
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        apply_colors(scatter, colors_rgba)

        nonlocal sphere_artist
        sphere_artist.remove()
        sphere_artist = ax.plot_surface(frame.center[0] + frame.half_mass_radius * sphere_x,
                                        frame.center[1] + frame.half_mass_radius * sphere_y,
                                        frame.center[2] + frame.half_mass_radius * sphere_z,
                                        color=sphere_rgba,
                                        alpha=sphere_alpha,
                                        linewidth=0.0,
                                        shade=False)
        title.set_text(f"t = {frame.time:.3f} | r_hm = {frame.half_mass_radius:.3f}")
        return scatter, sphere_artist, title

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=len(frames),
                                  interval=1000 / fps,
                                  blit=False,
                                  repeat=False)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        ani.save(output, writer="ffmpeg", fps=fps)
        print(f"✅ Guardado {output}")
    else:
        plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Animate snapshots with r_hm sphere and distance colouring.")
    parser.add_argument("snapshot_file", help="Archivo *_state.txt generado con --dump-snapshots.")
    parser.add_argument("--energy-file", required=True, help="Archivo *_energy.txt correspondiente.")
    parser.add_argument("--output", help="Archivo MP4 para guardar la animación.")
    parser.add_argument("--fps", type=int, default=30, help="Frames por segundo del video (default: 30).")
    parser.add_argument("--inside-color", default="tab:blue", help="Color para partículas con r ≤ r_hm.")
    parser.add_argument("--outside-color", default="tab:orange", help="Color para partículas con r > r_hm.")
    parser.add_argument("--sphere-color", default="silver", help="Color de la esfera de radio r_hm.")
    parser.add_argument("--sphere-alpha", type=float, default=0.2, help="Transparencia de la esfera (0-1).")
    parser.add_argument("--radius-tolerance", type=float, default=1e-9,
                        help="Tolerancia para clasificar partículas en el borde de r_hm.")
    parser.add_argument("--radius-percentile", type=float, default=99.0,
                        help="Percentil de distancias usado para dimensionar los ejes (0-100).")
    parser.add_argument("--axis-padding", type=float, default=0.15,
                        help="Padding fraccional aplicado al rango de los ejes.")
    parser.add_argument("--axis-abs-limit", type=float, default=20.0,
                        help="Valor máximo absoluto permitido para cada eje (default: 20). Usa <=0 para desactivar.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    snapshot_path = Path(args.snapshot_file).resolve()
    energy_path = Path(args.energy_file).resolve()
    if not snapshot_path.exists():
        raise SystemExit(f"No existe {snapshot_path}")
    if not energy_path.exists():
        raise SystemExit(f"No existe {energy_path}")

    energy_times, energy_radii = load_energy_series(energy_path)
    snapshot_rows = load_snapshot_rows(snapshot_path)
    frames = build_frames(snapshot_rows, energy_times, energy_radii)

    output_path = Path(args.output).resolve() if args.output else None
    animate(frames,
            output=output_path,
            fps=max(args.fps, 1),
            inside_color=args.inside_color,
            outside_color=args.outside_color,
            sphere_color=args.sphere_color,
            sphere_alpha=np.clip(args.sphere_alpha, 0.0, 1.0),
            radius_tolerance=max(args.radius_tolerance, 0.0),
            radius_percentile=args.radius_percentile,
            axis_padding=max(args.axis_padding, 0.0),
            axis_abs_limit=max(args.axis_abs_limit, 0.0))


if __name__ == "__main__":
    main()
