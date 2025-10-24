from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import (
    PROJECT_ROOT,
    add_simulation_arguments,
    ensure_simulation,
    group_states_by_step,
    load_states,
    params_from_args,
)

MPL_CACHE_DIR = PROJECT_ROOT / "tmp_mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
XDG_CACHE_DIR = PROJECT_ROOT / "tmp_cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generates an animation from simulation states.")
    add_simulation_arguments(parser)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every stride-th frame to speed up the animation.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Output animation file (mp4 or gif). Defaults to <output_dir>/animation.mp4.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for the saved animation.",
    )

    args = parser.parse_args()
    if args.stride <= 0:
        raise SystemExit("stride must be >= 1")

    params = params_from_args(args)
    output_dir = ensure_simulation(params)
    save_path = args.save
    if save_path is None:
        save_path = output_dir / "animation.mp4"
    else:
        save_path = save_path if save_path.is_absolute() else (output_dir / save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    states = load_states(output_dir / "states.txt")
    frames = group_states_by_step(states)
    frames = frames[:: args.stride]
    if not frames:
        raise SystemExit("No frames available in states.txt")

    build_animation(frames, params.domain, save_path, params.output_interval * args.stride, args.dpi)


def build_animation(frames, domain, save_path: Path, frame_dt: float, dpi: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import animation
        import numpy as np
        from matplotlib.patches import Circle
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required to generate animations") from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, domain)
    ax.set_ylim(0, domain)
    ax.set_aspect("equal")
    ax.set_title("Pedestrian simulation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    central_radius = 0.21
    central = Circle((domain / 2.0, domain / 2.0), central_radius, facecolor="gray", alpha=0.5, edgecolor="black")
    ax.add_patch(central)

    first_agents = sorted(frames[0][2], key=lambda rec: rec.agent_id)
    circles: dict[int, Circle] = {}
    for rec in first_agents:
        circle = Circle((rec.x, rec.y), rec.radius, facecolor="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.add_patch(circle)
        circles[rec.agent_id] = circle

    offsets = np.array([[rec.x, rec.y] for rec in first_agents])
    velocities = np.array([[rec.vx, rec.vy] for rec in first_agents])
    quiver = ax.quiver(
        offsets[:, 0],
        offsets[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        color="tab:orange",
    )

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        time_text.set_text("")
        return []

    def update(frame):
        step, time, agents = frame
        ordered = sorted(agents, key=lambda rec: rec.agent_id)
        coords = np.array([[rec.x, rec.y] for rec in ordered])
        vels = np.array([[rec.vx, rec.vy] for rec in ordered])

        for rec in ordered:
            circle = circles.get(rec.agent_id)
            if circle is None:
                circle = Circle((rec.x, rec.y), rec.radius, facecolor="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.add_patch(circle)
                circles[rec.agent_id] = circle
            circle.center = (rec.x, rec.y)
            circle.radius = rec.radius

        quiver.set_offsets(coords)
        quiver.set_UVC(vels[:, 0], vels[:, 1])
        time_text.set_text(f"t = {time:.2f} s (step {step})")
        return []

    interval_ms = max(1, int(frame_dt * 1000))
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval_ms,
        blit=False,
    )

    fps = 30
    writer = _pick_writer(animation, save_path.suffix.lower(), fps)
    ani.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {save_path}")


def _pick_writer(animation_module, suffix: str, fps: int):
    if suffix == ".gif":
        try:
            return animation_module.PillowWriter(fps=fps)
        except Exception as exc:  # pragma: no cover
            raise SystemExit("PillowWriter not available; install pillow or choose mp4 output") from exc
    try:
        return animation_module.FFMpegWriter(fps=fps)
    except Exception:  # pragma: no cover
        return animation_module.PillowWriter(fps=fps)


if __name__ == "__main__":
    main()
