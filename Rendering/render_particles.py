#!/usr/bin/env python
"""Render particle simulation from VTK frames to MP4 video using PyVista."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except OSError:
    pass


def load_vtk_frames(frames_dir: Path) -> np.ndarray:
    """Load all VTK frame files, return (n_frames, N, 3) float32 array."""
    files = sorted(frames_dir.glob("frame_*.vtk"))
    if not files:
        raise FileNotFoundError(f"No frame_*.vtk files in {frames_dir}")

    frames = []
    for f in files:
        mesh = pv.read(str(f))
        frames.append(np.array(mesh.points, dtype=np.float32))

    return np.stack(frames)


def render_particles_to_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    point_size: float = 5.0,
) -> None:
    positions = load_vtk_frames(frames_dir)
    n_frames = positions.shape[0]
    print(f"Loaded {n_frames} frames, {positions.shape[1]} particles from {frames_dir}")

    pl = pv.Plotter(off_screen=True, window_size=resolution)
    pl.set_background("#0a0a1a", top="#1a1a3a")

    # First frame
    cloud = pv.PolyData(positions[0])
    pl.add_mesh(
        cloud,
        color="white",
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=False,
    )

    # Lighting
    pl.remove_all_lights()
    key_light = pv.Light(position=(1.5, 1.5, 2.0), focal_point=(0.5, 0.5, 0.5), intensity=1.0)
    key_light.positional = True
    pl.add_light(key_light)
    fill_light = pv.Light(position=(-0.5, 0.5, 1.0), focal_point=(0.5, 0.5, 0.5), intensity=0.4)
    pl.add_light(fill_light)

    # Camera — side view to see wall impact
    pl.camera.position = (-0.3, 0.5, 0.6)
    pl.camera.focal_point = (0.5, 0.7, 0.4)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.view_angle = 45.0

    # Ground plane
    ground = pv.Plane(center=(0.5, 0.5, 0.0), direction=(0, 0, 1), i_size=1.0, j_size=1.0)
    pl.add_mesh(ground, color="#1a1a2e", opacity=0.5)

    # Back wall (y=1)
    wall = pv.Plane(center=(0.5, 1.0, 0.5), direction=(0, -1, 0), i_size=1.0, j_size=1.0)
    pl.add_mesh(wall, color="#2a1a2e", opacity=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.open_movie(str(output_path), framerate=fps, quality=9)

    for i in range(n_frames):
        cloud.points = positions[i]
        pl.write_frame()
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  rendered {i + 1}/{n_frames}")

    pl.close()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render VTK particle frames to MP4")
    parser.add_argument("frames_dir", type=Path, help="Path to directory with frame_*.vtk files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    out = args.output or (args.frames_dir.parent / "particles.mp4")
    render_particles_to_mp4(args.frames_dir, out, args.fps, tuple(args.resolution), args.point_size)
