#!/usr/bin/env python
"""Render splashsurf surface meshes to MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except OSError:
    pass


def render_meshes_to_mp4(
    mesh_dir: Path,
    output_path: Path,
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
) -> None:
    mesh_files = sorted(mesh_dir.glob("frame_surface_*.vtk"))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found in {mesh_dir}")

    print(f"Found {len(mesh_files)} mesh files")

    # Load first mesh to set up scene
    mesh0 = pv.read(mesh_files[0])

    pl = pv.Plotter(off_screen=True, window_size=resolution)
    pl.set_background("#0a0a1a", top="#1a1a3a")

    actor = pl.add_mesh(
        mesh0,
        color="#d0e8ff",
        specular=0.8,
        specular_power=30,
        diffuse=0.7,
        ambient=0.15,
        smooth_shading=True,
        show_edges=False,
    )

    # Lighting
    pl.remove_all_lights()
    key_light = pv.Light(position=(1.5, 1.5, 2.0), focal_point=(0.5, 0.5, 0.5), intensity=1.0)
    key_light.positional = True
    pl.add_light(key_light)
    fill_light = pv.Light(position=(-0.5, 0.5, 1.0), focal_point=(0.5, 0.5, 0.5), intensity=0.4)
    pl.add_light(fill_light)
    rim_light = pv.Light(position=(0.5, -0.5, -0.5), focal_point=(0.5, 0.5, 0.5), intensity=0.3)
    pl.add_light(rim_light)

    # Camera — slightly elevated 3/4 view of the [0,1]^3 domain
    pl.camera.position = (0.5, -0.8, 1.2)
    pl.camera.focal_point = (0.5, 0.5, 0.3)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.view_angle = 35.0

    # Add a subtle ground plane
    ground = pv.Plane(center=(0.5, 0.5, 0.0), direction=(0, 0, 1), i_size=1.0, j_size=1.0)
    pl.add_mesh(ground, color="#1a1a2e", opacity=0.5)

    pl.open_movie(str(output_path), framerate=fps, quality=9)

    total = len(mesh_files)
    for i, f in enumerate(mesh_files):
        mesh = pv.read(f)
        actor.mapper.SetInputData(mesh)
        pl.write_frame()
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  rendered {i + 1}/{total}")

    pl.close()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render surface meshes to MP4")
    parser.add_argument("--mesh-dir", type=Path, default=Path("output/snowball_meshes"))
    parser.add_argument("--output", type=Path, default=Path("output/snowball.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080])
    args = parser.parse_args()

    render_meshes_to_mp4(args.mesh_dir, args.output, args.fps, tuple(args.resolution))
