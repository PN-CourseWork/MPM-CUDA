#!/usr/bin/env python
"""Export H5 particle data to per-frame VTK files for splashsurf."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


def export_frames(h5_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        positions = f["positions"][:]  # (n_frames, N, 3)

    n_frames = positions.shape[0]
    print(f"Exporting {n_frames} frames from {h5_path} → {output_dir}/")

    for i in range(n_frames):
        cloud = pv.PolyData(positions[i])
        out_path = output_dir / f"frame_{i:04d}.vtk"
        cloud.save(str(out_path))

    print(f"Done — {n_frames} VTK files written")
    return n_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export H5 → per-frame VTK")
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.h5_path.with_suffix("") / "particles"
    export_frames(args.h5_path, out_dir)
