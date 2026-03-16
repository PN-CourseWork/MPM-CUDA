"""Per-frame VTK particle writer.

Writes one legacy VTK file per saved frame containing particle positions
as an Unstructured Grid. Compatible with splashsurf CLI, splashsurf_studio
Blender add-on, ParaView, and meshio.

Layout:
    output_dir/frame_00001.vtk
    output_dir/frame_00002.vtk
    ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_vtk(path: Path, positions: np.ndarray) -> None:
    """Write positions as a legacy VTK 4.2 binary Unstructured Grid."""
    n = len(positions)

    with open(path, "wb") as f:
        # Header (ASCII)
        f.write(b"# vtk DataFile Version 4.2\n")
        f.write(b"MPM particle data\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n} float\n".encode("ascii"))
        f.write(positions.astype(">f4").tobytes())

        # Cells: each point is a single-vertex cell
        f.write(f"\nCELLS {n} {2 * n}\n".encode("ascii"))
        cells = np.empty((n, 2), dtype=">i4")
        cells[:, 0] = 1
        cells[:, 1] = np.arange(n)
        f.write(cells.tobytes())

        # Cell types: VTK_VERTEX = 1
        f.write(f"\nCELL_TYPES {n}\n".encode("ascii"))
        f.write(np.full(n, 1, dtype=">i4").tobytes())
        f.write(b"\n")


class FrameWriter:
    """Writes per-frame VTK files for splashsurf and Blender."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame = 0

    def append(self, x):
        """Write one frame of positions (numpy or JAX array)."""
        self.frame += 1
        pos = np.asarray(x, dtype=np.float32)
        _write_vtk(self.output_dir / f"frame_{self.frame:05d}.vtk", pos)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
