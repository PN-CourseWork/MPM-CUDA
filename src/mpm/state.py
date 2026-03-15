"""Particle and grid state data structures."""

from __future__ import annotations

from typing import NamedTuple

import torch


def resolve_device(name: str) -> torch.device:
    """Resolve device name from config to a torch.device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class ParticleState(NamedTuple):
    """Full particle state — unpackable as (x, v, C, F, Jp)."""
    x: torch.Tensor       # (N, 3) positions
    v: torch.Tensor       # (N, 3) velocities
    C: torch.Tensor       # (N, 3, 3) APIC affine velocity field
    F: torch.Tensor       # (N, 3, 3) elastic deformation gradient
    Jp: torch.Tensor      # (N,) plastic determinant ratio


class GridState(NamedTuple):
    """Eulerian grid state."""
    velocity: torch.Tensor   # (GR, GR, GR, 3) momentum (pre-normalize) or velocity
    mass: torch.Tensor       # (GR, GR, GR)
