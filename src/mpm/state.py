"""Particle and grid state data structures."""

from __future__ import annotations

from typing import NamedTuple

import jax


def resolve_device(name: str) -> str:
    """Resolve device name from config to a string for logging purposes.

    JAX handles device placement automatically; this function exists only for
    API compatibility and informational logging.
    """
    if name == "auto":
        for backend in ["gpu", "tpu"]:
            try:
                if jax.devices(backend):
                    return backend
            except RuntimeError:
                pass
        return "cpu"
    if name == "cuda":
        return "gpu"
    return name


class ParticleState(NamedTuple):
    """Full particle state — unpackable as (x, v, C, F, Jp)."""
    x: jax.Array       # (N, 3) positions
    v: jax.Array       # (N, 3) velocities
    C: jax.Array       # (N, 3, 3) APIC affine velocity field
    F: jax.Array       # (N, 3, 3) elastic deformation gradient
    Jp: jax.Array      # (N,) plastic determinant ratio


class GridState(NamedTuple):
    """Eulerian grid state."""
    velocity: jax.Array   # (GR, GR, GR, 3) momentum (pre-normalize) or velocity
    mass: jax.Array       # (GR, GR, GR)
