"""Grid operations: normalize, gravity, boundary conditions, colliders."""

from __future__ import annotations

import jax.numpy as jnp

from mpm.params import SimParams, WallCollider, BoxCollider
from mpm.state import GridState


def update_grid(grid: GridState, params: SimParams) -> GridState:
    GR, DT, DX, B = params.grid_res, params.dt, params.dx, params.bound
    grid_v, grid_m = grid.velocity, grid.mass

    # Normalize momentum → velocity, zero empty cells
    safe_m = jnp.maximum(grid_m, 1e-30)
    grid_v = grid_v / safe_m[..., None]
    grid_v = jnp.where((grid_m > 0)[..., None], grid_v, 0.0)

    # Gravity
    gravity = jnp.array(params.gravity, dtype=jnp.float32)
    grid_v = grid_v + DT * gravity

    # Domain boundary: separating condition (no penetration)
    # x-axis low face: velocity x must be >= 0
    grid_v = grid_v.at[:B, :, :, 0].set(jnp.maximum(grid_v[:B, :, :, 0], 0.0))
    # x-axis high face: velocity x must be <= 0
    grid_v = grid_v.at[-B:, :, :, 0].set(jnp.minimum(grid_v[-B:, :, :, 0], 0.0))
    # y-axis low face
    grid_v = grid_v.at[:, :B, :, 1].set(jnp.maximum(grid_v[:, :B, :, 1], 0.0))
    # y-axis high face
    grid_v = grid_v.at[:, -B:, :, 1].set(jnp.minimum(grid_v[:, -B:, :, 1], 0.0))
    # z-axis low face
    grid_v = grid_v.at[:, :, :B, 2].set(jnp.maximum(grid_v[:, :, :B, 2], 0.0))
    # z-axis high face
    grid_v = grid_v.at[:, :, -B:, 2].set(jnp.minimum(grid_v[:, :, -B:, 2], 0.0))

    # Colliders
    for c in params.colliders:
        if isinstance(c, BoxCollider):
            lo = jnp.clip(
                ((jnp.array(c.center) - jnp.array(c.half_size)) / DX).astype(jnp.int32),
                0, GR - 1,
            )
            hi = jnp.clip(
                ((jnp.array(c.center) + jnp.array(c.half_size)) / DX).astype(jnp.int32),
                0, GR - 1,
            )
            # Convert to Python ints for slice indexing
            lx, ly, lz = int(lo[0]), int(lo[1]), int(lo[2])
            hx, hy, hz = int(hi[0]), int(hi[1]), int(hi[2])
            grid_v = grid_v.at[lx:hx+1, ly:hy+1, lz:hz+1].set(0.0)

        elif isinstance(c, WallCollider):
            n = jnp.array(c.normal, dtype=jnp.float32)
            n = n / jnp.linalg.norm(n)
            wp = jnp.array(c.point, dtype=jnp.float32) * params.inv_dx
            gi = jnp.arange(GR, dtype=jnp.float32)
            dist = (
                (gi[:, None, None] - wp[0]) * n[0]
                + (gi[None, :, None] - wp[1]) * n[1]
                + (gi[None, None, :] - wp[2]) * n[2]
            )
            vn = (grid_v * n).sum(axis=-1)
            mask = (dist < 1.0) & (vn < 0)
            correction = jnp.where(mask[..., None], vn[..., None] * n, 0.0)
            grid_v = grid_v - correction

    return GridState(velocity=grid_v, mass=grid_m)
