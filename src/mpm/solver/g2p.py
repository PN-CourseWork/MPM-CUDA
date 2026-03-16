"""Grid-to-particle gather: update velocities, positions, and F."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mpm.params import SimParams


class StencilData(NamedTuple):
    """Lightweight stencil data (no momentum/mass) for G2P gather."""
    node_i: jax.Array   # (N, 27)
    node_j: jax.Array   # (N, 27)
    node_k: jax.Array   # (N, 27)
    weights: jax.Array  # (N, 27)
    dpos: jax.Array     # (N, 27, 3)


# Stencil offsets: (27, 3) integer grid — computed once at module level
_STENCIL = jnp.stack(
    jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing="ij"),
    axis=-1,
).reshape(27, 3)  # (27, 3)


def compute_stencil(x: jax.Array, params: SimParams) -> StencilData:
    """Recompute B-spline stencil from particle positions (cheap)."""
    stencil_int = _STENCIL                        # (27, 3) int32
    stencil_f = stencil_int.astype(jnp.float32)  # (27, 3) float32

    base = (x * params.inv_dx - 0.5).astype(jnp.int32)          # (N, 3)
    fx = x * params.inv_dx - base.astype(jnp.float32)            # (N, 3)

    # Quadratic B-spline weights: w3 shape (N, 3, 3) — [particle, node-offset, axis]
    w3 = jnp.stack(
        [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ],
        axis=1,
    )  # (N, 3, 3)

    # Per-neighbour weight: product of per-axis weights
    w = (
        w3[:, stencil_int[:, 0], 0]   # (N, 27)
        * w3[:, stencil_int[:, 1], 1]
        * w3[:, stencil_int[:, 2], 2]
    )  # (N, 27)

    # Displacement from particle to neighbour node centre
    dpos = (stencil_f[None, :, :] - fx[:, None, :]) * params.dx  # (N, 27, 3)

    # Neighbour node indices, clamped to grid bounds
    nodes = jnp.clip(
        base[:, None, :] + stencil_int[None, :, :],
        0, params.grid_res - 1,
    )  # (N, 27, 3)

    return StencilData(nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2], w, dpos)


def gather(
    grid_v: jax.Array,
    stencil: StencilData,
    x: jax.Array,
    Fe_new: jax.Array,
    params: SimParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gather grid velocities back to particles. Returns (x, v, C, Fe).

    stencil can be P2GData or StencilData — both have node_i/j/k, weights, dpos.
    """
    g_vi = grid_v[stencil.node_i, stencil.node_j, stencil.node_k]  # (N, 27, 3)
    w_gvi = stencil.weights[..., None] * g_vi                        # (N, 27, 3)

    new_v = w_gvi.sum(axis=1)                                                      # (N, 3)
    new_C = 4.0 * params.inv_dx ** 2 * (jnp.swapaxes(w_gvi, -1, -2) @ stencil.dpos)  # (N, 3, 3)
    new_x = jnp.clip(
        x + params.dt * new_v,
        3 * params.dx,
        1.0 - 3 * params.dx,
    )
    new_Fe = (jnp.eye(3) + params.dt * new_C) @ Fe_new

    return new_x, new_v, new_C, new_Fe
