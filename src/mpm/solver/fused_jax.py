"""Fused stress + P2G via JAX (jit-compiled).

Same pipeline as the CUDA kernel but expressed in JAX ops,
letting XLA fuse/optimize the computation.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import torch

from mpm.params import SimParams
from mpm.state import GridState


# ---------------------------------------------------------------------------
# Core fused stress + P2G (pure JAX, jit-safe)
# ---------------------------------------------------------------------------

# Stencil offsets — computed once outside jit (constant, not traced)
_STENCIL = jnp.stack(jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing="ij"), axis=-1).reshape(27, 3)


@functools.partial(jax.jit, static_argnames=("grid_res",))
def _fused_stress_p2g(
    x, v, C, Fe, Jp,
    grid_res: int,
    dt, inv_dx, dx, p_vol, p_mass,
    theta_c, theta_s, hardening, mu_0, lambda_0,
):
    GR = grid_res
    stencil = _STENCIL  # (27, 3)

    # ---- SVD: Fe = U @ diag(sig) @ Vh ----
    U, sig, Vh = jnp.linalg.svd(Fe)  # (N,3,3), (N,3), (N,3,3)
    R = U @ Vh  # rotation

    # ---- Plasticity: clamp singular values ----
    sig_c = jnp.clip(sig, 1.0 - theta_c, 1.0 + theta_s)
    Jp_new = Jp * jnp.prod(sig, axis=-1) / jnp.prod(sig_c, axis=-1)

    # ---- Reconstruct Fe_new ----
    Fe_new = (U * sig_c[:, None, :]) @ Vh
    J = jnp.prod(sig_c, axis=-1)

    # ---- Stress: fixed corotated with hardening ----
    h = jnp.exp(hardening * (1.0 - Jp_new))
    mu = mu_0 * h
    la = lambda_0 * h

    I = jnp.eye(3)
    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ jnp.swapaxes(Fe_new, -1, -2))
              + (la * (J - 1.0) * J)[:, None, None] * I)

    # ---- Affine momentum (MLS-MPM eq. 29) ----
    affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress + p_mass * C

    # ---- B-spline weights ----
    base = jnp.floor(x * inv_dx - 0.5).astype(jnp.int32)  # (N, 3)
    fx = x * inv_dx - base.astype(jnp.float32)  # (N, 3)

    stencil_f = stencil.astype(jnp.float32)

    w3_0 = 0.5 * (1.5 - fx) ** 2
    w3_1 = 0.75 - (fx - 1.0) ** 2
    w3_2 = 0.5 * (fx - 0.5) ** 2
    w3 = jnp.stack([w3_0, w3_1, w3_2], axis=1)  # (N, 3, 3)

    w = (w3[:, stencil[:, 0], 0]
         * w3[:, stencil[:, 1], 1]
         * w3[:, stencil[:, 2], 2])  # (N, 27)

    # Offset from particle to grid node
    dpos = (stencil_f[None, :, :] - fx[:, None, :]) * dx  # (N, 27, 3)

    # Momentum per stencil node
    mv = w[:, :, None] * (p_mass * v[:, None, :] + jnp.einsum("nij,nkj->nki", affine, dpos))
    m = w * p_mass  # (N, 27)

    # ---- Scatter to grid ----
    nodes = jnp.clip(base[:, None, :] + stencil[None, :, :], 0, GR - 1)  # (N, 27, 3)
    flat = nodes[:, :, 0] * GR * GR + nodes[:, :, 1] * GR + nodes[:, :, 2]  # (N, 27)

    grid_v = jnp.zeros((GR * GR * GR, 3), dtype=jnp.float32)
    grid_m = jnp.zeros((GR * GR * GR,), dtype=jnp.float32)

    flat_exp = flat.reshape(-1)
    grid_v = grid_v.at[flat_exp].add(mv.reshape(-1, 3))
    grid_m = grid_m.at[flat_exp].add(m.reshape(-1))

    grid_v = grid_v.reshape(GR, GR, GR, 3)
    grid_m = grid_m.reshape(GR, GR, GR)

    return Fe_new, Jp_new, grid_v, grid_m


# ---------------------------------------------------------------------------
# Public API: torch tensors in/out, JAX inside
# ---------------------------------------------------------------------------

def _torch_to_jax(t: torch.Tensor):
    return jnp.from_dlpack(t.detach().contiguous())


def _jax_to_torch(a, device: torch.device):
    return torch.from_dlpack(a).to(device)


def fused_stress_p2g_jax(
    x, v, C, Fe, Jp, params: SimParams, block_size: int = 256,
):
    """Fused stress + P2G via JAX. Returns (Fe_new, Jp_new, GridState)."""
    dev = x.device

    # torch → jax (zero-copy via dlpack)
    jx = _torch_to_jax(x)
    jv = _torch_to_jax(v)
    jC = _torch_to_jax(C)
    jFe = _torch_to_jax(Fe)
    jJp = _torch_to_jax(Jp)

    Fe_new, Jp_new, grid_v, grid_m = _fused_stress_p2g(
        jx, jv, jC, jFe, jJp,
        params.grid_res,
        params.dt, params.inv_dx, params.dx, params.p_vol, params.p_mass,
        params.theta_c, params.theta_s, params.hardening,
        params.mu_0, params.lambda_0,
    )

    # jax → torch
    return (
        _jax_to_torch(Fe_new, dev),
        _jax_to_torch(Jp_new, dev),
        GridState(
            velocity=_jax_to_torch(grid_v, dev),
            mass=_jax_to_torch(grid_m, dev),
        ),
    )
