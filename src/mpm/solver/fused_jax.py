"""Fused stress + P2G via JAX (jit-compiled).

Same algorithm as the torch and CUDA backends:
  Newton-Schulz polar decomposition → Cardano eigendecomposition → stress → P2G scatter.
Letting XLA jit fuse/optimize the computation.
"""

from __future__ import annotations

import jax.numpy as jnp
import torch

from mpm.params import SimParams
from mpm.state import GridState


# Stencil offsets — computed once outside jit (constant, not traced)
_STENCIL = jnp.stack(
    jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing="ij"),
    axis=-1,
).reshape(27, 3)


# ---------------------------------------------------------------------------
# Polar decomposition via Newton-Schulz (matches torch backend)
# ---------------------------------------------------------------------------

def _polar_newton_schulz(F, n_iter=3):
    """Batched polar decomposition: F = R @ S, returns R."""
    I = jnp.eye(3)
    norms = jnp.sqrt(jnp.sum(F * F, axis=(-2, -1), keepdims=True).clip(min=1e-12))
    Y = F * (1.7320508 / norms)  # sqrt(3) / ||F||

    for _ in range(n_iter):
        Y = 0.5 * Y @ (3.0 * I - jnp.swapaxes(Y, -1, -2) @ Y)

    return Y


# ---------------------------------------------------------------------------
# Analytical 3x3 symmetric eigendecomposition (Cardano, matches torch backend)
# ---------------------------------------------------------------------------

def _cross(a, b):
    return jnp.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], axis=-1)


def _sym_eig3x3(S):
    """Analytical eigendecomposition of batched 3x3 symmetric matrices."""
    a11 = S[:, 0, 0]; a22 = S[:, 1, 1]; a33 = S[:, 2, 2]
    a12 = S[:, 0, 1]; a13 = S[:, 0, 2]; a23 = S[:, 1, 2]

    p = a11 + a22 + a33
    q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23
    r = a11*a22*a33 + 2*a12*a13*a23 - a11*a23*a23 - a22*a13*a13 - a33*a12*a12

    p3 = p / 3.0
    pp = (p * p - 3.0 * q) / 9.0
    qq = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0

    pp_safe = pp.clip(min=1e-30)
    sqrt_pp = jnp.sqrt(pp_safe)
    cos_arg = (qq / (pp_safe * sqrt_pp)).clip(-1.0, 1.0)
    phi = jnp.arccos(cos_arg) / 3.0

    two_sqrt_pp = 2.0 * sqrt_pp
    e0 = p3 - two_sqrt_pp * jnp.cos(phi - 2.094395102)
    e1 = p3 - two_sqrt_pp * jnp.cos(phi + 2.094395102)
    e2 = p3 - two_sqrt_pp * jnp.cos(phi)

    eigs = jnp.stack([e0, e1, e2], axis=-1)

    I = jnp.eye(3, dtype=S.dtype)
    vecs_list = []

    for k in range(3):
        M = S - eigs[:, k:k+1, None] * I[None, :, :]
        r0 = M[:, 0]; r1 = M[:, 1]; r2 = M[:, 2]

        c01 = _cross(r0, r1)
        c02 = _cross(r0, r2)
        c12 = _cross(r1, r2)

        n01 = jnp.sum(c01 * c01, axis=-1)
        n02 = jnp.sum(c02 * c02, axis=-1)
        n12 = jnp.sum(c12 * c12, axis=-1)

        best = c12; best_n = n12
        mask02 = n02 > best_n
        best = jnp.where(mask02[:, None], c02, best)
        best_n = jnp.where(mask02, n02, best_n)
        mask01 = n01 > best_n
        best = jnp.where(mask01[:, None], c01, best)
        best_n = jnp.where(mask01, n01, best_n)

        best = best / jnp.sqrt(best_n.clip(min=1e-30))[:, None]
        vecs_list.append(best)

    vecs = jnp.stack(vecs_list, axis=-1)  # (N, 3, 3)
    return eigs, vecs


# ---------------------------------------------------------------------------
# Fused stress + P2G kernel
# ---------------------------------------------------------------------------

def _fused_stress_p2g(
    x, v, C, Fe, Jp,
    grid_res: int,
    dt, inv_dx, dx, p_vol, p_mass,
    theta_c, theta_s, hardening, mu_0, lambda_0,
):
    GR = grid_res
    stencil = _STENCIL  # (27, 3)
    I = jnp.eye(3)

    # ---- Polar decomposition: Fe = R @ S ----
    R = _polar_newton_schulz(Fe)
    S = jnp.swapaxes(R, -1, -2) @ Fe

    # ---- Eigendecomposition of symmetric S ----
    sig, Q = _sym_eig3x3(S)

    # ---- Plasticity: clamp singular values ----
    sig_c = jnp.clip(sig, 1.0 - theta_c, 1.0 + theta_s)
    Jp_new = Jp * jnp.prod(sig, axis=-1) / jnp.prod(sig_c, axis=-1)

    # ---- Reconstruct Fe_new ----
    Fe_new = R @ ((Q * sig_c[:, None, :]) @ jnp.swapaxes(Q, -1, -2))
    J = jnp.prod(sig_c, axis=-1)

    # ---- Stress: fixed corotated with hardening ----
    h = jnp.exp(hardening * (1.0 - Jp_new))
    mu = mu_0 * h
    la = lambda_0 * h

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
