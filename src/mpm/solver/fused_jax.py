"""Fused JAX MPM step: stress+P2G → grid_ops → G2P in one jit.

Letting XLA jit fuse/optimize the entire timestep as one program,
and using lax.scan to eliminate Python loop overhead.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from mpm.params import SimParams, WallCollider, BoxCollider


# Stencil offsets — computed once outside jit (constant, not traced)
_STENCIL = jnp.stack(
    jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing="ij"),
    axis=-1,
).reshape(27, 3)


# ---------------------------------------------------------------------------
# Polar decomposition via Newton-Schulz
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
# Analytical 3x3 symmetric eigendecomposition (Cardano)
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
# Core step implementation (called from within jitted functions)
# ---------------------------------------------------------------------------

def _step_impl(x, v, C, Fe, Jp, params: SimParams):
    """One complete MPM timestep: stress+P2G → grid_ops → G2P.

    Not jitted on its own — called from full_step() or scan_steps().
    params must be static in the calling jit so that collider loops
    and grid_res are resolved at trace time.
    """
    GR = params.grid_res
    DT = params.dt
    DX = params.dx
    inv_dx = params.inv_dx
    B = params.bound
    I3 = jnp.eye(3)
    stencil = _STENCIL
    stencil_f = stencil.astype(jnp.float32)

    # ========== STRESS ==========

    R = _polar_newton_schulz(Fe)
    S = jnp.swapaxes(R, -1, -2) @ Fe

    sig, Q = _sym_eig3x3(S)

    sig_c = jnp.clip(sig, 1.0 - params.theta_c, 1.0 + params.theta_s)
    Jp_new = Jp * jnp.prod(sig, axis=-1) / jnp.prod(sig_c, axis=-1)

    Fe_new = R @ ((Q * sig_c[:, None, :]) @ jnp.swapaxes(Q, -1, -2))
    J = jnp.prod(sig_c, axis=-1)

    h = jnp.exp(params.hardening * (1.0 - Jp_new))
    mu = params.mu_0 * h
    la = params.lambda_0 * h

    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ jnp.swapaxes(Fe_new, -1, -2))
              + (la * (J - 1.0) * J)[:, None, None] * I3)

    affine = (-DT * params.p_vol * 4.0 * inv_dx * inv_dx) * stress + params.p_mass * C

    # ========== B-SPLINE STENCIL (computed once, used for P2G and G2P) ==========

    base = jnp.floor(x * inv_dx - 0.5).astype(jnp.int32)
    fx = x * inv_dx - base.astype(jnp.float32)

    w3 = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ], axis=1)

    w = (w3[:, stencil[:, 0], 0]
         * w3[:, stencil[:, 1], 1]
         * w3[:, stencil[:, 2], 2])

    dpos = (stencil_f[None, :, :] - fx[:, None, :]) * DX

    nodes = jnp.clip(base[:, None, :] + stencil[None, :, :], 0, GR - 1)

    # ========== P2G SCATTER ==========

    flat = nodes[:, :, 0] * GR * GR + nodes[:, :, 1] * GR + nodes[:, :, 2]

    mv = w[:, :, None] * (params.p_mass * v[:, None, :] + jnp.einsum("nij,nkj->nki", affine, dpos))
    m = w * params.p_mass

    grid_v = jnp.zeros((GR * GR * GR, 3), dtype=jnp.float32)
    grid_m = jnp.zeros((GR * GR * GR,), dtype=jnp.float32)
    flat_exp = flat.reshape(-1)
    grid_v = grid_v.at[flat_exp].add(mv.reshape(-1, 3))
    grid_m = grid_m.at[flat_exp].add(m.reshape(-1))
    grid_v = grid_v.reshape(GR, GR, GR, 3)
    grid_m = grid_m.reshape(GR, GR, GR)

    # ========== GRID OPS ==========

    safe_m = jnp.maximum(grid_m, 1e-30)
    grid_v = grid_v / safe_m[..., None]
    grid_v = jnp.where((grid_m > 0)[..., None], grid_v, 0.0)

    gravity = jnp.array(params.gravity, dtype=jnp.float32)
    grid_v = grid_v + DT * gravity

    # Boundary conditions
    grid_v = grid_v.at[:B, :, :, 0].set(jnp.maximum(grid_v[:B, :, :, 0], 0.0))
    grid_v = grid_v.at[-B:, :, :, 0].set(jnp.minimum(grid_v[-B:, :, :, 0], 0.0))
    grid_v = grid_v.at[:, :B, :, 1].set(jnp.maximum(grid_v[:, :B, :, 1], 0.0))
    grid_v = grid_v.at[:, -B:, :, 1].set(jnp.minimum(grid_v[:, -B:, :, 1], 0.0))
    grid_v = grid_v.at[:, :, :B, 2].set(jnp.maximum(grid_v[:, :, :B, 2], 0.0))
    grid_v = grid_v.at[:, :, -B:, 2].set(jnp.minimum(grid_v[:, :, -B:, 2], 0.0))

    # Colliders (unrolled at trace time since params is static)
    for c in params.colliders:
        if isinstance(c, BoxCollider):
            # Compute slice indices as Python ints (params is static)
            lx = max(0, min(int((c.center[0] - c.half_size[0]) / DX), GR - 1))
            ly = max(0, min(int((c.center[1] - c.half_size[1]) / DX), GR - 1))
            lz = max(0, min(int((c.center[2] - c.half_size[2]) / DX), GR - 1))
            hx = max(0, min(int((c.center[0] + c.half_size[0]) / DX), GR - 1))
            hy = max(0, min(int((c.center[1] + c.half_size[1]) / DX), GR - 1))
            hz = max(0, min(int((c.center[2] + c.half_size[2]) / DX), GR - 1))
            grid_v = grid_v.at[lx:hx+1, ly:hy+1, lz:hz+1].set(0.0)
        elif isinstance(c, WallCollider):
            n = jnp.array(c.normal, dtype=jnp.float32)
            n = n / jnp.linalg.norm(n)
            wp = jnp.array(c.point, dtype=jnp.float32) * inv_dx
            gi = jnp.arange(GR, dtype=jnp.float32)
            dist = ((gi[:, None, None] - wp[0]) * n[0]
                    + (gi[None, :, None] - wp[1]) * n[1]
                    + (gi[None, None, :] - wp[2]) * n[2])
            vn = (grid_v * n).sum(axis=-1)
            mask = (dist < 1.0) & (vn < 0)
            correction = jnp.where(mask[..., None], vn[..., None] * n, 0.0)
            grid_v = grid_v - correction

    # ========== G2P GATHER (reuses stencil from P2G) ==========

    g_vi = grid_v[nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2]]
    w_gvi = w[..., None] * g_vi

    new_v = w_gvi.sum(axis=1)
    new_C = 4.0 * inv_dx ** 2 * (jnp.swapaxes(w_gvi, -1, -2) @ dpos)
    new_x = jnp.clip(x + DT * new_v, 3 * DX, 1.0 - 3 * DX)
    new_Fe = (I3 + DT * new_C) @ Fe_new

    return new_x, new_v, new_C, new_Fe, Jp_new


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params",))
def full_step(x, v, C, Fe, Jp, params: SimParams):
    """Single jitted timestep."""
    return _step_impl(x, v, C, Fe, Jp, params)


@functools.partial(jax.jit, static_argnames=("params", "n_steps"))
def scan_steps(x, v, C, Fe, Jp, params: SimParams, n_steps: int):
    """Run n_steps timesteps fused into one XLA program via lax.scan."""
    def body(carry, _):
        return _step_impl(*carry, params), None
    (x, v, C, Fe, Jp), _ = jax.lax.scan(body, (x, v, C, Fe, Jp), None, length=n_steps)
    return x, v, C, Fe, Jp
