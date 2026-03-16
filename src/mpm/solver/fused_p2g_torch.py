"""Fused stress + P2G using PyTorch ops (no custom CUDA).

Same fusion as the CUDA kernel — stress, p2g data, and scatter in one call —
but implemented entirely with PyTorch tensor operations. This serves as the
baseline to measure how much custom CUDA gains over letting PyTorch handle it.
"""

from __future__ import annotations

import torch

from mpm.params import SimParams
from mpm.state import GridState

# 3x3x3 stencil offsets (27 neighbors)
_STENCIL = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))
_STENCIL_F = _STENCIL.float()
_stencil_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_grid_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _polar_newton_schulz(F: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    I = torch.eye(3, device=F.device, dtype=F.dtype)
    norms = torch.sqrt((F * F).sum((-2, -1), keepdim=True).clamp(min=1e-12))
    Y = F * (1.7320508 / norms)
    for _ in range(n_iter):
        Y = 0.5 * Y @ (3.0 * I - Y.mT @ Y)
    return Y


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], dim=-1)


def _sym_eig3x3(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    N = S.shape[0]
    dev = S.device

    a11 = S[:, 0, 0]; a22 = S[:, 1, 1]; a33 = S[:, 2, 2]
    a12 = S[:, 0, 1]; a13 = S[:, 0, 2]; a23 = S[:, 1, 2]

    p = a11 + a22 + a33
    q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23
    r = a11*a22*a33 + 2*a12*a13*a23 - a11*a23*a23 - a22*a13*a13 - a33*a12*a12

    p3 = p / 3.0
    pp = (p * p - 3.0 * q) / 9.0
    qq = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0

    pp_safe = pp.clamp(min=1e-30)
    sqrt_pp = torch.sqrt(pp_safe)
    cos_arg = (qq / (pp_safe * sqrt_pp)).clamp(-1.0, 1.0)
    phi = torch.acos(cos_arg) / 3.0

    two_sqrt_pp = 2.0 * sqrt_pp
    e0 = p3 - two_sqrt_pp * torch.cos(phi - 2.094395102)
    e1 = p3 - two_sqrt_pp * torch.cos(phi + 2.094395102)
    e2 = p3 - two_sqrt_pp * torch.cos(phi)

    eigs = torch.stack([e0, e1, e2], dim=-1)

    I = torch.eye(3, device=dev, dtype=S.dtype)
    vecs = torch.zeros(N, 3, 3, device=dev, dtype=S.dtype)

    for k in range(3):
        M = S - eigs[:, k:k+1, None] * I.unsqueeze(0)
        r0 = M[:, 0]; r1 = M[:, 1]; r2 = M[:, 2]

        c01 = _cross(r0, r1)
        c02 = _cross(r0, r2)
        c12 = _cross(r1, r2)

        n01 = (c01 * c01).sum(-1)
        n02 = (c02 * c02).sum(-1)
        n12 = (c12 * c12).sum(-1)

        best = c12; best_n = n12
        mask02 = n02 > best_n
        best = torch.where(mask02.unsqueeze(-1), c02, best)
        best_n = torch.where(mask02, n02, best_n)
        mask01 = n01 > best_n
        best = torch.where(mask01.unsqueeze(-1), c01, best)
        best_n = torch.where(mask01, n01, best_n)

        best = best / torch.sqrt(best_n.clamp(min=1e-30)).unsqueeze(-1)
        vecs[:, :, k] = best

    return eigs, vecs


def fused_stress_p2g_torch(
    x: torch.Tensor, v: torch.Tensor, C: torch.Tensor,
    Fe: torch.Tensor, Jp: torch.Tensor,
    params: SimParams, block_size: int = 256, newton_schulz_iters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, GridState]:
    """Fused stress + P2G in PyTorch ops. Returns (Fe_new, Jp_new, GridState)."""
    dev = x.device
    I = torch.eye(3, device=dev, dtype=Fe.dtype)

    # ---- Stress computation ----
    R = _polar_newton_schulz(Fe, newton_schulz_iters)
    S = R.mT @ Fe

    sig, Q = _sym_eig3x3(S)

    sig_c = sig.clamp(1.0 - params.theta_c, 1.0 + params.theta_s)
    Jp_new = Jp * sig.prod(-1) / sig_c.prod(-1)

    Fe_new = R @ ((Q * sig_c.unsqueeze(-2)) @ Q.mT)
    J = sig_c.prod(-1)

    h = torch.exp(params.hardening * (1.0 - Jp_new))
    mu = params.mu_0 * h
    la = params.lambda_0 * h

    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ Fe_new.mT)
              + (la * (J - 1.0) * J)[:, None, None] * I)

    # ---- P2G data ----
    if dev not in _stencil_cache:
        _stencil_cache[dev] = (_STENCIL.to(dev), _STENCIL_F.to(dev))
    stencil_int, stencil_f = _stencil_cache[dev]

    affine = (-params.dt * params.p_vol * 4.0 * params.inv_dx ** 2) * stress + params.p_mass * C

    base = (x * params.inv_dx - 0.5).int()
    fx = x * params.inv_dx - base.float()

    w3 = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=1)
    w = w3[:, stencil_int[:, 0], 0] * w3[:, stencil_int[:, 1], 1] * w3[:, stencil_int[:, 2], 2]

    dpos = (stencil_f.unsqueeze(0) - fx.unsqueeze(1)) * params.dx

    mv = w.unsqueeze(-1) * (params.p_mass * v.unsqueeze(1) + dpos @ affine.mT)
    m = w * params.p_mass

    nodes = (base.unsqueeze(1) + stencil_int.unsqueeze(0)).clamp(0, params.grid_res - 1)
    GR = params.grid_res
    flat = nodes[:, :, 0] * GR * GR + nodes[:, :, 1] * GR + nodes[:, :, 2]

    # ---- Scatter ----
    GR3 = GR ** 3
    key = (dev, GR3)
    if key not in _grid_cache:
        _grid_cache[key] = (torch.zeros(GR3, 3, device=dev), torch.zeros(GR3, device=dev))
    grid_v, grid_m = _grid_cache[key]
    grid_v.zero_()
    grid_m.zero_()

    flat_long = flat.reshape(-1).long()
    grid_v.scatter_add_(0, flat_long.unsqueeze(-1).expand(-1, 3), mv.reshape(-1, 3))
    grid_m.scatter_add_(0, flat_long, m.reshape(-1))

    grid = GridState(
        velocity=grid_v.reshape(GR, GR, GR, 3),
        mass=grid_m.reshape(GR, GR, GR),
    )
    return Fe_new, Jp_new, grid
