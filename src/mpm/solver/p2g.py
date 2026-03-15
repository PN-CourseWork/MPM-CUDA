"""P2G: compute stencil data and scatter to grid (PyTorch scatter_add)."""

from __future__ import annotations

from typing import NamedTuple

import torch

from mpm.params import SimParams
from mpm.state import GridState

# 3x3x3 stencil offsets (27 neighbors)
_STENCIL = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))  # (27, 3)
_STENCIL_F = _STENCIL.float()

# Per-device caches
_stencil_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_grid_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


class P2GData(NamedTuple):
    """Stencil data shared between P2G scatter and G2P gather."""
    node_i: torch.Tensor       # (N, 27)
    node_j: torch.Tensor       # (N, 27)
    node_k: torch.Tensor       # (N, 27)
    flat_indices: torch.Tensor  # (N, 27)
    weights: torch.Tensor       # (N, 27)
    dpos: torch.Tensor          # (N, 27, 3)
    momentum: torch.Tensor      # (N, 27, 3)
    mass: torch.Tensor          # (N, 27)


def compute_p2g_data(
    x: torch.Tensor, v: torch.Tensor, C: torch.Tensor,
    stress: torch.Tensor, params: SimParams,
) -> P2GData:
    dev = x.device
    if dev not in _stencil_cache:
        _stencil_cache[dev] = (_STENCIL.to(dev), _STENCIL_F.to(dev))
    stencil_int, stencil_f = _stencil_cache[dev]

    # Affine momentum: force + APIC terms fused
    affine = (-params.dt * params.p_vol * 4.0 * params.inv_dx ** 2) * stress + params.p_mass * C

    # Base cell and fractional position
    base = (x * params.inv_dx - 0.5).int()
    fx = x * params.inv_dx - base.float()

    # Quadratic B-spline weights → product for 27 neighbors
    w3 = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=1)
    w = w3[:, stencil_int[:, 0], 0] * w3[:, stencil_int[:, 1], 1] * w3[:, stencil_int[:, 2], 2]

    # Offset from particle to grid node
    dpos = (stencil_f.unsqueeze(0) - fx.unsqueeze(1)) * params.dx

    # Momentum and mass
    mv = w.unsqueeze(-1) * (params.p_mass * v.unsqueeze(1) + dpos @ affine.mT)
    m = w * params.p_mass

    # Grid node indices
    nodes = (base.unsqueeze(1) + stencil_int.unsqueeze(0)).clamp(0, params.grid_res - 1)
    GR = params.grid_res
    flat = nodes[:, :, 0] * GR * GR + nodes[:, :, 1] * GR + nodes[:, :, 2]

    return P2GData(nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2], flat, w, dpos, mv, m)


def scatter(p2g_data: P2GData, grid_res: int) -> GridState:
    GR3 = grid_res ** 3
    dev = p2g_data.flat_indices.device

    # Reuse grid buffers to avoid allocation each step
    key = (dev, GR3)
    if key not in _grid_cache:
        _grid_cache[key] = (torch.zeros(GR3, 3, device=dev), torch.zeros(GR3, device=dev))
    grid_v, grid_m = _grid_cache[key]
    grid_v.zero_()
    grid_m.zero_()

    flat = p2g_data.flat_indices.reshape(-1).long()
    grid_v.scatter_add_(0, flat.unsqueeze(-1).expand(-1, 3), p2g_data.momentum.reshape(-1, 3))
    grid_m.scatter_add_(0, flat, p2g_data.mass.reshape(-1))

    return GridState(
        velocity=grid_v.reshape(grid_res, grid_res, grid_res, 3),
        mass=grid_m.reshape(grid_res, grid_res, grid_res),
    )
