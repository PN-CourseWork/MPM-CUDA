"""Grid-to-particle gather: update velocities, positions, and F."""

from __future__ import annotations

from typing import NamedTuple

import torch

from mpm.params import SimParams
from mpm.solver.stress import _eye3


class StencilData(NamedTuple):
    """Lightweight stencil data (no momentum/mass) for G2P gather."""
    node_i: torch.Tensor     # (N, 27)
    node_j: torch.Tensor     # (N, 27)
    node_k: torch.Tensor     # (N, 27)
    weights: torch.Tensor    # (N, 27)
    dpos: torch.Tensor       # (N, 27, 3)


# Stencil offsets — lazily moved to device
_STENCIL = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))  # (27, 3)
_STENCIL_F = _STENCIL.float()
_stencil_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def compute_stencil(x: torch.Tensor, params: SimParams) -> StencilData:
    """Recompute B-spline stencil from particle positions (cheap)."""
    dev = x.device
    if dev not in _stencil_cache:
        _stencil_cache[dev] = (_STENCIL.to(dev), _STENCIL_F.to(dev))
    stencil_int, stencil_f = _stencil_cache[dev]

    base = (x * params.inv_dx - 0.5).int()
    fx = x * params.inv_dx - base.float()

    w3 = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=1)
    w = w3[:, stencil_int[:, 0], 0] * w3[:, stencil_int[:, 1], 1] * w3[:, stencil_int[:, 2], 2]

    dpos = (stencil_f.unsqueeze(0) - fx.unsqueeze(1)) * params.dx

    nodes = (base.unsqueeze(1) + stencil_int.unsqueeze(0)).clamp(0, params.grid_res - 1)

    return StencilData(nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2], w, dpos)


def gather(
    grid_v: torch.Tensor, stencil,
    x: torch.Tensor, Fe_new: torch.Tensor, params: SimParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather grid velocities back to particles. Returns (x, v, C, Fe).

    stencil can be P2GData or StencilData — both have node_i/j/k, weights, dpos.
    """
    g_vi = grid_v[stencil.node_i, stencil.node_j, stencil.node_k]  # (N, 27, 3)
    w_gvi = stencil.weights.unsqueeze(-1) * g_vi  # (N, 27, 3)

    new_v = w_gvi.sum(dim=1)
    new_C = 4.0 * params.inv_dx ** 2 * (w_gvi.mT @ stencil.dpos)  # (N, 3, 3)
    new_x = (x + params.dt * new_v).clamp(3 * params.dx, 1.0 - 3 * params.dx)
    new_Fe = (_eye3(x.device) + params.dt * new_C) @ Fe_new

    return new_x, new_v, new_C, new_Fe
