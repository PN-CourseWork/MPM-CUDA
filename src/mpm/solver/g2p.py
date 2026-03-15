"""Grid-to-particle gather: update velocities, positions, and F."""

from __future__ import annotations

import torch

from mpm.params import SimParams
from mpm.solver.p2g import P2GData
from mpm.solver.stress import _eye3


def gather(
    grid_v: torch.Tensor, p2g_data: P2GData,
    x: torch.Tensor, Fe_new: torch.Tensor, params: SimParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather grid velocities back to particles. Returns (x, v, C, Fe)."""
    g_vi = grid_v[p2g_data.node_i, p2g_data.node_j, p2g_data.node_k]  # (N, 27, 3)
    w_gvi = p2g_data.weights.unsqueeze(-1) * g_vi  # (N, 27, 3)

    new_v = w_gvi.sum(dim=1)
    new_C = 4.0 * params.inv_dx ** 2 * (w_gvi.mT @ p2g_data.dpos)  # (N, 3, 3)
    new_x = (x + params.dt * new_v).clamp(3 * params.dx, 1.0 - 3 * params.dx)
    new_Fe = (_eye3(x.device) + params.dt * new_C) @ Fe_new

    return new_x, new_v, new_C, new_Fe
