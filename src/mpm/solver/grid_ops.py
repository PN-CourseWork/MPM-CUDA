"""Grid operations: normalize, gravity, boundary conditions, colliders."""

from __future__ import annotations

import torch

from mpm.params import SimParams, WallCollider, BoxCollider
from mpm.state import GridState

# Cache gravity tensor per (device, gravity_tuple) to avoid re-creating each step
_gravity_cache: dict[tuple, torch.Tensor] = {}


def _get_gravity(gravity: tuple, dev: torch.device) -> torch.Tensor:
    key = (gravity, dev)
    if key not in _gravity_cache:
        _gravity_cache[key] = torch.tensor(gravity, dtype=torch.float32, device=dev)
    return _gravity_cache[key]


def update_grid(grid: GridState, params: SimParams) -> GridState:
    GR, DT, DX, B = params.grid_res, params.dt, params.dx, params.bound
    grid_v, grid_m = grid.velocity, grid.mass

    # Normalize momentum → velocity, zero empty cells
    grid_v.div_(grid_m.clamp(min=1e-30).unsqueeze(-1))
    grid_v *= (grid_m > 0).unsqueeze(-1)

    # Gravity
    grid_v += DT * _get_gravity(params.gravity, grid_v.device)

    # Domain boundary: separating condition (no penetration)
    grid_v[:B, :, :, 0].clamp_(min=0.0)
    grid_v[-B:, :, :, 0].clamp_(max=0.0)
    grid_v[:, :B, :, 1].clamp_(min=0.0)
    grid_v[:, -B:, :, 1].clamp_(max=0.0)
    grid_v[:, :, :B, 2].clamp_(min=0.0)
    grid_v[:, :, -B:, 2].clamp_(max=0.0)

    # Colliders
    for c in params.colliders:
        if isinstance(c, BoxCollider):
            lo = (torch.tensor(c.center) - torch.tensor(c.half_size)) / DX
            hi = (torch.tensor(c.center) + torch.tensor(c.half_size)) / DX
            lo, hi = lo.int().clamp(0, GR - 1), hi.int().clamp(0, GR - 1)
            grid_v[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = 0.0

        elif isinstance(c, WallCollider):
            dev = grid_v.device
            n = torch.tensor(c.normal, dtype=torch.float32, device=dev)
            n = n / n.norm()
            wp = torch.tensor(c.point, device=dev) * params.inv_dx
            gi = torch.arange(GR, dtype=torch.float32, device=dev)
            dist = ((gi[:, None, None] - wp[0]) * n[0]
                    + (gi[None, :, None] - wp[1]) * n[1]
                    + (gi[None, None, :] - wp[2]) * n[2])
            vn = (grid_v * n).sum(-1)
            mask = (dist < 1.0) & (vn < 0)
            grid_v[mask] -= vn[mask].unsqueeze(-1) * n

    return GridState(velocity=grid_v, mass=grid_m)
