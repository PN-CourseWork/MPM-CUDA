"""P2G scatter via custom CUDA kernel."""

from __future__ import annotations

import os

from torch.utils.cpp_extension import load

from mpm.params import SimParams
from mpm.state import GridState

_module = None


def _get_module():
    global _module
    if _module is None:
        src_dir = os.path.join(os.path.dirname(__file__), "kernels")
        _module = load(
            name="p2g_cuda",
            sources=[
                os.path.join(src_dir, "fused_stress_p2g.cpp"),
                os.path.join(src_dir, "fused_stress_p2g.cu"),
            ],
            extra_cuda_cflags=[
                "-arch=sm_90", "-O3", "--use_fast_math",
                "-lineinfo", "-Xptxas=-v",
            ],
        )
    return _module


def p2g_scatter_cuda(x, v, C, stress, params: SimParams, block_size: int = 256):
    """P2G scatter via custom CUDA kernel. Returns GridState."""
    mod = _get_module()
    grid_v, grid_m = mod.p2g(
        x.contiguous(), v.contiguous(), C.contiguous(),
        stress.contiguous(),
        params.grid_res, params.dt, params.inv_dx, params.dx,
        params.p_vol, params.p_mass,
        block_size,
    )
    return GridState(velocity=grid_v, mass=grid_m)
