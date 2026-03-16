"""Fused stress + P2G via custom CUDA kernel."""

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
            name="fused_stress_p2g",
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


def fused_stress_p2g(x, v, C, Fe, Jp, params: SimParams,
                     block_size: int = 256, newton_schulz_iters: int = 3):
    """Run fused stress + P2G. Returns (Fe_new, Jp_new, GridState)."""
    mod = _get_module()
    Fe_new, Jp_new, grid_v, grid_m = mod.fused_stress_p2g(
        x.contiguous(), v.contiguous(), C.contiguous(),
        Fe.contiguous(), Jp.contiguous(),
        params.grid_res, params.dt, params.inv_dx, params.dx,
        params.p_vol, params.p_mass,
        params.theta_c, params.theta_s, params.hardening,
        params.mu_0, params.lambda_0,
        block_size, newton_schulz_iters,
    )
    return Fe_new, Jp_new, GridState(velocity=grid_v, mass=grid_m)
