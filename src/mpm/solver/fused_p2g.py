"""P2G scatter and fused stress+P2G via custom CUDA kernels.

Compiled at runtime using cuda.core (NVRTC). No C++ wrapper needed.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch
from mpm.params import SimParams
from mpm.state import GridState

# Compiled kernel cache
_kernels: dict[str, object] = {}


def _get_stream():
    """Get PyTorch's current CUDA stream (shares stream to avoid sync issues)."""
    return torch.cuda.current_stream()


def _compile_kernels():
    """Compile the .cu file via NVRTC and extract both kernels."""
    if _kernels:
        return

    src_path = os.path.join(os.path.dirname(__file__), "kernels", "fused_stress_p2g.cu")
    with open(src_path) as f:
        source = f.read()

    dev = Device(torch.cuda.current_device())
    dev.set_current()
    arch = f"sm_{dev.compute_capability[0]}{dev.compute_capability[1]}"

    prog = Program(source, code_type="c++",
                   options=ProgramOptions(std="c++17", arch=arch,
                                          use_fast_math=True))
    mod = prog.compile("cubin")

    _kernels["p2g"] = mod.get_kernel("p2g_kernel")
    _kernels["fused"] = mod.get_kernel("fused_stress_p2g_kernel")


def p2g_scatter_cuda(x, v, C, stress, params: SimParams, block_size: int = 256):
    """P2G scatter via custom CUDA kernel. Returns GridState."""
    _compile_kernels()

    N = x.shape[0]
    GR = params.grid_res
    GR3 = GR ** 3

    grid_v = torch.zeros(GR3, 3, dtype=torch.float32, device=x.device)
    grid_m = torch.zeros(GR3, dtype=torch.float32, device=x.device)

    grid_dim = (N + block_size - 1) // block_size
    config = LaunchConfig(grid=grid_dim, block=block_size)

    launch(_get_stream(), config, _kernels["p2g"],
           x.data_ptr(), v.data_ptr(), C.data_ptr(),
           stress.data_ptr(),
           grid_v.data_ptr(), grid_m.data_ptr(),
           np.int32(N), np.int32(GR),
           np.float32(params.dt), np.float32(params.inv_dx),
           np.float32(params.dx), np.float32(params.p_vol),
           np.float32(params.p_mass))

    return GridState(
        velocity=grid_v.reshape(GR, GR, GR, 3),
        mass=grid_m.reshape(GR, GR, GR),
    )


def fused_stress_p2g_cuda(x, v, C, Fe, Jp, params: SimParams, block_size: int = 256):
    """Fused stress + P2G via custom CUDA kernel. Returns (Fe_new, Jp_new, GridState)."""
    _compile_kernels()

    N = x.shape[0]
    GR = params.grid_res
    GR3 = GR ** 3

    Fe_new = torch.empty(N, 3, 3, dtype=torch.float32, device=x.device)
    Jp_new = torch.empty(N, dtype=torch.float32, device=x.device)
    grid_v = torch.zeros(GR3, 3, dtype=torch.float32, device=x.device)
    grid_m = torch.zeros(GR3, dtype=torch.float32, device=x.device)

    grid_dim = (N + block_size - 1) // block_size
    config = LaunchConfig(grid=grid_dim, block=block_size)

    launch(_get_stream(), config, _kernels["fused"],
           x.data_ptr(), v.data_ptr(), C.data_ptr(),
           Fe.data_ptr(), Jp.data_ptr(),
           Fe_new.data_ptr(), Jp_new.data_ptr(),
           grid_v.data_ptr(), grid_m.data_ptr(),
           np.int32(N), np.int32(GR),
           np.float32(params.dt), np.float32(params.inv_dx),
           np.float32(params.dx), np.float32(params.p_vol),
           np.float32(params.p_mass),
           np.float32(params.theta_c), np.float32(params.theta_s),
           np.float32(params.hardening),
           np.float32(params.mu_0), np.float32(params.lambda_0))

    return Fe_new, Jp_new, GridState(
        velocity=grid_v.reshape(GR, GR, GR, 3),
        mass=grid_m.reshape(GR, GR, GR),
    )
