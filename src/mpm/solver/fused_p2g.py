"""Fused stress+P2G via custom CUDA kernel.
Compiled at runtime using cuda.core (NVRTC). No C++ wrapper needed.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch
from mpm.params import SimParams
from mpm.state import GridState

# Compiled kernel and stream cache
_kernel = None
_stream = None


def _get_stream():
    """Get (or create) a cuda.core stream for the default device."""
    global _stream
    if _stream is None:
        dev = Device(0)
        dev.set_current()
        _stream = dev.create_stream()
    return _stream


def _ptr(a):
    """Get CUDA device pointer from a JAX array."""
    return a.__cuda_array_interface__['data'][0]


def _compile_kernel():
    """Compile the .cu file via NVRTC and extract the fused kernel."""
    global _kernel
    if _kernel is not None:
        return

    src_path = os.path.join(os.path.dirname(__file__), "kernels", "fused_stress_p2g.cu")
    with open(src_path) as f:
        source = f.read()

    dev = Device(0)
    dev.set_current()
    arch = f"sm_{dev.compute_capability[0]}{dev.compute_capability[1]}"

    prog = Program(source, code_type="c++",
                   options=ProgramOptions(std="c++17", arch=arch,
                                          use_fast_math=True))
    mod = prog.compile("cubin")
    _kernel = mod.get_kernel("fused_stress_p2g_kernel")


def fused_stress_p2g_cuda(x, v, C, Fe, Jp, params: SimParams, block_size: int = 256):
    """Fused stress + P2G via custom CUDA kernel. Returns (Fe_new, Jp_new, GridState)."""
    _compile_kernel()

    N = x.shape[0]
    GR = params.grid_res
    GR3 = GR ** 3

    # Ensure all inputs are materialized on device before reading pointers
    jax.block_until_ready((x, v, C, Fe, Jp))

    # Allocate outputs
    Fe_new = jnp.zeros((N, 3, 3), dtype=jnp.float32)
    Jp_new = jnp.zeros((N,), dtype=jnp.float32)
    grid_v = jnp.zeros((GR3, 3), dtype=jnp.float32)
    grid_m = jnp.zeros((GR3,), dtype=jnp.float32)

    # Materialize output buffers before passing pointers to the kernel
    jax.block_until_ready((Fe_new, Jp_new, grid_v, grid_m))

    grid_dim = (N + block_size - 1) // block_size
    config = LaunchConfig(grid=grid_dim, block=block_size)

    stream = _get_stream()
    launch(stream, config, _kernel,
           _ptr(x), _ptr(v), _ptr(C),
           _ptr(Fe), _ptr(Jp),
           _ptr(Fe_new), _ptr(Jp_new),
           _ptr(grid_v), _ptr(grid_m),
           np.int32(N), np.int32(GR),
           np.float32(params.dt), np.float32(params.inv_dx),
           np.float32(params.dx), np.float32(params.p_vol),
           np.float32(params.p_mass),
           np.float32(params.theta_c), np.float32(params.theta_s),
           np.float32(params.hardening),
           np.float32(params.mu_0), np.float32(params.lambda_0))

    # Synchronize stream before returning so JAX sees completed writes
    stream.sync()

    return Fe_new, Jp_new, GridState(
        velocity=grid_v.reshape(GR, GR, GR, 3),
        mass=grid_m.reshape(GR, GR, GR),
    )
