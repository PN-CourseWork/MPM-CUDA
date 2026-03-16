"""CUDA solver: compile and launch hand-written CUDA kernels via cuda.core.

Supports three kernel versions (selectable via kernel_version):
  v1_naive  — 4 kernel launches per step (separate stress, P2G, grid_ops, G2P)
  v2_fused  — 3 launches (fused stress+P2G, grid_ops, G2P with __ldg)
  v3_warp   — 3 launches (warp-per-particle P2G+G2P with shuffle reduce)

Each version is a self-contained .cu file in src/mpm/solver/kernels/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from mpm.params import SimParams
from mpm.state import ParticleState

log = logging.getLogger(__name__)

_KERNEL_DIR = Path(__file__).parent / "kernels"

# Kernel version → (.cu file, kernel names, warp-based flag)
_VERSIONS = {
    "v1_naive": (
        "v1_naive.cu",
        ["stress_kernel", "p2g_kernel", "grid_ops_kernel", "g2p_kernel"],
        False,
    ),
    "v2_fused": (
        "v2_fused.cu",
        ["fused_stress_p2g_kernel", "grid_ops_kernel", "g2p_kernel"],
        False,
    ),
    "v3_warp": (
        "v3_warp.cu",
        ["fused_stress_p2g_warp_kernel", "grid_ops_kernel", "g2p_warp_kernel"],
        True,
    ),
}


def _compile_kernels(cu_file: str, kernel_names: list[str], arch: str = "sm_90"):
    """Compile CUDA source via NVRTC and return kernel handles."""
    from cuda.core.experimental import Device, Program

    dev = Device(0)
    dev.set_current()

    source = (_KERNEL_DIR / cu_file).read_text()

    prog = Program(source, "utf-8")
    mod = prog.compile(
        "cubin",
        options=[f"-arch={arch}", "--use_fast_math", "-std=c++17"],
    )

    kernels = {}
    for name in kernel_names:
        kernels[name] = mod.get_kernel(name)
    return dev, kernels


def _to_gpu(arr: np.ndarray):
    """Allocate device memory and copy numpy array to GPU."""
    from cuda.core.experimental import Device
    import cuda.core.experimental as cuda_core

    nbytes = arr.nbytes
    buf = Device(0).allocate(nbytes)
    buf.copy_from(arr.ctypes.data, nbytes)
    return buf


def _from_gpu(buf, shape, dtype=np.float32):
    """Copy device buffer to numpy array."""
    arr = np.empty(shape, dtype=dtype)
    buf.copy_to(arr.ctypes.data, arr.nbytes)
    return arr


def _zeros_gpu(shape, dtype=np.float32):
    """Allocate zeroed GPU memory."""
    from cuda.core.experimental import Device

    arr = np.zeros(shape, dtype=dtype)
    return _to_gpu(arr)


class CUDAStepper:
    """CUDA kernel-based MPM stepper.

    Manages GPU memory and kernel dispatch for a complete timestep.
    """

    def __init__(self, params: SimParams, kernel_version: str = "v2_fused",
                 block_size: int = 256):
        if kernel_version not in _VERSIONS:
            raise ValueError(f"Unknown kernel version: {kernel_version}. "
                             f"Available: {list(_VERSIONS.keys())}")

        self.params = params
        self.kernel_version = kernel_version
        self.block_size = block_size
        self._step_count = 0
        self._wall_time = 0.0

        cu_file, kernel_names, self.warp_based = _VERSIONS[kernel_version]

        log.info(f"Compiling CUDA kernels ({kernel_version}: {cu_file}) …")
        self.dev, self.kernels = _compile_kernels(cu_file, kernel_names)
        log.info(f"Compiled {len(self.kernels)} kernels: {kernel_names}")

        # Pre-allocate grid buffers (reused every step)
        GR = params.grid_res
        self._grid_v = _zeros_gpu((GR * GR * GR * 3,))
        self._grid_m = _zeros_gpu((GR * GR * GR,))

    def step(self, state: ParticleState) -> ParticleState:
        """Single timestep on GPU."""
        x_np = np.asarray(state.x, dtype=np.float32)
        v_np = np.asarray(state.v, dtype=np.float32)
        C_np = np.asarray(state.C, dtype=np.float32).reshape(-1)
        Fe_np = np.asarray(state.F, dtype=np.float32).reshape(-1)
        Jp_np = np.asarray(state.Jp, dtype=np.float32)

        N = x_np.shape[0]
        p = self.params

        # Upload to GPU
        d_x = _to_gpu(x_np.reshape(-1))
        d_v = _to_gpu(v_np.reshape(-1))
        d_C = _to_gpu(C_np)
        d_Fe = _to_gpu(Fe_np)
        d_Jp = _to_gpu(Jp_np)

        # Output buffers
        d_Fe_new = _zeros_gpu((N * 9,))
        d_Jp_new = _zeros_gpu((N,))
        d_x_out = _zeros_gpu((N * 3,))
        d_v_out = _zeros_gpu((N * 3,))
        d_C_out = _zeros_gpu((N * 9,))
        d_Fe_out = _zeros_gpu((N * 9,))

        # Zero grid
        self._zero_grid()

        GR = p.grid_res
        bs = self.block_size

        t0 = time.perf_counter()

        if self.kernel_version == "v1_naive":
            self._step_v1(d_x, d_v, d_C, d_Fe, d_Jp,
                          d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                          N, p, bs, GR)
        elif self.kernel_version == "v2_fused":
            self._step_v2(d_x, d_v, d_C, d_Fe, d_Jp,
                          d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                          N, p, bs, GR)
        elif self.kernel_version == "v3_warp":
            self._step_v3(d_x, d_v, d_C, d_Fe, d_Jp,
                          d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                          N, p, bs, GR)

        self.dev.sync()
        self._wall_time += time.perf_counter() - t0
        self._step_count += 1

        # Download results
        import jax.numpy as jnp
        x_out = jnp.array(_from_gpu(d_x_out, (N, 3)))
        v_out = jnp.array(_from_gpu(d_v_out, (N, 3)))
        C_out = jnp.array(_from_gpu(d_C_out, (N, 3, 3)))
        Fe_out = jnp.array(_from_gpu(d_Fe_out, (N, 3, 3)))
        Jp_out = jnp.array(_from_gpu(d_Jp_new, (N,)))

        return ParticleState(x_out, v_out, C_out, Fe_out, Jp_out)

    def _zero_grid(self):
        """Zero the grid velocity and mass buffers."""
        GR = self.params.grid_res
        total = GR * GR * GR
        zeros_v = np.zeros(total * 3, dtype=np.float32)
        zeros_m = np.zeros(total, dtype=np.float32)
        self._grid_v.copy_from(zeros_v.ctypes.data, zeros_v.nbytes)
        self._grid_m.copy_from(zeros_m.ctypes.data, zeros_m.nbytes)

    def _grid_blocks(self, GR, bs):
        total = GR * GR * GR
        return (total + bs - 1) // bs

    def _particle_blocks(self, N, bs):
        return (N + bs - 1) // bs

    def _warp_blocks(self, N, bs):
        """For warp-per-particle: N warps needed, bs threads per block."""
        total_threads = N * 32
        return (total_threads + bs - 1) // bs

    def _step_v1(self, d_x, d_v, d_C, d_Fe, d_Jp,
                 d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                 N, p, bs, GR):
        """v1: stress → P2G → grid_ops → G2P (4 kernels)."""
        # Stress output: affine matrix stored in d_C_out temporarily
        d_aff = _zeros_gpu((N * 9,))

        pb = self._particle_blocks(N, bs)
        gb = self._grid_blocks(GR, bs)

        # Kernel 1: stress
        self.kernels["stress_kernel"].launch(
            grid=(pb, 1, 1), block=(bs, 1, 1),
            args=[d_Fe.handle, d_Jp.handle, d_C.handle,
                  d_aff.handle, d_Fe_new.handle, d_Jp_new.handle,
                  np.int32(N),
                  np.float32(p.theta_c), np.float32(p.theta_s),
                  np.float32(p.hardening),
                  np.float32(p.mu_0), np.float32(p.lambda_0),
                  np.float32(p.dt), np.float32(p.p_vol),
                  np.float32(p.inv_dx), np.float32(p.p_mass)],
        )

        # Kernel 2: P2G
        self.kernels["p2g_kernel"].launch(
            grid=(pb, 1, 1), block=(bs, 1, 1),
            args=[d_x.handle, d_v.handle, d_aff.handle,
                  self._grid_v.handle, self._grid_m.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.inv_dx), np.float32(p.dx), np.float32(p.p_mass)],
        )

        # Kernel 3: grid ops
        self.kernels["grid_ops_kernel"].launch(
            grid=(gb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, self._grid_m.handle,
                  np.int32(GR), np.float32(p.dt),
                  np.float32(p.gravity[0]), np.float32(p.gravity[1]),
                  np.float32(p.gravity[2]),
                  np.int32(p.bound)],
        )

        # Kernel 4: G2P
        self.kernels["g2p_kernel"].launch(
            grid=(pb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, d_x.handle, d_Fe_new.handle,
                  d_x_out.handle, d_v_out.handle, d_C_out.handle, d_Fe_out.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.dt), np.float32(p.inv_dx), np.float32(p.dx)],
        )

    def _step_v2(self, d_x, d_v, d_C, d_Fe, d_Jp,
                 d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                 N, p, bs, GR):
        """v2: fused_stress_p2g → grid_ops → G2P (3 kernels)."""
        pb = self._particle_blocks(N, bs)
        gb = self._grid_blocks(GR, bs)

        # Kernel 1: fused stress + P2G
        self.kernels["fused_stress_p2g_kernel"].launch(
            grid=(pb, 1, 1), block=(bs, 1, 1),
            args=[d_x.handle, d_v.handle, d_C.handle, d_Fe.handle, d_Jp.handle,
                  d_Fe_new.handle, d_Jp_new.handle,
                  self._grid_v.handle, self._grid_m.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.dt), np.float32(p.inv_dx), np.float32(p.dx),
                  np.float32(p.p_vol), np.float32(p.p_mass),
                  np.float32(p.theta_c), np.float32(p.theta_s),
                  np.float32(p.hardening),
                  np.float32(p.mu_0), np.float32(p.lambda_0)],
        )

        # Kernel 2: grid ops
        self.kernels["grid_ops_kernel"].launch(
            grid=(gb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, self._grid_m.handle,
                  np.int32(GR), np.float32(p.dt),
                  np.float32(p.gravity[0]), np.float32(p.gravity[1]),
                  np.float32(p.gravity[2]),
                  np.int32(p.bound)],
        )

        # Kernel 3: G2P
        self.kernels["g2p_kernel"].launch(
            grid=(pb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, d_x.handle, d_Fe_new.handle,
                  d_x_out.handle, d_v_out.handle, d_C_out.handle, d_Fe_out.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.dt), np.float32(p.inv_dx), np.float32(p.dx)],
        )

    def _step_v3(self, d_x, d_v, d_C, d_Fe, d_Jp,
                 d_Fe_new, d_Jp_new, d_x_out, d_v_out, d_C_out, d_Fe_out,
                 N, p, bs, GR):
        """v3: warp-per-particle stress+P2G → grid_ops → warp G2P (3 kernels)."""
        wb = self._warp_blocks(N, bs)
        gb = self._grid_blocks(GR, bs)

        # Kernel 1: warp-per-particle stress + P2G
        self.kernels["fused_stress_p2g_warp_kernel"].launch(
            grid=(wb, 1, 1), block=(bs, 1, 1),
            args=[d_x.handle, d_v.handle, d_C.handle, d_Fe.handle, d_Jp.handle,
                  d_Fe_new.handle, d_Jp_new.handle,
                  self._grid_v.handle, self._grid_m.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.dt), np.float32(p.inv_dx), np.float32(p.dx),
                  np.float32(p.p_vol), np.float32(p.p_mass),
                  np.float32(p.theta_c), np.float32(p.theta_s),
                  np.float32(p.hardening),
                  np.float32(p.mu_0), np.float32(p.lambda_0)],
        )

        # Kernel 2: grid ops (same as v2)
        self.kernels["grid_ops_kernel"].launch(
            grid=(gb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, self._grid_m.handle,
                  np.int32(GR), np.float32(p.dt),
                  np.float32(p.gravity[0]), np.float32(p.gravity[1]),
                  np.float32(p.gravity[2]),
                  np.int32(p.bound)],
        )

        # Kernel 3: warp-per-particle G2P
        self.kernels["g2p_warp_kernel"].launch(
            grid=(wb, 1, 1), block=(bs, 1, 1),
            args=[self._grid_v.handle, d_x.handle, d_Fe_new.handle,
                  d_x_out.handle, d_v_out.handle, d_C_out.handle, d_Fe_out.handle,
                  np.int32(N), np.int32(GR),
                  np.float32(p.dt), np.float32(p.inv_dx), np.float32(p.dx)],
        )

    def scan(self, state: ParticleState, n_steps: int) -> ParticleState:
        """Run n_steps, keeping data on GPU between steps."""
        # For now, simple Python loop (each step does H2D/D2H).
        # A production version would keep all state on GPU.
        for _ in range(n_steps):
            state = self.step(state)
        self._step_count += n_steps - n_steps  # already counted in step()
        return state

    def trajectory(self, state: ParticleState, n_saves: int,
                   save_every: int):
        """Run n_saves * save_every steps, collecting positions."""
        import jax.numpy as jnp
        saved = []
        for _ in range(n_saves):
            for _ in range(save_every):
                state = self.step(state)
            saved.append(np.asarray(state.x))
        saved_x = jnp.stack([jnp.array(s) for s in saved])
        return state, saved_x

    def reset_timer(self):
        self._step_count = 0
        self._wall_time = 0.0

    def report(self, wall_time: float | None = None) -> str:
        t = wall_time or self._wall_time
        if self._step_count == 0:
            return "No steps recorded."
        ms = t / self._step_count * 1000
        return (f"Timings ({self._step_count} steps, {t:.3f}s wall, "
                f"{ms:.2f} ms/step)")
