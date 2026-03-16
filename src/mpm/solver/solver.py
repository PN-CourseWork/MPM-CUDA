"""Timestep orchestrator.

Two modes:
  - step(state)        — single jitted timestep (warmup, per-step saving)
  - scan(state, n)     — N steps fused via lax.scan (fast bulk stepping)
"""

from __future__ import annotations

import time

import jax

from mpm.params import SimParams
from mpm.state import ParticleState
from mpm.solver.fused_jax import full_step, scan_steps, scan_trajectory


class Stepper:
    """Callable timestep integrator."""

    def __init__(self, params: SimParams):
        self.params = params
        self._step_count = 0
        self._wall_time = 0.0

    def __call__(self, state: ParticleState) -> ParticleState:
        """Single timestep (jitted)."""
        x, v, C, F, Jp = state
        jax.block_until_ready(state)
        t0 = time.perf_counter()
        x, v, C, F, Jp = full_step(x, v, C, F, Jp, self.params)
        jax.block_until_ready((x, v, C, F, Jp))
        self._wall_time += time.perf_counter() - t0
        self._step_count += 1
        return ParticleState(x, v, C, F, Jp)

    def scan(self, state: ParticleState, n_steps: int) -> ParticleState:
        """Run n_steps fused into one XLA program."""
        x, v, C, F, Jp = state
        x, v, C, F, Jp = scan_steps(x, v, C, F, Jp, self.params, n_steps)
        self._step_count += n_steps
        return ParticleState(x, v, C, F, Jp)

    def trajectory(self, state: ParticleState, n_saves: int,
                   save_every: int) -> tuple[ParticleState, "jax.Array"]:
        """Run n_saves * save_every steps, return (final_state, saved_x).

        The entire trajectory is one XLA program (nested lax.scan).
        saved_x has shape (n_saves, N, 3).
        """
        x, v, C, F, Jp = state
        x, v, C, F, Jp, saved_x = scan_trajectory(
            x, v, C, F, Jp, self.params, n_saves, save_every)
        self._step_count += n_saves * save_every
        return ParticleState(x, v, C, F, Jp), saved_x

    def reset_timer(self):
        self._step_count = 0
        self._wall_time = 0.0

    def report(self, wall_time: float | None = None) -> str:
        t = wall_time or self._wall_time
        if self._step_count == 0:
            return "No steps recorded."
        return f"Timings ({self._step_count} steps, {t:.3f}s wall)"


def build_step(params: SimParams, kernel_cfg=None) -> Stepper:
    """Create a stepper based on kernel config.

    kernel_cfg.backend:
      "jax"        → JAX XLA solver (default)
      "fused_cuda" → Hand-written CUDA kernels (requires GPU)

    For CUDA backend, kernel_cfg.version selects the optimization level:
      "v1_naive"  — 4 kernel launches, global atomicAdd
      "v2_fused"  — 3 launches, fused stress+P2G
      "v3_warp"   — 3 launches, warp-per-particle scatter/gather
    """
    backend = getattr(kernel_cfg, "backend", "jax") if kernel_cfg else "jax"

    if backend == "fused_cuda":
        from mpm.solver.cuda_solver import CUDAStepper
        version = getattr(kernel_cfg, "version", "v2_fused")
        block_size = getattr(kernel_cfg, "block_size", 256)
        return CUDAStepper(params, kernel_version=version, block_size=block_size)

    return Stepper(params)
