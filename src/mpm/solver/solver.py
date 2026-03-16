"""Timestep orchestrator: stress+P2G → grid_ops → G2P.

Two backends for comparison:
  - jax:        JAX jit-compiled — baseline
  - fused_cuda: hand-written CUDA kernel
"""

from __future__ import annotations

import time

import jax

from mpm.params import SimParams
from mpm.state import ParticleState
from mpm.solver.grid_ops import update_grid
from mpm.solver.g2p import gather, compute_stencil


PHASES = ["stress_p2g", "grid_ops", "g2p"]


class StepTimings:
    def __init__(self):
        self.totals = {p: 0.0 for p in PHASES}
        self.count = 0

    def reset(self):
        self.totals = {p: 0.0 for p in PHASES}
        self.count = 0

    def record(self, t: dict[str, float]):
        for p in PHASES:
            self.totals[p] += t.get(p, 0.0)
        self.count += 1

    def report(self, wall_time: float | None = None) -> str:
        if self.count == 0:
            return "No steps recorded."
        total = wall_time or sum(self.totals.values())
        entries = sorted(self.totals.items(), key=lambda x: -x[1])
        if wall_time:
            entries.append(("overhead", wall_time - sum(self.totals.values())))
            entries.sort(key=lambda x: -x[1])
        lines = [f"Timings ({self.count} steps, {total:.3f}s wall):"]
        for name, t in entries:
            lines.append(f"  {name:15s}  {t:7.3f}s  ({100*t/total:5.1f}%)")
        return "\n".join(lines)


def _timed(fn, *args, **kwargs):
    """Time a function with proper JAX synchronization."""
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    return result, time.perf_counter() - t0


class Stepper:
    """Callable timestep integrator with built-in timing."""

    def __init__(self, params: SimParams, kernel_backend: str = "jax",
                 block_size: int = 256):
        self.params = params
        self.kernel_backend = kernel_backend
        self.block_size = block_size
        self._cached_fn = None
        self.timings = StepTimings()

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params

        t = {}
        do_sp = self._get_stress_p2g_fn()

        (Fe_new, Jp_new, grid, stencil), t["stress_p2g"] = _timed(
            do_sp, x, v, C, F, Jp, p)

        grid, t["grid_ops"] = _timed(update_grid, grid, p)

        (new_x, new_v, new_C, new_Fe), t["g2p"] = _timed(
            gather, grid.velocity, stencil, x, Fe_new, p)

        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, Jp_new)

    # --- Fused stress+P2G dispatch ---

    def _stress_p2g_jax(self, x, v, C, F, Jp, p):
        """JAX jit-compiled fused stress+P2G."""
        if self._cached_fn is None:
            from mpm.solver.fused_jax import fused_stress_p2g_jax
            self._cached_fn = fused_stress_p2g_jax
        Fe_new, Jp_new, grid = self._cached_fn(x, v, C, F, Jp, p, self.block_size)
        stencil = compute_stencil(x, p)
        return Fe_new, Jp_new, grid, stencil

    def _stress_p2g_fused_cuda(self, x, v, C, F, Jp, p):
        """Hand-written CUDA kernel: stress+P2G in one launch."""
        if self._cached_fn is None:
            from mpm.solver.fused_p2g import fused_stress_p2g_cuda
            self._cached_fn = fused_stress_p2g_cuda
        Fe_new, Jp_new, grid = self._cached_fn(x, v, C, F, Jp, p, self.block_size)
        stencil = compute_stencil(x, p)
        return Fe_new, Jp_new, grid, stencil

    def _get_stress_p2g_fn(self):
        if self.kernel_backend == "fused_cuda":
            return self._stress_p2g_fused_cuda
        else:
            return self._stress_p2g_jax


def build_step(params: SimParams, kernel_cfg=None) -> Stepper:
    backend = "jax"
    block_size = 256
    if kernel_cfg is not None:
        backend = getattr(kernel_cfg, "backend", "jax")
        block_size = getattr(kernel_cfg, "block_size", 256)
    return Stepper(params, backend, block_size)
