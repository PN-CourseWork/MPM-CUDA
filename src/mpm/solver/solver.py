"""Timestep orchestrator: stress+P2G → grid_ops → G2P.

All backends use fused phases for fair comparison:
  - torch: PyTorch ops (torch.compile for stress)
  - jax: JAX jit-compiled
  - fused_cuda: hand-written CUDA kernel
"""

from __future__ import annotations

import time

import torch

from mpm.params import SimParams
from mpm.state import ParticleState
from mpm.solver.stress import compute_stress
from mpm.solver.p2g import compute_p2g_data, scatter
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


def _timed_cuda(fn, start_event, end_event, *args, **kwargs):
    start_event.record()
    result = fn(*args, **kwargs)
    end_event.record()
    return result


class Stepper:
    """Callable timestep integrator with built-in timing."""

    def __init__(self, params: SimParams, kernel_backend: str = "torch",
                 block_size: int = 256):
        self.params = params
        self.kernel_backend = kernel_backend
        self.block_size = block_size
        self._cuda_events = None
        self._cached_fn = None
        self.timings = StepTimings()

    def _ensure_cuda_events(self):
        if self._cuda_events is None:
            self._cuda_events = {
                p: (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                for p in PHASES
            }
        return self._cuda_events

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params

        if x.is_cuda:
            return self._step_cuda(x, v, C, F, Jp, p)
        else:
            return self._step_cpu(x, v, C, F, Jp, p)

    # --- Fused stress+P2G dispatch ---

    def _stress_p2g_torch(self, x, v, C, F, Jp, p):
        """Torch fused stress+P2G: compute stress then scatter."""
        stress_result = compute_stress(F, Jp, p)
        p2g_data = compute_p2g_data(x, v, C, stress_result.stress, p)
        grid = scatter(p2g_data, p.grid_res)
        stencil = compute_stencil(x, p)
        return stress_result.Fe_new, stress_result.Jp_new, grid, stencil

    def _stress_p2g_fused_cuda(self, x, v, C, F, Jp, p):
        """Hand-written CUDA kernel: stress+P2G in one launch."""
        if self._cached_fn is None:
            from mpm.solver.fused_p2g import fused_stress_p2g_cuda
            self._cached_fn = fused_stress_p2g_cuda
        Fe_new, Jp_new, grid = self._cached_fn(x, v, C, F, Jp, p, self.block_size)
        stencil = compute_stencil(x, p)
        return Fe_new, Jp_new, grid, stencil

    def _stress_p2g_jax(self, x, v, C, F, Jp, p):
        """JAX jit-compiled fused stress+P2G."""
        if self._cached_fn is None:
            from mpm.solver.fused_jax import fused_stress_p2g_jax
            self._cached_fn = fused_stress_p2g_jax
        Fe_new, Jp_new, grid = self._cached_fn(x, v, C, F, Jp, p, self.block_size)
        stencil = compute_stencil(x, p)
        return Fe_new, Jp_new, grid, stencil

    def _get_stress_p2g_fn(self):
        if self.kernel_backend == "fused_cuda":
            return self._stress_p2g_fused_cuda
        elif self.kernel_backend == "jax":
            return self._stress_p2g_jax
        else:
            return self._stress_p2g_torch

    # --- Step implementations ---

    def _step_cpu(self, x, v, C, F, Jp, p):
        t = {}
        do_sp = self._get_stress_p2g_fn()

        t0 = time.perf_counter()
        Fe_new, Jp_new, grid, stencil = do_sp(x, v, C, F, Jp, p)
        t["stress_p2g"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        grid = update_grid(grid, p)
        t["grid_ops"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_x, new_v, new_C, new_Fe = gather(grid.velocity, stencil, x, Fe_new, p)
        t["g2p"] = time.perf_counter() - t0

        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, Jp_new)

    def _step_cuda(self, x, v, C, F, Jp, p):
        ev = self._ensure_cuda_events()
        do_sp = self._get_stress_p2g_fn()

        Fe_new, Jp_new, grid, stencil = _timed_cuda(
            do_sp, ev["stress_p2g"][0], ev["stress_p2g"][1],
            x, v, C, F, Jp, p)

        grid = _timed_cuda(
            update_grid, ev["grid_ops"][0], ev["grid_ops"][1], grid, p)

        (new_x, new_v, new_C, new_Fe) = _timed_cuda(
            gather, ev["g2p"][0], ev["g2p"][1],
            grid.velocity, stencil, x, Fe_new, p)

        torch.cuda.synchronize()
        t = {}
        for phase in PHASES:
            t[phase] = ev[phase][0].elapsed_time(ev[phase][1]) / 1000.0
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, Jp_new)


def build_step(params: SimParams, kernel_cfg=None) -> Stepper:
    backend = "torch"
    block_size = 256
    if kernel_cfg is not None:
        backend = getattr(kernel_cfg, "backend", "torch")
        block_size = getattr(kernel_cfg, "block_size", 256)
    return Stepper(params, backend, block_size)
