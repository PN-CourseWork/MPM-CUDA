"""Timestep orchestrator: stress -> P2G -> grid_ops -> G2P."""

from __future__ import annotations

import time

import torch

from mpm.params import SimParams
from mpm.state import ParticleState
from mpm.solver.stress import compute_stress
from mpm.solver.p2g import compute_p2g_data, scatter
from mpm.solver.grid_ops import update_grid
from mpm.solver.g2p import gather, compute_stencil


PHASES = ["stress", "p2g", "grid_ops", "g2p"]


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
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


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
        self.timings = StepTimings()
        self._cuda_events = None
        self._p2g_fn = None

    def _ensure_cuda_events(self):
        if self._cuda_events is None:
            self._cuda_events = {
                p: (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                for p in PHASES
            }
        return self._cuda_events

    def _get_p2g_fn(self):
        """Return the P2G function for the selected backend."""
        if self._p2g_fn is None:
            if self.kernel_backend == "cuda":
                from mpm.solver.fused_p2g import p2g_scatter_cuda
                self._p2g_fn = p2g_scatter_cuda
            # torch backend uses compute_p2g_data + scatter (inline below)
        return self._p2g_fn

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params

        if x.is_cuda:
            return self._step_cuda(x, v, C, F, Jp, p)
        else:
            return self._step_cpu(x, v, C, F, Jp, p)

    def _do_p2g_torch(self, x, v, C, stress, p):
        """P2G via PyTorch ops (compute_p2g_data + scatter)."""
        p2g_data = compute_p2g_data(x, v, C, stress, p)
        grid = scatter(p2g_data, p.grid_res)
        return grid, p2g_data  # p2g_data reused by G2P

    def _do_p2g_cuda(self, x, v, C, stress, p):
        """P2G via custom CUDA kernel."""
        p2g_fn = self._get_p2g_fn()
        grid = p2g_fn(x, v, C, stress, p, self.block_size)
        stencil = compute_stencil(x, p)  # recompute for G2P
        return grid, stencil

    def _step_cpu(self, x, v, C, F, Jp, p):
        t = {}
        stress_result, t["stress"] = _timed(compute_stress, F, Jp, p)
        (grid, stencil), t["p2g"] = _timed(self._do_p2g_torch, x, v, C, stress_result.stress, p)
        grid, t["grid_ops"] = _timed(update_grid, grid, p)
        (new_x, new_v, new_C, new_Fe), t["g2p"] = _timed(
            gather, grid.velocity, stencil, x, stress_result.Fe_new, p
        )
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, stress_result.Jp_new)

    def _step_cuda(self, x, v, C, F, Jp, p):
        ev = self._ensure_cuda_events()
        use_cuda_kernel = self.kernel_backend == "cuda"

        stress_result = _timed_cuda(
            compute_stress, ev["stress"][0], ev["stress"][1], F, Jp, p)

        do_p2g = self._do_p2g_cuda if use_cuda_kernel else self._do_p2g_torch
        grid, stencil = _timed_cuda(
            do_p2g, ev["p2g"][0], ev["p2g"][1],
            x, v, C, stress_result.stress, p)

        grid = _timed_cuda(
            update_grid, ev["grid_ops"][0], ev["grid_ops"][1], grid, p)

        (new_x, new_v, new_C, new_Fe) = _timed_cuda(
            gather, ev["g2p"][0], ev["g2p"][1],
            grid.velocity, stencil, x, stress_result.Fe_new, p)

        torch.cuda.synchronize()
        t = {}
        for phase in PHASES:
            t[phase] = ev[phase][0].elapsed_time(ev[phase][1]) / 1000.0
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, stress_result.Jp_new)


def build_step(params: SimParams, kernel_cfg=None) -> Stepper:
    backend = "torch"
    block_size = 256
    if kernel_cfg is not None:
        backend = getattr(kernel_cfg, "backend", "torch")
        block_size = getattr(kernel_cfg, "block_size", 256)
    return Stepper(params, backend, block_size)
