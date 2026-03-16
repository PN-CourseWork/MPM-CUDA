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


class StepTimings:
    def __init__(self, phases: list[str]):
        self.PHASES = phases
        self.totals = {p: 0.0 for p in phases}
        self.count = 0

    def reset(self):
        self.totals = {p: 0.0 for p in self.PHASES}
        self.count = 0

    def record(self, t: dict[str, float]):
        for p in self.PHASES:
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


TORCH_PHASES = ["stress", "p2g_data", "p2g_scatter", "grid_ops", "g2p"]
FUSED_PHASES = ["fused_stress_p2g", "grid_ops", "stencil", "g2p"]


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
                 block_size: int = 256, newton_schulz_iters: int = 3):
        self.params = params
        self.kernel_backend = kernel_backend
        self.block_size = block_size
        self.newton_schulz_iters = newton_schulz_iters

        phases = FUSED_PHASES if kernel_backend in ("fused_cuda", "fused_torch") else TORCH_PHASES
        self.timings = StepTimings(phases)
        self._cuda_events = None
        self._fused_fn = None

    def _ensure_cuda_events(self):
        if self._cuda_events is None:
            self._cuda_events = {
                p: (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                for p in self.timings.PHASES
            }
        return self._cuda_events

    def _get_fused_fn(self):
        if self._fused_fn is None:
            if self.kernel_backend == "fused_cuda":
                from mpm.solver.fused_p2g import fused_stress_p2g
                self._fused_fn = fused_stress_p2g
            else:
                from mpm.solver.fused_p2g_torch import fused_stress_p2g_torch
                self._fused_fn = fused_stress_p2g_torch
        return self._fused_fn

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params

        if self.kernel_backend in ("fused_cuda", "fused_torch"):
            return self._step_fused_cuda(x, v, C, F, Jp, p)
        elif x.is_cuda:
            return self._step_cuda(x, v, C, F, Jp, p)
        else:
            return self._step_cpu(x, v, C, F, Jp, p)

    def _step_cpu(self, x, v, C, F, Jp, p):
        t = {}
        stress_result, t["stress"] = _timed(compute_stress, F, Jp, p)
        p2g_data, t["p2g_data"] = _timed(compute_p2g_data, x, v, C, stress_result.stress, p)
        grid, t["p2g_scatter"] = _timed(scatter, p2g_data, p.grid_res)
        grid, t["grid_ops"] = _timed(update_grid, grid, p)
        (new_x, new_v, new_C, new_Fe), t["g2p"] = _timed(
            gather, grid.velocity, p2g_data, x, stress_result.Fe_new, p
        )
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, stress_result.Jp_new)

    def _step_cuda(self, x, v, C, F, Jp, p):
        ev = self._ensure_cuda_events()

        stress_result = _timed_cuda(
            compute_stress, ev["stress"][0], ev["stress"][1], F, Jp, p)
        p2g_data = _timed_cuda(
            compute_p2g_data, ev["p2g_data"][0], ev["p2g_data"][1],
            x, v, C, stress_result.stress, p)
        grid = _timed_cuda(
            scatter, ev["p2g_scatter"][0], ev["p2g_scatter"][1],
            p2g_data, p.grid_res)
        grid = _timed_cuda(
            update_grid, ev["grid_ops"][0], ev["grid_ops"][1], grid, p)
        (new_x, new_v, new_C, new_Fe) = _timed_cuda(
            gather, ev["g2p"][0], ev["g2p"][1],
            grid.velocity, p2g_data, x, stress_result.Fe_new, p)

        torch.cuda.synchronize()
        t = {}
        for phase in self.timings.PHASES:
            t[phase] = ev[phase][0].elapsed_time(ev[phase][1]) / 1000.0
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, stress_result.Jp_new)

    def _step_fused_cuda(self, x, v, C, F, Jp, p):
        ev = self._ensure_cuda_events()
        fused_fn = self._get_fused_fn()

        Fe_new, Jp_new, grid = _timed_cuda(
            fused_fn, ev["fused_stress_p2g"][0], ev["fused_stress_p2g"][1],
            x, v, C, F, Jp, p, self.block_size, self.newton_schulz_iters)

        grid = _timed_cuda(
            update_grid, ev["grid_ops"][0], ev["grid_ops"][1], grid, p)

        stencil = _timed_cuda(
            compute_stencil, ev["stencil"][0], ev["stencil"][1], x, p)

        (new_x, new_v, new_C, new_Fe) = _timed_cuda(
            gather, ev["g2p"][0], ev["g2p"][1],
            grid.velocity, stencil, x, Fe_new, p)

        torch.cuda.synchronize()
        t = {}
        for phase in self.timings.PHASES:
            t[phase] = ev[phase][0].elapsed_time(ev[phase][1]) / 1000.0
        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, Jp_new)


def build_step(params: SimParams, kernel_cfg=None) -> Stepper:
    backend = "torch"
    block_size = 256
    ns_iters = 3
    if kernel_cfg is not None:
        backend = getattr(kernel_cfg, "backend", "torch")
        block_size = getattr(kernel_cfg, "block_size", 256)
        ns_iters = getattr(kernel_cfg, "newton_schulz_iters", 3)
    return Stepper(params, backend, block_size, ns_iters)
