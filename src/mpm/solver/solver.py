"""Timestep orchestrator: stress → P2G → grid_ops → G2P."""

from __future__ import annotations

import functools
import time

import torch

from mpm.params import SimParams
from mpm.state import ParticleState
from mpm.solver.stress import compute_stress
from mpm.solver.p2g import compute_p2g_data, scatter
from mpm.solver.grid_ops import update_grid
from mpm.solver.g2p import gather


class StepTimings:
    PHASES = ["stress", "p2g_data", "p2g_scatter", "grid_ops", "g2p"]

    def __init__(self):
        self.totals = {p: 0.0 for p in self.PHASES}
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

    def __init__(self, params: SimParams):
        self.params = params
        self.timings = StepTimings()
        self._cuda_events = None

    def _ensure_cuda_events(self):
        if self._cuda_events is None:
            phases = StepTimings.PHASES
            self._cuda_events = {
                p: (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                for p in phases
            }
        return self._cuda_events

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params

        if x.is_cuda:
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

        # Synchronize and collect timings
        torch.cuda.synchronize()
        t = {}
        for phase in StepTimings.PHASES:
            t[phase] = ev[phase][0].elapsed_time(ev[phase][1]) / 1000.0  # ms → s

        self.timings.record(t)
        return ParticleState(new_x, new_v, new_C, new_Fe, stress_result.Jp_new)


@functools.lru_cache(maxsize=8)
def build_step(params: SimParams) -> Stepper:
    return Stepper(params)
