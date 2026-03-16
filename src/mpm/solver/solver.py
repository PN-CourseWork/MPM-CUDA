"""Timestep orchestrator: stress → P2G → grid_ops → G2P."""

from __future__ import annotations

import functools
import time

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


class Stepper:
    """Callable timestep integrator with built-in timing."""

    def __init__(self, params: SimParams):
        self.params = params
        self.timings = StepTimings()

    def __call__(self, state: ParticleState) -> ParticleState:
        x, v, C, F, Jp = state
        p = self.params
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


@functools.lru_cache(maxsize=8)
def build_step(params: SimParams) -> Stepper:
    return Stepper(params)
