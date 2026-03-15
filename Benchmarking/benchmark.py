"""Benchmark solver step: CPU vs MPS with proper sync."""

import time
import torch
from mpm.problems.particles import make_sphere, init_state
from mpm.params import SimParams
from mpm.solver.solver import build_step

N = 10000
WARMUP = 5
ITERS = 50

pos, vel = make_sphere([0.5, 0.5, 0.5], 0.1, N)

for dev_name in ["cpu", "mps"]:
    if dev_name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, skipping")
        continue

    dev = torch.device(dev_name)
    state = init_state(pos, vel, device=dev)
    params = SimParams(grid_res=128)
    build_step.cache_clear()
    step = build_step(params)

    # Warmup
    s = state
    for _ in range(WARMUP):
        s = step(s)
    if dev_name == "mps":
        torch.mps.synchronize()

    # Pure wall-clock timing (no per-phase overhead)
    step.timings.reset()
    if dev_name == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        s = step(s)
    if dev_name == "mps":
        torch.mps.synchronize()
    wall = time.perf_counter() - t0

    ms_per_step = wall / ITERS * 1000
    print(f"\n=== {dev_name.upper()} ({N} particles, {ITERS} steps) ===")
    print(f"  {ms_per_step:.2f} ms/step  ({1000/ms_per_step:.0f} steps/sec)")
    print(step.timings.report(wall))
