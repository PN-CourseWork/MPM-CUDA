#!/usr/bin/env python
"""MPM Snow Simulation — Hydra-configured driver."""

from __future__ import annotations

import logging
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from mpm.problems.scene import build_scene
from mpm.solver.solver import build_step
from mpm.io.writer import FrameWriter

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    scene_name, params, state = build_scene(cfg)
    step = build_step(params)

    total_steps = cfg.run.steps
    save_every = cfg.run.get("save_every", 1)
    log_every = cfg.run.get("log_every", 1)

    output_dir = HydraConfig.get().runtime.output_dir

    n_particles = state.x.shape[0]
    sim_time = total_steps * params.dt
    log.info(f"Scene: {scene_name} — {cfg.scene.description}")
    log.info(f"Particles: {n_particles} | Grid: {params.grid_res}³ | "
             f"Steps: {total_steps} | dt: {params.dt} | "
             f"Physics time: {sim_time:.4f}s | Device: {state.x.device.type}")

    log.info("Warmup …")
    state = step(state)
    log.info("Warmup done.")

    log.info(f"Simulating {total_steps} steps …")
    step.timings.reset()
    t0 = time.time()
    with FrameWriter(f"{output_dir}/frames") as writer:
        for i in range(total_steps):
            state = step(state)
            if (i + 1) % save_every == 0:
                writer.append(state.x)
            if (i + 1) % log_every == 0 or i == total_steps - 1:
                log.info(f"  step {i + 1}/{total_steps} ({time.time() - t0:.1f}s)")

    wall_time = time.time() - t0
    log.info(f"Done → {output_dir}/ ({writer.frame} frames)")
    log.info(step.timings.report(wall_time))


if __name__ == "__main__":
    main()
