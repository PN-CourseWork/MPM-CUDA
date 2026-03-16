#!/usr/bin/env python
"""MPM Snow Simulation — Hydra-configured driver."""

from __future__ import annotations

import logging
import time

import jax
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
    step = build_step(params, kernel_cfg=cfg.kernel)

    total_steps = cfg.run.steps
    save_every = cfg.run.get("save_every", 1)

    output_dir = HydraConfig.get().runtime.output_dir

    n_particles = state.x.shape[0]
    sim_time = total_steps * params.dt
    log.info(f"Scene: {scene_name} — {cfg.scene.description}")
    log.info(f"Particles: {n_particles} | Grid: {params.grid_res}³ | "
             f"Steps: {total_steps} | dt: {params.dt} | "
             f"Physics time: {sim_time:.4f}s | Device: {state.x.devices()}")

    log.info("Warmup (JIT compile) …")
    state = step(state)
    log.info("Warmup done.")

    step.reset_timer()
    t0 = time.time()

    if save_every > 0:
        n_saves = total_steps // save_every
        remainder = total_steps % save_every

        log.info(f"Simulating {n_saves * save_every} steps "
                 f"({n_saves} saves × {save_every} steps/save) …")

        # Run entire trajectory as one XLA program
        state, saved_x = step.trajectory(state, n_saves, save_every)
        jax.block_until_ready(saved_x)
        compute_time = time.time() - t0
        log.info(f"Compute done in {compute_time:.1f}s")

        # Handle remainder steps
        if remainder > 0:
            state = step.scan(state, remainder)
            jax.block_until_ready(state)

        # Write all frames to disk
        log.info(f"Writing {n_saves} frames …")
        with FrameWriter(f"{output_dir}/frames") as writer:
            for i in range(n_saves):
                writer.append(saved_x[i])
            if remainder > 0:
                writer.append(state.x)
        n_frames = n_saves + (1 if remainder > 0 else 0)
    else:
        log.info(f"Simulating {total_steps} steps (no saving) …")
        state = step.scan(state, total_steps)
        jax.block_until_ready(state)
        n_frames = 0

    wall_time = time.time() - t0
    log.info(f"Done → {output_dir}/ ({n_frames} frames)")
    log.info(step.report(wall_time))


if __name__ == "__main__":
    main()
