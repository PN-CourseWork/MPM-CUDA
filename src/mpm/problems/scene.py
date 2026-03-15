"""Build simulation parameters and initial state from Hydra config."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mpm.params import SimParams, WallCollider, BoxCollider
from mpm.state import ParticleState, resolve_device
from mpm.problems.particles import make_sphere, make_box, make_mesh, init_state


_COLLIDER_TYPES = {"wall": WallCollider, "box": BoxCollider}


def _parse_colliders(raw: list[dict]) -> tuple:
    result = []
    for c in raw:
        c = dict(c)
        cls = _COLLIDER_TYPES.get(c.pop("type"))
        if cls is None:
            raise ValueError(f"Unknown collider type: {c}")
        result.append(cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()}))
    return tuple(result)


def _parse_params(physics_cfg, solver_cfg) -> SimParams:
    phys = OmegaConf.to_container(physics_cfg, resolve=True)
    solv = OmegaConf.to_container(solver_cfg, resolve=True)

    material = phys.pop("material", {})
    phys["gravity"] = tuple(phys["gravity"])
    phys["colliders"] = _parse_colliders(phys.get("colliders", []))

    return SimParams(**solv, **phys, **material)


def _create_particles(groups, device) -> ParticleState:
    all_pos, all_vel = [], []
    for g in groups:
        rng = np.random.default_rng(g.get("seed", 42))
        if g.type == "sphere":
            pos, vel = make_sphere(
                center=tuple(g.center), radius=g.radius, n_particles=g.n_particles,
                velocity=tuple(g.velocity), rng=rng,
            )
        elif g.type == "box":
            pos, vel = make_box(
                center=tuple(g.center), half_size=tuple(g.half_size),
                n_particles=g.n_particles,
                velocity=tuple(g.velocity), rng=rng,
            )
        elif g.type == "mesh":
            pos, vel = make_mesh(
                stl_path=g.stl_path, center=tuple(g.center),
                height=g.height, n_particles=g.n_particles,
                velocity=tuple(g.velocity), rng=rng,
            )
        else:
            raise ValueError(f"Unknown particle type: {g.type}")
        all_pos.append(pos)
        all_vel.append(vel)
    return init_state(np.concatenate(all_pos), np.concatenate(all_vel), device=device)


def build_scene(cfg: DictConfig) -> tuple[str, SimParams, ParticleState]:
    """Parse Hydra config into scene name, SimParams, and initial ParticleState."""
    params = _parse_params(cfg.physics, cfg.solver)
    device = resolve_device(cfg.kernel.device)
    state = _create_particles(cfg.scene.particles, device)
    return cfg.scene.name, params, state
