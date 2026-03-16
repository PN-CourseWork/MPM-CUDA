# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D MLS-MPM (Moving Least Squares Material Point Method) snow simulation with JAX (CPU/CUDA) and a pluggable CUDA P2G kernel. Inspired by Disney Research's snow paper (Stomakhin et al., SIGGRAPH 2013) and the MLS-MPM reformulation by Hu et al. Part of DTU CUDA course (Spring 2026).

## Build & Run Commands

```bash
# Run a simulation (uses uv for dependency management)
uv run python simulate.py                              # default (snowball, test solver)
uv run python simulate.py scene=castle solver=medium   # pick scene + solver
uv run python simulate.py solver=large run.steps=500   # override run params
uv run python simulate.py +experiment=snowball_hires   # experiment preset
uv run python simulate.py physics.E=2e5                # override material param
uv run python simulate.py --cfg job                    # inspect resolved config
```

## Output Format

Per-frame VTK files: `outputs/{scene_name}/{timestamp}/frames/frame_NNNNN.vtk`.
Compatible with splashsurf, ParaView, and Blender.

Run config `save_every` controls how often frames are saved.

## Architecture

### Directory Structure

- **`simulate.py`** — Thin Hydra driver: build scene → trajectory → save VTK
- **`conf/`** — Hydra config groups (see Config Structure below)
- **`src/mpm/`** — Library package:
  - **`params.py`** / **`state.py`** — Shared types at package root
  - **`solver/`** — Physics engine (JAX, runs on CPU or CUDA)
  - **`problems/`** — Scene construction and initial conditions
  - **`io/`** — VTK frame output
- **`Benchmarking/`** — Profiling and performance measurement scripts
- **`Rendering/`** — Blender rendering, particle visualization, video, and VTK export scripts

### Config Structure

```
conf/
├── config.yaml              # defaults: [scene, solver, physics, kernel, run]
├── scene/                   # Particle placement and initial conditions
│   ├── snowball.yaml        # Two colliding snowballs
│   ├── snowball_hires.yaml  # High-res snowball variant
│   ├── castle.yaml          # Snowball vs snow castle
│   └── snowman.yaml         # Snowball vs mesh snowman
├── solver/                  # Numerical discretization
│   ├── test.yaml            # grid_res: 16
│   ├── small.yaml           # grid_res: 32
│   ├── medium.yaml          # grid_res: 64
│   └── large.yaml           # grid_res: 128
├── physics/                 # Forces and material
│   ├── default.yaml         # gravity, colliders → defaults to material: snow
│   └── material/
│       └── snow.yaml        # E, nu, theta_c, theta_s, hardening, p_rho
├── kernel/                  # Backend selection + GPU tuning
│   ├── jax.yaml             # JAX jit-compiled (default)
│   └── fused_cuda.yaml      # Hand-written CUDA kernel
├── run/                     # Execution: steps, save frequency, device
│   ├── default.yaml         # 3000 steps, device: cuda
│   └── hires.yaml           # 15000 steps
└── experiment/              # Presets combining scene + solver + run
    └── snowball_hires.yaml  # snowball_hires + large solver + hires run
```

Each config group is independently swappable on the CLI: `scene=castle solver=large run=hires`.

### Key Modules

- **`src/mpm/params.py`** — `SimParams` frozen dataclass, `WallCollider`/`BoxCollider`.
- **`src/mpm/state.py`** — `ParticleState`, `GridState` NamedTuples (JAX arrays), `resolve_device()`.
- **`src/mpm/solver/solver.py`** — `Stepper` class with `step()`, `scan()`, and `trajectory()` methods. `build_step(params)` creates a stepper.
- **`src/mpm/solver/fused_jax.py`** — Core physics: entire timestep (stress+P2G+grid_ops+G2P) fused into one `@jax.jit`. Includes `full_step()`, `scan_steps()` (lax.scan), and `scan_trajectory()` (nested lax.scan with frame saving).
- **`src/mpm/solver/kernels/`** — Hand-written CUDA kernel (`fused_stress_p2g.cu`) for GPU acceleration.
- **`src/mpm/problems/particles.py`** — `make_sphere`, `make_box`, `make_mesh`, `init_state`. Numpy sampling → JAX arrays.
- **`src/mpm/problems/scene.py`** — `build_scene(cfg)`: Hydra config → `(scene_name, SimParams, ParticleState)`.
- **`src/mpm/io/writer.py`** — `FrameWriter`: per-frame VTK binary writer.

### Simulation Pipeline (each timestep)

All four phases are fused into a single `@jax.jit` function (`_step_impl`):

1. **Stress** — Newton-Schulz polar decomposition, Cardano eigendecomposition, singular value clamping (plasticity), hardening, fixed-corotated stress
2. **P2G** — Scatter particle momentum/mass to 3D grid via quadratic B-spline weights (27-point stencil)
3. **Grid update** — Normalize, apply gravity, enforce boundary conditions and colliders
4. **G2P** — Gather grid velocities back to particles, update APIC affine field `C`, advect positions, evolve deformation gradient

B-spline stencil weights are computed once per step and reused for both P2G and G2P.

### Design Patterns

- **JAX arrays throughout** — solver uses `jax.numpy` with automatic device placement (CUDA → CPU).
- **Fused `@jax.jit`** — entire timestep compiled as one XLA program, eliminating dispatch overhead.
- **`jax.lax.scan`** — multi-step execution without Python loop overhead. `scan_trajectory()` uses nested scans to collect frames.
- **Hydra config composition** — Top-level config groups (scene, solver, physics, kernel, run) are independently composable.
- **`SimParams` is frozen/hashable** — used as static arg in jit for trace-time unrolling of collider loops.
- **Simulation domain** is unit cube `[0,1]^3` with default grid resolution 128.

### Adding a New Scene

Create `conf/scene/my_scene.yaml` with `name`, `description`, and `particles` list. Each particle group has `type: sphere|box|mesh` plus params (`center`, `radius`, `n_particles`, `velocity`, `seed`). Override physics if needed: `physics.colliders=[...]`.
