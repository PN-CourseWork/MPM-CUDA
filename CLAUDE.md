# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D MLS-MPM (Moving Least Squares Material Point Method) snow simulation with PyTorch (CPU/CUDA) and a pluggable CUDA P2G kernel. Inspired by Disney Research's snow paper (Stomakhin et al., SIGGRAPH 2013) and the MLS-MPM reformulation by Hu et al. Part of DTU CUDA course (Spring 2026).

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

One `.h5` file per simulation run: `output/{scene_name}.h5`. Contains:
- `/positions` — `(n_saved, N, 3)` float32
- `/velocities` — `(n_saved, N, 3)` float32
- Attributes: `dt`, `grid_res`

Run config `save_every` controls how often frames are saved.

## Architecture

### Directory Structure

- **`simulate.py`** — Thin Hydra driver: build scene → step loop → save HDF5
- **`conf/`** — Hydra config groups (see Config Structure below)
- **`src/mpm/`** — Library package, split into three areas:
  - **`params.py`** / **`state.py`** — Shared types at package root
  - **`solver/`** — Physics engine (PyTorch, runs on CPU or CUDA)
  - **`problems/`** — Scene construction and initial conditions
  - **`io/`** — HDF5 output
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
├── kernel/                  # P2G backend selection + GPU tuning
│   └── torch.yaml           # PyTorch scatter_add_ (default)
├── run/                     # Execution: steps, save frequency, device
│   ├── default.yaml         # 300 steps, device: auto
│   └── hires.yaml           # 15000 steps, device: mps
└── experiment/              # Presets combining scene + solver + run
    └── snowball_hires.yaml  # snowball_hires + large solver + hires run
```

Each config group is independently swappable on the CLI: `scene=castle solver=large run=hires`.

### Key Modules

- **`src/mpm/params.py`** — `SimParams` frozen dataclass, `WallCollider`/`BoxCollider`.
- **`src/mpm/state.py`** — `ParticleState`, `GridState` NamedTuples (torch tensors), `resolve_device()`.
- **`src/mpm/solver/solver.py`** — Orchestrator: stress → P2G → grid_ops → G2P. `build_step(params)` returns cached step function with `StepTimings`.
- **`src/mpm/solver/stress.py`** — `compute_stress`: polar-decomposition-based fixed-corotated stress with snow plasticity.
- **`src/mpm/solver/polar.py`** — Polar decomposition via Jacobi eigendecomposition for 3×3 matrices.
- **`src/mpm/solver/grid_ops.py`** — `update_grid`: normalize momentum, gravity, BCs, colliders.
- **`src/mpm/solver/g2p.py`** — `gather`: grid-to-particle velocity/position/F update.
- **`src/mpm/solver/p2g/`** — P2G scatter (pluggable for custom CUDA kernel).
  - `data.py` — `P2GData`, `compute_p2g_data`, `quadratic_weights`
  - `torch.py` — PyTorch `scatter_add_` implementation
- **`src/mpm/problems/particles.py`** — `make_sphere`, `init_state`. Numpy sampling → torch tensors.
- **`src/mpm/problems/scene.py`** — `build_scene(cfg)`: Hydra config → `(scene_name, SimParams, ParticleState)`.
- **`src/mpm/io/writer.py`** — `SimWriter`: append-mode HDF5 writer for simulation frames.

### Simulation Pipeline (each timestep)

1. **Stress** — Polar decomposition of elastic deformation gradient, singular value clamping (plasticity), hardening, fixed-corotated stress
2. **P2G** — Scatter particle momentum/mass to 3D grid via quadratic B-spline weights (27-point stencil)
3. **Grid update** — Normalize, apply gravity, enforce boundary conditions and colliders
4. **G2P** — Gather grid velocities back to particles, update APIC affine field `C`, advect positions, evolve deformation gradient

### Design Patterns

- **PyTorch tensors throughout** — solver uses `torch.Tensor` on `resolve_device()` (CUDA → MPS → CPU auto-detection).
- **Hydra config composition** — Top-level config groups (scene, solver, physics, kernel, run) are independently composable. Experiment presets combine them.
- **`SimParams` is frozen/hashable** — enables `lru_cache` on `build_step()`
- **Step timings built in** — `step.timings.report()` shows per-phase ms/step breakdown
- **Simulation domain** is unit cube `[0,1]^3` with default grid resolution 128

### Adding a Custom CUDA P2G Kernel

1. Create `src/mpm/solver/p2g/cuda.py` with a `scatter(p2g_data, grid_res) -> GridState` function.
2. Create `conf/kernel/cuda.yaml` with `backend: cuda` and any tuning params (block_size, etc.).
3. Update the import in `solver.py` to dispatch based on kernel config.

### Adding a New Scene

Create `conf/scene/my_scene.yaml` with `name`, `description`, and `particles` list. Each particle group has `type: sphere|box|mesh` plus params (`center`, `radius`, `n_particles`, `velocity`, `seed`). Override physics if needed: `physics.colliders=[...]`.
