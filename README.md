# MLS-MPM Snow Solver

3D Material Point Method solver in PyTorch, inspired by Disney Research's
*"A material point method for snow simulation"* (Stomakhin et al., SIGGRAPH 2013)
and the MLS-MPM reformulation by Hu et al. Part of DTU CUDA course (Spring 2026).

## Features

- **MLS-MPM** (Moving Least Squares MPM) with APIC transfers
- **Snow constitutive model** — fixed-corotated elasticity with SVD-based plasticity and hardening
- **Fully vectorized** — no Python loops in the simulation step
- **Pluggable P2G kernel** — PyTorch `scatter_add_` reference, CUDA planned
- **Hydra configuration** — composable YAML configs for physics, solver, and simulations

## Quickstart

```bash
# Default simulation (500 particles, 32^3 grid, 10 steps)
python simulate.py

# Run the snowball simulation (40k particles, 128^3 grid)
python simulate.py sim=snowball

# Override solver settings
python simulate.py sim.solver.device=cuda sim.solver.grid_res=64

# Override physics
python simulate.py sim.physics.E=2e5 sim.physics.gravity='[0,0,-400]'

# Inspect resolved config
python simulate.py --cfg job
```

## Configuration

Configs live in `conf/` and use [Hydra](https://hydra.cc/) composition:

```
conf/
├── config.yaml                # top-level defaults
└── sim/
    ├── default.yaml           # test: 500 particles, solver: small
    ├── snowball.yaml          # 40k particles, solver: default
    ├── solver/
    │   ├── default.yaml       # grid_res: 128, device: cpu
    │   └── small.yaml         # grid_res: 32, device: cpu
    └── physics/
        └── default.yaml       # material constants, gravity, colliders
```

Each simulation config selects its solver and physics via Hydra's native defaults list:

```yaml
defaults:
  - solver: small
  - physics: default

name: test
steps: 10
particles:
  - type: sphere
    center: [0.5, 0.5, 0.5]
    ...
```

## Simulations

| Sim        | Description                   | Particles | Grid   |
|------------|-------------------------------|-----------|--------|
| `default`  | Tiny scene for quick testing  | 500       | 32^3   |
| `snowball` | Two snowballs collide mid-air | 40,000    | 128^3  |

## How it works

Each timestep:

1. **Constitutive model** — SVD of the elastic deformation gradient, singular value
   clamping for plasticity, hardening, and fixed-corotated stress computation
2. **P2G** — scatter particle momentum and mass to a 3D Eulerian grid
   using quadratic B-spline weights (27-point stencil)
3. **Grid update** — normalize momentum to velocity, apply gravity and boundary conditions
4. **G2P** — gather grid velocities back to particles, update APIC affine field,
   advect positions, and evolve the deformation gradient

## Output

Simulation results are saved as HDF5 files in `output/<name>.h5` containing particle positions and velocities per saved frame.
