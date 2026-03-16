"""Particle samplers and initialisation (3D, pure numpy → JAX)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from mpm.state import ParticleState


def make_sphere(center, radius, n_particles, velocity=(0.0, 0.0, 0.0), rng=None):
    """Sample particles uniformly inside a 3D sphere."""
    if rng is None:
        rng = np.random.default_rng(42)
    center = np.asarray(center, dtype=np.float64)
    # Direct sampling: uniform radius via inverse CDF, uniform direction via Gaussian
    r = radius * rng.random(n_particles) ** (1.0 / 3.0)
    direction = rng.standard_normal((n_particles, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    pos = direction * r[:, np.newaxis] + center
    vel = np.broadcast_to(np.array(velocity), pos.shape).copy()
    return pos, vel


def make_box(center, half_size, n_particles, velocity=(0.0, 0.0, 0.0), rng=None):
    """Sample particles uniformly inside a 3D axis-aligned box."""
    if rng is None:
        rng = np.random.default_rng(42)
    center = np.asarray(center, dtype=np.float64)
    half_size = np.asarray(half_size, dtype=np.float64)
    pos = rng.uniform(-half_size, half_size, size=(n_particles, 3)) + center
    vel = np.broadcast_to(np.array(velocity), pos.shape).copy()
    return pos, vel


def make_mesh(stl_path, center, height, n_particles, velocity=(0.0, 0.0, 0.0), rng=None):
    """Sample particles uniformly inside a 3D mesh, scaled to fit given height.

    Uses voxelization for robust inside/outside testing (works even if mesh
    is not perfectly watertight).
    """
    import trimesh
    if rng is None:
        rng = np.random.default_rng(42)
    mesh = trimesh.load(stl_path)
    # Scale so the mesh's tallest extent equals `height`
    scale = height / mesh.extents.max()
    mesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    # Position: place bottom of mesh at z=center[2], centered in x/y
    current_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    offset = np.array(center, dtype=np.float64) - current_center
    offset[2] = center[2] - mesh.bounds[0][2]
    mesh.apply_translation(offset)
    # Voxelize and sample from filled voxels
    pitch = height / 80.0  # ~80 voxels along tallest axis
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    # Get filled voxel centers
    voxel_pts = voxel_grid.points
    print(f"  make_mesh: {len(voxel_pts)} filled voxels (pitch={pitch:.5f})")
    # Uniformly sample from voxel cells with jitter
    indices = rng.integers(0, len(voxel_pts), size=n_particles)
    jitter = rng.uniform(-pitch / 2, pitch / 2, size=(n_particles, 3))
    pts = voxel_pts[indices] + jitter
    vel = np.broadcast_to(np.array(velocity), pts.shape).copy()
    return pts.astype(np.float64), vel


def init_state(positions, velocities, device=None) -> ParticleState:
    """Create initial simulation state arrays as JAX arrays.

    The `device` parameter is kept for API compatibility but ignored;
    JAX handles device placement automatically.
    """
    n = len(positions)
    return ParticleState(
        x=jnp.array(positions, dtype=jnp.float32),
        v=jnp.array(velocities, dtype=jnp.float32),
        C=jnp.zeros((n, 3, 3), dtype=jnp.float32),
        F=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n, 3, 3)),
        Jp=jnp.ones(n, dtype=jnp.float32),
    )
