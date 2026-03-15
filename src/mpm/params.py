"""Simulation parameters and collider data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WallCollider:
    normal: tuple[float, ...] = (1.0, 0.0, 0.0)
    point: tuple[float, ...] = (0.0, 0.0, 0.0)
    friction: float = 0.0


@dataclass(frozen=True)
class BoxCollider:
    center: tuple[float, ...] = (0.5, 0.5, 0.5)
    half_size: tuple[float, ...] = (0.05, 0.05, 0.05)
    friction: float = 0.0


@dataclass(frozen=True)
class SimParams:
    grid_res: int = 128
    dt: float = 1e-4
    gravity: tuple[float, ...] = (0.0, 0.0, -9.81)
    bound: int = 3

    # Material (snow defaults from Stomakhin et al. 2013)
    E: float = 1.4e5
    nu: float = 0.2
    theta_c: float = 2.5e-2
    theta_s: float = 7.5e-3
    hardening: float = 10.0
    p_rho: float = 400.0

    colliders: tuple[WallCollider | BoxCollider, ...] = ()

    # Derived quantities — precomputed in __post_init__
    mu_0: float = 0.0
    lambda_0: float = 0.0
    dx: float = 0.0
    inv_dx: float = 0.0
    p_vol: float = 0.0
    p_mass: float = 0.0

    def __post_init__(self):
        s = object.__setattr__  # bypass frozen
        s(self, 'dx', 1.0 / self.grid_res)
        s(self, 'inv_dx', float(self.grid_res))
        s(self, 'mu_0', self.E / (2.0 * (1.0 + self.nu)))
        s(self, 'lambda_0', self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu)))
        s(self, 'p_vol', (self.dx * 0.5) ** 3)
        s(self, 'p_mass', self.p_vol * self.p_rho)
