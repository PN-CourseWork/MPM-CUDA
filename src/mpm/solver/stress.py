"""Fixed-corotated stress with snow plasticity (Stomakhin et al. 2013)."""

from __future__ import annotations

from typing import NamedTuple

import torch

from mpm.params import SimParams


class StressResult(NamedTuple):
    stress: torch.Tensor   # (N, 3, 3) Kirchhoff stress
    Fe_new: torch.Tensor   # (N, 3, 3) clamped elastic deformation gradient
    Jp_new: torch.Tensor   # (N,) updated plastic ratio


# Cached eye matrix per device
_eye_cache: dict[torch.device, torch.Tensor] = {}


def _eye3(dev: torch.device) -> torch.Tensor:
    if dev not in _eye_cache:
        _eye_cache[dev] = torch.eye(3, device=dev)
    return _eye_cache[dev]


def compute_stress(Fe: torch.Tensor, Jp: torch.Tensor, params: SimParams) -> StressResult:
    """SVD-based fixed-corotated stress with snow plasticity.

    Uses torch.linalg.svd for robustness under extreme deformation.
    """
    U, sig, Vh = torch.linalg.svd(Fe)
    R = U @ Vh  # rotation from polar decomposition

    # Plasticity: clamp singular values, push excess into Jp
    sig_c = sig.clamp(1.0 - params.theta_c, 1.0 + params.theta_s)
    Jp_new = Jp * sig.prod(-1) / sig_c.prod(-1)

    # Reconstruct clamped Fe: U @ diag(sig_c) @ Vh  (avoid diag_embed)
    Fe_new = (U * sig_c.unsqueeze(-2)) @ Vh
    J = sig_c.prod(-1)

    # Hardening
    h = torch.exp(params.hardening * (1.0 - Jp_new))
    mu = params.mu_0 * h
    la = params.lambda_0 * h

    # Kirchhoff stress: τ = 2μ(Fe - R)Fe^T + λ(J-1)J·I
    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ Fe_new.mT)
              + (la * (J - 1.0) * J)[:, None, None] * _eye3(Fe.device))

    return StressResult(stress, Fe_new, Jp_new)
