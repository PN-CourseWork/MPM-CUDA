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


def _polar_newton_schulz(F: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """Polar decomposition F = R @ S via Newton-Schulz iterations.

    Returns the rotation R. Only uses batched matmuls — very GPU-friendly.
    """
    I = _eye3(F.device)
    norms = torch.sqrt((F * F).sum((-2, -1), keepdim=True).clamp(min=1e-12))
    Y = F * (1.7320508 / norms)  # sqrt(3) / ||F||

    for _ in range(n_iter):
        Y = 0.5 * Y @ (3.0 * I - Y.mT @ Y)

    return Y


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched cross product of (N, 3) vectors."""
    return torch.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], dim=-1)


def _sym_eig3x3(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical eigendecomposition of batched 3x3 symmetric matrices.

    Cardano's formula for eigenvalues, cross-product method for eigenvectors.
    No LAPACK calls — pure tensor ops, fully GPU-compatible.

    Args:
        S: (N, 3, 3) symmetric matrices
    Returns:
        eigenvalues: (N, 3) sorted ascending
        eigenvectors: (N, 3, 3) columns are eigenvectors
    """
    N = S.shape[0]
    dev = S.device

    # --- Eigenvalues via Cardano's formula ---
    a11 = S[:, 0, 0]; a22 = S[:, 1, 1]; a33 = S[:, 2, 2]
    a12 = S[:, 0, 1]; a13 = S[:, 0, 2]; a23 = S[:, 1, 2]

    p = a11 + a22 + a33  # trace
    q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23
    r = a11*a22*a33 + 2*a12*a13*a23 - a11*a23*a23 - a22*a13*a13 - a33*a12*a12

    p3 = p / 3.0
    pp = (p * p - 3.0 * q) / 9.0
    qq = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0

    pp_safe = pp.clamp(min=1e-30)
    sqrt_pp = torch.sqrt(pp_safe)
    cos_arg = (qq / (pp_safe * sqrt_pp)).clamp(-1.0, 1.0)
    phi = torch.acos(cos_arg) / 3.0

    two_sqrt_pp = 2.0 * sqrt_pp
    # Sorted ascending
    e0 = p3 - two_sqrt_pp * torch.cos(phi - 2.094395102)  # smallest
    e1 = p3 - two_sqrt_pp * torch.cos(phi + 2.094395102)  # middle
    e2 = p3 - two_sqrt_pp * torch.cos(phi)                 # largest

    eigs = torch.stack([e0, e1, e2], dim=-1)  # (N, 3)

    # --- Eigenvectors via cross-product method ---
    # For eigenvalue λ, the eigenvector is in the null space of (S - λI).
    # For a rank-2 matrix, the null vector = cross product of any two rows.
    # We compute all three row cross products and pick the longest for robustness.

    I = _eye3(dev)
    vecs = torch.zeros(N, 3, 3, device=dev)

    for k in range(3):
        # M = S - eigs[:, k] * I
        M = S - eigs[:, k:k+1, None] * I.unsqueeze(0)  # (N, 3, 3)

        r0 = M[:, 0]  # (N, 3)
        r1 = M[:, 1]
        r2 = M[:, 2]

        # Three candidate null vectors
        c01 = _cross(r0, r1)
        c02 = _cross(r0, r2)
        c12 = _cross(r1, r2)

        # Pick the one with largest squared norm
        n01 = (c01 * c01).sum(-1)  # (N,)
        n02 = (c02 * c02).sum(-1)
        n12 = (c12 * c12).sum(-1)

        # Branchless max selection
        best = c12  # default
        best_n = n12
        mask02 = n02 > best_n
        best = torch.where(mask02.unsqueeze(-1), c02, best)
        best_n = torch.where(mask02, n02, best_n)
        mask01 = n01 > best_n
        best = torch.where(mask01.unsqueeze(-1), c01, best)
        best_n = torch.where(mask01, n01, best_n)

        # Normalize
        best = best / torch.sqrt(best_n.clamp(min=1e-30)).unsqueeze(-1)
        vecs[:, :, k] = best

    return eigs, vecs


def compute_stress(Fe: torch.Tensor, Jp: torch.Tensor, params: SimParams) -> StressResult:
    """Newton-Schulz polar decomp + analytical eigendecomp stress.

    Uses Newton-Schulz iterations (batched matmuls) for R, then a fully
    analytical 3x3 symmetric eigendecomposition (Cardano + cross products)
    for singular values and eigenvectors. Zero LAPACK calls.
    """
    I = _eye3(Fe.device)

    # Polar decomposition: Fe = R @ S
    R = _polar_newton_schulz(Fe)
    S = R.mT @ Fe  # symmetric positive semi-definite stretch

    # Analytical eigendecomposition of S
    sig, Q = _sym_eig3x3(S)

    # Plasticity: clamp singular values, push excess into Jp
    sig_c = sig.clamp(1.0 - params.theta_c, 1.0 + params.theta_s)
    Jp_new = Jp * sig.prod(-1) / sig_c.prod(-1)

    # Reconstruct clamped Fe: R @ Q @ diag(sig_c) @ Q^T
    Fe_new = R @ ((Q * sig_c.unsqueeze(-2)) @ Q.mT)
    J = sig_c.prod(-1)

    # Hardening
    h = torch.exp(params.hardening * (1.0 - Jp_new))
    mu = params.mu_0 * h
    la = params.lambda_0 * h

    # Kirchhoff stress: τ = 2μ(Fe - R)Fe^T + λ(J-1)J·I
    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ Fe_new.mT)
              + (la * (J - 1.0) * J)[:, None, None] * I)

    return StressResult(stress, Fe_new, Jp_new)
