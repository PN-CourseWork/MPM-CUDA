"""Fixed-corotated stress with snow plasticity (Stomakhin et al. 2013)."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from mpm.params import SimParams


class StressResult(NamedTuple):
    stress: torch.Tensor   # (N, 3, 3) Kirchhoff stress
    Fe_new: torch.Tensor   # (N, 3, 3) clamped elastic deformation gradient
    Jp_new: torch.Tensor   # (N,) updated plastic ratio


# Cached eye matrix per device (used outside torch.compile only)
_eye_cache: dict[torch.device, torch.Tensor] = {}


def _eye3(dev: torch.device) -> torch.Tensor:
    if dev not in _eye_cache:
        _eye_cache[dev] = torch.eye(3, device=dev)
    return _eye_cache[dev]


# ---------------------------------------------------------------------------
# Polar decomposition via Newton-Schulz iterations (batched matmuls only)
# ---------------------------------------------------------------------------

def _polar_newton_schulz(F: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    I = torch.eye(3, device=F.device, dtype=F.dtype)
    norms = torch.sqrt((F * F).sum((-2, -1), keepdim=True).clamp(min=1e-12))
    Y = F * (1.7320508 / norms)  # sqrt(3) / ||F||

    for _ in range(n_iter):
        Y = 0.5 * Y @ (3.0 * I - Y.mT @ Y)

    return Y


# ---------------------------------------------------------------------------
# Analytical 3x3 symmetric eigendecomposition (zero LAPACK calls)
# ---------------------------------------------------------------------------

def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], dim=-1)


def _sym_eig3x3(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical eigendecomposition of batched 3x3 symmetric matrices.

    Cardano's formula for eigenvalues, cross-product method for eigenvectors.
    """
    N = S.shape[0]
    dev = S.device

    a11 = S[:, 0, 0]; a22 = S[:, 1, 1]; a33 = S[:, 2, 2]
    a12 = S[:, 0, 1]; a13 = S[:, 0, 2]; a23 = S[:, 1, 2]

    p = a11 + a22 + a33
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
    e0 = p3 - two_sqrt_pp * torch.cos(phi - 2.094395102)
    e1 = p3 - two_sqrt_pp * torch.cos(phi + 2.094395102)
    e2 = p3 - two_sqrt_pp * torch.cos(phi)

    eigs = torch.stack([e0, e1, e2], dim=-1)

    I = torch.eye(3, device=dev, dtype=S.dtype)
    vecs = torch.zeros(N, 3, 3, device=dev, dtype=S.dtype)

    for k in range(3):
        M = S - eigs[:, k:k+1, None] * I.unsqueeze(0)
        r0 = M[:, 0]; r1 = M[:, 1]; r2 = M[:, 2]

        c01 = _cross(r0, r1)
        c02 = _cross(r0, r2)
        c12 = _cross(r1, r2)

        n01 = (c01 * c01).sum(-1)
        n02 = (c02 * c02).sum(-1)
        n12 = (c12 * c12).sum(-1)

        best = c12; best_n = n12
        mask02 = n02 > best_n
        best = torch.where(mask02.unsqueeze(-1), c02, best)
        best_n = torch.where(mask02, n02, best_n)
        mask01 = n01 > best_n
        best = torch.where(mask01.unsqueeze(-1), c01, best)
        best_n = torch.where(mask01, n01, best_n)

        best = best / torch.sqrt(best_n.clamp(min=1e-30)).unsqueeze(-1)
        vecs[:, :, k] = best

    return eigs, vecs


# ---------------------------------------------------------------------------
# Stress computation — core function (torch.compile-safe: no global caches)
# ---------------------------------------------------------------------------

def _compute_stress_analytical(
    Fe: torch.Tensor, Jp: torch.Tensor,
    theta_c: float, theta_s: float, hardening: float,
    mu_0: float, lambda_0: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core stress computation with analytical decomposition."""
    I = torch.eye(3, device=Fe.device, dtype=Fe.dtype)

    R = _polar_newton_schulz(Fe)
    S = R.mT @ Fe

    sig, Q = _sym_eig3x3(S)

    sig_c = sig.clamp(1.0 - theta_c, 1.0 + theta_s)
    Jp_new = Jp * sig.prod(-1) / sig_c.prod(-1)

    Fe_new = R @ ((Q * sig_c.unsqueeze(-2)) @ Q.mT)
    J = sig_c.prod(-1)

    h = torch.exp(hardening * (1.0 - Jp_new))
    mu = mu_0 * h
    la = lambda_0 * h

    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ Fe_new.mT)
              + (la * (J - 1.0) * J)[:, None, None] * I)

    return stress, Fe_new, Jp_new


# ---------------------------------------------------------------------------
# torch.compile wrapper (lazy init)
# ---------------------------------------------------------------------------

_compiled_stress = None


def _get_compiled_stress():
    global _compiled_stress
    if _compiled_stress is None:
        _compiled_stress = torch.compile(
            _compute_stress_analytical, mode="default"
        )
    return _compiled_stress


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Set MPM_STRESS=svd|analytical|compile|auto (default: auto).
# auto = SVD on CUDA (single fused kernel), analytical on CPU (no LAPACK).
_STRESS_BACKEND = os.environ.get("MPM_STRESS", "auto")


def compute_stress(Fe: torch.Tensor, Jp: torch.Tensor, params: SimParams) -> StressResult:
    backend = _STRESS_BACKEND
    if backend == "auto":
        backend = "compile" if Fe.is_cuda else "analytical"

    if backend == "svd":
        return _stress_svd(Fe, Jp, params)
    elif backend == "compile":
        fn = _get_compiled_stress()
        stress, Fe_new, Jp_new = fn(
            Fe, Jp, params.theta_c, params.theta_s,
            params.hardening, params.mu_0, params.lambda_0,
        )
        return StressResult(stress, Fe_new, Jp_new)
    else:
        stress, Fe_new, Jp_new = _compute_stress_analytical(
            Fe, Jp, params.theta_c, params.theta_s,
            params.hardening, params.mu_0, params.lambda_0,
        )
        return StressResult(stress, Fe_new, Jp_new)


def _stress_svd(Fe: torch.Tensor, Jp: torch.Tensor, params: SimParams) -> StressResult:
    """Original SVD-based stress (fallback / reference)."""
    U, sig, Vh = torch.linalg.svd(Fe)
    R = U @ Vh

    sig_c = sig.clamp(1.0 - params.theta_c, 1.0 + params.theta_s)
    Jp_new = Jp * sig.prod(-1) / sig_c.prod(-1)

    Fe_new = (U * sig_c.unsqueeze(-2)) @ Vh
    J = sig_c.prod(-1)

    h = torch.exp(params.hardening * (1.0 - Jp_new))
    mu = params.mu_0 * h
    la = params.lambda_0 * h

    stress = (2.0 * mu[:, None, None] * ((Fe_new - R) @ Fe_new.mT)
              + (la * (J - 1.0) * J)[:, None, None] * _eye3(Fe.device))

    return StressResult(stress, Fe_new, Jp_new)
