"""Benchmark: torch.linalg.svd vs Newton-Schulz polar + analytical eigenvalues."""

import time
import torch

# --- Method 1: Current SVD approach ---

def stress_svd(Fe):
    U, sig, Vh = torch.linalg.svd(Fe)
    R = U @ Vh
    return R, sig, U, Vh


# --- Method 2: Newton-Schulz polar decomposition + analytical eigenvalues ---

def polar_newton_schulz(F, n_iter=5):
    """Polar decomposition F = R @ S via Newton-Schulz iterations.

    Computes R directly. Very GPU-friendly: only batched matmuls.
    Assumes F is not too far from a rotation (det(F) > 0, singular values
    in roughly [0.3, 3.0] — which holds for snow with typical plasticity bounds).
    """
    # Normalise to improve convergence: F_hat = F / ||F||_F * sqrt(3)
    # For near-identity F this is ~1, so iterations converge in 3-5 steps.
    norms = torch.sqrt((F * F).sum((-2, -1), keepdim=True).clamp(min=1e-12))
    Y = F * (1.7320508 / norms)  # sqrt(3) / ||F||

    for _ in range(n_iter):
        YtY = Y.mT @ Y
        Y = 0.5 * Y @ (3.0 * torch.eye(3, device=F.device) - YtY)

    R = Y
    return R


def symmetric_eigenvalues_cardano(S):
    """Analytical eigenvalues of batched 3x3 symmetric matrices via Cardano's formula.

    S: (N, 3, 3) symmetric matrices
    Returns: (N, 3) eigenvalues sorted descending
    """
    # Extract unique elements
    a11 = S[:, 0, 0]; a22 = S[:, 1, 1]; a33 = S[:, 2, 2]
    a12 = S[:, 0, 1]; a13 = S[:, 0, 2]; a23 = S[:, 1, 2]

    # Characteristic polynomial: λ³ - p·λ² + q·λ - r = 0
    p = a11 + a22 + a33  # trace
    q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23
    r = (a11*a22*a33 + 2*a12*a13*a23
         - a11*a23*a23 - a22*a13*a13 - a33*a12*a12)  # determinant

    # Shift: let λ = t + p/3
    p3 = p / 3.0
    pp = (p * p - 3.0 * q) / 9.0
    qq = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0

    # Clamp to avoid numerical issues
    pp_safe = pp.clamp(min=1e-30)
    sqrt_pp = torch.sqrt(pp_safe)
    cos_arg = (qq / (pp_safe * sqrt_pp)).clamp(-1.0, 1.0)
    phi = torch.acos(cos_arg) / 3.0

    # Three roots
    two_sqrt_pp = 2.0 * sqrt_pp
    e1 = p3 + two_sqrt_pp * torch.cos(phi)
    e2 = p3 + two_sqrt_pp * torch.cos(phi - 2.094395)  # 2π/3
    e3 = p3 + two_sqrt_pp * torch.cos(phi + 2.094395)

    # Sort descending
    eigs = torch.stack([e1, e2, e3], dim=-1)
    eigs, _ = eigs.sort(dim=-1, descending=True)
    return eigs


def stress_newton_schulz(Fe, n_iter=5):
    R = polar_newton_schulz(Fe, n_iter=n_iter)
    S = R.mT @ Fe  # symmetric stretch
    sig = symmetric_eigenvalues_cardano(S)
    # For reconstruction we need U, sig, Vh — but for just R + singular values
    # this is sufficient. Full reconstruction: Fe_new = R @ U_s @ diag(sig_c) @ U_s^T
    return R, sig


# --- Benchmark ---

def bench(fn, Fe, label, n_warmup=10, n_runs=100):
    for _ in range(n_warmup):
        fn(Fe)
    if Fe.device.type != "cpu":
        torch.mps.synchronize() if Fe.device.type == "mps" else torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(Fe)
    if Fe.device.type != "cpu":
        torch.mps.synchronize() if Fe.device.type == "mps" else torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  {label:30s}  {elapsed/n_runs*1000:7.2f} ms/call  ({n_runs} calls)")
    return elapsed / n_runs


def verify_accuracy(Fe):
    """Check that Newton-Schulz gives similar results to SVD."""
    R_svd, sig_svd, _, _ = stress_svd(Fe)
    R_ns, sig_ns = stress_newton_schulz(Fe, n_iter=10)

    # R should match (up to sign flips on columns)
    r_err = (R_svd - R_ns).abs().max().item()
    # Singular values should match
    sig_svd_sorted, _ = sig_svd.sort(dim=-1, descending=True)
    s_err = (sig_svd_sorted - sig_ns).abs().max().item()
    print(f"  Max R error:   {r_err:.2e}")
    print(f"  Max sig error: {s_err:.2e}")
    return r_err, s_err


if __name__ == "__main__":
    for n_particles in [5000, 15000, 50000]:
        print(f"\n{'='*60}")
        print(f"N = {n_particles} particles")
        print(f"{'='*60}")

        for device_name in ["cpu", "mps"]:
            if device_name == "mps" and not torch.backends.mps.is_available():
                continue
            if device_name == "cuda" and not torch.cuda.is_available():
                continue

            dev = torch.device(device_name)
            # Near-identity F with some perturbation (typical early simulation)
            Fe = torch.eye(3, device=dev).unsqueeze(0).expand(n_particles, -1, -1).clone()
            Fe += 0.05 * torch.randn(n_particles, 3, 3, device=dev)

            print(f"\n  Device: {device_name}")
            print(f"  Accuracy check (10 Newton-Schulz iters):")
            verify_accuracy(Fe)

            print(f"  Timing:")
            bench(stress_svd, Fe, "torch.linalg.svd")
            for n_iter in [3, 5, 8]:
                bench(lambda F, ni=n_iter: stress_newton_schulz(F, ni), Fe,
                      f"Newton-Schulz ({n_iter} iters)")

            # Also test with more deformed F (impact scenario)
            Fe_deformed = Fe.clone()
            Fe_deformed[:, 0, 0] *= 1.5
            Fe_deformed[:, 1, 1] *= 0.7
            print(f"\n  Deformed F (stretch 0.7-1.5):")
            print(f"  Accuracy check:")
            verify_accuracy(Fe_deformed)
            print(f"  Timing:")
            bench(stress_svd, Fe_deformed, "torch.linalg.svd")
            bench(lambda F: stress_newton_schulz(F, 5), Fe_deformed,
                  "Newton-Schulz (5 iters)")
