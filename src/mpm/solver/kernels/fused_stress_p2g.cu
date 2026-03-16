/* Fused stress + P2G kernel for 3D MLS-MPM snow simulation.
 *
 * One thread per particle:
 *   1. Load F, Jp, x, v, C
 *   2. Newton-Schulz polar decomposition → R
 *   3. Cardano eigendecomposition of S = R^T F → eigenvalues + eigenvectors
 *   4. Snow plasticity clamping, stress computation
 *   5. Quadratic B-spline weight computation
 *   6. Scatter momentum + mass to 27 grid nodes via atomicAdd
 *   7. Write Fe_new, Jp_new
 *
 * Target: NVIDIA Hopper (sm_90), compiled with --use_fast_math.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 3x3 matrix helpers (all in registers, row-major: m[row][col])
// ---------------------------------------------------------------------------

struct Mat3 {
    float m[3][3];
};

struct Vec3 {
    float x, y, z;
};

__device__ __forceinline__ Mat3 mat3_load(const float* p, int stride) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = p[i * 3 + j];
    return r;
}

__device__ __forceinline__ void mat3_store(float* p, const Mat3& a) {
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            p[i * 3 + j] = a.m[i][j];
}

__device__ __forceinline__ Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            r.m[i][j] = a.m[i][0] * b.m[0][j]
                       + a.m[i][1] * b.m[1][j]
                       + a.m[i][2] * b.m[2][j];
        }
    return r;
}

__device__ __forceinline__ Mat3 mat3_transpose(const Mat3& a) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[j][i];
    return r;
}

__device__ __forceinline__ Mat3 mat3_add(const Mat3& a, const Mat3& b) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][j] + b.m[i][j];
    return r;
}

__device__ __forceinline__ Mat3 mat3_sub(const Mat3& a, const Mat3& b) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][j] - b.m[i][j];
    return r;
}

__device__ __forceinline__ Mat3 mat3_scale(const Mat3& a, float s) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][j] * s;
    return r;
}

__device__ __forceinline__ Mat3 mat3_eye() {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = (i == j) ? 1.0f : 0.0f;
    return r;
}

__device__ __forceinline__ float mat3_frobenius_sq(const Mat3& a) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            s += a.m[i][j] * a.m[i][j];
    return s;
}

// Mat-vec: result = A * v
__device__ __forceinline__ Vec3 mat3_vec(const Mat3& a, const Vec3& v) {
    return {a.m[0][0]*v.x + a.m[0][1]*v.y + a.m[0][2]*v.z,
            a.m[1][0]*v.x + a.m[1][1]*v.y + a.m[1][2]*v.z,
            a.m[2][0]*v.x + a.m[2][1]*v.y + a.m[2][2]*v.z};
}

// ---------------------------------------------------------------------------
// Newton-Schulz polar decomposition: F → R (rotation)
// ---------------------------------------------------------------------------

__device__ Mat3 polar_newton_schulz(const Mat3& F, int n_iter) {
    Mat3 I = mat3_eye();
    float norm_sq = mat3_frobenius_sq(F);
    float norm = sqrtf(fmaxf(norm_sq, 1e-12f));
    float scale = 1.7320508f / norm;  // sqrt(3) / ||F||

    Mat3 Y = mat3_scale(F, scale);

    for (int iter = 0; iter < n_iter; iter++) {
        // Y = 0.5 * Y * (3I - Y^T * Y)
        Mat3 YtY = mat3_mul(mat3_transpose(Y), Y);
        Mat3 inner = mat3_sub(mat3_scale(I, 3.0f), YtY);
        Y = mat3_scale(mat3_mul(Y, inner), 0.5f);
    }
    return Y;
}

// ---------------------------------------------------------------------------
// Cardano eigendecomposition of 3x3 symmetric matrix
// ---------------------------------------------------------------------------

__device__ void sym_eig3x3(
    const Mat3& S, float eigs[3], float vecs[3][3]
) {
    float a11 = S.m[0][0], a22 = S.m[1][1], a33 = S.m[2][2];
    float a12 = S.m[0][1], a13 = S.m[0][2], a23 = S.m[1][2];

    float p = a11 + a22 + a33;
    float q = a11*a22 + a11*a33 + a22*a33 - a12*a12 - a13*a13 - a23*a23;
    float r = a11*a22*a33 + 2.0f*a12*a13*a23
            - a11*a23*a23 - a22*a13*a13 - a33*a12*a12;

    float p3 = p / 3.0f;
    float pp = (p*p - 3.0f*q) / 9.0f;
    float qq = (2.0f*p*p*p - 9.0f*p*q + 27.0f*r) / 54.0f;

    float pp_safe = fmaxf(pp, 1e-30f);
    float sqrt_pp = sqrtf(pp_safe);
    float cos_arg = fminf(fmaxf(qq / (pp_safe * sqrt_pp), -1.0f), 1.0f);
    float phi = acosf(cos_arg) / 3.0f;

    float two_sqrt_pp = 2.0f * sqrt_pp;
    eigs[0] = p3 - two_sqrt_pp * cosf(phi - 2.094395102f);
    eigs[1] = p3 - two_sqrt_pp * cosf(phi + 2.094395102f);
    eigs[2] = p3 - two_sqrt_pp * cosf(phi);

    // Eigenvectors via cross-product method
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        // M = S - eigs[k] * I
        float M[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int j = 0; j < 3; j++)
                M[i][j] = S.m[i][j] - ((i == j) ? eigs[k] : 0.0f);

        // Cross products of row pairs
        float c01[3], c02[3], c12[3];
        c01[0] = M[0][1]*M[1][2] - M[0][2]*M[1][1];
        c01[1] = M[0][2]*M[1][0] - M[0][0]*M[1][2];
        c01[2] = M[0][0]*M[1][1] - M[0][1]*M[1][0];

        c02[0] = M[0][1]*M[2][2] - M[0][2]*M[2][1];
        c02[1] = M[0][2]*M[2][0] - M[0][0]*M[2][2];
        c02[2] = M[0][0]*M[2][1] - M[0][1]*M[2][0];

        c12[0] = M[1][1]*M[2][2] - M[1][2]*M[2][1];
        c12[1] = M[1][2]*M[2][0] - M[1][0]*M[2][2];
        c12[2] = M[1][0]*M[2][1] - M[1][1]*M[2][0];

        float n01 = c01[0]*c01[0] + c01[1]*c01[1] + c01[2]*c01[2];
        float n02 = c02[0]*c02[0] + c02[1]*c02[1] + c02[2]*c02[2];
        float n12 = c12[0]*c12[0] + c12[1]*c12[1] + c12[2]*c12[2];

        // Pick the cross product with largest norm
        float* best = c12;
        float best_n = n12;
        if (n02 > best_n) { best = c02; best_n = n02; }
        if (n01 > best_n) { best = c01; best_n = n01; }

        float inv_len = rsqrtf(fmaxf(best_n, 1e-30f));
        vecs[k][0] = best[0] * inv_len;
        vecs[k][1] = best[1] * inv_len;
        vecs[k][2] = best[2] * inv_len;
    }
}

// ---------------------------------------------------------------------------
// Main fused kernel
// ---------------------------------------------------------------------------

__global__ void fused_stress_p2g_kernel(
    const float* __restrict__ x_in,       // (N, 3)
    const float* __restrict__ v_in,       // (N, 3)
    const float* __restrict__ C_in,       // (N, 3, 3)
    const float* __restrict__ Fe_in,      // (N, 3, 3)
    const float* __restrict__ Jp_in,      // (N,)
    float* __restrict__ Fe_out,           // (N, 3, 3)
    float* __restrict__ Jp_out,           // (N,)
    float* __restrict__ grid_v,           // (GR^3, 3)
    float* __restrict__ grid_m,           // (GR^3,)
    int N, int grid_res,
    float dt, float inv_dx, float dx, float p_vol, float p_mass,
    float theta_c, float theta_s, float hardening,
    float mu_0, float lambda_0,
    int newton_schulz_iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // ---- Load particle data ----
    float px = x_in[tid*3 + 0];
    float py = x_in[tid*3 + 1];
    float pz = x_in[tid*3 + 2];

    float vx = v_in[tid*3 + 0];
    float vy = v_in[tid*3 + 1];
    float vz = v_in[tid*3 + 2];

    Mat3 Cm = mat3_load(C_in + tid * 9, 0);
    Mat3 Fe = mat3_load(Fe_in + tid * 9, 0);
    float Jp = Jp_in[tid];

    // ---- 1. Polar decomposition: Fe → R ----
    Mat3 R = polar_newton_schulz(Fe, newton_schulz_iters);

    // ---- 2. Symmetric stretch: S = R^T * Fe ----
    Mat3 S = mat3_mul(mat3_transpose(R), Fe);

    // ---- 3. Eigendecomposition of S ----
    float eigs[3];
    float Q_cols[3][3];  // eigenvectors as columns: Q_cols[col][row]
    sym_eig3x3(S, eigs, Q_cols);

    // ---- 4. Plasticity clamping ----
    float sig_c[3], sig_prod = 1.0f, sig_c_prod = 1.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        sig_prod *= eigs[i];
        sig_c[i] = fminf(fmaxf(eigs[i], 1.0f - theta_c), 1.0f + theta_s);
        sig_c_prod *= sig_c[i];
    }
    float Jp_new = Jp * sig_prod / sig_c_prod;

    // ---- 5. Reconstruct Fe_new = R * Q * diag(sig_c) * Q^T ----
    // Q is stored as Q_cols[col][row], build Q matrix: Q.m[row][col] = Q_cols[col][row]
    Mat3 Qm;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            Qm.m[i][j] = Q_cols[j][i];

    // Q * diag(sig_c)
    Mat3 QS;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            QS.m[i][j] = Qm.m[i][j] * sig_c[j];

    Mat3 Fe_new = mat3_mul(R, mat3_mul(QS, mat3_transpose(Qm)));

    // ---- 6. Stress computation ----
    float J = sig_c_prod;
    float h = expf(hardening * (1.0f - Jp_new));
    float mu = mu_0 * h;
    float la = lambda_0 * h;

    // stress = 2*mu*(Fe_new - R)*Fe_new^T + la*(J-1)*J*I
    Mat3 diff = mat3_sub(Fe_new, R);
    Mat3 Fe_new_T = mat3_transpose(Fe_new);
    Mat3 stress = mat3_scale(mat3_mul(diff, Fe_new_T), 2.0f * mu);
    float la_term = la * (J - 1.0f) * J;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        stress.m[i][i] += la_term;

    // ---- 7. Affine momentum: affine = -dt * p_vol * 4 * inv_dx^2 * stress + p_mass * C ----
    float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;
    Mat3 affine = mat3_add(mat3_scale(stress, coeff), mat3_scale(Cm, p_mass));

    // ---- 8. B-spline weights ----
    float fx_x = px * inv_dx - floorf(px * inv_dx - 0.5f) - 1.0f;
    float fx_y = py * inv_dx - floorf(py * inv_dx - 0.5f) - 1.0f;
    float fx_z = pz * inv_dx - floorf(pz * inv_dx - 0.5f) - 1.0f;

    int base_x = (int)floorf(px * inv_dx - 0.5f);
    int base_y = (int)floorf(py * inv_dx - 0.5f);
    int base_z = (int)floorf(pz * inv_dx - 0.5f);

    // w[axis][0..2] — quadratic B-spline
    float wx[3], wy[3], wz[3];
    wx[0] = 0.5f * (1.5f - fx_x) * (1.5f - fx_x);
    wx[1] = 0.75f - (fx_x - 1.0f) * (fx_x - 1.0f);
    wx[2] = 0.5f * (fx_x - 0.5f) * (fx_x - 0.5f);
    wy[0] = 0.5f * (1.5f - fx_y) * (1.5f - fx_y);
    wy[1] = 0.75f - (fx_y - 1.0f) * (fx_y - 1.0f);
    wy[2] = 0.5f * (fx_y - 0.5f) * (fx_y - 0.5f);
    wz[0] = 0.5f * (1.5f - fx_z) * (1.5f - fx_z);
    wz[1] = 0.75f - (fx_z - 1.0f) * (fx_z - 1.0f);
    wz[2] = 0.5f * (fx_z - 0.5f) * (fx_z - 0.5f);

    // ---- 9. Scatter to 27 grid nodes ----
    int GR = grid_res;
    #pragma unroll
    for (int di = 0; di < 3; di++) {
        #pragma unroll
        for (int dj = 0; dj < 3; dj++) {
            #pragma unroll
            for (int dk = 0; dk < 3; dk++) {
                float w = wx[di] * wy[dj] * wz[dk];

                int ni = min(max(base_x + di, 0), GR - 1);
                int nj = min(max(base_y + dj, 0), GR - 1);
                int nk = min(max(base_z + dk, 0), GR - 1);
                int flat = ni * GR * GR + nj * GR + nk;

                // dpos = ((di,dj,dk) - fx) * dx
                float dpx = ((float)di - fx_x) * dx;
                float dpy = ((float)dj - fx_y) * dx;
                float dpz = ((float)dk - fx_z) * dx;

                // mv = w * (p_mass * v + affine * dpos)
                float mvx = w * (p_mass * vx + affine.m[0][0]*dpx + affine.m[0][1]*dpy + affine.m[0][2]*dpz);
                float mvy = w * (p_mass * vy + affine.m[1][0]*dpx + affine.m[1][1]*dpy + affine.m[1][2]*dpz);
                float mvz = w * (p_mass * vz + affine.m[2][0]*dpx + affine.m[2][1]*dpy + affine.m[2][2]*dpz);
                float m = w * p_mass;

                atomicAdd(&grid_v[flat * 3 + 0], mvx);
                atomicAdd(&grid_v[flat * 3 + 1], mvy);
                atomicAdd(&grid_v[flat * 3 + 2], mvz);
                atomicAdd(&grid_m[flat], m);
            }
        }
    }

    // ---- 10. Write outputs ----
    mat3_store(Fe_out + tid * 9, Fe_new);
    Jp_out[tid] = Jp_new;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_stress_p2g_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor C,
    torch::Tensor Fe, torch::Tensor Jp,
    int grid_res, float dt, float inv_dx, float dx,
    float p_vol, float p_mass,
    float theta_c, float theta_s, float hardening,
    float mu_0, float lambda_0,
    int block_size, int newton_schulz_iters
) {
    int N = x.size(0);
    int GR3 = grid_res * grid_res * grid_res;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto Fe_new = torch::empty({N, 3, 3}, opts);
    auto Jp_new = torch::empty({N}, opts);
    auto grid_v = torch::zeros({GR3, 3}, opts);
    auto grid_m = torch::zeros({GR3}, opts);

    int grid_dim = (N + block_size - 1) / block_size;

    fused_stress_p2g_kernel<<<grid_dim, block_size>>>(
        x.data_ptr<float>(), v.data_ptr<float>(), C.data_ptr<float>(),
        Fe.data_ptr<float>(), Jp.data_ptr<float>(),
        Fe_new.data_ptr<float>(), Jp_new.data_ptr<float>(),
        grid_v.data_ptr<float>(), grid_m.data_ptr<float>(),
        N, grid_res, dt, inv_dx, dx, p_vol, p_mass,
        theta_c, theta_s, hardening, mu_0, lambda_0,
        newton_schulz_iters
    );

    return std::make_tuple(
        Fe_new,
        Jp_new,
        grid_v.reshape({grid_res, grid_res, grid_res, 3}),
        grid_m.reshape({grid_res, grid_res, grid_res})
    );
}
