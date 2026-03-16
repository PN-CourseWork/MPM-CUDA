/* Fused stress + P2G kernel for 3D MLS-MPM snow simulation.
 *
 * One thread per particle — everything in registers:
 *   1. Load Fe, Jp, x, v, C
 *   2. Jacobi SVD of Fe (3x3) -> U, sig, V
 *   3. Snow plasticity: clamp singular values, update Jp
 *   4. Stress: fixed corotated with hardening
 *   5. Compute Q_p (affine momentum matrix, MLS-MPM eq. 29)
 *   6. B-spline weights + scatter to 27 grid nodes via atomicAdd
 *   7. Write Fe_new, Jp_new
 *
 * Also includes a P2G-only kernel (no stress) for comparison.
 * Target: NVIDIA Hopper (sm_90), compiled with --use_fast_math.
 *
 * Compiled via NVRTC from Python (cuda.core). No C++ host code here.
 */

// ===== 3x3 matrix in registers (row-major) =====

struct Mat3 { float m[3][3]; };

__device__ __forceinline__ Mat3 mat3_load(const float* p) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 9; i++) ((float*)r.m)[i] = p[i];
    return r;
}

__device__ __forceinline__ void mat3_store(float* p, const Mat3& a) {
    #pragma unroll
    for (int i = 0; i < 9; i++) p[i] = ((const float*)a.m)[i];
}

__device__ __forceinline__ Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][0]*b.m[0][j] + a.m[i][1]*b.m[1][j] + a.m[i][2]*b.m[2][j];
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

__device__ __forceinline__ Mat3 mat3_eye() {
    Mat3 r;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            r.m[i][j] = (i == j) ? 1.0f : 0.0f;
    return r;
}

// ===== 3x3 Jacobi SVD =====
// Based on McAdams et al. 2011 "Computing the Singular Value Decomposition
// of 3x3 matrices with minimal branching and elementary floating point
// operations". Simplified for our use case.

__device__ __forceinline__ void jacobi_rotation(
    Mat3& A, Mat3& V, int p, int q
) {
    float app = 0.0f, aqq = 0.0f, apq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        app += A.m[i][p] * A.m[i][p];
        aqq += A.m[i][q] * A.m[i][q];
        apq += A.m[i][p] * A.m[i][q];
    }

    if (fabsf(apq) < 1e-12f) return;

    float tau = (aqq - app) / (2.0f * apq);
    float t = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
    float c = rsqrtf(1.0f + t * t);
    float s = t * c;

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        float a_ip = A.m[i][p], a_iq = A.m[i][q];
        A.m[i][p] = c * a_ip - s * a_iq;
        A.m[i][q] = s * a_ip + c * a_iq;
    }
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        float v_ip = V.m[i][p], v_iq = V.m[i][q];
        V.m[i][p] = c * v_ip - s * v_iq;
        V.m[i][q] = s * v_ip + c * v_iq;
    }
}

__device__ __forceinline__ Mat3 qr_givens(Mat3& A) {
    Mat3 Q = mat3_eye();
    // Zero out A[1][0], A[2][0], A[2][1] via Givens rotations
    for (int col = 0; col < 2; col++) {
        for (int row = col + 1; row < 3; row++) {
            float a = A.m[col][col], b = A.m[row][col];
            float r = sqrtf(a*a + b*b);
            float c, s;
            if (r < 1e-12f) { c = 1.0f; s = 0.0f; }
            else { c = a / r; s = -b / r; }
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                float tc = c*A.m[col][j] - s*A.m[row][j];
                float tr = s*A.m[col][j] + c*A.m[row][j];
                A.m[col][j] = tc; A.m[row][j] = tr;
                float qc = c*Q.m[col][j] - s*Q.m[row][j];
                float qr = s*Q.m[col][j] + c*Q.m[row][j];
                Q.m[col][j] = qc; Q.m[row][j] = qr;
            }
        }
    }
    return mat3_transpose(Q);
}

__device__ void svd3x3(const Mat3& F, Mat3& U, float sigma[3], Mat3& V) {
    Mat3 A = F;
    V = mat3_eye();

    #pragma unroll
    for (int sweep = 0; sweep < 4; sweep++) {
        jacobi_rotation(A, V, 0, 1);
        jacobi_rotation(A, V, 0, 2);
        jacobi_rotation(A, V, 1, 2);
    }

    U = qr_givens(A);
    sigma[0] = A.m[0][0];
    sigma[1] = A.m[1][1];
    sigma[2] = A.m[2][2];

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        if (sigma[i] < 0.0f) {
            sigma[i] = -sigma[i];
            #pragma unroll
            for (int j = 0; j < 3; j++)
                U.m[j][i] = -U.m[j][i];
        }
    }
}


// ===== P2G scatter (shared by both kernels) =====

__device__ __forceinline__ void scatter_to_grid(
    float px, float py, float pz,
    float vx, float vy, float vz,
    const float aff[9],
    float* __restrict__ grid_v,
    float* __restrict__ grid_m,
    int grid_res, float inv_dx, float dx, float p_mass
) {
    int base_x = (int)floorf(px * inv_dx - 0.5f);
    int base_y = (int)floorf(py * inv_dx - 0.5f);
    int base_z = (int)floorf(pz * inv_dx - 0.5f);

    float fx_x = px * inv_dx - (float)base_x;
    float fx_y = py * inv_dx - (float)base_y;
    float fx_z = pz * inv_dx - (float)base_z;

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

                float dpx = ((float)di - fx_x) * dx;
                float dpy = ((float)dj - fx_y) * dx;
                float dpz = ((float)dk - fx_z) * dx;

                float mvx = w * (p_mass * vx + aff[0]*dpx + aff[1]*dpy + aff[2]*dpz);
                float mvy = w * (p_mass * vy + aff[3]*dpx + aff[4]*dpy + aff[5]*dpz);
                float mvz = w * (p_mass * vz + aff[6]*dpx + aff[7]*dpy + aff[8]*dpz);
                float m = w * p_mass;

                atomicAdd(&grid_v[flat * 3 + 0], mvx);
                atomicAdd(&grid_v[flat * 3 + 1], mvy);
                atomicAdd(&grid_v[flat * 3 + 2], mvz);
                atomicAdd(&grid_m[flat], m);
            }
        }
    }
}


// ===== Kernel 1: P2G only (stress pre-computed by PyTorch) =====

extern "C" __global__ void p2g_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ C_in,
    const float* __restrict__ stress_in,
    float* __restrict__ grid_v,
    float* __restrict__ grid_m,
    int N, int grid_res,
    float dt, float inv_dx, float dx, float p_vol, float p_mass
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = x_in[tid*3+0], py = x_in[tid*3+1], pz = x_in[tid*3+2];
    float vx = v_in[tid*3+0], vy = v_in[tid*3+1], vz = v_in[tid*3+2];

    float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;
    float aff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++)
        aff[i] = coeff * stress_in[tid*9 + i] + p_mass * C_in[tid*9 + i];

    scatter_to_grid(px, py, pz, vx, vy, vz, aff,
                    grid_v, grid_m, grid_res, inv_dx, dx, p_mass);
}


// ===== Kernel 2: Fused stress + P2G (full pipeline in registers) =====

extern "C" __global__ void fused_stress_p2g_kernel(
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
    float mu_0, float lambda_0
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = x_in[tid*3+0], py = x_in[tid*3+1], pz = x_in[tid*3+2];
    float vx = v_in[tid*3+0], vy = v_in[tid*3+1], vz = v_in[tid*3+2];
    Mat3 Cm = mat3_load(C_in + tid * 9);
    Mat3 Fe = mat3_load(Fe_in + tid * 9);
    float Jp = Jp_in[tid];

    // ---- SVD: Fe = U * diag(sig) * V^T ----
    Mat3 U, V;
    float sig[3];
    svd3x3(Fe, U, sig, V);

    // ---- Plasticity: clamp singular values ----
    float sig_c[3], sig_prod = 1.0f, sig_c_prod = 1.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        sig_prod *= sig[i];
        sig_c[i] = fminf(fmaxf(sig[i], 1.0f - theta_c), 1.0f + theta_s);
        sig_c_prod *= sig_c[i];
    }
    float Jp_new = Jp * sig_prod / sig_c_prod;

    // ---- Reconstruct Fe_new = U * diag(sig_c) * V^T, R = U * V^T ----
    Mat3 Fe_new, R;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            float fe = 0.0f, rv = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; k++) {
                fe += U.m[i][k] * sig_c[k] * V.m[j][k];
                rv += U.m[i][k] * V.m[j][k];
            }
            Fe_new.m[i][j] = fe;
            R.m[i][j] = rv;
        }

    // ---- Stress: fixed corotated with hardening ----
    float J = sig_c_prod;
    float h = expf(hardening * (1.0f - Jp_new));
    float mu = mu_0 * h;
    float la = lambda_0 * h;

    float stress[9];
    float la_term = la * (J - 1.0f) * J;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            float diff_dot_feT = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; k++)
                diff_dot_feT += (Fe_new.m[i][k] - R.m[i][k]) * Fe_new.m[j][k];
            stress[i*3+j] = 2.0f * mu * diff_dot_feT + ((i == j) ? la_term : 0.0f);
        }

    // ---- Q_p: affine momentum (MLS-MPM eq. 29) ----
    float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;
    float aff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++)
        aff[i] = coeff * stress[i] + p_mass * ((float*)Cm.m)[i];

    // ---- P2G scatter ----
    scatter_to_grid(px, py, pz, vx, vy, vz, aff,
                    grid_v, grid_m, grid_res, inv_dx, dx, p_mass);

    // ---- Write outputs ----
    mat3_store(Fe_out + tid * 9, Fe_new);
    Jp_out[tid] = Jp_new;
}
