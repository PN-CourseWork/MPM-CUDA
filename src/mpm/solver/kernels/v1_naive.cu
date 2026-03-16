/* v1_naive.cu — Naive baseline: 4 separate kernels, global memory only.
 *
 * Kernel launches per timestep: 4
 *   1. stress_kernel        — SVD + plasticity + stress tensor (1 thread/particle)
 *   2. p2g_kernel           — B-spline scatter via global atomicAdd (1 thread/particle)
 *   3. grid_ops_kernel      — Normalize + gravity + boundary (1 thread/cell)
 *   4. g2p_kernel           — Gather + update positions/velocity/F (1 thread/particle)
 *
 * No shared memory, no kernel fusion, no warp-level tricks.
 * Target: sm_90 (Hopper), compiled via NVRTC with --use_fast_math.
 */

// ===== 3x3 matrix helpers (row-major, register-resident) =====

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

// ===== 3x3 Jacobi SVD (McAdams et al. 2011) =====

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
    sigma[0] = A.m[0][0]; sigma[1] = A.m[1][1]; sigma[2] = A.m[2][2];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        if (sigma[i] < 0.0f) {
            sigma[i] = -sigma[i];
            #pragma unroll
            for (int j = 0; j < 3; j++) U.m[j][i] = -U.m[j][i];
        }
    }
}


// ===== Kernel 1: Stress (SVD + plasticity + stress tensor) =====

extern "C" __global__ void stress_kernel(
    const float* __restrict__ Fe_in,      // (N, 3, 3)
    const float* __restrict__ Jp_in,      // (N,)
    const float* __restrict__ C_in,       // (N, 3, 3)
    float* __restrict__ stress_out,       // (N, 3, 3)
    float* __restrict__ Fe_out,           // (N, 3, 3) — Fe_new after plasticity
    float* __restrict__ Jp_out,           // (N,)
    int N,
    float theta_c, float theta_s, float hardening,
    float mu_0, float lambda_0, float dt, float p_vol, float inv_dx, float p_mass
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    Mat3 Fe = mat3_load(Fe_in + tid * 9);
    float Jp = Jp_in[tid];

    // SVD
    Mat3 U, V;
    float sig[3];
    svd3x3(Fe, U, sig, V);

    // Plasticity: clamp singular values
    float sig_c[3], sig_prod = 1.0f, sig_c_prod = 1.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        sig_prod *= sig[i];
        sig_c[i] = fminf(fmaxf(sig[i], 1.0f - theta_c), 1.0f + theta_s);
        sig_c_prod *= sig_c[i];
    }
    float Jp_new = Jp * sig_prod / sig_c_prod;

    // Reconstruct Fe_new = U * diag(sig_c) * V^T and R = U * V^T
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

    // Stress: fixed corotated with hardening
    float J = sig_c_prod;
    float h = expf(hardening * (1.0f - Jp_new));
    float mu = mu_0 * h;
    float la = lambda_0 * h;
    float la_term = la * (J - 1.0f) * J;

    // Affine = -dt * p_vol * 4 * inv_dx^2 * stress + p_mass * C
    float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;
    Mat3 Cm = mat3_load(C_in + tid * 9);
    float aff[9];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            float diff_dot_feT = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; k++)
                diff_dot_feT += (Fe_new.m[i][k] - R.m[i][k]) * Fe_new.m[j][k];
            float s = 2.0f * mu * diff_dot_feT + ((i == j) ? la_term : 0.0f);
            aff[i*3+j] = coeff * s + p_mass * Cm.m[i][j];
        }

    // Write outputs
    mat3_store(stress_out + tid * 9, *(Mat3*)aff);
    mat3_store(Fe_out + tid * 9, Fe_new);
    Jp_out[tid] = Jp_new;
}


// ===== Kernel 2: P2G scatter (global atomicAdd) =====

extern "C" __global__ void p2g_kernel(
    const float* __restrict__ x_in,       // (N, 3)
    const float* __restrict__ v_in,       // (N, 3)
    const float* __restrict__ affine_in,  // (N, 3, 3) — precomputed affine matrix
    float* __restrict__ grid_v,           // (GR^3, 3)
    float* __restrict__ grid_m,           // (GR^3,)
    int N, int grid_res, float inv_dx, float dx, float p_mass
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = x_in[tid*3+0], py = x_in[tid*3+1], pz = x_in[tid*3+2];
    float vx = v_in[tid*3+0], vy = v_in[tid*3+1], vz = v_in[tid*3+2];
    float aff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) aff[i] = affine_in[tid*9 + i];

    int base_x = (int)floorf(px * inv_dx - 0.5f);
    int base_y = (int)floorf(py * inv_dx - 0.5f);
    int base_z = (int)floorf(pz * inv_dx - 0.5f);
    float fx_x = px * inv_dx - (float)base_x;
    float fx_y = py * inv_dx - (float)base_y;
    float fx_z = pz * inv_dx - (float)base_z;

    float wx[3], wy[3], wz[3];
    wx[0] = 0.5f*(1.5f-fx_x)*(1.5f-fx_x); wx[1] = 0.75f-(fx_x-1.0f)*(fx_x-1.0f); wx[2] = 0.5f*(fx_x-0.5f)*(fx_x-0.5f);
    wy[0] = 0.5f*(1.5f-fx_y)*(1.5f-fx_y); wy[1] = 0.75f-(fx_y-1.0f)*(fx_y-1.0f); wy[2] = 0.5f*(fx_y-0.5f)*(fx_y-0.5f);
    wz[0] = 0.5f*(1.5f-fx_z)*(1.5f-fx_z); wz[1] = 0.75f-(fx_z-1.0f)*(fx_z-1.0f); wz[2] = 0.5f*(fx_z-0.5f)*(fx_z-0.5f);

    int GR = grid_res;
    #pragma unroll
    for (int di = 0; di < 3; di++)
        #pragma unroll
        for (int dj = 0; dj < 3; dj++)
            #pragma unroll
            for (int dk = 0; dk < 3; dk++) {
                float w = wx[di] * wy[dj] * wz[dk];
                int ni = min(max(base_x + di, 0), GR-1);
                int nj = min(max(base_y + dj, 0), GR-1);
                int nk = min(max(base_z + dk, 0), GR-1);
                int flat = ni*GR*GR + nj*GR + nk;

                float dpx = ((float)di - fx_x) * dx;
                float dpy = ((float)dj - fx_y) * dx;
                float dpz = ((float)dk - fx_z) * dx;

                float mvx = w * (p_mass*vx + aff[0]*dpx + aff[1]*dpy + aff[2]*dpz);
                float mvy = w * (p_mass*vy + aff[3]*dpx + aff[4]*dpy + aff[5]*dpz);
                float mvz = w * (p_mass*vz + aff[6]*dpx + aff[7]*dpy + aff[8]*dpz);

                atomicAdd(&grid_v[flat*3+0], mvx);
                atomicAdd(&grid_v[flat*3+1], mvy);
                atomicAdd(&grid_v[flat*3+2], mvz);
                atomicAdd(&grid_m[flat], w * p_mass);
            }
}


// ===== Kernel 3: Grid operations (normalize, gravity, boundary) =====

extern "C" __global__ void grid_ops_kernel(
    float* __restrict__ grid_v,       // (GR^3, 3) momentum → velocity
    const float* __restrict__ grid_m, // (GR^3,)
    int grid_res, float dt,
    float grav_x, float grav_y, float grav_z,
    int bound
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_res * grid_res * grid_res;
    if (idx >= total) return;

    float m = grid_m[idx];
    if (m > 0.0f) {
        float inv_m = 1.0f / m;
        grid_v[idx*3+0] = grid_v[idx*3+0] * inv_m + dt * grav_x;
        grid_v[idx*3+1] = grid_v[idx*3+1] * inv_m + dt * grav_y;
        grid_v[idx*3+2] = grid_v[idx*3+2] * inv_m + dt * grav_z;
    } else {
        grid_v[idx*3+0] = 0.0f;
        grid_v[idx*3+1] = 0.0f;
        grid_v[idx*3+2] = 0.0f;
    }

    // Boundary conditions
    int GR = grid_res;
    int iz = idx % GR;
    int iy = (idx / GR) % GR;
    int ix = idx / (GR * GR);

    if (ix < bound)      grid_v[idx*3+0] = fmaxf(grid_v[idx*3+0], 0.0f);
    if (ix >= GR-bound)  grid_v[idx*3+0] = fminf(grid_v[idx*3+0], 0.0f);
    if (iy < bound)      grid_v[idx*3+1] = fmaxf(grid_v[idx*3+1], 0.0f);
    if (iy >= GR-bound)  grid_v[idx*3+1] = fminf(grid_v[idx*3+1], 0.0f);
    if (iz < bound)      grid_v[idx*3+2] = fmaxf(grid_v[idx*3+2], 0.0f);
    if (iz >= GR-bound)  grid_v[idx*3+2] = fminf(grid_v[idx*3+2], 0.0f);
}


// ===== Kernel 4: G2P gather (grid → particles) =====

extern "C" __global__ void g2p_kernel(
    const float* __restrict__ grid_v,   // (GR^3, 3)
    const float* __restrict__ x_in,     // (N, 3)
    const float* __restrict__ Fe_in,    // (N, 3, 3) — Fe_new from stress kernel
    float* __restrict__ x_out,          // (N, 3)
    float* __restrict__ v_out,          // (N, 3)
    float* __restrict__ C_out,          // (N, 3, 3)
    float* __restrict__ Fe_out,         // (N, 3, 3) — (I + dt*C) @ Fe_new
    int N, int grid_res,
    float dt, float inv_dx, float dx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = x_in[tid*3+0], py = x_in[tid*3+1], pz = x_in[tid*3+2];

    int base_x = (int)floorf(px * inv_dx - 0.5f);
    int base_y = (int)floorf(py * inv_dx - 0.5f);
    int base_z = (int)floorf(pz * inv_dx - 0.5f);
    float fx_x = px * inv_dx - (float)base_x;
    float fx_y = py * inv_dx - (float)base_y;
    float fx_z = pz * inv_dx - (float)base_z;

    float wx[3], wy[3], wz[3];
    wx[0] = 0.5f*(1.5f-fx_x)*(1.5f-fx_x); wx[1] = 0.75f-(fx_x-1.0f)*(fx_x-1.0f); wx[2] = 0.5f*(fx_x-0.5f)*(fx_x-0.5f);
    wy[0] = 0.5f*(1.5f-fx_y)*(1.5f-fx_y); wy[1] = 0.75f-(fx_y-1.0f)*(fx_y-1.0f); wy[2] = 0.5f*(fx_y-0.5f)*(fx_y-0.5f);
    wz[0] = 0.5f*(1.5f-fx_z)*(1.5f-fx_z); wz[1] = 0.75f-(fx_z-1.0f)*(fx_z-1.0f); wz[2] = 0.5f*(fx_z-0.5f)*(fx_z-0.5f);

    int GR = grid_res;
    float new_vx = 0.0f, new_vy = 0.0f, new_vz = 0.0f;
    float new_C[9] = {0.0f};

    #pragma unroll
    for (int di = 0; di < 3; di++)
        #pragma unroll
        for (int dj = 0; dj < 3; dj++)
            #pragma unroll
            for (int dk = 0; dk < 3; dk++) {
                float w = wx[di] * wy[dj] * wz[dk];
                int ni = min(max(base_x + di, 0), GR-1);
                int nj = min(max(base_y + dj, 0), GR-1);
                int nk = min(max(base_z + dk, 0), GR-1);
                int flat = ni*GR*GR + nj*GR + nk;

                float gvx = grid_v[flat*3+0];
                float gvy = grid_v[flat*3+1];
                float gvz = grid_v[flat*3+2];

                new_vx += w * gvx;
                new_vy += w * gvy;
                new_vz += w * gvz;

                float dpx = ((float)di - fx_x) * dx;
                float dpy = ((float)dj - fx_y) * dx;
                float dpz = ((float)dk - fx_z) * dx;

                float scale = 4.0f * inv_dx * inv_dx;
                new_C[0] += scale * w * gvx * dpx;
                new_C[1] += scale * w * gvx * dpy;
                new_C[2] += scale * w * gvx * dpz;
                new_C[3] += scale * w * gvy * dpx;
                new_C[4] += scale * w * gvy * dpy;
                new_C[5] += scale * w * gvy * dpz;
                new_C[6] += scale * w * gvz * dpx;
                new_C[7] += scale * w * gvz * dpy;
                new_C[8] += scale * w * gvz * dpz;
            }

    // Update position (clamped to boundary)
    float bound = 3.0f * dx;
    x_out[tid*3+0] = fminf(fmaxf(px + dt * new_vx, bound), 1.0f - bound);
    x_out[tid*3+1] = fminf(fmaxf(py + dt * new_vy, bound), 1.0f - bound);
    x_out[tid*3+2] = fminf(fmaxf(pz + dt * new_vz, bound), 1.0f - bound);

    // Write velocity
    v_out[tid*3+0] = new_vx;
    v_out[tid*3+1] = new_vy;
    v_out[tid*3+2] = new_vz;

    // Write C
    #pragma unroll
    for (int i = 0; i < 9; i++) C_out[tid*9+i] = new_C[i];

    // Update Fe = (I + dt*C) @ Fe_new
    Mat3 Fe_new = mat3_load(Fe_in + tid * 9);
    Mat3 I_dtC;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            I_dtC.m[i][j] = ((i == j) ? 1.0f : 0.0f) + dt * new_C[i*3+j];
    Mat3 Fe_final = mat3_mul(I_dtC, Fe_new);
    mat3_store(Fe_out + tid * 9, Fe_final);
}
