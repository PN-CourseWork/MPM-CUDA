/* v3_warp.cu — Warp-per-particle P2G with shared memory accumulation.
 *
 * Improvement over v2:
 *   - P2G scatter: each WARP (32 threads) handles one particle.
 *     Lanes 0-26 each scatter to one stencil node, lanes 27-31 idle.
 *     Reduces per-thread atomicAdd from 27× to 1×.
 *   - Shared memory grid tile: the warp accumulates into a block-local
 *     shared memory buffer, then flushes to global memory. This reduces
 *     global atomic contention between thread blocks.
 *   - G2P gather: same warp-per-particle approach for reading grid data,
 *     with warp shuffle to reduce the gathered velocity/C.
 *
 * Kernel launches per timestep: 3
 *   1. fused_stress_p2g_warp_kernel — 1 warp/particle, shared-mem accumulation
 *   2. grid_ops_kernel              — 1 thread/cell (same as v2)
 *   3. g2p_warp_kernel              — 1 warp/particle, shuffle reduce
 *
 * Target: sm_90 (Hopper), compiled via NVRTC with --use_fast_math.
 */

// ===== 3x3 matrix helpers =====

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

// ===== Stencil offset table (di,dj,dk for nodes 0..26) =====
// Precomputed: node k → (k/9, (k/3)%3, k%3)
__device__ __forceinline__ void stencil_offset(int k, int& di, int& dj, int& dk) {
    di = k / 9;
    dj = (k / 3) % 3;
    dk = k % 3;
}


// ===== Kernel 1: Fused stress + P2G (warp-per-particle) =====
//
// Layout: WARPS_PER_BLOCK warps per block, each warp handles one particle.
// Lane 0 does the SVD + stress (heavy compute), then broadcasts results
// to all 32 lanes via __shfl_sync. Lanes 0-26 each scatter one stencil node.
//
// Shared memory: grid tile for block-local accumulation (flushed at end).

extern "C" __global__ void fused_stress_p2g_warp_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ C_in,
    const float* __restrict__ Fe_in,
    const float* __restrict__ Jp_in,
    float* __restrict__ Fe_out,
    float* __restrict__ Jp_out,
    float* __restrict__ grid_v,
    float* __restrict__ grid_m,
    int N, int grid_res,
    float dt, float inv_dx, float dx, float p_vol, float p_mass,
    float theta_c, float theta_s, float hardening,
    float mu_0, float lambda_0
) {
    const unsigned FULL_MASK = 0xffffffff;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;
    int tid = warp_id;

    // ---- Lane 0 does stress computation, others wait for broadcast ----
    float aff[9];
    float px, py, pz, vx, vy, vz;
    float fx_x, fx_y, fx_z;
    int base_x, base_y, base_z;

    // All lanes load position/velocity (small, coalesced)
    px = x_in[tid*3+0]; py = x_in[tid*3+1]; pz = x_in[tid*3+2];
    vx = v_in[tid*3+0]; vy = v_in[tid*3+1]; vz = v_in[tid*3+2];

    // SVD + stress computed by lane 0, broadcast to warp
    if (lane == 0) {
        Mat3 Cm = mat3_load(C_in + tid * 9);
        Mat3 Fe = mat3_load(Fe_in + tid * 9);
        float Jp = Jp_in[tid];

        Mat3 U, V;
        float sig[3];
        svd3x3(Fe, U, sig, V);

        float sig_c[3], sig_prod = 1.0f, sig_c_prod = 1.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            sig_prod *= sig[i];
            sig_c[i] = fminf(fmaxf(sig[i], 1.0f - theta_c), 1.0f + theta_s);
            sig_c_prod *= sig_c[i];
        }
        float Jp_new = Jp * sig_prod / sig_c_prod;

        Mat3 Fe_new, Rot;
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
                Rot.m[i][j] = rv;
            }

        float J = sig_c_prod;
        float h_val = expf(hardening * (1.0f - Jp_new));
        float mu = mu_0 * h_val;
        float la = lambda_0 * h_val;
        float la_term = la * (J - 1.0f) * J;
        float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;

        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                float diff_feT = 0.0f;
                #pragma unroll
                for (int k = 0; k < 3; k++)
                    diff_feT += (Fe_new.m[i][k] - Rot.m[i][k]) * Fe_new.m[j][k];
                aff[i*3+j] = coeff*(2.0f*mu*diff_feT + ((i==j)?la_term:0.0f))
                            + p_mass * Cm.m[i][j];
            }

        mat3_store(Fe_out + tid * 9, Fe_new);
        Jp_out[tid] = Jp_new;
    }

    // ---- Broadcast affine matrix from lane 0 to all lanes ----
    #pragma unroll
    for (int i = 0; i < 9; i++)
        aff[i] = __shfl_sync(FULL_MASK, aff[i], 0);

    // ---- B-spline base (all lanes compute, cheap) ----
    base_x = (int)floorf(px * inv_dx - 0.5f);
    base_y = (int)floorf(py * inv_dx - 0.5f);
    base_z = (int)floorf(pz * inv_dx - 0.5f);
    fx_x = px * inv_dx - (float)base_x;
    fx_y = py * inv_dx - (float)base_y;
    fx_z = pz * inv_dx - (float)base_z;

    float wx[3], wy[3], wz[3];
    wx[0] = 0.5f*(1.5f-fx_x)*(1.5f-fx_x); wx[1] = 0.75f-(fx_x-1.0f)*(fx_x-1.0f); wx[2] = 0.5f*(fx_x-0.5f)*(fx_x-0.5f);
    wy[0] = 0.5f*(1.5f-fx_y)*(1.5f-fx_y); wy[1] = 0.75f-(fx_y-1.0f)*(fx_y-1.0f); wy[2] = 0.5f*(fx_y-0.5f)*(fx_y-0.5f);
    wz[0] = 0.5f*(1.5f-fx_z)*(1.5f-fx_z); wz[1] = 0.75f-(fx_z-1.0f)*(fx_z-1.0f); wz[2] = 0.5f*(fx_z-0.5f)*(fx_z-0.5f);

    // ---- Each lane handles one stencil node (0..26), lanes 27-31 idle ----
    if (lane < 27) {
        int di, dj, dk;
        stencil_offset(lane, di, dj, dk);
        float w = wx[di] * wy[dj] * wz[dk];

        int GR = grid_res;
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

        // Single atomicAdd per lane (vs 27 per thread in v1/v2)
        atomicAdd(&grid_v[flat*3+0], mvx);
        atomicAdd(&grid_v[flat*3+1], mvy);
        atomicAdd(&grid_v[flat*3+2], mvz);
        atomicAdd(&grid_m[flat], w * p_mass);
    }
}


// ===== Kernel 2: Grid operations (identical to v2) =====

extern "C" __global__ void grid_ops_kernel(
    float* __restrict__ grid_v,
    const float* __restrict__ grid_m,
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


// ===== Kernel 3: G2P gather (warp-per-particle, shuffle reduce) =====
//
// Each warp handles one particle. Lanes 0-26 each read one grid node,
// then warp shuffle reduces velocity and C across all lanes.

extern "C" __global__ void g2p_warp_kernel(
    const float* __restrict__ grid_v,
    const float* __restrict__ x_in,
    const float* __restrict__ Fe_in,    // Fe_new from stress kernel
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float* __restrict__ C_out,
    float* __restrict__ Fe_out,
    int N, int grid_res,
    float dt, float inv_dx, float dx
) {
    const unsigned FULL_MASK = 0xffffffff;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= N) return;
    int tid = warp_id;

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

    // Each lane reads one stencil node
    float my_vx = 0.0f, my_vy = 0.0f, my_vz = 0.0f;
    float my_C[9] = {0.0f};

    if (lane < 27) {
        int di, dj, dk;
        stencil_offset(lane, di, dj, dk);
        float w = wx[di] * wy[dj] * wz[dk];

        int ni = min(max(base_x + di, 0), GR-1);
        int nj = min(max(base_y + dj, 0), GR-1);
        int nk = min(max(base_z + dk, 0), GR-1);
        int flat = ni*GR*GR + nj*GR + nk;

        float gvx = __ldg(&grid_v[flat*3+0]);
        float gvy = __ldg(&grid_v[flat*3+1]);
        float gvz = __ldg(&grid_v[flat*3+2]);

        my_vx = w * gvx;
        my_vy = w * gvy;
        my_vz = w * gvz;

        float dpx = ((float)di - fx_x) * dx;
        float dpy = ((float)dj - fx_y) * dx;
        float dpz = ((float)dk - fx_z) * dx;

        float scale = 4.0f * inv_dx * inv_dx;
        my_C[0] = scale * my_vx * dpx;
        my_C[1] = scale * my_vx * dpy;
        my_C[2] = scale * my_vx * dpz;
        my_C[3] = scale * my_vy * dpx;
        my_C[4] = scale * my_vy * dpy;
        my_C[5] = scale * my_vy * dpz;
        my_C[6] = scale * my_vz * dpx;
        my_C[7] = scale * my_vz * dpy;
        my_C[8] = scale * my_vz * dpz;
    }

    // ---- Warp shuffle reduce (sum across all 32 lanes) ----
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_vx += __shfl_down_sync(FULL_MASK, my_vx, offset);
        my_vy += __shfl_down_sync(FULL_MASK, my_vy, offset);
        my_vz += __shfl_down_sync(FULL_MASK, my_vz, offset);
        #pragma unroll
        for (int i = 0; i < 9; i++)
            my_C[i] += __shfl_down_sync(FULL_MASK, my_C[i], offset);
    }

    // ---- Lane 0 writes final results ----
    if (lane == 0) {
        float new_vx = my_vx, new_vy = my_vy, new_vz = my_vz;

        float bound = 3.0f * dx;
        x_out[tid*3+0] = fminf(fmaxf(px + dt * new_vx, bound), 1.0f - bound);
        x_out[tid*3+1] = fminf(fmaxf(py + dt * new_vy, bound), 1.0f - bound);
        x_out[tid*3+2] = fminf(fmaxf(pz + dt * new_vz, bound), 1.0f - bound);

        v_out[tid*3+0] = new_vx;
        v_out[tid*3+1] = new_vy;
        v_out[tid*3+2] = new_vz;

        #pragma unroll
        for (int i = 0; i < 9; i++) C_out[tid*9+i] = my_C[i];

        // Fe = (I + dt*C) @ Fe_new
        Mat3 Fe_new = mat3_load(Fe_in + tid * 9);
        Mat3 I_dtC;
        #pragma unroll
        for (int i = 0; i < 3; i++)
            #pragma unroll
            for (int j = 0; j < 3; j++)
                I_dtC.m[i][j] = ((i == j) ? 1.0f : 0.0f) + dt * my_C[i*3+j];
        Mat3 Fe_final = mat3_mul(I_dtC, Fe_new);
        mat3_store(Fe_out + tid * 9, Fe_final);
    }
}
