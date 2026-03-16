/* P2G scatter kernel for 3D MLS-MPM.
 *
 * One thread per particle:
 *   1. Load x, v, C, stress
 *   2. Compute quadratic B-spline weights
 *   3. Scatter momentum + mass to 27 grid nodes via atomicAdd
 *
 * Replaces compute_p2g_data + scatter (avoids (N,27,...) intermediates).
 * Target: NVIDIA Hopper (sm_90).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void p2g_kernel(
    const float* __restrict__ x_in,       // (N, 3)
    const float* __restrict__ v_in,       // (N, 3)
    const float* __restrict__ C_in,       // (N, 3, 3)
    const float* __restrict__ stress_in,  // (N, 3, 3)
    float* __restrict__ grid_v,           // (GR^3, 3)
    float* __restrict__ grid_m,           // (GR^3,)
    int N, int grid_res,
    float dt, float inv_dx, float dx, float p_vol, float p_mass
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Load particle data
    float px = x_in[tid*3 + 0];
    float py = x_in[tid*3 + 1];
    float pz = x_in[tid*3 + 2];

    float vx = v_in[tid*3 + 0];
    float vy = v_in[tid*3 + 1];
    float vz = v_in[tid*3 + 2];

    // Load C and stress as flat arrays, compute affine in-place
    // affine = -dt * p_vol * 4 * inv_dx^2 * stress + p_mass * C
    float coeff = -dt * p_vol * 4.0f * inv_dx * inv_dx;
    float aff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++)
        aff[i] = coeff * stress_in[tid*9 + i] + p_mass * C_in[tid*9 + i];

    // Base cell and fractional position
    int base_x = (int)floorf(px * inv_dx - 0.5f);
    int base_y = (int)floorf(py * inv_dx - 0.5f);
    int base_z = (int)floorf(pz * inv_dx - 0.5f);

    float fx_x = px * inv_dx - (float)base_x;
    float fx_y = py * inv_dx - (float)base_y;
    float fx_z = pz * inv_dx - (float)base_z;

    // Quadratic B-spline weights per axis
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

    // Scatter to 27 grid nodes
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

// Launcher
std::tuple<torch::Tensor, torch::Tensor>
p2g_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor C,
    torch::Tensor stress,
    int grid_res, float dt, float inv_dx, float dx,
    float p_vol, float p_mass,
    int block_size
) {
    int N = x.size(0);
    int GR3 = grid_res * grid_res * grid_res;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto grid_v = torch::zeros({GR3, 3}, opts);
    auto grid_m = torch::zeros({GR3}, opts);

    int grid_dim = (N + block_size - 1) / block_size;

    p2g_kernel<<<grid_dim, block_size>>>(
        x.data_ptr<float>(), v.data_ptr<float>(), C.data_ptr<float>(),
        stress.data_ptr<float>(),
        grid_v.data_ptr<float>(), grid_m.data_ptr<float>(),
        N, grid_res, dt, inv_dx, dx, p_vol, p_mass
    );

    return std::make_tuple(
        grid_v.reshape({grid_res, grid_res, grid_res, 3}),
        grid_m.reshape({grid_res, grid_res, grid_res})
    );
}
