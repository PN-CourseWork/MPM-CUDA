#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_stress_p2g_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor C,
    torch::Tensor Fe, torch::Tensor Jp,
    int grid_res, float dt, float inv_dx, float dx,
    float p_vol, float p_mass,
    float theta_c, float theta_s, float hardening,
    float mu_0, float lambda_0,
    int block_size, int newton_schulz_iters
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_stress_p2g", &fused_stress_p2g_cuda,
          "Fused stress + P2G scatter (CUDA)");
}
