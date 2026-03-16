#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
p2g_cuda(
    torch::Tensor x, torch::Tensor v, torch::Tensor C,
    torch::Tensor stress,
    int grid_res, float dt, float inv_dx, float dx,
    float p_vol, float p_mass,
    int block_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("p2g", &p2g_cuda, "P2G scatter (CUDA)");
}
