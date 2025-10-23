#include <torch/extension.h>
#include <cuda_fp16.h>

extern "C" {
    void profile_load_x(const half* x, half* out, int M, int N, int K);
    void profile_load_packed(const int* qweight, half* out, int M, int N, int K);
    void profile_extract_bits(const int* qweight, half* out, int M, int N, int K);
    void profile_load_scale_zp(const half* scale, const half* zero_point, half* out,
                                int M, int N, int K, int group_size);
    void profile_dequant(const int* qweight, const half* scale, const half* zero_point,
                         half* out, int M, int N, int K, int group_size);
    void profile_full_kernel(const half* x, const int* qweight, const half* scale,
                             const half* zero_point, half* out,
                             int M, int N, int K, int group_size);
}

void profile_load_x_py(torch::Tensor x, torch::Tensor out) {
    int M = x.size(0), K = x.size(1), N = out.size(1);
    profile_load_x(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_load_packed_py(torch::Tensor qweight, torch::Tensor out) {
    int N = qweight.size(0), packed_K = qweight.size(1);
    int M = out.size(0);
    int K = packed_K * 4;  // 8-bit
    profile_load_packed(
        qweight.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_extract_bits_py(torch::Tensor qweight, torch::Tensor out) {
    int N = qweight.size(0), packed_K = qweight.size(1);
    int M = out.size(0);
    int K = packed_K * 4;  // 8-bit
    profile_extract_bits(
        qweight.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_load_scale_zp_py(torch::Tensor scale, torch::Tensor zero_point,
                               torch::Tensor out, int group_size) {
    int N = scale.size(0);
    int M = out.size(0);
    int num_groups = scale.size(1);
    int K = num_groups * group_size;
    profile_load_scale_zp(
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size
    );
}

void profile_dequant_py(torch::Tensor qweight, torch::Tensor scale,
                        torch::Tensor zero_point, torch::Tensor out, int group_size) {
    int N = qweight.size(0), packed_K = qweight.size(1);
    int M = out.size(0);
    int K = packed_K * 4;  // 8-bit
    profile_dequant(
        qweight.data_ptr<int>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size
    );
}

void profile_full_kernel_py(torch::Tensor x, torch::Tensor qweight, torch::Tensor scale,
                             torch::Tensor zero_point, torch::Tensor out, int group_size) {
    int M = x.size(0), K = x.size(1);
    int N = qweight.size(0);
    profile_full_kernel(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        qweight.data_ptr<int>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("profile_load_x", &profile_load_x_py, "Profile: Load X");
    m.def("profile_load_packed", &profile_load_packed_py, "Profile: Load packed");
    m.def("profile_extract_bits", &profile_extract_bits_py, "Profile: Extract bits");
    m.def("profile_load_scale_zp", &profile_load_scale_zp_py, "Profile: Load scale/zp");
    m.def("profile_dequant", &profile_dequant_py, "Profile: Dequantization");
    m.def("profile_full_kernel", &profile_full_kernel_py, "Profile: Full kernel");
}
