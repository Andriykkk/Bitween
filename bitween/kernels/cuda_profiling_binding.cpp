#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declarations of profiling CUDA kernels
extern "C" void profile_load_x_cuda(const half* x, half* out, int M, int N, int K);
extern "C" void profile_load_packed_cuda(const int* qweight, half* out, int M, int N, int K);
extern "C" void profile_extract_bits_cuda(const int* qweight, half* out, int M, int N, int K);
extern "C" void profile_load_scale_zp_cuda(const half* scale, const half* zero_point, half* out,
                                            int M, int N, int K, int group_size);
extern "C" void profile_dequantize_cuda(const int* qweight, const half* scale, const half* zero_point,
                                         half* out, int M, int N, int K, int group_size);
extern "C" void profile_fp16_matmul_cuda(const half* x, const half* w, half* out, int M, int N, int K);

// PyTorch wrappers
void profile_load_x(torch::Tensor x, torch::Tensor out) {
    int M = x.size(0), K = x.size(1);
    int N = out.size(1);
    profile_load_x_cuda(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_load_packed(torch::Tensor qweight, torch::Tensor out) {
    int N = qweight.size(0);
    int M = out.size(0);
    int packed_K = qweight.size(1);
    int K = packed_K * 4;  // Assuming 8-bit
    profile_load_packed_cuda(
        qweight.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_extract_bits(torch::Tensor qweight, torch::Tensor out) {
    int N = qweight.size(0);
    int M = out.size(0);
    int packed_K = qweight.size(1);
    int K = packed_K * 4;  // Assuming 8-bit
    profile_extract_bits_cuda(
        qweight.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

void profile_load_scale_zp(torch::Tensor scale, torch::Tensor zero_point,
                            torch::Tensor out, int group_size) {
    int N = scale.size(0);
    int M = out.size(0);
    int num_groups = scale.size(1);
    int K = num_groups * group_size;
    profile_load_scale_zp_cuda(
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size
    );
}

void profile_dequantize(torch::Tensor qweight, torch::Tensor scale,
                         torch::Tensor zero_point, torch::Tensor out, int group_size) {
    int N = qweight.size(0);
    int M = out.size(0);
    int packed_K = qweight.size(1);
    int K = packed_K * 4;  // Assuming 8-bit
    profile_dequantize_cuda(
        qweight.data_ptr<int>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size
    );
}

void profile_fp16_matmul(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    int M = x.size(0), K = x.size(1);
    int N = w.size(0);
    profile_fp16_matmul_cuda(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("profile_load_x", &profile_load_x, "Profile: Load X only");
    m.def("profile_load_packed", &profile_load_packed, "Profile: Load packed weights");
    m.def("profile_extract_bits", &profile_extract_bits, "Profile: Extract bits");
    m.def("profile_load_scale_zp", &profile_load_scale_zp, "Profile: Load scale/zp");
    m.def("profile_dequantize", &profile_dequantize, "Profile: Full dequantization");
    m.def("profile_fp16_matmul", &profile_fp16_matmul, "Profile: FP16 tensor core matmul");
}
