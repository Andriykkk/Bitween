#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration of CUDA kernel
extern "C" void quantized_matmul_cuda(
    const half* x,
    const int* qweight,
    const half* scale,
    const half* zero_point,
    const half* bias,
    half* out,
    int M, int N, int K,
    int bits, int group_size
);

// PyTorch wrapper
torch::Tensor quantized_matmul_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scale,
    torch::Tensor zero_point,
    torch::optional<torch::Tensor> bias,
    int bits,
    int group_size
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(zero_point.is_cuda(), "zero_point must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(qweight.dtype() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(scale.dtype() == torch::kFloat16, "scale must be float16");
    TORCH_CHECK(zero_point.dtype() == torch::kFloat16, "zero_point must be float16");

    // Get dimensions
    auto x_shape = x.sizes();
    int M = x.size(-2);  // Last two dimensions
    int K = x.size(-1);
    int N = qweight.size(0);

    // Reshape x to 2D if needed
    auto original_shape = x.sizes();
    auto x_2d = x.reshape({-1, K});
    M = x_2d.size(0);

    // Allocate output
    auto out = torch::empty({M, N}, x.options());

    // Get pointers
    const half* x_ptr = reinterpret_cast<const half*>(x_2d.data_ptr<at::Half>());
    const int* qweight_ptr = qweight.data_ptr<int>();
    const half* scale_ptr = reinterpret_cast<const half*>(scale.data_ptr<at::Half>());
    const half* zp_ptr = reinterpret_cast<const half*>(zero_point.data_ptr<at::Half>());
    const half* bias_ptr = bias.has_value() ?
        reinterpret_cast<const half*>(bias.value().data_ptr<at::Half>()) : nullptr;
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

    // Launch kernel
    quantized_matmul_cuda(
        x_ptr, qweight_ptr, scale_ptr, zp_ptr, bias_ptr, out_ptr,
        M, N, K, bits, group_size
    );

    // Reshape output back to original batch dimensions
    std::vector<int64_t> output_shape;
    for (int i = 0; i < original_shape.size() - 1; ++i) {
        output_shape.push_back(original_shape[i]);
    }
    output_shape.push_back(N);

    return out.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantized_matmul_forward", &quantized_matmul_forward, "Quantized matmul forward (CUDA)");
}
