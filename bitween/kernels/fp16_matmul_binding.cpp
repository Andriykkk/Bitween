#include <torch/extension.h>
#include <cuda_fp16.h>

extern "C" {
    void fp16_matmul_wmma_cuda(const half* x, const half* w, half* out, int M, int N, int K);
    void fp16_matmul_naive_cuda(const half* x, const half* w, half* out, int M, int N, int K);
}

torch::Tensor fp16_matmul_wmma(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(w.dtype() == torch::kFloat16, "w must be float16");

    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    TORCH_CHECK(w.size(1) == K, "Dimension mismatch");

    auto out = torch::empty({M, N}, x.options());

    fp16_matmul_wmma_cuda(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );

    return out;
}

torch::Tensor fp16_matmul_naive(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(w.dtype() == torch::kFloat16, "w must be float16");

    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    TORCH_CHECK(w.size(1) == K, "Dimension mismatch");

    auto out = torch::empty({M, N}, x.options());

    fp16_matmul_naive_cuda(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_matmul_wmma", &fp16_matmul_wmma, "FP16 matmul with WMMA");
    m.def("fp16_matmul_naive", &fp16_matmul_naive, "FP16 matmul naive");
}
