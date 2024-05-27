#ifndef __CUBLAS_CUH__
#define __CUBLAS_CUH__

#include <cublas_v2.h>

class Cublas
{
    cublasHandle_t _handle;

public:
    Cublas(cudaStream_t stream)
    {
        cublasCreate(&_handle);
        cublasSetStream(_handle, stream);
    }
    ~Cublas() noexcept
    {
        cublasDestroy(_handle);
    }

    Cublas(Cublas const &) = delete;
    Cublas(Cublas &&) noexcept = delete;
    Cublas &operator=(Cublas const &) = delete;
    Cublas &operator=(Cublas &&) noexcept = delete;

    void gemm(half *c, half const *a, half const *b,
              size_t m, size_t n, size_t k)
    {
        half alpha = 1, beta = 0;
        cublasHgemm(_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, k,
                    a, k,
                    &beta,
                    c, n);
    }
    void gemm(double *c, double const *a, double const *b,
              size_t m, size_t n, size_t k)
    {
        double alpha = 1, beta = 0;
        cublasDgemm(_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, k,
                    a, k,
                    &beta,
                    c, n);
    }
    void gemm(float *c, float const *a, float const *b,
              size_t m, size_t n, size_t k)
    {
        float alpha = 1.0, beta = 0.0;
        cublasSgemm(_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    b, k,
                    a, k,
                    &beta,
                    c, n);
    }
};

#endif // __CUBLAS_CUH__
