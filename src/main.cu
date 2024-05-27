#include "cast.cuh"
#include "compare.cuh"
#include "cublas.cuh"
#include "gemm_simple.cuh"
#include "gemm_v2.cuh"
#include "rng.cuh"

void test_v2(size_t m, size_t n, size_t k, size_t repeat)
{
    float *c_cublas, *a, *b;
    half *a_half, *b_half, *ct_half, *cb_half;
    auto size_a = m * k;
    auto size_b = k * n;
    auto size_c = m * n;

    // init memory
    CUDA_CALL(cudaMalloc(&c_cublas, size_c * sizeof(float)));
    CUDA_CALL(cudaMalloc(&a, size_a * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b, size_b * sizeof(float)));
    CUDA_CALL(cudaMalloc(&a_half, size_a * sizeof(half)));
    CUDA_CALL(cudaMalloc(&b_half, size_b * sizeof(half)));
    CUDA_CALL(cudaMalloc(&ct_half, size_c * sizeof(half)));
    CUDA_CALL(cudaMalloc(&cb_half, size_c * sizeof(half)));

    RNG gen(1234ULL);
    gen.rand(a, size_a);
    cast(a_half, a, size_a);
    gen.rand(b, size_b);
    cast(b_half, b, size_b);
    CUDA_CALL(cudaMemset(c_cublas, 0, size_c * sizeof(float)));
    CUDA_CALL(cudaMemset(ct_half, 0, size_c * sizeof(half)));
    CUDA_CALL(cudaMemset(cb_half, 0, size_c * sizeof(half)));

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    float cublas_time = 0, our_time = 0;
    Cublas cublas(stream);
    // run fp32 cublas
    cublas.gemm(c_cublas, a, b, m, n, k);
    // run fp16 cutlass and cublas
    for (size_t i = 0; i < repeat; i++)
    {
        float ms;
        CUDA_CALL(cudaEventRecord(start, stream));
        cublas.gemm(cb_half, a_half, b_half, m, n, k);
        CUDA_CALL(cudaEventRecord(end, stream));
        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        cublas_time += ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        gemm_v2<half>(ct_half, a_half, b_half, m, n, k, stream);
        CUDA_CALL(cudaEventRecord(end, stream));
        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        our_time += ms;
    }
    printf("cublas time: %.3f ms\n", cublas_time / repeat);
    printf("our time: %.3f ms\n", our_time / repeat);

    // fp32 cublas compare with fp16 cutlass
    // compare(c_cublas, ct_half, 1e-3, size_c);

    // {
    //     std::vector<float> host(m * k);
    //     CUDA_CALL(cudaMemcpy(host.data(), a, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    //     printf("A %ldx%ld\n", m, k);
    //     for (size_t i = 0; i < m; i++)
    //     {
    //         for (size_t j = 0; j < k; j++)
    //         {
    //             printf("%.3f ", host[i * k + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // {
    //     std::vector<half> host(m * k);
    //     CUDA_CALL(cudaMemcpy(host.data(), a_half, m * k * sizeof(half), cudaMemcpyDeviceToHost));
    //     printf("A_half %ldx%ld\n", m, k);
    //     for (size_t i = 0; i < m; i++)
    //     {
    //         for (size_t j = 0; j < k; j++)
    //         {
    //             printf("%.3f ", (float)host[i * k + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // {
    //     std::vector<float> host(n * k);
    //     CUDA_CALL(cudaMemcpy(host.data(), b, n * k * sizeof(float), cudaMemcpyDeviceToHost));
    //     printf("B %ldx%ld\n", k, n);
    //     for (size_t i = 0; i < k; i++)
    //     {
    //         for (size_t j = 0; j < n; j++)
    //         {
    //             printf("%.3f ", host[i * n + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // {
    //     std::vector<half> host(n * k);
    //     CUDA_CALL(cudaMemcpy(host.data(), b_half, n * k * sizeof(half), cudaMemcpyDeviceToHost));
    //     printf("B_half %ldx%ld\n", k, n);
    //     for (size_t i = 0; i < k; i++)
    //     {
    //         for (size_t j = 0; j < n; j++)
    //         {
    //             printf("%.3f ", (float)host[i * n + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // {
    //     std::vector<half> host(m * n);
    //     CUDA_CALL(cudaMemcpy(host.data(), ct_half, m * n * sizeof(half), cudaMemcpyDeviceToHost));
    //     printf("C CUTLASS %ldx%ld\n", m, n);
    //     for (size_t i = 0; i < m; i++)
    //     {
    //         for (size_t j = 0; j < n; j++)
    //         {
    //             printf("%.3f ", (float)host[i * n + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // {
    //     std::vector<float> host(m * n);
    //     CUDA_CALL(cudaMemcpy(host.data(), c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    //     printf("C CUBLAS %ldx%ld\n", m, n);
    //     for (size_t i = 0; i < m; i++)
    //     {
    //         for (size_t j = 0; j < n; j++)
    //         {
    //             printf("%.3f ", host[i * n + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // {
    //     std::vector<float> cublas_host(m * n);
    //     std::vector<half> cutlass_host(m * n);
    //     CUDA_CALL(cudaMemcpy(cublas_host.data(), c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    //     CUDA_CALL(cudaMemcpy(cutlass_host.data(), ct_half, m * n * sizeof(half), cudaMemcpyDeviceToHost));
    //     printf("CUBLAS COMPARE CUTLASS %ldx%ld\n", m, n);
    //     for (size_t i = 0; i < m; i++)
    //     {
    //         for (size_t j = 0; j < n; j++)
    //         {
    //             float t_cublas = cublas_host[i * n + j];
    //             float t_cutlass = (float)cutlass_host[i * n + j];
    //             float diff = std::abs(t_cublas - t_cutlass);
    //             printf("%.3f ", diff);
    //         }
    //         printf("\n");
    //     }
    // }
    compare(c_cublas, cb_half, 1e-3, size_c);

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(c_cublas));
    CUDA_CALL(cudaFree(a));
    CUDA_CALL(cudaFree(b));
    CUDA_CALL(cudaFree(a_half));
    CUDA_CALL(cudaFree(b_half));
    CUDA_CALL(cudaFree(ct_half));
    CUDA_CALL(cudaFree(cb_half));
}

void test(size_t m, size_t n, size_t k, size_t repeat)
{
    half *c_cublas, *c_our, *a, *b;
    CUDA_CALL(cudaMalloc(&c_cublas, m * n * sizeof(half)));
    CUDA_CALL(cudaMalloc(&c_our, m * n * sizeof(half)));
    CUDA_CALL(cudaMalloc(&a, m * k * sizeof(half)));
    CUDA_CALL(cudaMalloc(&b, k * n * sizeof(half)));

    RNG gen(1234ULL);
    float *rand;
    CUDA_CALL(cudaMalloc(&rand, std::max(m * k, k * n) * sizeof(float)));
    gen.rand(rand, m * k);
    cast(a, rand, m * k);
    gen.rand(rand, k * n);
    cast(b, rand, k * n);

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    float cublas_time = 0, our_time = 0;

    Cublas cublas(stream);
    for (size_t i = 0; i < repeat; i++)
    {
        float ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        cublas.gemm(c_cublas, a, b, m, n, k);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        cublas_time += ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        gemm_simple(c_our, a, b, m, n, k, stream);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        our_time += ms;
    }
    printf("cublas time: %.3f ms\n", cublas_time / repeat);
    printf("our time: %.3f ms\n", our_time / repeat);

    compare(c_cublas, c_our, 1e-3, m * n);

    {
        std::vector<half> host(m * k);
        CUDA_CALL(cudaMemcpy(host.data(), a, m * k * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(a));
        printf("A %ldx%ld\n", m, k);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < k; j++) {
        //         printf("%.3f ", (float) host[i * k + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(k * n);
        CUDA_CALL(cudaMemcpy(host.data(), b, k * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(b));
        printf("B %ldx%ld\n", k, n);
        // for (size_t i = 0; i < k; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i + j * k]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_cublas, m * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_cublas));
        printf("C CUBLAS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<half> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_our, m * n * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_our));
        printf("C CUTLASS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaStreamDestroy(stream));
}

void test_fp64(size_t m, size_t n, size_t k, size_t repeat)
{
    using T = double;
    T *c_cublas, *c_our, *a, *b;
    CUDA_CALL(cudaMalloc(&c_cublas, m * n * sizeof(T)));
    CUDA_CALL(cudaMalloc(&c_our, m * n * sizeof(T)));
    CUDA_CALL(cudaMalloc(&a, m * k * sizeof(T)));
    CUDA_CALL(cudaMalloc(&b, k * n * sizeof(T)));

    RNG gen(1234ULL);
    // T *rand;
    // CUDA_CALL(cudaMalloc(&rand, std::max(m * k, k * n) * sizeof(T)));
    gen.rand(a, m * k);
    gen.rand(b, k * n);

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    float cublas_time = 0, our_time = 0;

    Cublas cublas(stream);
    for (size_t i = 0; i < repeat; i++)
    {
        float ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        cublas.gemm(c_cublas, a, b, m, n, k);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        cublas_time += ms;

        CUDA_CALL(cudaEventRecord(start, stream));
        gemm_simple_fp64(c_our, a, b, m, n, k, stream);
        // gemm_v2(a, b, c_our, m, n, k, stream);
        CUDA_CALL(cudaEventRecord(end, stream));

        CUDA_CALL(cudaEventSynchronize(end));
        CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        our_time += ms;
    }
    printf("cublas time: %.3f ms\n", cublas_time / repeat);
    printf("our time: %.3f ms\n", our_time / repeat);

    compare(c_cublas, c_our, 1e-3, m * n);

    {
        std::vector<T> host(m * k);
        CUDA_CALL(cudaMemcpy(host.data(), a, m * k * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(a));
        printf("A %ldx%ld\n", m, k);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < k; j++) {
        //         printf("%.3f ", (float) host[i * k + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<T> host(k * n);
        CUDA_CALL(cudaMemcpy(host.data(), b, k * n * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(b));
        printf("B %ldx%ld\n", k, n);
        // for (size_t i = 0; i < k; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i + j * k]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<T> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_cublas, m * n * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_cublas));
        printf("C CUBLAS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }
    {
        std::vector<T> host(m * n);
        CUDA_CALL(cudaMemcpy(host.data(), c_our, m * n * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(c_our));
        printf("C CUTLASS %ldx%ld\n", m, n);
        // for (size_t i = 0; i < m; i++) {
        //     for (size_t j = 0; j < n; j++) {
        //         printf("%.3f ", (float) host[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
    CUDA_CALL(cudaStreamDestroy(stream));
}
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Usage: %s m n k\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t m = atoi(argv[1]);
    size_t n = atoi(argv[2]);
    size_t k = atoi(argv[3]);
    // test(m, n, k, 100);
    // test_fp64(m, n, k, 1);
    test_v2(m, n, k, 100);
    return EXIT_SUCCESS;
}
