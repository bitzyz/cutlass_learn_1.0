{
    files = {
        "build/.objs/gemm/linux/x86_64/release/src/main.cu.o"
    },
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/bin/nvcc",
        {
            "-L/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/lib64",
            "-lcurand",
            "-lcublas",
            "-lcudadevrt",
            "-lcudart_static",
            "-lrt",
            "-lpthread",
            "-ldl",
            "-m64",
            "-dlink"
        }
    }
}