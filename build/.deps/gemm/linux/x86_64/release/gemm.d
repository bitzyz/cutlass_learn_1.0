{
    files = {
        "build/.objs/gemm/linux/x86_64/release/src/main.cu.o",
        "build/.objs/gemm/linux/x86_64/release/rules/cuda/devlink/gemm_gpucode.cu.o"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-L/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/lib64",
            "-Wl,-rpath=/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/lib64",
            "-s",
            "-lcurand",
            "-lcublas",
            "-lcudadevrt",
            "-lcudart_static",
            "-lrt",
            "-lpthread",
            "-ldl"
        }
    }
}