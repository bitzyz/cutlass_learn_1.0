{
    files = {
        "src/main.cu"
    },
    depfiles_gcc = "build/.objs/gemm/linux/x86_64/release/src/main.cu.o : src/main.cu     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_runtime.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/host_config.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/builtin_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/device_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/host_defines.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/driver_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/vector_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/surface_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/texture_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/library_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/channel_descriptor.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_runtime_api.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_device_runtime_api.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/driver_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/vector_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/vector_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/common_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/math_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/math_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_surface_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_texture_types.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/device_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/device_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/device_atomic_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/device_atomic_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/device_double_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/device_double_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_20_atomic_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_20_atomic_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_32_atomic_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_32_atomic_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_35_atomic_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_60_atomic_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_60_atomic_functions.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_20_intrinsics.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_20_intrinsics.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_30_intrinsics.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_30_intrinsics.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_32_intrinsics.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_32_intrinsics.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_35_intrinsics.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_61_intrinsics.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/sm_61_intrinsics.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_70_rt.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_70_rt.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_80_rt.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_80_rt.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_90_rt.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/sm_90_rt.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/surface_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/texture_fetch_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/texture_indirect_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/surface_indirect_functions.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/crt/cudacc_ext.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/device_launch_parameters.h     src/cast.cuh     src/common.cuh     cutlass/include/cute/tensor.hpp     cutlass/include/cute/config.hpp     cutlass/include/cute/util/type_traits.hpp     cutlass/include/cute/numeric/numeric_types.hpp     cutlass/include/cutlass/numeric_types.h     cutlass/include/cutlass/cutlass.h     cutlass/include/cutlass/detail/helper_macros.hpp     cutlass/include/cutlass/platform/platform.h     cutlass/include/cutlass/numeric_size.h     cutlass/include/cutlass/integer_subbyte.h     cutlass/include/cutlass/half.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_fp16.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_fp16.hpp     cutlass/include/cutlass/float8.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_fp8.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_bf16.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_bf16.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda_fp8.hpp     cutlass/include/cutlass/bfloat16.h     cutlass/include/cutlass/tfloat32.h     cutlass/include/cutlass/uint128.h     cutlass/include/cute/numeric/int.hpp     cutlass/include/cute/numeric/real.hpp     cutlass/include/cute/util/print.hpp     cutlass/include/cute/util/debug.hpp     cutlass/include/cute/numeric/integral_constant.hpp     cutlass/include/cute/numeric/math.hpp     cutlass/include/cute/numeric/integer_sequence.hpp     cutlass/include/cute/container/tuple.hpp     cutlass/include/cute/container/cuda_types.hpp     cutlass/include/cute/container/array_aligned.hpp     cutlass/include/cute/container/array.hpp     cutlass/include/cute/container/alignment.hpp     cutlass/include/cute/container/array_subbyte.hpp     cutlass/include/cute/pointer.hpp     cutlass/include/cute/pointer_base.hpp     cutlass/include/cute/pointer_swizzle.hpp     cutlass/include/cute/swizzle.hpp     cutlass/include/cute/algorithm/tuple_algorithms.hpp     cutlass/include/cute/algorithm/functional.hpp     cutlass/include/cute/numeric/complex.hpp     cutlass/include/cutlass/complex.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuComplex.h     cutlass/include/cutlass/functional.h     cutlass/include/cutlass/real.h     cutlass/include/cutlass/fast_math.h     cutlass/include/cutlass/array.h     cutlass/include/cutlass/array_subbyte.h     cutlass/include/cutlass/coord.h     cutlass/include/cute/layout.hpp     cutlass/include/cute/underscore.hpp     cutlass/include/cute/int_tuple.hpp     cutlass/include/cute/stride.hpp     cutlass/include/cute/numeric/arithmetic_tuple.hpp     cutlass/include/cute/numeric/integral_ratio.hpp     cutlass/include/cute/swizzle_layout.hpp     cutlass/include/cute/layout_composed.hpp     cutlass/include/cute/pointer_flagged.hpp     cutlass/include/cute/arch/util.hpp     cutlass/include/cute/algorithm/tensor_algorithms.hpp     cutlass/include/cute/algorithm/fill.hpp     cutlass/include/cute/algorithm/prefer.hpp     cutlass/include/cute/algorithm/clear.hpp     cutlass/include/cute/algorithm/copy.hpp     cutlass/include/cute/tensor_predicate.hpp     cutlass/include/cute/atom/copy_atom.hpp     cutlass/include/cute/arch/copy.hpp     cutlass/include/cute/atom/copy_traits.hpp     cutlass/include/cute/atom/mma_atom.hpp     cutlass/include/cute/arch/mma.hpp     cutlass/include/cute/atom/mma_traits.hpp     cutlass/include/cute/atom/mma_traits_sm61.hpp     cutlass/include/cute/arch/mma_sm61.hpp     cutlass/include/cute/atom/mma_traits_sm70.hpp     cutlass/include/cute/arch/mma_sm70.hpp     cutlass/include/cute/atom/mma_traits_sm75.hpp     cutlass/include/cute/arch/mma_sm75.hpp     cutlass/include/cute/atom/mma_traits_sm80.hpp     cutlass/include/cute/arch/mma_sm80.hpp     cutlass/include/cute/atom/mma_traits_sm90.hpp     cutlass/include/cute/arch/mma_sm90.hpp     cutlass/include/cute/arch/mma_sm90_desc.hpp     cutlass/include/cute/arch/mma_sm90_gmma.hpp     cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp     cutlass/include/cute/atom/copy_traits_sm50.hpp     cutlass/include/cute/arch/copy_sm50.hpp     cutlass/include/cute/atom/copy_traits_sm75.hpp     cutlass/include/cute/arch/copy_sm75.hpp     cutlass/include/cute/atom/copy_traits_sm80.hpp     cutlass/include/cute/arch/copy_sm80.hpp     cutlass/include/cute/atom/copy_traits_sm90.hpp     cutlass/include/cute/arch/copy_sm90.hpp     cutlass/include/cute/arch/copy_sm90_desc.hpp     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cuda.h     cutlass/include/cute/container/bit_field.hpp     cutlass/include/cute/arch/copy_sm90_tma.hpp     cutlass/include/cute/algorithm/prefetch.hpp     cutlass/include/cute/algorithm/axpby.hpp     cutlass/include/cute/algorithm/gemm.hpp     cutlass/include/cute/algorithm/cooperative_copy.hpp     cutlass/include/cute/algorithm/cooperative_gemm.hpp     src/compare.cuh     src/cublas.cuh     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cublas_v2.h     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/cublas_api.h     src/gemm_simple.cuh     src/rng.cuh     /home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include/curand.h\
",
    values = {
        "/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/bin/nvcc",
        {
            "-Xcompiler",
            "-fPIE",
            "-O3",
            "-Icutlass/include",
            "-I/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-11.8.0-cd6aisckdu4ykiqz3wbmsdl64r6eneab/include",
            "--std",
            "c++17",
            "-arch=sm_80",
            "--expt-relaxed-constexpr",
            "-m64",
            "-rdc=true"
        }
    }
}