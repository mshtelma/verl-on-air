// Compile-only probe: does nvcc compile what FlashInfer's Gated-DeltaNet JIT
// compiles? A kernel including ONLY cuda_runtime.h succeeds even with a skewed
// toolchain, because the version assert lives in CCCL. So pull CCCL in, and use
// code=sm_90a so ptxas actually runs (that is how the cicc/ptxas PTX-ISA
// mismatch surfaced: "Unsupported .version 9.2; current version is '9.0'").
//
// Kept as a real file rather than a printf: escaping \n through
// Dockerfile -> /bin/sh -> printf produced a one-line file whose #include had
// "extra tokens at end of directive".
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

__global__ void k(float* x) { x[threadIdx.x] = 1.0f; }

static_assert(cuda::std::is_same<float, float>::value, "CCCL headers usable");
