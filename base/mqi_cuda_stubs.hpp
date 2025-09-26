#ifndef MQI_CUDA_STUBS_HPP
#define MQI_CUDA_STUBS_HPP

// This header provides mock/stub implementations for CUDA API calls
// to allow compilation in an environment without the CUDA SDK.
// It should be included in C++ files that call CUDA API functions from host code.

#ifndef __CUDACC__

#include <iostream>

// Define mock types for CUDA constructs to prevent compile errors.
// These are placeholders and have no functional equivalence.
struct cudaError_t {};
struct cudaChannelFormatDesc {};
struct cudaArray {};
struct cudaResourceDesc {};
struct cudaTextureDesc {};
struct cudaTextureObject_t {};

// Mock CUDA API calls to allow compilation without the SDK
inline void check_cuda_last_error() { /* No-op */ }
namespace mqi {
    inline void check_cuda_last_error() { /* No-op */ }
}

inline cudaError_t cudaMalloc3DArray(...) { return {}; }
inline cudaError_t cudaMemcpy3D(...) { return {}; }
inline cudaError_t cudaCreateTextureObject(...) { return {}; }
inline cudaError_t cudaDestroyTextureObject(...) { return {}; }
inline cudaError_t cudaFreeArray(...) { return {}; }
inline cudaError_t cudaMemcpyToSymbol(...) { return {}; }
// Add any other necessary stubs for CUDA functions called from host code.

#endif // __CUDACC__

#endif // MQI_CUDA_STUBS_HPP