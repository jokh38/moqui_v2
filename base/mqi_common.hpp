#ifndef MQI_COMMON_HPP
#define MQI_COMMON_HPP

/// \file mqi_common.hpp
///
/// \brief A header file containing common definitions, macros, and type aliases for the project.
///
/// \details This file is intended to be included in most other files in the project. It serves
/// as a central place to define fundamental types, constants, and macros. This improves
/// code readability and maintainability. For example, by using a type alias like `phsp_t`,
/// the code becomes more self-documenting, and changing the underlying numerical precision
/// (e.g., from `float` to `double`) only requires a change in this one file.
///
/// It also includes CUDA-related headers and defines macros to handle code compilation
/// for both CPU (`g++`) and GPU (`nvcc`), ensuring cross-compiler compatibility.

// This block is compiled only by the NVIDIA CUDA Compiler (nvcc).
#if defined(__CUDACC__)

#include <cublas.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nvfunctional>
#include <stdio.h>

/// \def CUDA_HOST_DEVICE
/// \brief A macro that expands to `__host__ __device__` when compiling with NVCC.
/// This allows a function to be compiled for and executed on both the CPU (host) and the GPU (device).
#define CUDA_HOST_DEVICE __host__ __device__
/// \def CUDA_HOST
/// \brief A macro that expands to `__host__` when compiling with NVCC, marking a function for CPU execution only.
#define CUDA_HOST __host__
/// \def CUDA_DEVICE
/// \brief A macro that expands to `__device__` when compiling with NVCC, marking a function for GPU execution only.
#define CUDA_DEVICE __device__
/// \def CUDA_GLOBAL
/// \brief A macro that expands to `__global__`, marking a function as a CUDA kernel that can be launched from the host.
#define CUDA_GLOBAL __global__
/// \def CUDA_LOCAL
/// \brief A macro that expands to `__local__`, a CUDA-specific memory space qualifier (though less common).
#define CUDA_LOCAL __local__
/// \def CUDA_SHARED
/// \brief A macro that expands to `__shared__`, marking a variable to be placed in the fast, on-chip shared memory of a GPU block.
#define CUDA_SHARED __shared__
/// \def CUDA_CONSTANT
/// \brief A macro that expands to `__constant__`, marking a variable to be placed in the cached, read-only constant memory of the GPU.
#define CUDA_CONSTANT __constant__

#else
// This block is compiled by a standard C++ compiler. The CUDA keywords are not recognized,
// so the macros are defined as empty, effectively removing them from the code.
#define CUDA_HOST_DEVICE
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL
#define CUDA_LOCAL
#define CUDA_SHARED
#define CUDA_CONSTANT
#endif

#include <cmath>
#include <cstdint>
#include <limits>

namespace mqi
{
/// \typedef phsp_t
/// \brief Type alias for phase-space variables (e.g., position, energy, direction). Using a type alias
/// makes the code more readable and allows for easy changes to numerical precision later.
typedef float phsp_t;

/// \typedef cnb_t
/// \brief Type alias for cumulative number types (e.g., number of histories), which can become large.
typedef uint64_t cnb_t;
/// \typedef ijk_t
/// \brief Type alias for grid indices (i, j, k).
typedef int32_t ijk_t;

#if defined(__CUDACC__)
/// \typedef density_t
/// \brief Type alias for density values.
/// \note This is defined separately to allow for future optimization on the GPU using 16-bit floats (`__half`),
/// which would reduce memory usage and potentially increase performance on compatible hardware.
typedef float density_t;
#else
typedef float density_t;
#endif

/// The maximum number of blocks allowed in a single dimension of a CUDA grid (historically 65535, 2^16-1).
const uint16_t block_limit = 65535;
/// The maximum number of threads allowed in a CUDA block, often limited by hardware architecture.
const uint16_t thread_limit = 512;

/// \typedef key_t
/// \brief Type alias for keys used in hash tables or maps.
typedef uint32_t key_t;
/// A constant representing an empty or invalid key in a hash table.
/// `0xffffffff` is the maximum value for a 32-bit unsigned integer, making it a good sentinel value
/// as it's unlikely to be used as a valid key.
const key_t empty_pair = 0xffffffff;

/// \enum cell_side
/// \brief Enumerates the six possible faces of a voxel that a particle can intersect and cross.
/// Used in the geometry transport algorithm to identify which boundary was hit.
typedef enum {
    XM = 0,   ///< The face on the minimum-x side ("x-minus").
    XP = 1,   ///< The face on the maximum-x side ("x-plus").
    YM = 2,   ///< The face on the minimum-y side.
    YP = 3,   ///< The face on the maximum-y side.
    ZM = 4,   ///< The face on the minimum-z side.
    ZP = 5,   ///< The face on the maximum-z side.
    NONE_XYZ_PLANE = 6   ///< Represents no intersection or an invalid side.
} cell_side;

/// \enum aperture_type_t
/// \brief Enumerates the types of apertures used to shape the beam.
typedef enum {
    MASK   = 0,   ///< A 2D mask-type aperture.
    VOLUME = 1    ///< A 3D volumetric aperture.
} aperture_type_t;

/// \enum sim_type_t
/// \brief Enumerates the different simulation modes, which can affect how data is scored and aggregated.
typedef enum {
    PER_BEAM    = 0,   ///< Simulation is run and results are aggregated on a per-beam basis.
    PER_SPOT = 1,   ///< Simulation is run and results are aggregated on a per-spot basis (for pencil beam scanning).
    PER_PATIENT = 2    ///< Simulation is run for the entire patient plan at once.
} sim_type_t;

/// \enum transport_type
/// \brief Enumerates different particle transport conditions, particularly in relation to apertures.
typedef enum {
    APERTURE_CLOSE = 1,   ///< Particle is outside the aperture opening and should be stopped.
    APERTURE_OPEN  = 2,   ///< Particle is inside the aperture opening and can continue.
    NORMAL_PHYSICS = 3    ///< Standard physics transport, not currently interacting with an aperture.
} transport_type;

/// A global constant for the maximum step size (in mm) allowed in the simulation.
const float max_step_global = 1.0;

}   // namespace mqi

#endif
