/**
 * @file
 * @brief Defines the node structure for the geometry hierarchy (scene graph).
 * @details This file provides the `node_t` struct, which is the fundamental building block
 * for creating a tree-like hierarchy of geometric objects. This "scene graph" structure
 * allows complex geometries (like a treatment machine) to be built from smaller, nested
 * components, each with its own coordinate system and properties.
 */
#ifndef MQI_NODE_HPP
#define MQI_NODE_HPP

#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_scorer.hpp>

// This is a preprocessor directive. The code inside this block will only be compiled
// if the `__CUDACC__` macro is defined, which is true only when using the
// NVIDIA CUDA Compiler (nvcc). This is used to include CUDA-specific headers.
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace mqi
{

/**
 * @struct node_t
 * @brief Represents a node in a hierarchical geometry, containing its own geometry, scorers, and child nodes.
 * @tparam R The floating-point type for scoring and geometry values (e.g., float or double).
 * @details
 * This structure is the core component of the simulation's scene graph. Each `node_t` acts as
 * a container in a tree. It can have its own geometry definition (e.g., a block of material
 * defined by a `grid3d`) and can also have children nodes, which are positioned and oriented
 * relative to it.
 *
 * For example, a `snout` node might contain a `rangeshifter` node and a `block` node as its
 * children. A particle being transported through the `snout` must also be tested for
 * intersections with the `rangeshifter` and `block` in their own local coordinate systems.
 *
 * This struct is designed to be used on both the CPU and GPU. It consists almost entirely of
 * pointers, meaning a `node_t` object itself is small and can be easily passed around, while
 * the large data arrays (for geometry, scorers, etc.) reside elsewhere in memory (e.g., in GPU global memory).
 */
template<typename R>
struct node_t
{
    /// A pointer to the node's own geometry, defined as a 3D grid. Can be `nullptr` if the node is just a container for transforms.
    grid3d<mqi::density_t, R>* geo = nullptr;

    uint16_t    n_scorers = 0;   ///< The number of scorers attached to this node.
    scorer<R>** scorers   = nullptr;   ///< An array of pointers to scorer objects, which define what quantities to score.

    // The following are arrays of pointers to the actual data arrays for each scorer.
    // This allows a single node to have multiple scorers, each with its own data buffers.
    // The double pointer (e.g., `key_value**`) represents a C-style array of pointers.

    /// An array of pointers to the primary data for each scorer (e.g., dose sum or dose-squared sum).
    mqi::key_value** scorers_data = nullptr;
    /// An array of pointers to the count data for each scorer (e.g., number of particles hitting each voxel).
    mqi::key_value** scorers_count = nullptr;
    /// An array of pointers to the mean value data for each scorer (for running statistics).
    mqi::key_value** scorers_mean = nullptr;
    /// An array of pointers to the variance data for each scorer (for running statistics).
    mqi::key_value** scorers_variance = nullptr;

    uint16_t           n_children = 0;      ///< The number of child nodes nested inside this one.
    struct node_t<R>** children   = nullptr;   ///< An array of pointers to the child nodes.
};

}   // namespace mqi
#endif
