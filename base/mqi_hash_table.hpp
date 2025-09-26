/// \file mqi_hash_table.hpp
///
/// \brief Defines the basic data structures and utility functions for a simple hash table.
///
/// \details This file provides the building blocks for a simple, open-addressing hash table.
/// A hash table is a data structure that maps keys to values for highly efficient lookup.
/// "Open addressing" is a collision resolution strategy: if two keys hash to the same table index,
/// the algorithm probes for the next available empty slot.
///
/// This type of data structure is often used in high-performance computing to efficiently
/// store and retrieve sparse data. In this simulation, it is likely used on the GPU
/// to create a sparse "influence matrix" (Dij), which stores the dose deposited by each
/// individual beamlet into each voxel it hits. This is far more memory-efficient than
/// storing a dense matrix, as most beamlets only deposit dose in a small fraction of the voxels.
#ifndef MQI_HASH_TABLE_CLASS_HPP
#define MQI_HASH_TABLE_CLASS_HPP

#include <cstring>
#include <moqui/base/mqi_common.hpp>

namespace mqi
{

/// \struct key_value
/// \brief Represents a single key-value entry in the hash table.
///
/// \details In C++, a `struct` is similar to a `class` but its members are public by default.
/// It is often used for simple data aggregate types like this one.
///
/// This structure uses a composite key (made of two parts) to map to a single
/// floating-point value. This is useful for associating a value with a unique
/// combination of two identifiers. For example, in a sparse influence matrix (Dij),
/// the key might be `(beamlet_id, voxel_id)` and the value would be the dose.
struct key_value
{
    mqi::key_t key1;  ///< The first component of the composite key (e.g., beamlet ID).
    mqi::key_t key2;  ///< The second component of the composite key (e.g., voxel ID).
    double     value; ///< The value associated with the key pair (e.g., dose).
};

/// \brief Initializes a hash table on the host (CPU).
///
/// \details This function iterates through the entire hash table array and sets all entries
/// to a known "empty" state. The keys are set to `mqi::empty_pair` (a special value,
/// often the maximum possible integer, to signify an unused slot), and the value
/// is set to 0. This prepares the table for new data to be inserted.
///
/// \param[out] table A pointer to the host memory array of `key_value` structs to be initialized.
/// \param[in] max_capacity The total number of entries in the hash table.
void
init_table(key_value* table, uint32_t max_capacity) {
    for (int i = 0; i < max_capacity; i++) {
        table[i].key1  = mqi::empty_pair;
        table[i].key2  = mqi::empty_pair;
        table[i].value = 0;
    }
}

/// \brief Initializes a hash table on a CUDA device (GPU).
///
/// \details This is a CUDA kernel, a function designed to be launched from the host and executed
/// in parallel by many threads on the GPU. The `CUDA_GLOBAL` keyword indicates that
/// it is a kernel. Each thread in the launch grid will typically initialize a subset
/// of the table entries.
///
/// \tparam R The floating-point type (e.g., float or double), included for template consistency.
/// \param[out] table A pointer to the device memory where the `key_value` structs are stored.
/// \param[in] max_capacity The maximum capacity of the hash table.
/// \note This kernel only initializes the `value` field of each entry to 0. The keys
///       are expected to be managed separately. For optimal performance, the entire table
///       (including keys) is often initialized to a default state (like all zeros or all 0xFF)
///       using a single, highly optimized `cudaMemset` call from the host before this kernel is launched.
template<typename R>
CUDA_GLOBAL void
init_table_cuda(key_value* table, uint32_t max_capacity) {
    // This kernel is simple and likely not used in the final implementation,
    // as a cudaMemset would be much faster for zeroing the whole table.
    // It might be used if only the 'value' field needs to be reset between runs.
    for (int i = 0; i < max_capacity; i++) {
        table[i].value = 0;
    }
}

/// \brief A CUDA kernel for debugging, used to print a specific entry from the hash table.
///
/// \details Debugging on the GPU can be challenging because you cannot simply use `printf`
/// everywhere as you would on the CPU. A common technique is to write a special kernel like
/// this one that is launched with a single thread to print the value of a specific memory
/// location on the device, helping to verify its contents during a simulation.
///
/// \tparam R The floating-point type, for template consistency.
/// \param[in] data A pointer to the device memory of the hash table.
template<typename R>
CUDA_GLOBAL void
test_print(mqi::key_value* data) {
    // A hardcoded index for testing purposes. In a real scenario, you might pass the
    // index to check as a parameter to the kernel.
    uint32_t ind = 512 * 512 * 200 * 4 - 1;
    printf("data[0].key1 %d data[0].key2 %d data[0].value %f\n",
           data[ind].key1,
           data[ind].key2,
           data[ind].value);
}
}   // namespace mqi
#endif
