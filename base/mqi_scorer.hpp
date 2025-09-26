#ifndef MQI_SCORER_HPP
#define MQI_SCORER_HPP

#include <mutex>

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_roi.hpp>

namespace mqi
{
/**
 * @enum scorer_t
 * @brief Defines the types of physical quantities that can be scored in the simulation.
 */
typedef enum
{
    VIRTUAL           = 0, ///< A virtual or placeholder scorer type.
    ENERGY_DEPOSITION = 1, ///< Scores the total energy deposited in a voxel.
    DOSE              = 2, ///< Scores the absorbed dose (energy deposited per unit mass).
    DOSE_Dij          = 3, ///< Scores a dose-influence matrix (Dij), often used for optimization.
    LETd              = 4, ///< Scores the dose-weighted Linear Energy Transfer (LET).
    LETt              = 5, ///< Scores the track-weighted Linear Energy Transfer (LET).
    TRACK_LENGTH      = 6  ///< Scores the total track length of particles within a voxel.
} scorer_t;

// Forward declarations
template<typename R>
class track_t;

template<typename T, typename R>
class grid3d;

/**
 * @typedef fp_compute_hit
 * @brief A function pointer type for a callback that calculates the physical quantity for a given particle interaction.
 * @tparam R The floating-point type (e.g., float, double).
 * @param track_t The particle track that produced the hit.
 * @param cnb_t The index of the voxel where the hit occurred.
 * @param grid3d The geometry grid containing material/density information.
 * @return The calculated physical quantity (e.g., deposited energy) as a double.
 */
template<typename R>
using fp_compute_hit = double (*)(const track_t<R>&, const mqi::cnb_t&, grid3d<mqi::density_t, R>&);

/**
 * @class scorer
 * @brief A generic class for accumulating physical quantities (e.g., dose) in a simulation.
 * @details This class uses a hash table to store scored data, which is memory-efficient for sparse scoring regions.
 * It is designed to be thread-safe for both CPU and GPU execution. The specific quantity to be scored is determined
 * by a callback function, making the scorer versatile.
 * @tparam R The floating-point type for simulation data (e.g., float, double).
 */
template<typename R>
class scorer
{

public:
    const char* name_; ///< The name of the scorer (e.g., "Dose_to_water").

    const fp_compute_hit<R> compute_hit_; ///< A function pointer to the callback that calculates the scored quantity per hit.

    mqi::key_value* data_             = nullptr; ///< Pointer to the hash table memory where scored values are accumulated.
    uint32_t        max_capacity_     = 0;       ///< The maximum capacity of the hash table.
    uint32_t        current_capacity_ = 0;       ///< The current number of entries in the hash table.

    scorer_t type_; ///< The type of quantity being scored.

    roi_t* roi_; ///< A pointer to a Region of Interest object that filters which voxels are scored.

    // Members for variance calculation
    bool            score_variance_ = false;   ///< A flag to enable or disable variance calculation.
    mqi::key_value* count_          = nullptr; ///< A hash table to store the number of hits per voxel.
    mqi::key_value* mean_           = nullptr; ///< A hash table to store the running mean of the scored quantity.
    mqi::key_value* variance_       = nullptr; ///< A hash table to store the running variance using Welford's algorithm.

#if !defined(__CUDACC__)
    std::mutex mtx; ///< A mutex to ensure thread safety during CPU execution.
#endif

    /**
     * @brief Constructs a scorer object.
     * @param name The name of the scorer.
     * @param max_capacity The maximum size of the hash table.
     * @param func_pointer A pointer to the function that calculates the value to be scored per hit.
     */
    CUDA_HOST_DEVICE
    scorer(const char* name, const uint32_t max_capacity, const fp_compute_hit<R> func_pointer) :
        name_(name), max_capacity_(max_capacity), current_capacity_(max_capacity),
        compute_hit_(func_pointer) {
        this->delete_data_if_used();
    }

    /**
     * @brief Destructor. Frees memory allocated for the data arrays.
     */
    CUDA_HOST_DEVICE
    ~scorer() {
        this->delete_data_if_used();
    }

    /**
     * @brief Frees the memory for all data, count, mean, and variance arrays if they have been allocated.
     */
    CUDA_HOST_DEVICE
    void
    delete_data_if_used(void) {
        if (data_ != nullptr) delete[] data_;
        if (count_ != nullptr) delete[] count_;
        if (mean_ != nullptr) delete[] mean_;
        if (variance_ != nullptr) delete[] variance_;
    }

    /**
     * @brief A hash function to map a key to an index in the hash table.
     * @param k The key to be hashed.
     * @return The hash table index.
     */
    CUDA_DEVICE
    unsigned long long int
    hash_fun(unsigned long long int k) {
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k % (this->max_capacity_ - 1);
    }

    /**
     * @brief A host-side implementation of the atomic Compare-And-Swap (CAS) operation.
     * @param address The memory address to operate on.
     * @param compare The value to compare with the value at `address`.
     * @param val The new value to set if the comparison is successful.
     * @return The original value at `address` before the operation.
     */
    CUDA_HOST_DEVICE
    uint32_t
    CAS(uint32_t* address, uint32_t compare, uint32_t val) {
        uint32_t old = *address;
        if (old == compare) {
            *address = val;
        } else {
        }
        return old;
    }

    /**
     * @brief Atomically inserts or adds a value to the hash table.
     * @details This function uses atomic operations and linear probing to handle hash collisions
     * and ensure that updates from multiple threads are correctly accumulated.
     * @param key1 The primary key (e.g., voxel index).
     * @param key2 An optional secondary key (e.g., beamlet index for Dij matrices).
     * @param value The value to add to the corresponding key's entry.
     * @param scorer_offset An offset used in hashing when a secondary key is present.
     */
    CUDA_DEVICE
    void
    insert_pair(mqi::key_t key1, mqi::key_t key2, R value, unsigned long long int scorer_offset) {
        mqi::key_t slot;
        if (key2 == mqi::empty_pair) {
            slot = key1;
            key2 = 0;
        } else {
            slot = hash_fun(key1 + (key2 * scorer_offset));
        }

        uint32_t prev1, prev2;
        while (true) {
#if defined(__CUDACC__)
            prev1 = atomicCAS(&this->data_[slot].key1, mqi::empty_pair, key1);
            prev2 = atomicCAS(&this->data_[slot].key2, mqi::empty_pair, key2);
#else
            prev1 = CAS(&this->data_[slot].key1, mqi::empty_pair, key1);
            prev2 = CAS(&this->data_[slot].key2, mqi::empty_pair, key2);
#endif
            if ((prev1 == mqi::empty_pair || prev1 == key1) &&
                (prev2 == mqi::empty_pair || prev2 == key2)) {
#if defined(__CUDACC__)
                atomicAdd(&this->data_[slot].value, value);
#else
                this->data_[slot].value += value;
#endif
                return;
            }
            slot = (slot + 1) % (this->max_capacity_ - 1);
        }
    }

    /**
     * @brief Processes a single particle interaction (a "hit").
     * @details This is the main scoring function called during the simulation. It checks if the hit
     * is within the ROI, calls the `compute_hit_` function to get the physical quantity,
     * and then accumulates the result in the hash table. It also updates variance statistics if enabled.
     * @param trk The particle track that had the interaction.
     * @param cnb The index of the voxel where the interaction occurred.
     * @param geo The geometry grid, used to get material properties like density.
     * @param offset An optional secondary key for scoring (e.g., beamlet index).
     * @param scorer_offset An offset used in hashing when a secondary key is present.
     */
    CUDA_DEVICE
    virtual void
    process_hit(const track_t<R>&          trk,
                const int32_t&             cnb,
                grid3d<mqi::density_t, R>& geo,
                const uint32_t&            offset,
                unsigned long long int     scorer_offset = 0) {
        // Calculate index to store hit
        // idx : -1 => a hit occured out of ROI. nothing to do.
        int32_t idx = roi_->idx(cnb);
        if (idx == -1) return;

        ///< calculate quantity
        R quantity = (*this->compute_hit_)(trk, cnb, geo);

        ///< store quantity and variance if it is set.
#if defined(__CUDACC__)
        insert_pair(cnb, offset, quantity, scorer_offset);

        if (this->score_variance_) {
            atomicAdd(&count_[cnb].value, 1.0);
            R delta = quantity - mean_[cnb].value;
            atomicAdd(&mean_[cnb].value, delta / count_[cnb].value);
            atomicAdd(&variance_[cnb].value, delta * (quantity - mean_[cnb].value));
        }
#else
        mtx.lock();
        insert_pair(cnb, offset, quantity, scorer_offset);
        data_[idx].value += quantity;
        if (this->score_variance_) {
            count_[cnb].value += 1.0;
            R delta = quantity - mean_[cnb].value;
            mean_[cnb].value += delta / count_[cnb].value;
            variance_[cnb].value += delta * (quantity - mean_[cnb].value);
        }

        mtx.unlock();
#endif
    }

    /**
     * @brief Clears all scored data.
     * @details Resets the data, count, mean, and variance arrays to an empty state (0xffffffff).
     * This is typically done before starting a new simulation run.
     */
    CUDA_HOST
    void
    clear_data() {
        std::memset(data_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        if (this->score_variance_) {
            std::memset(count_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
            std::memset(mean_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
            std::memset(variance_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        }
    }
};

}   // namespace mqi

#endif
