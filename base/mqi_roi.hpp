#ifndef MQI_ROI_HPP
#define MQI_ROI_HPP
#include <moqui/base/mqi_common.hpp>

namespace mqi
{
/**
 * @enum roi_mapping_t
 * @brief Defines the method for mapping a transport grid index to a Region of Interest (ROI) scoring index.
 * @details In simulations, we often want to calculate results (like dose) only in specific areas
 * (Regions of Interest, or ROIs) to save memory and computation time, especially on GPUs. This
 * enum defines the strategy used to map a voxel's index from the full simulation grid to its
 * corresponding index within the much smaller ROI data array.
 */
typedef enum {
    /// Direct 1-to-1 mapping. The scoring grid is the same as the transport grid.
    /// The ROI index is the same as the transport grid index.
    DIRECT = 0,
    /// Indirect mapping via a lookup table. `roi_index = lookup_table[transport_index]`.
    /// This is useful when the ROI is composed of sparse, disconnected voxels. The `start_`
    /// array acts as the `lookup_table`.
    INDIRECT = 1,
    /// The ROI is defined by a set of contours, which are compressed using run-length encoding.
    /// This is efficient for large, complex, but contiguous ROIs (like organs).
    /// The mapping is calculated using the `start_`, `stride_`, and `acc_stride_` arrays.
    CONTOUR = 2
} roi_mapping_t;

/**
 * @class roi_t
 * @brief Manages the mapping from a global transport grid to a sparse Region of Interest (ROI) for scoring.
 * @details
 * This class is a key optimization for Monte Carlo simulations. The "transport grid" can be very
 * large (e.g., a 512x512x200 CT scan), but we often only care about the dose in a smaller,
 * irregularly shaped "scoring grid" or ROI (e.g., a tumor). This class provides the logic
 * to quickly determine if a given voxel from the large grid is inside the ROI, and if so,
 * what its index is in the smaller, densely packed ROI data array.
 *
 * This is crucial for performance on GPUs, where memory is limited and memory access patterns
 * are critical. Instead of allocating a huge array for the full grid, we only allocate a smaller
 * one for the ROI and use this class to map to it.
 *
 * For a Python developer, think of this as managing a sparse matrix format. Instead of a giant
 * NumPy array full of zeros, you store only the non-zero values and their indices.
 *
 * The class does not own the mapping arrays (`start_`, `stride_`, etc.). It only holds
 * pointers to them, assuming they are allocated and managed elsewhere (e.g., on the GPU).
 */
class roi_t
{
public:
    roi_mapping_t method_;          ///< The mapping method to use (DIRECT, INDIRECT, or CONTOUR).
    uint32_t      original_length_; ///< The total number of voxels in the original, full transport grid (e.g., 512*512*200).
    uint32_t      length_;          ///< The number of elements in the mapping arrays (`start_`, `stride_`, `acc_stride_`). For INDIRECT, this is the size of the lookup table. For CONTOUR, it's the number of "runs".

    // GPU memory pointers. This class does not own the memory.
    uint32_t* start_;      ///< For INDIRECT: A lookup table where `start_[transport_idx] = roi_idx`. For CONTOUR: The transport index where each run of ROI voxels begins.
    uint32_t* stride_;     ///< For CONTOUR only: The length (number of consecutive voxels) of each run.
    uint32_t* acc_stride_; ///< For CONTOUR only: The accumulated length of all preceding runs. `acc_stride_[i]` gives the starting index in the ROI array for the i-th run.

public:
    /**
     * @brief Constructs an roi_t object.
     * @param[in] m The mapping method.
     * @param[in] n The total size of the original transport grid.
     * @param[in] l The length of the mapping arrays.
     * @param[in] s Pointer to the start/indirect-index array.
     * @param[in] t Pointer to the stride array (for CONTOUR mapping).
     * @param[in] a Pointer to the accumulated stride array (for CONTOUR mapping).
     */
    CUDA_HOST_DEVICE
    roi_t(roi_mapping_t m,
          uint32_t      n,
          int32_t       l = 0,
          uint32_t*     s = nullptr,
          uint32_t*     t = nullptr,
          uint32_t*     a = nullptr) :
        method_(m),
        original_length_(n), length_(l), start_(s), stride_(t), acc_stride_(a) {
        ;
    }

    /**
     * @brief Determines if a transport index `v` is inside the ROI and gets its mapped index.
     * @details This is a general-purpose function. For CONTOUR mapping, it only checks for inclusion (returns 1 or -1).
     * For more direct index retrieval, see `get_mask_idx`.
     * @param[in] v The transport grid index to check.
     * @return For CONTOUR, returns 1 if inside, -1 if outside. For other methods, returns the mapped ROI index or -1 if outside.
     */
    CUDA_HOST_DEVICE
    int32_t
    idx(const uint32_t& v) const {

        switch (method_) {
        case INDIRECT:
            return start_[v];   // Returns ROI index from lookup table, or -1 if not in ROI.
        case CONTOUR:
            return idx_contour(v);   // Returns 1 if in ROI, -1 if not.
        default:                     // DIRECT
            return v;                // Index is unchanged.
        }
    }

    /**
     * @brief Gets the final index within the flattened ROI data array for a given transport index.
     * @param[in] v The transport grid index.
     * @return The corresponding index in the ROI's own data array (from 0 to `get_mask_size()-1`), or -1 if outside the ROI.
     */
    CUDA_HOST_DEVICE
    int32_t
    get_mask_idx(const uint32_t& v) const {
        switch (method_) {
        case INDIRECT:
            return start_[v];
        case CONTOUR:
            return get_contour_idx(v);
        default:   // DIRECT
            return v;
        }
    }

    /**
     * @brief Gets the total number of voxels in the ROI.
     * @return The size of the ROI data array.
     */
    CUDA_HOST_DEVICE
    int32_t
    get_mask_size() const {
        switch (method_) {
        case INDIRECT:
            // For indirect mapping, `length_` is explicitly set to the ROI size.
            return length_;
        case CONTOUR:
            // The last element of the accumulated stride array holds the total number of voxels in all runs.
            return acc_stride_[length_ - 1];
        default:   // DIRECT
            // The ROI is the entire grid.
            return original_length_;
        }
    }

    /**
     * @brief Calculates the ROI mask index for a transport index using the CONTOUR (run-length-encoded) method.
     * @details
     * Example: Imagine an ROI consists of voxels 10-14 (5 voxels) and 25-26 (2 voxels).
     * The mapping arrays would be:
     * - `start_`:      [10, 25]
     * - `stride_`:     [5,  2]
     * - `acc_stride_`: [5,  7] (i.e., [5, 5+2])
     *
     * If we query for transport index `v = 13`:
     * 1. `lower_bound_cpp(13)` finds the index of the first element in `start_` that is > 13. This is index 1 (value 25).
     * 2. We check the preceding run by using `c = 1 - 1 = 0`.
     * 3. We find the distance from the start of this run: `distance = v - start_[c] = 13 - 10 = 3`.
     * 4. We check if this distance is within the run's length: `distance (3) < stride_[c] (5)`. It is.
     * 5. The base index for this run `c=0` is 0 (since `c < 1`). The final ROI index is `0 + distance = 3`.
     *
     * If we query for `v = 25`:
     * 1. `lower_bound_cpp(25)` finds the index of the first element > 25. Let's assume it returns 2 (end of array).
     * 2. We check the preceding run: `c = 2 - 1 = 1`.
     * 3. `distance = v - start_[c] = 25 - 25 = 0`.
     * 4. `distance (0) < stride_[c] (2)`. It is.
     * 5. The base index for run `c=1` is `acc_stride_[c-1] = acc_stride_[0] = 5`.
     * 6. The final ROI index is `5 + distance = 5`.
     *
     * @param[in] v The transport grid index.
     * @return The final ROI index if `v` is within a contour segment, otherwise -1.
     */
    CUDA_HOST_DEVICE
    int32_t
    get_contour_idx(const uint32_t& v) const {
        // Find the first run that STARTS AFTER v.
        int32_t c = this->lower_bound_cpp(v) - 1;
        // Check if v falls into the run that PRECEDES it.
        uint32_t distance = v - start_[c];
        if (distance < stride_[c]) {
            // It's inside the run. Calculate the final index.
            // The base index is the accumulated stride of all previous runs.
            if (c >= 1) distance += acc_stride_[c - 1];
            return distance;
        }
        return -1;   // Not in any run, so invalid.
    }

    /**
     * @brief Checks if a transport index is within any contour segment.
     * @param v The transport grid index.
     * @return 1 if `v` is inside a contour segment, otherwise -1.
     */
    CUDA_HOST_DEVICE
    int32_t
    idx_contour(const uint32_t& v) const {
        int32_t  c        = this->lower_bound_cpp(v) - 1;
        uint32_t distance = v - start_[c];
        if (distance < stride_[c]) {
            /// is in stride
            return 1;
        }
        return -1;   //invalid
    }

    /**
     * @brief Gets the mapped index for the INDIRECT method.
     * @param v The transport grid index.
     * @return The mapped ROI index from the lookup table.
     */
    CUDA_HOST_DEVICE
    int32_t
    idx_indirect(const uint32_t& v) const {
        return start_[v];
    }

    /**
     * @brief A custom binary search implementation to find the lower bound of a value in the `start_` array.
     * @details
     * This is a key performance component for the CONTOUR method, as it efficiently finds the
     * run-length-encoded segment that might contain the transport index `v`.
     * `std::lower_bound` is part of the C++ standard library, but it may not be available or may have
     * performance implications when compiling for a GPU (`__CUDA_ARCH__`). This custom implementation
     * ensures the code works correctly and efficiently on both CPU and GPU.
     * @param[in] value The transport index to search for.
     * @return The index of the first element in `start_` that is greater than `value`.
     */
    CUDA_HOST_DEVICE
    int32_t
    lower_bound_cpp(const int32_t& value) const {
        int32_t first = 0;
        int32_t count = length_;
        int32_t step;

        int32_t it;
        while (count > 0) {

            it   = first;
            step = count / 2;
            it += step;

            // If the middle element is less than or equal to the target value,
            // then the lower bound must be in the second half of the range.
            if (start_[it] <= value) {
                first = ++it;
                count -= step + 1;
            } else
                // Otherwise, the lower bound is in the first half.
                count = step;
        }
        return first;
    }
};

}   // namespace mqi

#endif
