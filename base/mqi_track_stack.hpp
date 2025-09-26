
#ifndef MQI_TRACK_STACK_HPP
#define MQI_TRACK_STACK_HPP

/// \file
/// \brief Defines a fixed-size stack for managing secondary particle tracks.
///
/// \details In a Monte Carlo simulation, a primary particle can create numerous secondary
/// particles (e.g., delta electrons, secondary protons from nuclear interactions). To ensure
/// all particles are simulated, these secondaries are temporarily stored on a stack. The
/// transport engine typically simulates one particle to completion, and then processes any
/// secondaries it created by popping them from this stack. This continues until the stack is
/// empty. This file defines the simple LIFO (Last-In, First-Out) stack used for this purpose.

#include <moqui/base/mqi_node.hpp>
#include <moqui/base/mqi_track.hpp>

namespace mqi
{

/**
 * @class track_stack_t
 * @brief A simple, fixed-size stack for managing secondary particle tracks.
 * @details This class provides a Last-In, First-Out (LIFO) container for `track_t` objects.
 * When a particle interaction creates new secondary particles, they are pushed onto this stack.
 * After the primary particle's transport is complete, tracks are popped from the stack to be
 * transported until the stack is empty. This ensures all particle lineages are fully simulated.
 *
 * The stack uses a fixed-size C-style array for storage, which is simple and efficient,
 * especially for GPU execution where dynamic memory allocation is costly. The capacity is
 * intentionally larger in debug builds (`__PHYSICS_DEBUG__`) to facilitate more detailed tracking of secondaries.
 *
 * @tparam R The floating-point type for the track data (e.g., float, double).
 */
template<typename R>
class track_stack_t
{

public:
// Use preprocessor directives to change the stack size based on the build configuration.
#ifdef __PHYSICS_DEBUG__
    const uint16_t limit = 200;   ///< The maximum number of tracks the stack can hold in debug mode.
    track_t<R>     tracks[200]; ///< The underlying C-style array to store track objects in debug mode.
#else
    const uint16_t limit = 10;    ///< The maximum number of tracks the stack can hold in release mode.
    track_t<R>     tracks[10];  ///< The underlying C-style array to store track objects in release mode.
#endif
    /// The current number of tracks on the stack. It acts as a stack pointer, always pointing
    /// to the next available empty slot. An index of 0 means the stack is empty.
    uint16_t idx = 0;

    /**
     * @brief Default constructor.
     */
    CUDA_HOST_DEVICE
    track_stack_t() {
        ;
    }

    /**
     * @brief Destructor.
     */
    CUDA_HOST_DEVICE
    ~track_stack_t() {
        ;
    }

    /**
     * @brief Pushes a secondary track onto the top of the stack.
     * @details If the stack is already full (i.e., `idx >= limit`), the operation is silently ignored
     * to prevent buffer overflows. This is a potential source of "lost" particles if the stack
     * size is too small for a given physics interaction.
     * @param[in] trk The track to be added to the stack.
     */
    CUDA_HOST_DEVICE
    void
    push_secondary(const track_t<R>& trk) {

        if (idx < limit) {
            tracks[idx] = trk;
            ++idx;
        }
    }

    /**
     * @brief Checks if the stack is empty.
     * @return `true` if the stack contains no tracks, `false` otherwise.
     */
    CUDA_HOST_DEVICE
    bool
    is_empty(void) {
        return idx == 0;
    }

    /**
     * @brief Removes and returns the track from the top of the stack.
     * @details This performs the pop operation by first decrementing the stack pointer (`idx`)
     * and then returning a copy of the track at that new index.
     * @return A copy of the track that was at the top of the stack.
     */
    CUDA_HOST_DEVICE
    track_t<R>
    pop(void) {
        ///copy
        return tracks[--idx];
    }

    /**
     * @brief Provides direct array-like access to the tracks in the stack.
     * @param[in] i The index of the track to access.
     * @return A reference to the track at the specified index.
     */
    CUDA_HOST_DEVICE
    track_t<R>&
    operator[](uint16_t i) {
        return tracks[i];
    }
};

}   // namespace mqi
#endif
