#ifndef MQI_THREAD_HPP
#define MQI_THREAD_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>

namespace mqi
{

/**
 * @struct thrd_t
 * @brief A structure to hold thread-local data for a single execution thread in the simulation.
 * @details Each thread in the simulation requires its own state to work independently. This struct
 * primarily holds a dedicated random number generator to avoid contention and ensure statistically
 * independent random sequences across threads.
 */
struct thrd_t {
    uint32_t histories[2];  ///< The range of simulation histories [from, to] assigned to this thread.
    mqi_rng  rnd_generator; ///< The thread-local random number generator object.
};

/**
 * @brief Initializes an array of thread-local data structures, primarily for seeding random number generators.
 * @details This function is designed to be run as a CUDA kernel on the GPU or as a standard function on the CPU.
 * It iterates through each thread structure and initializes its random number generator with a unique seed
 * derived from a master seed and the thread's unique ID. This is a critical step to ensure proper
 * parallel execution of the Monte Carlo simulation.
 * @param thrds A pointer to the array of `thrd_t` structures to be initialized.
 * @param n_threads The total number of threads (and `thrd_t` structures) in the array. This is primarily used for the CPU implementation.
 * @param master_seed The main seed for the random number generators.
 * @param offset An offset used in the seeding process, particularly for CUDA's `curand_init`.
 */
CUDA_GLOBAL
void
initialize_threads(mqi::thrd_t*   thrds,
                   const uint32_t n_threads,
                   unsigned long  master_seed = 0,
                   unsigned long  offset      = 0) {
#if defined(__CUDACC__)
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(master_seed + blockIdx.x, threadIdx.x, offset, &thrds[thread_id].rnd_generator);
#else
    for (uint32_t i = 0; i < n_threads; ++i) {
        std::seed_seq seed{ master_seed + i };
        thrds[i].rnd_generator.seed(master_seed);
    }
#endif
}

}   // namespace mqi

#endif
