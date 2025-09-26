/**
 * @file
 * @brief Defines structures and classes for managing physics processes and parameters.
 */
#ifndef MQI_PHYSICS_LIST_HPP
#define MQI_PHYSICS_LIST_HPP

#include <moqui/base/mqi_interaction.hpp>

namespace mqi
{

/**
 * @struct dL_t
 * @brief A struct to hold a proposed step length and the ID of the physics process that generated it.
 * @tparam R The floating-point type for the step length.
 */
template<typename R>
struct dL_t {
    R      L;   ///< The proposed step length.
    int8_t P;   ///< The ID of the physics process. (-1 for geometry boundary). P-> 0: CSDA, 1: delta_ion, 2: pp-elastic, 3: po-elastic, 4: po-inelastic
};

/**
 * @class physics_list
 * @brief A class to manage physics parameters and settings for the simulation.
 * @tparam R The floating-point type for energy cuts and step limits.
 * @details This class holds various energy cutoffs and step size limits that control the behavior of physics interactions.
 */
template<typename R>
class physics_list
{
public:
    const physics_constants<R> units;               ///< A struct holding physical constants and conversion factors.
    R       Te_cut   = 0.08511 * units.MeV;   ///< Kinetic energy cut for secondary electrons.
    const R Tp_cut   = 0.5 * units.MeV;         ///< Kinetic energy cut for secondary protons.
    const R Tp_max   = 330.0 * units.MeV;       ///< Maximum proton energy to be handled by the simulation.
    const R Tp_up    = 2.0 * units.MeV;         ///< Upper limit for applying restricted stopping power.
    const R max_step = 1.0 * units.cm;          ///< The absolute maximum allowed step length.

public:
    /**
     * @brief Default constructor.
     */
    CUDA_HOST_DEVICE
    physics_list() {
        ;
    }

    /**
     * @brief Destructor.
     */
    CUDA_HOST_DEVICE
    ~physics_list() {}
};

}   // namespace mqi
#endif
