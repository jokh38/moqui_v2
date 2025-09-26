#ifndef MQI_REL_QUANTITIES_HPP
#define MQI_REL_QUANTITIES_HPP

/// \file
/// \brief Defines a helper class for calculating and storing common relativistic kinematic quantities.

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>

namespace mqi
{

/**
 * @class relativistic_quantities
 * @brief A struct to calculate and store essential relativistic quantities for a particle.
 * @details
 * In particle transport simulations, many physics formulas depend on the same set of kinematic
 * quantities derived from a particle's energy. This class takes a particle's kinetic energy
 * and rest mass and pre-calculates these values once. This is an optimization that avoids
 * redundant computations in various physics models that are called repeatedly during the simulation.
 *
 * The calculations are based on the principles of special relativity, which are necessary
 * for particles traveling at a significant fraction of the speed of light.
 *
 * @tparam R The floating-point type for calculations (e.g., `float` or `double`).
 */
template<typename R>
class relativistic_quantities
{
public:
    R beta_sq;    ///< The square of the particle's velocity relative to the speed of light (v^2/c^2). Beta is the speed of the particle as a fraction of the speed of light.
    R gamma_sq;   ///< The square of the Lorentz factor.
    R gamma;      ///< The Lorentz factor, gamma = 1 / sqrt(1 - beta^2). It quantifies the time dilation and length contraction effects of special relativity.
    R Et;         ///< The total energy of the particle (kinetic energy + rest mass energy), E_total = Ek + m0*c^2.
    R Et_sq;      ///< The square of the total energy.
    R Ek;         ///< The kinetic energy of the particle in MeV.
    R mc2;        ///< The rest mass energy of the particle in MeV (m0*c^2).
    R tau;        ///< The ratio of kinetic energy to rest mass energy (Ek / m0*c^2), a common dimensionless parameter.
    R Te_max;     ///< The maximum kinetic energy in MeV that can be transferred to a stationary electron in a single collision. This is a key kinematic limit for collision physics.

    /**
     * @brief Constructs and computes the relativistic quantities.
     * @details Initializes all member variables based on the provided kinetic energy and rest mass.
     *
     * @warning This constructor has a critical limitation: it uses a hardcoded value for the
     *          proton rest mass (`Mp`) to calculate total energy, gamma, and tau, regardless of the
     *          `rest_mass_MeV` value passed to it. The `mc2` member is initialized with the argument
     *          but is not used in most calculations. This means the class currently only produces
     *          correct results for protons. This should be refactored if other particle types are needed.
     *
     * @param[in] kinetic_energy The kinetic energy of the particle in MeV.
     * @param[in] rest_mass_MeV The rest mass of the particle in MeV.
     */
    CUDA_HOST_DEVICE
    relativistic_quantities(R kinetic_energy, R rest_mass_MeV) :
        Ek(kinetic_energy), mc2(rest_mass_MeV) {
        // Hardcoded rest mass values in MeV.
        const R Mp = 938.272046;    // Proton rest mass energy
        const R Me = 0.510998928;   // Electron rest mass energy

        // NOTE: The following calculations are based on the hardcoded PROTON mass (Mp),
        // not the `mc2` member variable passed to the constructor.
        Et       = Ek + Mp;       // E_total = E_kinetic + m_proton*c^2
        Et_sq    = Et * Et;
        gamma    = Et / Mp;       // gamma = E_total / m0*c^2
        gamma_sq = gamma * gamma;
        beta_sq = 1.0 - 1.0 / gamma_sq;   // From gamma = 1/sqrt(1-beta^2)
        tau     = Ek / Mp;                // Kinetic energy in units of rest mass

        // Calculation of maximum energy transfer to a free electron
        const R MeMp    = Me / Mp;       // Ratio of electron to proton mass
        const R MeMp_sq = MeMp * MeMp;   // Square of the ratio
        Te_max          = (2.0 * Me * beta_sq * gamma_sq);
        Te_max /= (1.0 + 2.0 * gamma * MeMp + MeMp_sq);
    }

    /**
     * @brief Destructor.
     */
    CUDA_HOST_DEVICE
    ~relativistic_quantities() {
        ;
    }

    /**
     * @brief Calculates the relativistic momentum of the particle.
     * @return The momentum of the particle in units of MeV/c.
     * @details Calculated from the energy-momentum relation: E^2 = (p*c)^2 + (m0*c^2)^2.
     * Note that it uses the `mc2` member variable, which might be inconsistent with other calculations
     * in the constructor if the particle is not a proton.
     */
    CUDA_HOST_DEVICE
    R
    momentum() {
        return mqi::mqi_sqrt(Et * Et - mc2 * mc2);
    }
};

}   // namespace mqi
#endif
