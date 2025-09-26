/// \file mqi_interaction.hpp
///
/// \brief Defines the abstract base class for all physics interaction models.
///
/// This file defines `mqi::interaction`, a "pure virtual" class that serves as a
/// template or "interface" for all specific physics processes in the simulation.
/// By defining a common set of functions that every physics process must have, it
/// allows the simulation's physics list to handle different types of interactions
/// (like ionization, elastic scattering, etc.) in a generic and polymorphic way.
#ifndef MQI_INTERACTION_HPP
#define MQI_INTERACTION_HPP
#include <random>

#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include <moqui/base/mqi_relativistic_quantities.hpp>
#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_track_stack.hpp>

namespace mqi
{

/// \class interaction
/// \brief A pure virtual class representing the interaction between a particle and a material.
///
/// In C++, a class containing at least one "pure virtual" function (one marked with `= 0`)
/// is an abstract base class. It cannot be instantiated directly. Instead, other classes
/// must inherit from it and provide implementations for all pure virtual functions.
/// This is a fundamental concept in object-oriented programming for creating a common
/// "contract" or interface that different components must adhere to.
///
/// \tparam R The floating-point type (e.g., `float` or `double`) for calculations.
/// \tparam P The particle type (e.g., `mqi::electron`, `mqi::proton`).
template<typename R, mqi::particle_t P>
class interaction
{
public:
    ///< A reference to the singleton object containing physical constants.
    const physics_constants<R> units;
#ifdef __PHYSICS_DEBUG__
    ///< Kinetic energy cutoff for creating secondary particles (delta rays).
    ///< If a secondary particle is created with less energy than this, its energy is deposited locally.
    R T_cut = 0.08511 * units.MeV;
#else
    R T_cut = 0.0815 * units.MeV;
#endif
    ///< Maximum step size used for generating dE/dx and cross-section data tables.
    const R max_step = 0.01 * units.cm;
    ///< Kinetic energy cutoff for transporting primary particles (protons).
    ///< If a proton's energy falls below this, it is stopped and its energy is deposited.
    R Tp_cut = 0.5 * units.MeV;
    ///< A constant vector representing the z-axis, used for scattering calculations.
    const mqi::vec3<R> dir_z;

public:
    /// \brief Default constructor.
    CUDA_HOST_DEVICE
    interaction() : dir_z(0, 0, -1) {
        ;
    }

    /// \brief Virtual destructor to ensure proper cleanup in derived classes.
    CUDA_HOST_DEVICE
    virtual ~interaction() {
        ;
    }

    /// \brief Samples a step length based on the interaction's cross-section.
    ///
    /// The distance a particle travels before an interaction is probabilistic and can be
    /// described by an exponential distribution. This function samples from that
    /// distribution using the mean free path (MFP), which is the inverse of the
    /// macroscopic cross-section (cs).
    ///
    /// \param[in] rel Relativistic quantities of the particle.
    /// \param[in] mat The material through which the particle is traveling.
    /// \param[in,out] rng A pointer to the random number generator.
    /// \return The sampled step length in cm.
    CUDA_HOST_DEVICE
    virtual R
    sample_step_length(const relativistic_quantities<R>& rel,
                       const material_t<R>&              mat,
                       mqi_rng*                          rng) {
        R cs   = mat.rho_mass * this->cross_section(rel, mat);
        R mfp  = (cs == 0.0) ? max_step : 1.0 / cs;
        R prob = mqi_uniform<R>(rng);
        return -1.0 * mfp * mqi_ln(prob);
    }

    /// \brief Samples a step length given a pre-calculated cross-section.
    ///
    /// This is an overloaded version of `sample_step_length` for cases where the
    /// total cross-section of multiple processes is already known.
    ///
    /// \param[in] cs The total macroscopic cross-section.
    /// \param[in,out] rng A pointer to the random number generator.
    /// \return The sampled step length in cm.
    CUDA_HOST_DEVICE
    virtual R
    sample_step_length(const R cs, mqi_rng* rng) {
        R mfp  = (cs == 0.0) ? max_step : 1.0 / cs;
        R prob = mqi_uniform<R>(rng);
        return -1.0 * mfp * mqi_ln(prob);
    }

    /// \brief Pure virtual function to calculate the microscopic cross-section for the interaction.
    ///
    /// This function must be implemented by all derived classes. It calculates the
    /// probability of this specific interaction occurring per unit path length.
    /// The `= 0` syntax makes this a pure virtual function.
    ///
    /// \param[in] rel Relativistic quantities of the particle.
    /// \param[in] mat The material.
    /// \return The microscopic cross-section in cm^2.
    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, const material_t<R>& mat) = 0;

    /// \brief Pure virtual function to apply continuous effects during a particle's step.
    ///
    /// This function must be implemented by derived classes. It models processes that
    /// occur continuously along the particle's path, such as energy loss due to ionization.
    ///
    /// \param[in,out] trk The particle track to be updated.
    /// \param[in,out] stk The stack for any newly created secondary particles.
    /// \param[in,out] rng A pointer to the random number generator.
    /// \param[in] len The length of the step.
    /// \param[in,out] mat The material properties.
    CUDA_HOST_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>&    mat) = 0;

    /// \brief Pure virtual function to apply discrete effects at the end of a particle's step.
    ///
    /// This function must be implemented by derived classes. It models discrete events
    /// that occur at a single point, such as a scattering event that changes the
    /// particle's direction or the creation of a secondary particle.
    ///
    /// \param[in,out] trk The particle track to be updated.
    /// \param[in,out] stk The stack for any newly created secondary particles.
    /// \param[in,out] rng A pointer to the random number generator.
    /// \param[in] len The length of the step.
    /// \param[in,out] mat The material properties.
    /// \param[in] score_local_deposit A flag to indicate whether to score energy locally.
    CUDA_HOST_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>&    mat,
              bool              score_local_deposit) = 0;
};

}   // namespace mqi

#endif
