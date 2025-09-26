/// \file mqi_fippel_physics.hpp
/// \brief Implements a "physics list" for proton transport based on the Fippel model.
///
/// In a Monte Carlo simulation, a "physics list" is a crucial component that defines:
/// 1. All the possible physical interactions a particle can undergo (e.g., ionization, scattering).
/// 2. The logic for determining how far a particle travels in a single "step".
/// 3. The process for choosing which interaction occurs at the end of a step.
/// This file provides a specific implementation for proton transport using models
/// developed by Fippel and others.
#ifndef MQI_FIPPEL_PHYSICS_HPP
#define MQI_FIPPEL_PHYSICS_HPP

#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_p_ionization.hpp>
#include <moqui/base/mqi_physics_list.hpp>
#include <moqui/base/mqi_po_elastic.hpp>
#include <moqui/base/mqi_po_inelastic.hpp>
#include <moqui/base/mqi_pp_elastic.hpp>

namespace mqi
{

/// \class fippel_physics
/// \brief A physics list that defines the Fippel model for proton transport.
///
/// \tparam R The floating-point type (e.g., `float`, `double`) used for calculations.
///
/// This class acts as the "brain" for the proton simulation. It orchestrates the
/// different physical processes and manages the particle's journey through the material.
/// It combines continuous energy loss from ionization with discrete, random interactions
/// like elastic and inelastic scattering.
template<typename R>
class fippel_physics : public physics_list<R>
{
public:
    ///< A reference to the singleton object containing physical constants.
    const physics_constants<R>& units = physics_list<R>::units;

    ///< The maximum geometric distance (in mm) a proton can travel in a single step.
    const float max_step = 1.0;
    ///< The maximum energy a proton can lose in a single step, as a fraction of its current energy.
    ///< This prevents single steps from being too large in high-density or low-energy regions.
    const float max_energy_loss = 0.25;

    ///< Physics model for proton ionization (continuous energy loss).
    mqi::p_ionization_tabulated<R> p_ion;
    ///< Physics model for proton-proton (p-p) elastic scattering.
    mqi::pp_elastic_tabulated<R> pp_e;
    ///< Physics model for proton-oxygen (p-o) elastic scattering.
    mqi::po_elastic_tabulated<R> po_e;
    ///< Physics model for proton-oxygen (p-o) inelastic scattering.
    mqi::po_inelastic_tabulated<R> po_i;

    /// \brief Default constructor.
    ///
    /// Initializes the tabulated physics models by providing them with energy ranges
    /// and pointers to pre-calculated data tables (e.g., `cs_p_ion_table`).
    /// \note `CUDA_HOST_DEVICE` allows this to be called from both CPU and GPU code.
    CUDA_HOST_DEVICE
    fippel_physics() :
        p_ion(0.1,
              299.6,
              0.5,
              mqi::cs_p_ion_table,
              mqi::restricted_stopping_power_table,
              mqi::range_steps),
        pp_e(0.5, 300.0, 0.5, mqi::cs_pp_e_g4_table), po_e(0.5, 300.0, 0.5, mqi::cs_po_e_g4_table),
        po_i(0.5, 300.0, 0.5, mqi::cs_po_i_g4_table)

    {
        ;
    }

    /// \brief Constructor with a specified energy cutoff for secondary electrons.
    /// \param[in] e_cut The energy cutoff for secondary electrons (delta rays).
    CUDA_HOST_DEVICE
    fippel_physics(R e_cut) :
        physics_list<R>::Te_cut(e_cut), p_ion(0.1,
                                              299.6,
                                              0.5,
                                              mqi::cs_p_ion_table,
                                              mqi::restricted_stopping_power_table,
                                              mqi::range_steps),

        pp_e(0.5, 300.0, 0.5, mqi::cs_pp_e_g4_table), po_e(0.5, 300.0, 0.5, mqi::cs_po_e_g4_table),
        po_i(0.5, 300.0, 0.5, mqi::cs_po_i_g4_table)

    {
        ;
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~fippel_physics() {
        ;
    }

    /// \brief Determines the step length and samples discrete interactions for a particle track.
    ///
    /// This method implements the core stepping logic for transporting a proton.
    /// It decides how far the proton will travel in the current step and what interactions
    /// will occur. The logic is as follows:
    ///
    /// 1.  **Check Energy:** If the proton's energy is below the transport cutoff, deposit
    ///     all remaining energy and stop the track.
    /// 2.  **Calculate Step Limits:** Determine the maximum allowed step size based on
    ///     both a fixed geometric limit (`max_step`) and a limit based on the maximum
    ///     allowed energy loss (`max_energy_loss`).
    /// 3.  **Calculate Cross-Sections:** Sum the probabilities (cross-sections) of all
    ///     possible discrete interactions to get a total interaction probability.
    /// 4.  **Sample Mean Free Path (MFP):** Randomly sample the distance the particle
    ///     will travel before its next discrete interaction. This is the MFP.
    /// 5.  **Determine Final Step Length:** The actual step length is the *minimum* of
    ///     the MFP, the step limits from step 2, and the distance to the next voxel boundary.
    /// 6.  **Apply Continuous Effects:** Update the particle's energy and position based on
    ///     the continuous energy loss over the final step length.
    /// 7.  **Sample Discrete Interaction:** If the step was limited by the MFP (meaning a
    ///     discrete interaction occurred), randomly select which interaction happens based
    ///     on their relative probabilities and update the particle's state (e.g., change
    ///     its direction after a scattering event).
    ///
    /// \param[in,out] trk The particle track to be transported.
    /// \param[in,out] stk The stack for created secondary particles.
    /// \param[in,out] rng A pointer to the random number generator.
    /// \param[in] rho_mass The mass density of the current material.
    /// \param[in] mat The material properties of the current voxel.
    /// \param[in] distance_to_boundary The distance to the next geometric boundary.
    /// \param[in] score_local_deposit A flag indicating whether to score locally deposited energy.
    CUDA_HOST_DEVICE
    virtual void
    stepping(track_t<R>&       trk,
             track_stack_t<R>& stk,
             mqi_rng*          rng,
             const R&          rho_mass,
             material_t<R>&    mat,
             const R&          distance_to_boundary,
             bool              score_local_deposit) {

        // If particle energy is below the transport cutoff, deposit remaining energy and stop.
        if (trk.vtx0.ke < this->Tp_cut) {
            if (trk.vtx0.ke < 0) trk.vtx0.ke = 0;
            assert(trk.vtx0.ke >= 0);
            trk.deposit(trk.vtx0.ke);
            trk.update_post_vertex_energy(trk.vtx0.ke);
            p_ion.last_step(trk, mat);
            trk.stop();
            return;
        }

        mqi::relativistic_quantities<R> rel(trk.vtx0.ke, units.Mp);
        R                               length = 0.0;
        // Determine the maximum allowed step size based on energy loss and geometric constraints.
        R max_loss_step    = max_energy_loss * -1.0 * rel.Ek / p_ion.dEdx(rel, mat);
        R current_min_step = this->max_step;
        current_min_step   = current_min_step * mat.stopping_power_ratio(rel.Ek) * mat.rho_mass / this->units.water_density;
        current_min_step  = (current_min_step <= max_loss_step) ? current_min_step : max_loss_step;
        R max_loss_energy = -1.0 * current_min_step * p_ion.dEdx(rel, mat);

        // Calculate the total cross-section (interaction probability) at the current energy
        // and at the energy after the maximum possible energy loss.
        R cs1[4]          = { p_ion.cross_section(rel, mat),
                     pp_e.cross_section(rel, mat),
                     po_e.cross_section(rel, mat),
                     po_i.cross_section(rel, mat) };
        R cs1_sum         = cs1[0] + cs1[1] + cs1[2] + cs1[3];

        mqi::relativistic_quantities<R> rel_de(trk.vtx0.ke - max_loss_energy, units.Mp);
        R                               cs2[4]  = { p_ion.cross_section(rel_de, mat),
                     pp_e.cross_section(rel_de, mat),
                     po_e.cross_section(rel_de, mat),
                     po_i.cross_section(rel_de, mat) };
        R                               cs2_sum = cs2[0] + cs2[1] + cs2[2] + cs2[3];

        // Use the larger of the two cross-sections for a conservative estimate.
        R  cs_sum = (cs1_sum >= cs2_sum) ? cs1_sum : cs2_sum;
        R* cs     = (cs1_sum >= cs2_sum) ? cs1 : cs2;

        // Sample the Mean Free Path (MFP) - the distance to the next discrete interaction.
        R prob       = mqi_uniform<R>(rng);
        R mfp        = -1.0f * logf(prob) / cs_sum;
        R step_limit = current_min_step * this->units.water_density / (mat.stopping_power_ratio(rel.Ek) * mat.rho_mass);

        // Branching logic to determine the final step length.
        // Case 1: Step is limited by the distance to the next voxel boundary.
        if (distance_to_boundary < mfp && distance_to_boundary < step_limit) {
            // Take a continuous step to the boundary.
            p_ion.along_step(trk, stk, rng, distance_to_boundary, mat);
            assert_track<R>(trk, 0);
        // Case 2: Step is limited by the Mean Free Path (a discrete interaction occurs).
        } else if ((mfp < distance_to_boundary ||
                    mqi::mqi_abs(mfp - distance_to_boundary) < mqi::geometry_tolerance) &&
                   (mfp < step_limit || mqi::mqi_abs(mfp - step_limit) < mqi::geometry_tolerance)) {
            // Take a continuous step for the length of the MFP.
            p_ion.along_step(trk, stk, rng, mfp, mat);
            assert_track<R>(trk, 10);
            if (trk.vtx1.ke < this->Tp_cut) { return; }

            // Sample which discrete interaction occurs.
            R u          = cs_sum * mqi_uniform<R>(rng);
            trk.vtx1.dir = trk.vtx0.dir;
            if (u < cs[0]) { // Ionization
                p_ion.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 1);
            } else if (u < (cs[0] + cs[1])) { // p-p elastic
                pp_e.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 2);
            } else if (u < (cs[0] + cs[1] + cs[2])) { // p-o elastic
                po_e.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 3);
            } else if (u < (cs[0] + cs[1] + cs[2] + cs[3])) { // p-o inelastic
                po_i.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 4);
            }
        // Case 3: Step is limited by the maximum allowed step size.
        } else {
            // Take a continuous step.
            p_ion.along_step(trk, stk, rng, step_limit, mat);
            assert_track<R>(trk, 5);
        }
        return;
    }
};

}   // namespace mqi
#endif