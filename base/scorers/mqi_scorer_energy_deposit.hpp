/// \file mqi_scorer_energy_deposit.hpp
///
/// \brief Defines functions for "scoring" (calculating) physics quantities like dose and LET.
///
/// \details
/// In Monte Carlo simulations, a "scorer" is a function that calculates and records a specific
/// quantity of interest each time a particle interacts with or passes through a region
/// of the simulation geometry. This file provides a set of standard scorers for radiotherapy
/// applications, such as calculating dose and Linear Energy Transfer (LET). These functions
/// are designed to be executed efficiently on CUDA-enabled GPUs, which is why they are
/// marked with the `CUDA_DEVICE` macro.
#ifndef MQI_SCORER_ENERGY_DEPOSIT_HPP
#define MQI_SCORER_ENERGY_DEPOSIT_HPP

#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_track.hpp>

namespace mqi
{

/// \brief Scores the total energy deposited by a particle as it passes through a voxel.
///
/// \tparam R The floating-point type (e.g., `float` or `double`). This is a C++
///           "template" that makes the function generic for different precisions.
///
/// \param[in] trk A `track_t` object representing the particle's most recent step. It contains
///                information like the energy deposited (`dE`, `local_dE`) and whether the
///                particle is a primary or secondary.
/// \param[in] cnb The index of the current voxel (a 3D pixel in the geometry) where the
///                energy is being deposited. `cnb_t` is typically an integer type.
/// \param[in] geo The simulation's 3D grid geometry, which stores information about voxel
///                sizes, positions, and the material (e.g., density) in each voxel.
///
/// \return The total energy deposited in the voxel, in MeV. This includes both
///         energy lost along the track (`dE`) and any locally deposited energy (`local_dE`).
///
/// \note The `CUDA_DEVICE` macro indicates this function is intended to run on a GPU.
template<typename R>
CUDA_DEVICE double
energy_deposit(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    return trk.dE + trk.local_dE;
}

/// \brief Scores the energy deposited by primary particles only.
///
/// \details
/// Primary particles are those originating from the initial radiation source, as
/// opposed to secondary particles, which are created from interactions within the target.
/// This scorer is useful for separating the dose contribution of primaries from secondaries.
///
/// \param[in] trk The particle track, which includes a `primary` flag.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return The energy deposited (`dE`) if the particle is a primary, otherwise 0.
template<typename R>
CUDA_DEVICE double
energy_deposit_primary(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    return trk.primary == true ? trk.dE : 0.0;
}

/// \brief Scores the energy deposited by secondary particles only.
///
/// \details
/// Secondary particles are produced when primary particles interact with the medium.
/// Scoring their energy separately is important for understanding the full dose distribution.
///
/// \param[in] trk The particle track, which includes a `primary` flag.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return The energy deposited (`dE`) if the particle is a secondary, otherwise 0.
template<typename R>
CUDA_DEVICE double
energy_deposit_secondary(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    return trk.primary == false ? trk.dE : 0.0;
}

/// \brief Calculates the dose-to-water for a given particle track.
///
/// \details
/// "Dose-to-water" is a standard quantity in radiotherapy. It represents the dose that
/// would be deposited in a small volume of water placed at that point in the medium.
/// It is calculated using the stopping power ratio between the current medium and water,
/// which accounts for differences in how the two materials absorb energy.
///
/// \param[in] trk The particle track. Contains the deposited energy and the particle's
///                kinetic energy (`trk.vtx0.ke`) needed to look up the stopping power ratio.
/// \param[in] cnb The current voxel index. Used to look up the voxel's density and volume.
/// \param[in] geo The simulation geometry grid.
/// \return The dose-to-water in Grays (Gy), where 1 Gy = 1 Joule/kg.
template<typename R>
CUDA_DEVICE double
dose_to_water(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
#if defined(__CUDACC__)
    density = geo.get_data()[cnb];
#else
    density = geo.get_data()[cnb];
#endif
    R             volume = geo.get_volume(cnb);
    mqi::h2o_t<R> water;
    if (density < 1.0e-7) {
        return 0.0;
    } else {
        water.rho_mass = density;
        // The formula converts energy from MeV to Joules (1.602e-13) and divides by the
        // mass of the voxel (volume * density) to get dose. The stopping power ratio
        // adjusts this value to what it would be in water.
        return (trk.dE + trk.local_dE) * 1.60218e-10 /
               (volume * density * water.stopping_power_ratio(trk.vtx0.ke));
    }
}

/// \brief Calculates the dose-to-medium for a given particle track.
///
/// \details
/// "Dose-to-medium" is the actual dose absorbed by the material at that point.
/// It is calculated as the energy deposited per unit mass of the medium. This scorer
/// only considers primary particles.
///
/// \param[in] trk The particle track.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return The dose to the medium in Grays (Gy).
template<typename R>
CUDA_DEVICE double
dose_to_medium(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density  = geo.get_data()[cnb];
    R volume = geo.get_volume(cnb);
    // Convert deposited energy (MeV) to Joules (1.602e-13) and divide by mass (kg) to get dose in Gy.
    return trk.primary ? trk.dE * 1.60218e-13 * 1000.0 / (volume * density)
                       : 0.0;
}

/// \brief Calculates the dose-weighted Linear Energy Transfer (LETd).
///
/// \details
/// LET is a measure of how much energy a particle transfers to the material per unit
/// distance, typically in keV/micrometer. It is an important indicator of the biological
/// effectiveness of radiation. Dose-weighted LET (LETd) is calculated by weighting the LET of
/// each particle by the dose it deposits. This function includes an additional weighting
/// factor where LET values above 25 are excluded.
///
/// \param[in] trk The particle track, containing the start and end positions of the step
///                (`vtx0.pos`, `vtx1.pos`) to calculate step length.
/// \param[in] cnb The current voxel index, used to find the medium density.
/// \param[in] geo The simulation geometry grid.
/// \return The dose-weighted LET value (dE * LET).
template<typename R>
CUDA_DEVICE double
LETd_weight1(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;   // Convert g/cm^3 to kg/m^3? Or other unit conversion.
    // Calculate the straight-line distance of the particle's step.
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    // LET = (Energy deposited) / (path length) / (density)
    double let = trk.dE / length / density;
    if (let >= 25.0) {   // High LET cut-off
        return 0;
    } else {
        return trk.dE * let;
    }
}

/// \brief Calculates the dose-weighted LET (LETd) with a different weighting.
///
/// \details
/// This is another variant of the dose-weighted LET calculation. Instead of weighting by
/// LET, it simply scores the deposited energy (`dE`) if the LET is below the 25.0 threshold.
/// The final LETd would be calculated by dividing the sum of these scores by the total dose.
///
/// \param[in] trk The particle track.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return `trk.dE` if LET < 25, otherwise 0.
template<typename R>
CUDA_DEVICE double
LETd_weight2(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    double let = trk.dE / length / density;
    if (let >= 25.0) {
        return 0;
    } else {
        return trk.dE * 1.0;
    }
}

/// \brief Calculates the track-weighted LET (LETt).
///
/// \details
/// Track-weighted LET (LETt) is calculated by averaging the LET over the particle track length.
/// This scorer function calculates the numerator of the LETt average, which is (track length * LET).
/// The final LETt is the sum of these values divided by the sum of all track lengths.
///
/// \param[in] trk The particle track.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return The track-length-weighted LET value (length * LET).
template<typename R>
CUDA_DEVICE double
LETt_weight1(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    double let = trk.dE / length / density;
    return length * let;
}

/// \brief Calculates the track-weighted LET (LETt) with a different weighting.
///
/// \details
/// This is another variant of the track-weighted LET calculation. This scorer simply
/// returns the length of the particle's step. The final LETt would be calculated by
/// dividing the sum of (length * LET) from another scorer by the sum of these lengths.
///
/// \param[in] trk The particle track.
/// \param[in] cnb The current voxel index.
/// \param[in] geo The simulation geometry grid.
/// \return The length of the track segment.
template<typename R>
CUDA_DEVICE double
LETt_weight2(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    return length;
}

#if defined(__CUDACC__)
/// \brief Function pointers for dynamically assigning scorers on the GPU.
///
/// \details
/// In C++, a function pointer is a variable that stores the memory address of a
/// function. This allows functions to be passed around and called dynamically at
/// runtime, which is similar to passing functions as arguments in Python.
///
/// This block is compiled only when using the CUDA compiler (`__CUDACC__`). It
/// creates GPU-side pointers to the scorer functions. This allows the main simulation
/// kernel to dynamically select which scorer to use based on user configuration,
/// making the simulation code more flexible and extensible without needing to be recompiled
/// for every different scoring quantity.
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> energy_deposit_pointer = mqi::energy_deposit;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> energy_deposit_primary_pointer =
  mqi::energy_deposit_primary;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> Dm_pointer           = mqi::dose_to_medium;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> Dw_pointer           = mqi::dose_to_water;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETd_weight1_pointer = mqi::LETd_weight1;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETd_weight2_pointer = mqi::LETd_weight2;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETt_weight1_pointer = mqi::LETt_weight1;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETt_weight2_pointer = mqi::LETt_weight2;

#endif
}   // namespace mqi
#endif
