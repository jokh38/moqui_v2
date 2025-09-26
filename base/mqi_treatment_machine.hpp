#ifndef MQI_TREATMENT_MACHINE_H
#define MQI_TREATMENT_MACHINE_H

/// \file
///
/// Abstraction for treatment machine

#include <moqui/base/mqi_beam_module.hpp>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_beamlet.hpp>
#include <moqui/base/mqi_beamline.hpp>
#include <moqui/base/mqi_beamsource.hpp>

namespace mqi
{

/// @class treatment_machine
/// @brief An abstract base class for all types of radiotherapy treatment machines (RT and ION).
///
/// This class provides a common interface for defining treatment machine models.
/// It typically consists of geometries (beam limiting devices) and sources (particle fluence).
/// Subclasses must implement the pure virtual functions to create a beamline, a beam source,
/// and a coordinate transformation.
///
/// @tparam T The floating-point type for phase-space variables (e.g., float or double).
/// @note The specific treatment machine is identified by its name.
template<typename T>
class treatment_machine
{
protected:
    ///< Machine name in string format (e.g., "site:system:mc_code").
    const std::string name_;

    ///< Source-to-Axis Distance (SAD) in mm. Necessary for calculating beam divergence.
    std::array<float, 2> SAD_;

    ///< Distance from the phase-space plane to the isocenter in mm.
    float source_to_isocenter_mm_;

public:
    /// @brief Default constructor.
    treatment_machine() {
        ;
    }

    /// @brief Virtual destructor for the abstract base class.
    virtual ~treatment_machine() {
        ;
    }

    /// @brief Creates and returns the beamline model for the treatment machine.
    /// @param ds Pointer to the dataset containing treatment plan information.
    /// @param m The modality type (e.g., RTIP, US, PASSIVE).
    /// @return A `mqi::beamline<T>` object representing the machine's beamline.
    virtual mqi::beamline<T>
    create_beamline(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// @brief Creates and returns the beam source model based on dataset information.
    /// @param ds Pointer to the dataset (e.g., DICOM-RT plan).
    /// @param m The modality type (e.g., RT-Ion Plan, RT-Ion Beam Treatment Record).
    /// @param pcoord The coordinate system of the beam geometry.
    /// @param particles_per_history The total number of histories to simulate.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamsource<T>` object for the simulation.
    virtual mqi::beamsource<T>
    create_beamsource(const mqi::dataset*                ds,
                      const mqi::modality_type           m,
                      const mqi::coordinate_transform<T> pcoord,
                      const float                        particles_per_history  = -1,
                      const float                        source_to_isocenter_mm = 390.0) = 0;

    /// @brief Creates and returns the beam source model from a vector of spots.
    ///
    /// This is used when beam data is provided directly, e.g., from a file like `tramp` for MGH.
    /// @param spots A vector of `mqi::beam_module_ion::spot` objects.
    /// @param m The modality type.
    /// @param pcoord The coordinate system of the beam geometry.
    /// @param particles_per_history The total number of histories to simulate.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamsource<T>` object for the simulation.
    virtual mqi::beamsource<T>
    create_beamsource(const std::vector<mqi::beam_module_ion::spot>& spots,
                      const mqi::modality_type                       m,
                      const mqi::coordinate_transform<T>             pcoord,
                      const float                                    particles_per_history = -1,
                      const float source_to_isocenter_mm = 390.0) = 0;

    /// @brief Creates a beam source model from log file data.
    ///
    /// This implementation is based on log files from Samsung Medical Center.
    /// @param logfileData A struct containing data parsed from the log files.
    /// @param pcoord The coordinate system of the beam geometry.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return A `mqi::beamsource<T>` object for the simulation.
    /// @note Added by Chanil Jeon in 2023.
    virtual mqi::beamsource<T>
    create_beamsource(const mqi::logfiles_t& logfileData,
                      const mqi::coordinate_transform<T> pcoord,
                      const float source_to_isocenter_mm = 465.0,
                      const bool rsuse = false) = 0;

    /// @brief Creates and returns the coordinate transformation information for the beam.
    /// @param ds Pointer to the dataset.
    /// @param m The modality type (e.g., IMPT, US, PASSIVE).
    /// @return A `mqi::coordinate_transform<T>` object for the beam geometry.
    virtual mqi::coordinate_transform<T>
    create_coordinate_transform(const mqi::dataset* ds, const mqi::modality_type m) = 0;

protected:
    /// @brief Characterizes a single beamlet from a spot definition.
    /// @param s A `spot` object containing energy, position, FWHM, and particle count.
    /// @return A `mqi::beamlet<T>` object created from the given parameters.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) = 0;

    /// @brief Characterizes a single beamlet from a spot definition with a specified source-to-isocenter distance.
    /// @param s A `spot` object containing energy, position, FWHM, and particle count.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamlet<T>` object created from the given parameters.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm) = 0;

    /// @brief Characterizes a beamlet by interpolating between two spots.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @return A `mqi::beamlet<T>` object representing the interpolated beamlet.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1) = 0;

    /// @brief Characterizes a modulated beamlet based on spot information from a log file.
    /// @param s A `logspot` object containing spot information from the log.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return A `mqi::beamlet<T>` object for the modulated beamlet.
    /// @note This method is connected to `mqi_treatment_machine_smc_gtr2.hpp`. Added by Chanil Jeon in 2023.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::logspot& s,
        const float                       source_to_isocenter_mm,
        const bool rsuse) = 0;
};

}   // namespace mqi

#endif
