#ifndef MQI_TREATMENT_MACHINE_ION_H
#define MQI_TREATMENT_MACHINE_ION_H

/// \file
///
/// Treatment machine for particle therapy

#include <moqui/base/mqi_aperture.hpp>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_rangeshifter.hpp>
#include <moqui/base/mqi_treatment_machine.hpp>

#define SIGMA2FWHM 2.35482004503   // 2.*std::sqrt(2.*std::log(2.))

namespace mqi
{

/// @class treatment_machine_ion
/// @brief A class representing a particle therapy system, inheriting from `treatment_machine`.
///
/// This class provides a concrete implementation for ion therapy machines, handling the creation
/// of beam sources, beamlines, and coordinate transformations based on DICOM data or other inputs.
/// It includes methods for characterizing beamlets, apertures, and range shifters.
///
/// @tparam T The floating-point type for phase-space variables (e.g., `float` or `double`).
template<typename T>
class treatment_machine_ion : public treatment_machine<T>
{

protected:
public:
    /// @brief Default constructor.
    treatment_machine_ion() {
        ;
    }

    /// @brief Default destructor.
    ~treatment_machine_ion() {
        ;
    }

    /// @brief Creates a coordinate transformation from a dataset.
    ///
    /// This method extracts gantry, collimator, and couch angles, as well as the isocenter position,
    /// from the provided dataset to define the beam's coordinate system.
    /// @param ds Pointer to the dataset containing treatment plan information.
    /// @param m The modality type (e.g., `RTPLAN`, `IONPLAN`).
    /// @return A `mqi::coordinate_transform<T>` object.
    /// @note Rotation angle direction is assumed to be counter-clockwise (CCW).
    /// @note For RTIBTR, which lacks isocenter data, a position of (0,0,0) is returned.
    virtual mqi::coordinate_transform<T>
    create_coordinate_transform(const mqi::dataset* ds, const mqi::modality_type m) {
        auto                     seq_tags = &mqi::seqtags_per_modality.at(m);
        auto                     layer0   = (*ds)(seq_tags->at("ctrl"))[0];   //layer0
        std::vector<float>       tmp;
        std::vector<std::string> tmp_dir;
        std::array<T, 4>         angles;
        ///< As CW, CRW, NONE, conditionally required, we don't use them at this moment
        layer0->get_values("BeamLimitingDeviceAngle", tmp);
        angles[0] = tmp[0];
        layer0->get_values("GantryAngle", tmp);
        angles[1] = tmp[0];

        layer0->get_values("PatientSupportAngle", tmp);
        angles[2] = -1.0 * tmp[0];
        angles[3] = 0.0;

        mqi::vec3<T> pos;

        ///< Rotation and Position don't exist in RTRECORD and IONRECORD
        if ((m == RTPLAN) || (m == IONPLAN)) {
            layer0->get_values("IsocenterPosition", tmp);
            pos.x = tmp[0];
            pos.y = tmp[1];
            pos.z = tmp[2];
        }
        return mqi::coordinate_transform<T>(angles, pos);
    }

    /// @brief Creates a beam source from a dataset.
    ///
    /// This method constructs a beam source by parsing an `IonBeamSequence` from the dataset.
    /// It characterizes each spot to create beamlets and determines the number of histories to simulate.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type (e.g., `RTIP` or `RTIBTR`).
    /// @param pcoord The coordinate transform to map to the global coordinate system.
    /// @param particles_per_history A scaling factor for the number of particles per history. If -1, each beamlet generates one history.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamsource<T>` object.
    /// @note Currently, only `MODULATED` scan mode is supported.
    virtual mqi::beamsource<T>
    create_beamsource(const mqi::dataset*                ds,
                      const mqi::modality_type           m,
                      const mqi::coordinate_transform<T> pcoord,
                      const float                        particles_per_history  = -1,
                      const float                        source_to_isocenter_mm = 390.0) {
        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;

        ///< Parse DICOM beam module for Ion
        beam_module_ion ion_beam(ds, m);

        std::vector<std::string> scan_mode(1);
        ds->get_values("ScanMode", scan_mode);
        if (!scan_mode[0].compare("MODULATED")) {
            std::runtime_error("Only MODULATED scan mode is supported");
        }
        mqi::beamsource<T> beamsource;

        ///< Modulated BEAM
        const auto&                       spots     = *(ion_beam.get_sequence());
        const size_t                      nb_spots  = spots.size();
        const mqi::beam_module_ion::spot* null_spot = nullptr;

        for (size_t i = 0; i < nb_spots; ++i) {
            /// Calculate number of histories to be simulated per beamlet
            size_t nb_histories = (particles_per_history == -1)
                                    ? 1
                                    : this->characterize_history(spots[i], particles_per_history);

            /// Calculate on & off time per beamlet
            /// By default, on is set to 1 sec but off is set to 0 sec.

            std::array<T, 2> time_on_off;
            if (i == (nb_spots - 1)) {
                time_on_off = this->characterize_beamlet_time(spots[i], *null_spot);

            } else {
                time_on_off = this->characterize_beamlet_time(spots[i], spots[i + 1]);
            }
            /// Then, add a beamlet with
            ///            its number of histories,
            ///            coordinate system
            ///            beamlet time on/off
            beamsource.append_beamlet(this->characterize_beamlet(spots[i]),
                                      nb_histories,
                                      pcoord,
                                      time_on_off[0],
                                      time_on_off[1]);
        }

        return beamsource;
    }

    /// @brief Creates a beam source from a vector of spots.
    ///
    /// This overload constructs a beam source directly from a list of spot definitions.
    /// @param spots A vector of `mqi::beam_module_ion::spot` objects.
    /// @param m The modality type.
    /// @param pcoord The coordinate transform to map to the global coordinate system.
    /// @param particles_per_history A scaling factor for the number of particles per history.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamsource<T>` object.
    virtual mqi::beamsource<T>
    create_beamsource(const std::vector<mqi::beam_module_ion::spot>& spots,
                      const mqi::modality_type                       m,
                      const mqi::coordinate_transform<T>             pcoord,
                      const float                                    particles_per_history = -1,
                      const float source_to_isocenter_mm = 390.0) {

        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;
        ///< Parse DICOM beam module for ION
        mqi::beamsource<T> beamsource;

        ///< Modulated BEAM
        const size_t                      nb_spots  = spots.size();
        const mqi::beam_module_ion::spot* null_spot = nullptr;
        for (size_t i = 0; i < nb_spots; ++i) {
            /// Calculate number of histories to be simulated per beamlet

            //// TODO: history counting is different from the other method and would be very confisuing
            size_t nb_histories = (particles_per_history <= 1)
                                    ? 1
                                    : std::ceil(spots[i].meterset * particles_per_history);

            /// Calculate on & off time per beamlet
            /// By default, on is set to 1 sec but off is set to 0 sec.
            std::array<T, 2> time_on_off;
            if (i == (nb_spots - 1)) {
                time_on_off = this->characterize_beamlet_time(spots[i], *null_spot);
            } else {
                time_on_off = this->characterize_beamlet_time(spots[i], spots[i + 1]);
            }

            /// Then, add a beamlet with
            ///            its number of histories,
            ///            coordinate system
            ///            beamlet time on/off
            beamsource.append_beamlet(this->characterize_beamlet(spots[i]),
                                      nb_histories,
                                      pcoord,
                                      time_on_off[0],
                                      time_on_off[1]);
        }
        return beamsource;
    }

    /// @brief Pure virtual method to characterize a modulated beamlet from DICOM spot information.
    /// @param s A `spot` object from the DICOM data.
    /// @return A `mqi::beamlet<T>` object.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) = 0;

    /// @brief Pure virtual method to characterize a modulated beamlet with a specified source-to-isocenter distance.
    /// @param s A `spot` object from the DICOM data.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamlet<T>` object.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm) = 0;

    /// @brief Pure virtual method to characterize a uniform or modulated-spec beamlet from two spots.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @return A `mqi::beamlet<T>` object.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1) = 0;

    /// @brief Pure virtual method to characterize a modulated beamlet from log file spot information.
    /// @param s A `logspot` object from the log file data.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return A `mqi::beamlet<T>` object.
    /// @note Connected to `mqi_treatment_machine_smc_gtr2.hpp`. Added by Chanil Jeon in 2023.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::logspot& s,
        const float                       source_to_isocenter_mm,
        const bool rsuse) = 0;

    /// @brief Characterizes the beam delivery time for a beamlet.
    /// @param s_current The current spot.
    /// @param s_next The next spot in the sequence.
    /// @return An array containing the on-time and off-time in seconds. Defaults to `{1.0, 0.0}`.
    virtual std::array<T, 2>
    characterize_beamlet_time(const mqi::beam_module_ion::spot& s_current,
                              const mqi::beam_module_ion::spot& s_next) {
        return { 1.0, 0.0 };
    }

    /// @brief Calculates the number of histories for a modulated beamlet.
    /// @param s The spot information from DICOM.
    /// @param scale The scaling factor (particles per history).
    /// @return The number of histories to simulate.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s, float scale) {
        return s.meterset / scale;
    }

    /// @brief Calculates the number of histories for a uniform or modulated-spec beamlet.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @param scale The scaling factor (particles per history).
    /// @return The number of histories to simulate.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1,
                         float                             scale) {
        return s1.meterset / scale;
    }

    /// @brief Creates a beamline model from a dataset.
    ///
    /// This method constructs the beamline by identifying and characterizing components like
    /// range shifters and apertures from the dataset.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type (e.g., `RTIP` or `RTIBTR`).
    /// @return A `mqi::beamline<T>` object.
    /// @note This method does not handle coordinate transformations, which are assumed to be managed by the MC engine (e.g., TOPAS).
    virtual mqi::beamline<T>
    create_beamline(const mqi::dataset* ds, mqi::modality_type m) {
        mqi::beamline<T> beamline;

        ///< Access to tag LUT
        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        ///< geometry creation of snout?
        ///< position from control point 0
        ///< beamline.append_geometry(this->characterize_snout);
        std::vector<int> itmp;
        std::vector<int> ftmp;

        ///< 1. number of rangeshifter sequence
        ds->get_values("NumberOfRangeShifters", itmp);
        std::cout << "Creating beam line object.. : Number of range shifter sequence --> " << itmp[0] << std::endl;
        if (itmp[0] >= 1) { beamline.append_geometry(this->characterize_rangeshifter(ds, m)); }

        ///< 2. number of blocks
        ds->get_values("NumberOfBlocks", itmp);
        std::cout << "Creating beam line object.. : Number of blocks --> " << itmp[0] << std::endl;
        if (itmp[0] >= 1) { beamline.append_geometry(this->characterize_aperture(ds, m)); }
        return beamline;
    }

    /// @brief Pure virtual method to characterize a range shifter.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::rangeshifter` object.
    /// @note Users must implement their own rules to convert DICOM information to range shifter specifications.
    virtual mqi::rangeshifter*
    characterize_rangeshifter(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// @brief Pure virtual method to characterize an aperture.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::aperture` object.
    /// @note Users must implement their own rules to convert DICOM information to aperture specifications.
    virtual mqi::aperture*
    characterize_aperture(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// @brief Characterizes the aperture opening from a dataset.
    ///
    /// This method extracts the (x, y) coordinates defining the aperture opening(s) from the `BlockSequence` in the dataset.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A vector of vectors, where each inner vector contains the (x, y) points for a single aperture hole.
    /// @note Multiple holes are supported.
    const std::vector<std::vector<std::array<float, 2>>>
    characterize_aperture_opening(const mqi::dataset* ds, mqi::modality_type m) {
        std::vector<std::vector<std::array<float, 2>>> apt_xy_points;
        auto                                           seq_tags = &mqi::seqtags_per_modality.at(m);

        //0. aperture sequence
        auto apt_ds = (*ds)(seq_tags->at("blk"));
        assert(apt_ds.size() >= 1);

        for (auto apt : apt_ds) {
            std::vector<std::array<float, 2>> block_data;
            std::vector<int>                  nb_xy_points;
            std::vector<float>                xy_points;

            apt->get_values("BlockNumberOfPoints", nb_xy_points);
            apt->get_values("BlockData", xy_points);

            for (int j = 0; j < nb_xy_points[0]; ++j) {
                block_data.push_back({ xy_points[j * 2], xy_points[j * 2 + 1] });
            }
            apt_xy_points.push_back(block_data);
        }
        return apt_xy_points;
    }

    /// @brief Calculates the beam starting position based on the isocenter.
    ///
    /// This method projects the isocenter position back to a given z-plane to determine the beam's starting (x, y) coordinates.
    /// @param iso The isocenter position.
    /// @param z The z-position where the beam starts.
    /// @return A `mqi::vec3<T>` representing the beam's starting position.
    virtual mqi::vec3<T>
    beam_starting_position(const mqi::vec3<T>& iso, T z) {
        mqi::vec3<T> beam(0, 0, z);
        beam.x =
          iso.x * (treatment_machine_ion<T>::SAD_[0] - beam.z) / treatment_machine_ion<T>::SAD_[0];
        beam.y =
          iso.y * (treatment_machine_ion<T>::SAD_[1] - beam.z) / treatment_machine_ion<T>::SAD_[1];

          //std::cout << "Beam starting position : X --> " << beam.x << " Y --> " << beam.y << " Z --> " << beam.z << std::endl;

        return beam;
    }
};

}   // namespace mqi

#endif
