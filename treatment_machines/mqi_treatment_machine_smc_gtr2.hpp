#ifndef MQI_TREATMENT_MACHINE_SMC_GTR2_H
#define MQI_TREATMENT_MACHINE_SMC_GTR2_H

// Samsung Medical Center Log file based QA treatment machine
// Modified by Chanil Jeon (2024-01-19 ver)
// Sungkyunkwan University, Samsung Medical Center

#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_treatment_machine_ion.hpp>
#include <moqui/base/distributions/mqi_phsp6d_ray.hpp>
#include <moqui/treatment_machines/spline_interp.hpp>

namespace mqi{

/// @class gtr2_material_t
/// @brief A material definition class specific to the GTR2 machine.
///
/// This class inherits from `patient_material_t` and is intended for use with the `gtr2`
/// treatment machine model. Custom material properties are not implemented in this version.
///
/// @tparam R The floating-point type for material properties (e.g., `float` or `double`).
template<typename R>
class gtr2_material_t: public patient_material_t<R> {
public:
    /// @brief Default constructor.
    CUDA_HOST_DEVICE
    gtr2_material_t(): patient_material_t<R>(){;}

    /// @brief Constructor initializing the material from a Hounsfield Unit value.
    /// @param hu The Hounsfield Unit (HU) value.
    CUDA_HOST_DEVICE
    gtr2_material_t(int16_t hu): patient_material_t<R>(hu){;}

    /// @brief Default destructor.
    CUDA_HOST_DEVICE
    ~gtr2_material_t(){;}
};

/// @class gtr2
/// @brief Represents the beam model for the Sumitomo IMPT machine (Gantry 2) at Samsung Medical Center (SMC).
///
/// This class provides a specific implementation of `treatment_machine_ion` for the GTR2 machine,
/// primarily based on log file data. It uses spline interpolation to model various energy-dependent
/// beam properties like spot size, angular spread, and particle count calibration.
///
/// @tparam T The floating-point type for phase-space variables (e.g., `float` or `double`).
template <typename T>
class gtr2 : public treatment_machine_ion<T> {
protected:

public:

    ///< Spline for interpolating particle count from dose.
    tk::spline protonPerDoseInterp;
    ///< Spline for interpolating dose from MU count.
    tk::spline dosePerMUCountInterp;

    ///< Spline for interpolating beam energy spread.
    tk::spline beamEnergySpreadInterp;
    ///< Spline for interpolating beam spot size.
    tk::spline beamSpotSizeInterp;
    ///< Spline for interpolating beam angular spread.
    tk::spline beamAngularSpreadInterp;
    ///< Spline for interpolating beam divergence.
    tk::spline beamDivergenceInterp;

    /// @brief Default constructor.
    ///
    /// Initializes the GTR2 machine model by setting the Source-to-Axis Distance (SAD) and
    /// configuring the spline interpolations for various beam characteristics based on
    /// machine-specific calibration data.
    gtr2()
    {
        treatment_machine_ion<T>::SAD_[0] = 2597.0 ;
        treatment_machine_ion<T>::SAD_[1] = 2223.0 ;

        // Particle count interpolation (Proton / Dose)
        std::vector<double> beamEnergyForProtonPerDose = { 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230 };
        std::vector<double> correctionProtonPerDose = { 1, 1.087142586, 1.194654014, 1.30672804, 1.418493333, 1.528336828, 1.635736563, 1.743366903, 1.85700629, 1.956866973, 2.065896427, 2.165839877, 2.272688046, 2.389197949, 2.491620628, 2.607891145, 2.728008023 };
        protonPerDoseInterp.set_points(beamEnergyForProtonPerDose, correctionProtonPerDose, tk::spline::cspline);
        protonPerDoseInterp.make_monotonic();

        // Particle count interpolation (Dose / MU count)
        std::vector<double> beamEnergyForDosePerMUcount = { 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230 };
        std::vector<double> correctionDosePerMUCount = { 1, 1.013734192, 1.012925693, 0.998307125, 0.983342911, 0.979028918, 0.967048299, 0.958585747, 0.946491263, 0.937642292, 0.921448278, 0.904786882, 0.893353261, 0.884179113, 0.884545787, 0.879304188, 0.871658754 };
        dosePerMUCountInterp.set_points(beamEnergyForDosePerMUcount, correctionDosePerMUCount, tk::spline::cspline);
        dosePerMUCountInterp.make_monotonic();

        // Beam Energy spread interpolation
        std::vector<double> beamEnergyForBeamEnergySpread = { 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230 };
        std::vector<double> correctionBeamEnergySpread = { 0.502793, 0.577766, 0.648957, 0.714868, 0.774127, 0.825533, 0.868103, 0.901097, 0.92404, 0.936734, 0.939254, 0.931934, 0.915346, 0.89027, 0.857653, 0.818572, 0.774184 };
        beamEnergySpreadInterp.set_points(beamEnergyForBeamEnergySpread, correctionBeamEnergySpread, tk::spline::cspline);
        beamEnergySpreadInterp.make_monotonic();

        // Beam Spot size interpolation
        std::vector<double> beamEnergyForBeamSpotSize = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamSpotSize = { 8.62, 8.02, 7.38, 6.74, 6.23, 5.94, 5.66, 5.41, 5.12, 4.87, 4.61, 4.53, 4.42, 4.44, 4.44, 4.46, 4.38, 4.29, 4.34, 4.13, 4.05, 3.98, 3.73, 3.76, 3.68, 3.52, 3.42, 3.25, 3.06, 2.72 };
        beamSpotSizeInterp.set_points(beamEnergyForBeamSpotSize, correctionBeamSpotSize, tk::spline::cspline);
        beamSpotSizeInterp.make_monotonic();

        // Beam angular spread interpolation
        std::vector<double> beamEnergyForBeamAngularSpread = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamAngularSpread = { 0.003615, 0.003495, 0.003815, 0.003695, 0.00354, 0.00334, 0.003185, 0.00258, 0.002545, 0.00238, 0.002675, 0.00231, 0.002285, 0.002135, 0.002255, 0.001845, 0.00201, 0.00197, 0.00089, 0.001855, 0.00194, 0.00195, 0.00178,0.001655, 0.001745, 0.00178, 0.00178, 0.001965, 0.00168, 0.001625 };
        beamAngularSpreadInterp.set_points(beamEnergyForBeamAngularSpread, correctionBeamAngularSpread, tk::spline::cspline);
        beamAngularSpreadInterp.make_monotonic();

        // Beam divergence interpolation
        std::vector<double> beamEnergyForBeamDivergence = { 70, 76, 84, 92, 100, 106, 112, 118, 126, 134, 142, 146, 150, 154, 158, 162, 170, 174, 178, 182, 186, 190, 198, 202, 206, 214, 218, 222, 226, 230 };
        std::vector<double> correctionBeamDivergence = { 0.0571, 0.0491, 0.0404, 0.0347, 0.0292, 0.0254, 0.0229, 0.0203, 0.0181, 0.0162, 0.0144, 0.0136, 0.0129, 0.0116, 0.0104, 0.00936, 0.00724, 0.00625, 0.00459, 0.00546, 0.00501, 0.00461, 0.00720, 0.00464, 0.00494, 0.00558, 0.00576, 0.00571, 0.00566, 0.00470 }; 
        beamDivergenceInterp.set_points(beamEnergyForBeamDivergence, correctionBeamDivergence, tk::spline::cspline);
        beamDivergenceInterp.make_monotonic();
    }

    /// @brief Default destructor.
    ~gtr2(){;}

    /// @brief Characterizes a beamlet from DICOM spot information.
    /// @param s A `spot` object from the DICOM data.
    /// @return A default-constructed `mqi::beamlet<T>` object.
    /// @note This function is not implemented for the GTR2 log file-based workflow.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) 
    {
        return mqi::beamlet<T>();
    }

    /// @brief Characterizes a beamlet from DICOM spot information with a specified source-to-isocenter distance.
    /// @param s A `spot` object from the DICOM data.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A default-constructed `mqi::beamlet<T>` object.
    /// @note This function is not implemented for the GTR2 log file-based workflow.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm)
    {
        return mqi::beamlet<T>();
    }

    /// @brief Characterizes a beamlet from two DICOM spots.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @return A default-constructed `mqi::beamlet<T>` object.
    /// @note This function is not implemented for the GTR2 log file-based workflow.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1)
    {
        return mqi::beamlet<T>();
    }

    /// @brief Calculates the number of histories for a spot based on log file data.
    ///
    /// This method converts the Monitor Units (MU) from the log file into the number of primary particles
    /// for the simulation, using an energy-dependent calibration curve.
    /// @param s The `logspot` object containing the MU count and energy.
    /// @return The number of particles (histories) to simulate for this spot.
    size_t
    characterize_history(
        const mqi::beam_module_ion::logspot& s)
    {
        int particleFromMUCount = s.muCount * protonPerDoseInterp(s.e) * dosePerMUCountInterp(s.e);
        return particleFromMUCount;
    }

    /// @brief Characterizes a beamlet from log file spot information.
    ///
    /// This method uses the energy-calibrated splines to define the beamlet's energy distribution,
    /// spot size, angular spread, and divergence based on the spot data from the machine log file.
    /// @param s A `logspot` object containing the spot's energy, position, and MU count.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return A `mqi::beamlet<T>` object representing the characterized spot.
    mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::logspot& s,
        const float                       source_to_isocenter_mm,
        const bool rsuse)
    {
        // Range shifter correction
        float newBeamStartingPos{ -source_to_isocenter_mm };
        if (rsuse) newBeamStartingPos = newBeamStartingPos - 40;

        // Spot beam's energy
        double energySpread = this->beamEnergySpreadInterp(s.e);

        // Gaussian energy spread distribution
        auto energy = new mqi::norm_1d<T>({ s.e }, { energySpread });

        // Caculate direction based on SAD and spot's position
        mqi::vec3<T> dir(std::atan(s.x/treatment_machine_ion<T>::SAD_[0]),
                         std::atan(s.y/treatment_machine_ion<T>::SAD_[1]),
                         -1.0);

        // Determine X,Y position of isocenter
        mqi::vec3<T> pos(0, 0, -newBeamStartingPos);
        pos.x = (treatment_machine_ion<T>::SAD_[0] - pos.z) * dir.x ;
        pos.y = (treatment_machine_ion<T>::SAD_[1] - pos.z) * dir.y ;

        // Spot size interpolation equation (70-230 MeV)
        double spotSize = this->beamSpotSizeInterp(s.e);

        // Angular spread interpolation equation (70-230 MeV)
        double angularSpread = this->beamAngularSpreadInterp(s.e);

        // Divergence interpolation equation (70-230 MeV)
        double divergence = this->beamDivergenceInterp(s.e);

        //Define phsp distribution
        std::array<T,6> beamlet_mean = { pos.x, pos.y, pos.z, dir.x, dir.y, dir.z };
        std::array<T,6> beamlet_sigm = { spotSize , spotSize, 0, angularSpread, angularSpread, 0};
        std::array<T,2> beamlet_divergence = { divergence, divergence };
        auto beamlet = new mqi::phsp_6d_ray<T>(beamlet_mean, beamlet_sigm, beamlet_divergence, newBeamStartingPos);

        return mqi::beamlet<T>(energy, beamlet);
    }

    /// @brief Characterizes the range shifter based on DICOM data.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::rangeshifter` object.
    mqi::rangeshifter*
    characterize_rangeshifter(
        const mqi::dataset* ds,
        mqi::modality_type m)
    {
        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        //1. rangeshifter sequence
        auto  rs_ds = (*ds)( seq_tags->at("rs")) ;
        assert(rs_ds.size() >=1);

       //2. Snout position from control point 0
        std::vector<float> ftmp;
        auto layer0    = (*ds)(seq_tags->at("ctrl"))[0]; //layer0 for snout position
        layer0->get_values( "SnoutPosition", ftmp);

        mqi::vec3<float>    lxyz(400.0, 400.0, 0.0);
        mqi::vec3<float>    pxyz(0.0, 0.0, ftmp[0]);
        mqi::mat3x3<float>  rxyz(0.0, 0.0, 0.0);

        //3. There must be at least one range shifter sequence
        for(auto r : rs_ds)
        {
            std::vector<std::string> rs_id(0);
            r->get_values("RangeShifterID", rs_id);

            std::cout<< "RangeShifterID detected.. : " << rs_id[0] << std::endl;
            rs_id[0].erase(std::remove(rs_id[0].begin(), rs_id[0].end(), ' '), rs_id[0].end()); // Erase blank

            if (rs_id[0] == "SNOUT_DEG_B") lxyz.z = 39.37;
            else lxyz.z = 0.0;
            assert(lxyz.z > 0);
        }

        pxyz.z += lxyz.z / 2;
        std::cout << "Range shifter thickness determined.. : " << lxyz.z <<" (mm) and position: " << pxyz.z <<" (mm)" << std::endl;
        pxyz.dump();

        return new mqi::rangeshifter(lxyz, pxyz, rxyz);
    }

    /// @brief Characterizes the aperture based on DICOM data.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::aperture` object.
    mqi::aperture*
    characterize_aperture(
        const mqi::dataset* ds,
        mqi::modality_type m
    ){
        auto xypts = this->characterize_aperture_opening(ds,m);

        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        //1. block sequence
        auto  blk_ds = (*ds)( seq_tags->at("blk")) ;
        assert(blk_ds.size() >=1);

        std::vector<float> ftmp;

        //note:
        // 1. Assumed opening points are physical points of geometry.
        //    (there must be tag for this) => this should be taken cared by users
        //apt_lunder -> set_dimension(150.0, 67.5);
        //400 is temporal
        blk_ds[0]->get_values( "BlockThickness", ftmp);
        mqi::vec3<float> lxyz(400.0, 400.0, ftmp[0]);

        //I omitted reading patient_side or snout_side
        blk_ds[0]->get_values("IsocenterToBlockTrayDistance", ftmp);
        mqi::vec3<float> pxyz(0.0, 0.0, ftmp[0]);

        mqi::mat3x3<float> rxyz(0.0, 0.0, 0.0);

        return new mqi::aperture(xypts, lxyz, pxyz, rxyz);
    }

    /// @brief Creates a beam source model from log file data.
    ///
    /// This implementation is specific to the Samsung Medical Center's log file format. It iterates
    /// through the parsed log file data, characterizes each spot, and appends the resulting
    /// beamlets to the beam source.
    /// @param logfileData A struct containing data parsed from the log files.
    /// @param pcoord The coordinate system of the beam geometry.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return A `mqi::beamsource<T>` object for the simulation.
    mqi::beamsource<T>
    create_beamsource(const mqi::logfiles_t& logfileData,
                      const mqi::coordinate_transform<T> pcoord,
                      const float source_to_isocenter_mm = 465.0,
                      const bool rsuse = false) 
    {
        // Definition of source to isocenter distance
        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;

        // Creating beam source with log file information
        mqi::beamsource<T> beamsource;

        for (int i = 0; i < logfileData.beamInfo.size(); i++)
        {
            for (int j = 0; j < logfileData.beamInfo[i].size(); j++)
            {
                mqi::beam_module_ion::logspot logSpotInfo;
                logSpotInfo.e = logfileData.beamEnergyInfo[i][j]; // Log file spot energy (single energy layer)

                // In single energy layer information
                for (int k = 0; k < logfileData.beamInfo[i][j].muCount.size(); k++)
                {
                    logSpotInfo.muCount = logfileData.beamInfo[i][j].muCount[k];
                    logSpotInfo.x = logfileData.beamInfo[i][j].posX[k];
                    logSpotInfo.y = logfileData.beamInfo[i][j].posY[k];

                    beamsource.append_beamlet_log(this->characterize_beamlet(logSpotInfo, treatment_machine<T>::source_to_isocenter_mm_, rsuse), this->characterize_history(logSpotInfo), pcoord);
                }
            }
        }
        return beamsource;
    }
};

}
//}
#endif
