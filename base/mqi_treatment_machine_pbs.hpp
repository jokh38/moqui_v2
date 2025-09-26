#ifndef MQI_TREATMENT_MACHINE_PBS_H
#define MQI_TREATMENT_MACHINE_PBS_H

/// \file
///
/// Treatment machine for pencil beam scanning machines

#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_treatment_machine_ion.hpp>
#include <moqui/base/mqi_utils.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <regex>
#include <sstream>

namespace mqi
{

/// @class pbs_material_t
/// @brief A material definition class for Pencil Beam Scanning (PBS) simulations.
///
/// This class extends `patient_material_t` and can be used to define materials
/// based on Hounsfield Units (HU).
///
/// @tparam R The floating-point type for material properties (e.g., `float` or `double`).
template<typename R>
class pbs_material_t : public patient_material_t<R>
{
public:
    /// @brief Default constructor.
    CUDA_HOST_DEVICE
    pbs_material_t() : patient_material_t<R>() {
        ;
    }

    /// @brief Constructor initializing the material from a Hounsfield Unit value.
    /// @param hu The Hounsfield Unit (HU) value.
    CUDA_HOST_DEVICE
    pbs_material_t(int16_t hu) : patient_material_t<R>(hu) {
        ;
    }

    /// @brief Default destructor.
    CUDA_HOST_DEVICE
    ~pbs_material_t() {
        ;
    }
};

/// @class pbs
/// @brief A generic Pencil Beam Scanning (PBS) treatment machine model.
///
/// This class implements the `treatment_machine_ion` interface for a PBS system.
/// It reads beam model specifications, geometry, and timing information from a configuration file.
/// It is responsible for characterizing beamlets, apertures, and range shifters based on this data.
///
/// @tparam T The floating-point type for phase-space variables (e.g., `float` or `double`).
template<typename T>
class pbs : public treatment_machine_ion<T>
{
protected:
    const T cm2mm;
    const T mm2cm;

    const T nA      = 1e-9;
    const T sec     = 1.0;
    const T ms      = 1e-3 * sec;
    const T us      = 1e-6 * sec;
    const T ns      = 1e-9 * sec;
    const T mm      = 1.0;
    const T m       = 1000.0 * mm;
    const T const_e = 1.6021766208e-19;

    /// @struct spot_specification
    /// @brief Holds the parameters defining a single spot in the beam model.
    struct spot_specification {
        T E;     ///< Nominal energy.
        T dE;    ///< Energy spread.
        T x;     ///< Spot size in x-direction.
        T y;     ///< Spot size in y-direction.
        T xp;    ///< Divergence in x-direction.
        T yp;    ///< Divergence in y-direction.
        T ratio; ///< Conversion ratio (e.g., meterset to particles).
    };
    ///< Map of beam data, keyed by energy.
    std::map<T, spot_specification> beamdata_;

    /// @struct geometry_specification
    /// @brief Holds the geometric parameters of the treatment machine.
    std::map<std::string, T> rangeshifter_;

    /// @struct geometry_specification
    /// @brief Holds the geometric parameters of the treatment machine.
    struct geometry_specification {
        std::array<T, 2> SAD;                  ///< Source-to-Axis Distance in x and y.
        std::array<T, 2> rangeshifter;         ///< Dimensions of the range shifter.
        T                rangeshifter_snout_gap; ///< Gap between range shifter and snout.
        std::array<T, 2> aperture;             ///< Dimensions of the aperture.
    } geometry_spec_;

    /// @struct time_specification
    /// @brief Holds the parameters related to beam delivery timing.
    struct time_specification {
        T i;       ///< Beam current (nA).
        T e;       ///< Layer switching time (sec).
        T dt_down; ///< Latency down (ms).
        T dt_up;   ///< Latency up (ms).
        T set_x;   ///< Settling time for x-scanning (ms).
        T set_y;   ///< Settling time for y-scanning (ms).
        T vx;      ///< Scanning speed in x (m/sec).
        T vy;      ///< Scanning speed in y (m/sec).
    } time_spec_;

    ///< Linear interpolation function.
    std::function<T(T, T, T, T, T)> intpl = [](T x, T x0, T x1, T y0, T y1) {
        return (x1 == x0) ? y0 : y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    };

public:
    /// @brief Default constructor. Initializes SAD to infinity.
    pbs() : cm2mm(10.0), mm2cm(0.1) {
        treatment_machine_ion<T>::SAD_[0] = std::numeric_limits<T>::infinity();
        treatment_machine_ion<T>::SAD_[1] = std::numeric_limits<T>::infinity();
    }

    /// @brief Constructor that loads beam data from a file.
    /// @param filename The path to the beam model configuration file.
    pbs(const std::string filename) : cm2mm(10.0), mm2cm(0.0) {
        this->load_beamdata(filename);
        if (geometry_spec_.SAD[0] == 0)
            treatment_machine_ion<T>::SAD_[0] = std::numeric_limits<T>::infinity();
        if (geometry_spec_.SAD[1] == 0)
            treatment_machine_ion<T>::SAD_[1] = std::numeric_limits<T>::infinity();
    }

    /// @brief Default destructor.
    ~pbs() {
        ;
    }

    /// @brief Calculates the number of histories for a uniform or modulated-spec beamlet.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @param scale The scaling factor (particles per history).
    /// @return The number of histories to simulate.
    /// @note Not yet implemented as the focus is on spot-scanning. Returns 0.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1,
                         float                             scale) {
        return 0;
    }

    /// @brief Calculates the number of histories for a modulated beamlet.
    ///
    /// This method interpolates the beam data to find the meterset-to-particle ratio for the given energy.
    /// @param s The spot information.
    /// @param scale The scaling factor (particles per history).
    /// @return The number of histories to simulate.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s, float scale) {
        auto beam_up = beamdata_.lower_bound(s.e);
        if (beam_up == beamdata_.begin()) std::cerr << "out-of-bound\n";
        auto beam_down = std::prev(beam_up, 1);
        auto down      = beam_down->second;
        auto up        = beam_up->second;
        T    mid_ratio = intpl(s.e, beam_down->first, beam_up->first, down.ratio, up.ratio);
        return s.meterset * mid_ratio / scale;
    }

    /// @brief Characterizes a beamlet for continuous-scan uniform beam delivery.
    ///
    /// This method creates a `phsp_6d_fanbeam` distribution to sample particle positions and directions
    /// between two given spots.
    /// @param s0 The starting spot.
    /// @param s1 The ending spot.
    /// @return A `mqi::beamlet<T>` object representing the continuous scan segment.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1) {
        //energy distribution
        auto beam_up = beamdata_.lower_bound(s0.e);
        if (beam_up == beamdata_.begin()) std::cerr << "out-of-bound\n";
        auto beam_down = std::prev(beam_up, 1);
        auto down      = beam_down->second;
        auto up        = beam_up->second;
        T    mid_e     = intpl(s0.e, beam_down->first, beam_up->first, down.E, up.E);
        T    mid_de    = intpl(s0.e, down.E, up.E, down.dE, up.dE);
        T    mid_x     = intpl(s0.e, down.E, up.E, down.x, up.x);
        T    mid_y     = intpl(s0.e, down.E, up.E, down.y, up.y);
        T    mid_xp    = intpl(s0.e, down.E, up.E, down.xp, up.xp);
        T    mid_yp    = intpl(s0.e, down.E, up.E, down.yp, up.yp);
        //        std::cout<<"Energy: "<< s.e <<", " << down.E << ", " << up.E << ", "<<mid_e<< "\n";
        auto energy = new mqi::norm_1d<T>({ mid_e }, { mid_de });

        mqi::vec3<T> dir0(std::atan(s0.x / treatment_machine_ion<T>::SAD_[0]),
                          std::atan(s0.y / treatment_machine_ion<T>::SAD_[1]),
                          -1.0);
        mqi::vec3<T> dir1(std::atan(s1.x / treatment_machine_ion<T>::SAD_[0]),
                          std::atan(s1.y / treatment_machine_ion<T>::SAD_[1]),
                          -1.0);

        mqi::vec3<T> pos0(0, 0, mqi::treatment_machine_ion<T>::source_to_isocenter_mm_);
        pos0.x = (treatment_machine_ion<T>::SAD_[0] - pos0.z) * dir0.x;
        pos0.y = (treatment_machine_ion<T>::SAD_[1] - pos0.z) * dir0.y;

        mqi::vec3<T> pos1(0, 0, mqi::treatment_machine_ion<T>::source_to_isocenter_mm_);
        pos1.x = (treatment_machine_ion<T>::SAD_[0] - pos1.z) * dir1.x;
        pos1.y = (treatment_machine_ion<T>::SAD_[1] - pos1.z) * dir1.y;

        std::array<T, 6> spot_pos_range = { pos0.x, pos1.x, pos0.y, pos1.y, pos0.z, pos1.z };
        std::array<T, 6> spot_sigma     = { mid_x, mid_y, 0.0, mid_xp, mid_yp, 0.0 };
        std::array<T, 2> corr           = { 0.0, 0.0 };

        //spot direction mean is calculated from the sampled position between x0 and x1
        //so passing spot-range to a specialized distribution for handling this case is prefered option.
        auto spot_ = new mqi::phsp_6d_fanbeam<T>(
          spot_pos_range, spot_sigma, corr, treatment_machine_ion<T>::SAD_);

        return mqi::beamlet<T>(energy, spot_);
    }

    /// @brief Characterizes a beamlet for a modulated (spot-scanning) beam.
    ///
    /// This method interpolates the machine's beam data to define the properties of a single spot beamlet,
    /// including energy, position, and divergence.
    /// @param s The spot information from the treatment plan.
    /// @return A `mqi::beamlet<T>` object for the spot.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) {

        //energy distribution
        auto beam_up = beamdata_.lower_bound(s.e);
        if (beam_up == beamdata_.begin()) std::cerr << "out-of-bound\n";
        auto beam_down = std::prev(beam_up, 1);
        auto down      = beam_down->second;
        auto up        = beam_up->second;
        T    mid_e     = intpl(s.e, beam_down->first, beam_up->first, down.E, up.E);
        T    mid_de    = intpl(s.e, down.E, up.E, down.dE, up.dE);
        T    mid_x     = intpl(s.e, down.E, up.E, down.x, up.x);
        T    mid_y     = intpl(s.e, down.E, up.E, down.y, up.y);
        T    mid_xp    = intpl(s.e, down.E, up.E, down.xp, up.xp);
        T    mid_yp    = intpl(s.e, down.E, up.E, down.yp, up.yp);

        auto energy = new mqi::norm_1d<T>({ mid_e }, { mid_de });

        //X-Y position at at z
        mqi::vec3<T> iso(s.x, s.y, 0);
        mqi::vec3<T> beam =
          this->beam_starting_position(iso, mqi::treatment_machine_ion<T>::source_to_isocenter_mm_);
        mqi::vec3<T> dir = iso - beam;
        dir.normalize();

        if (beam.z > 0) {
            //Update spot_size at source_to_isocenter
            //update energy at source_to_isocenter
        }

        //Define phsp distribution
        std::array<T, 6> beamlet_mean = { beam.x, beam.y, beam.z, dir.x, dir.y, dir.z };
        std::array<T, 6> beamlet_sigm = { mid_x, mid_y, 0.0, mid_xp, mid_yp, 0.0 };
        std::array<T, 2> corr         = { 0.0, 0.0 };
        auto             beamlet      = new mqi::phsp_6d<T>(beamlet_mean, beamlet_sigm, corr);

        return mqi::beamlet<T>(energy, beamlet);
    }

    /// @brief Characterizes a beamlet for a modulated beam with a specified source-to-isocenter distance.
    /// @param s The spot information from the treatment plan.
    /// @param source_to_isocenter_mm The distance from the source to the isocenter in mm.
    /// @return A `mqi::beamlet<T>` object for the spot.
    mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm = 390.0) {
        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;
        //energy distribution
        auto beam_up = beamdata_.lower_bound(s.e);
        if (beam_up == beamdata_.begin()) std::cerr << "out-of-bound\n";
        auto beam_down = std::prev(beam_up, 1);
        auto down      = beam_down->second;
        auto up        = beam_up->second;
        T    mid_e     = intpl(s.e, beam_down->first, beam_up->first, down.E, up.E);
        T    mid_de    = intpl(s.e, down.E, up.E, down.dE, up.dE);
        T    mid_x     = intpl(s.e, down.E, up.E, down.x, up.x);
        T    mid_y     = intpl(s.e, down.E, up.E, down.y, up.y);
        T    mid_xp    = intpl(s.e, down.E, up.E, down.xp, up.xp);
        T    mid_yp    = intpl(s.e, down.E, up.E, down.yp, up.yp);

        auto energy = new mqi::norm_1d<T>({ mid_e }, { mid_de });

        //X-Y position at at z
        mqi::vec3<T> iso(s.x, s.y, 0);
        mqi::vec3<T> beam =
          this->beam_starting_position(iso, mqi::treatment_machine_ion<T>::source_to_isocenter_mm_);
        mqi::vec3<T> dir = iso - beam;
        dir.normalize();

        if (beam.z > 0) {
            //Update spot_size at source_to_isocenter
            //update energy at source_to_isocenter
        }

        //Define phsp distribution
        std::array<T, 6> beamlet_mean = { beam.x, beam.y, beam.z, dir.x, dir.y, dir.z };
        std::array<T, 6> beamlet_sigm = { mid_x, mid_y, 0.0, mid_xp, mid_yp, 0.0 };
        std::array<T, 2> corr         = { 0.0, 0.0 };
        auto             beamlet      = new mqi::phsp_6d<T>(beamlet_mean, beamlet_sigm, corr);

        return mqi::beamlet<T>(energy, beamlet);
    }

    /// @brief Characterizes a range shifter based on DICOM data and machine geometry.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::rangeshifter` object.
    mqi::rangeshifter*
    characterize_rangeshifter(const mqi::dataset* ds, mqi::modality_type m) {
        auto seq_tags = &mqi::seqtags_per_modality.at(m);
        //1. rangeshifter sequence
        auto rs_ds = (*ds)(seq_tags->at("rs"));
        assert(rs_ds.size() >= 1);

        //2. layer0 for snout and rangeshifter setting sequence
        auto layer0 = (*ds)(seq_tags->at("ctrl"))[0];   //layer0 for snout position

        mqi::vec3<T>   lxyz;
        mqi::vec3<T>   pxyz;
        mqi::mat3x3<T> rxyz(0.0, 0.0, 0.0);

        lxyz.x = geometry_spec_.rangeshifter[0];
        lxyz.y = geometry_spec_.rangeshifter[1];   //if this is 0, rangeshifter is circle

        if (rangeshifter_.size() == 0) {
            std::vector<float> ftmp;
            std::cout << "Thickness is defined from RangeShifter Setting sequence\n";
            auto rs_ss0 = (*layer0)(seq_tags->at("rsss"))[0];

            //% If MDACC simulates non-water material for rangeshifter,
            //better to divide density ration for the material thickness.
            rs_ss0->get_values("RangeShifterWaterEquivalentThickness", ftmp);
            ///!!!! TODO
            lxyz.z = ftmp[0] / 1.15;   //needs to check later

            //? distance is center to center or to downside? upside?
            rs_ss0->get_values("IsocenterToRangeShifterDistance", ftmp);
            pxyz.z = ftmp[0];

            pxyz.z -= lxyz.z;

        } else {

            std::cout << "Thickness is defined from Rangeshifter ID\n";
            std::vector<float> ftmp;
            layer0->get_values("SnoutPosition", ftmp);
            pxyz.z = ftmp[0];

            //3. There must be at least one range shifter sequence
            for (auto r : rs_ds) {
                std::vector<std::string> rs_id(0);
                r->get_values("RangeShifterID", rs_id);
                auto it = rangeshifter_.find(mqi::trim_copy(rs_id[0]));
                if (it != rangeshifter_.end()) lxyz.z += it->second;
            }
            assert(lxyz.z > 0);
            pxyz.z -= (lxyz.z * 0.5 + geometry_spec_.rangeshifter_snout_gap);
        }
        std::cout << "Range shifter thickness: " << lxyz.z << " (mm) and position: " << pxyz.z
                  << " (mm)" << std::endl;

        return new mqi::rangeshifter(lxyz, pxyz, rxyz);
    }

    /// @brief Characterizes the beam delivery time for a spot.
    ///
    /// This method calculates the beam-on time based on the spot's meterset and the beam current,
    /// and the beam-off time based on scanning speeds, settling times, and layer switching times.
    /// @param s_curr The current spot to be delivered.
    /// @param s_next The next spot in the sequence.
    /// @return An array containing the on-time and off-time in seconds.
    virtual std::array<T, 2>
    characterize_beamlet_time(const mqi::beam_module_ion::spot& s_curr,
                              const mqi::beam_module_ion::spot& s_next) {
        // dT_on: time to beam on
        // dT_off: time to beam off including layer switing, settling spot, latencies
        // s_curr.meterset is number of particles
        const T dT_on  = s_curr.meterset * const_e / (time_spec_.i * nA);
        T       dT_off = 0;

        /// calculate beam-off time after beam is delivered.
        /// except last beamlet
        if (&s_next != nullptr) {
            if (s_next.e != s_curr.e) {
                /// Layer changae.
                /// Note: The spot-settling time is not required or included in layer-switching time.
                dT_off = time_spec_.e * sec;
            } else {
                /// When a spot moves in same layer
                /// Calculate time required to move spot to next and add
                const T dTx = std::abs(s_next.x - s_curr.x) / (time_spec_.vx * m / sec);
                const T dTy = std::abs(s_next.y - s_curr.y) / (time_spec_.vy * m / sec);
                if (dTx > 0 || dTy > 0)
                    dT_off =
                      (dTy >= dTx) ? dTy + (time_spec_.set_y * ms) : dTx + (time_spec_.set_x * ms);
            }

            ///add latencies (up and down)
            dT_off += (time_spec_.dt_up * ms);
            dT_off += (time_spec_.dt_down * ms);
            ;
        }

        return { dT_on, dT_off };
    }

    /// @brief Characterizes an aperture based on DICOM data and machine geometry.
    /// @param ds Pointer to the dataset for one item of an `IonBeamSequence`.
    /// @param m The modality type.
    /// @return A pointer to a `mqi::aperture` object.
    mqi::aperture*
    characterize_aperture(const mqi::dataset* ds, mqi::modality_type m) {
        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        std::vector<std::string> stmp;
        std::vector<float>       ftmp;

        ///< 1. block sequence
        auto blk_ds = (*ds)(seq_tags->at("blk"));
        assert(blk_ds.size() >= 1);
        blk_ds[0]->get_values("BlockThickness", ftmp);
        mqi::vec3<float> lxyz(
          geometry_spec_.aperture[0], geometry_spec_.aperture[1], ftmp[0]);   //360 or 0?

        blk_ds[0]->get_values("IsocenterToBlockTrayDistance", ftmp);
        mqi::vec3<float> pxyz(0.0, 0.0, ftmp[0] + lxyz.z * 0.5);
        std::cout << "BlockThickness and Position (center) : " << lxyz.z << ", " << pxyz.z << " mm"
                  << std::endl;
        mqi::mat3x3<float> rxyz(0.0, 0.0, 0.0);
        //3. x-y points for opening
        auto xypts = this->characterize_aperture_opening(ds, m);
        return new mqi::aperture(xypts, lxyz, pxyz, rxyz, false);
    }

    /// @brief Loads machine parameters from a configuration file.
    ///
    /// This method parses a text file containing sections for geometry, range shifter thickness,
    /// spot characteristics, and timing parameters.
    /// @param f The path to the configuration file.
    void
    load_beamdata(const std::string f) {
        const std::regex section_header("(.*)\\[(.*)\\](.*)");
        std::ifstream    file(f);

        std::string line;
        while (std::getline(file, line)) {
            //check '#' and ignore
            line = line.substr(0, line.find_first_of('#'));
            if (line.size() == 0) continue;

            //Geometry section
            if (line.compare("[geometry]") == 0) {
                std::cout << line << "\n";
                while (std::getline(file, line)) {
                    line = line.substr(0, line.find_first_of('#'));
                    if (line.size() == 0) continue;
                    if (std::regex_match(line, section_header)) break;

                    std::stringstream data(line);
                    std::string       key;
                    T                 val1 = 0.0;
                    T                 val2 = 0.0;
                    data >> key >> val1 >> val2;
                    if (key.compare("SAD(mm)") == 0) {
                        geometry_spec_.SAD[0] = val1;
                        geometry_spec_.SAD[1] = val2;
                    };

                    if (key.compare("rangeshifter(mm)") == 0) {
                        geometry_spec_.rangeshifter[0] = val1;
                        geometry_spec_.rangeshifter[1] = val2;
                    }

                    if (key.compare("rangeshifter_snout_gap(mm)") == 0) {
                        geometry_spec_.rangeshifter_snout_gap = val1;
                    }

                    if (key.compare("aperture(mm)") == 0) {
                        geometry_spec_.aperture[0] = val1;
                        geometry_spec_.aperture[1] = val2;
                    }

                    std::cout << key << " : " << val1 << ", " << val2 << "\n";
                }
            }   //geometry

            //Range-shifter thickness
            if (line.compare("[rangeshifter_thickness]") == 0) {
                std::cout << line << "\n";
                while (std::getline(file, line)) {
                    line = line.substr(0, line.find_first_of('#'));
                    if (line.size() == 0) continue;
                    if (std::regex_match(line, section_header)) break;

                    //Unfortunately, std::quoted(in) available in C++14
                    size_t      from = line.find_first_of('"');
                    size_t      to   = line.find_last_of('"');
                    std::string k(line.substr(from + 1, to - 1));
                    T           thickness = std::stof(line.substr(to + 1, line.size()));
                    std::cout << k << ": " << thickness << "\n";
                    rangeshifter_.insert(std::make_pair(k, thickness));
                }

            }   //rangeshifter

            //Spot section
            if (line.compare("[spot]") == 0) {
                std::cout << line << "\n";

                while (std::getline(file, line)) {
                    line = line.substr(0, line.find_first_of('#'));
                    if (line.size() == 0) continue;
                    if (std::regex_match(line, section_header)) break;

                    std::stringstream  data(line);
                    spot_specification s;
                    T                  e;
                    data >> e >> s.E >> s.dE >> s.x >> s.y >> s.xp >> s.yp >> s.ratio;

                    beamdata_.insert(std::make_pair(e, s));
                    /*
					//print out
                    std::cout<< e <<": " <<
                        s.E <<", " << s.dE <<", " <<
                        s.x <<", " << s.y  <<", " <<
                        s.xp <<", " << s.yp <<", " <<
                        s.ratio <<"\n";
					*/
                }
            }   //spot

            //Time section
            if (line.compare("[time]") == 0) {
                std::cout << line << "\n";
                while (std::getline(file, line)) {
                    line = line.substr(0, line.find_first_of('#'));
                    if (line.size() == 0) continue;
                    if (std::regex_match(line, section_header)) break;

                    std::stringstream data(line);
                    std::string       k;
                    T                 val;
                    data >> k >> val;
                    if (k.compare("E(sec)") == 0) time_spec_.e = val;
                    if (k.compare("I(nA)") == 0) time_spec_.i = val;
                    if (k.compare("dT_up(ms)") == 0) time_spec_.dt_up = val;
                    if (k.compare("dT_down(ms)") == 0) time_spec_.dt_down = val;
                    if (k.compare("set_x(ms)") == 0) time_spec_.set_x = val;
                    if (k.compare("set_y(ms)") == 0) time_spec_.set_y = val;
                    if (k.compare("velocity_x(m/sec)") == 0) time_spec_.vx = val;
                    if (k.compare("velocity_y(m/sec)") == 0) time_spec_.vy = val;
                    //read-in time variables
                }
            }   //time
        }       //while(readline)
    }
};

}   // namespace mqi
#endif
