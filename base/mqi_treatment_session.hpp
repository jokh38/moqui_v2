#ifndef MQI_TREATMENT_SESSION_HPP
#define MQI_TREATMENT_SESSION_HPP

/// \file
///
/// Treatment session

#include <algorithm>
#include <type_traits>

#include "gdcmReader.h"

#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_ct.hpp>
#include <moqui/base/mqi_dataset.hpp>
#include <moqui/base/mqi_treatment_machine.hpp>
#include <moqui/base/mqi_treatment_machine_pbs.hpp>   //generic treatment machine
#include <moqui/treatment_machines/mqi_treatment_machine_smc_gtr2.hpp> // SMC PBS dedicated nozzle
#include <moqui/treatment_machines/mqi_treatment_machine_smc_gtr1.hpp> // SMC PBS multi-purpose nozzle
#include <moqui/base/mqi_utils.hpp>

namespace mqi
{

/// @class treatment_session
/// @brief Manages a radiotherapy treatment session, acting as the primary interface to the Monte Carlo engine.
///
/// This class is the main entry point for a simulation. It reads and parses RT-ION DICOM files
/// (Plan or Treatment Record), determines the modality, and instantiates the appropriate
/// treatment machine model. It provides methods to access various components of the
/// treatment plan, such as the beamline, beam source, and coordinate systems for each beam.
///
/// @tparam T The floating-point type for phase-space variables (e.g., `float` or `double`).
/// @note Future development may include patient and dose grid management within this class.
template<typename T>
class treatment_session
{
protected:
    ///< Sequence tag dictionary for modality-specific DICOM tags.
    const std::map<const std::string, const gdcm::Tag>* seq_tags_;

    ///< RT Modality type (e.g., RTPLAN, RTRECORD, IONPLAN, IONRECORD).
    mqi::modality_type mtype_;

    ///< Machine name, typically in "institution:name" format.
    std::string machine_name_;

    ///< Pointer to the instantiated treatment machine object.
    mqi::treatment_machine<T>* tx_machine_ = nullptr;

    ///< Pointer to the top-level DICOM dataset (RTIP or RTIBTR).
    mqi::dataset* mqi_ds_ = nullptr;

public:
    ///< Patient material properties.
    mqi::patient_material_t<T> material_;

    /// @brief Constructs a treatment session from a DICOM file.
    ///
    /// This constructor reads the specified DICOM file, determines the modality (e.g., IONPLAN, IONRECORD),
    /// parses the dataset, and creates the corresponding treatment machine model.
    ///
    /// @param file_for_tx_machine Path to the DICOM file (RTPLAN, IONPLAN, RTRECORD, IONRECORD).
    /// @param m_name Optional machine name. If empty, it's read from the DICOM file.
    /// @param mc_code The name and version of the Monte Carlo code being used (e.g., "TOPAS 3.5").
    /// @param gantryNum The gantry number, used to select the specific nozzle/machine model.
    /// @note Currently, only IONPLAN and IONRECORD modalities are fully supported.
    treatment_session(std::string file_for_tx_machine,   //for beamline and source,
                      std::string m_name  = "",
                      std::string mc_code = "TOPAS 3.5",
                      int gantryNum = 2) {

        // Read RT ION PLAN/RECORD with GDCM
        gdcm::Reader reader;
        reader.SetFileName(file_for_tx_machine.c_str());

        const bool is_file_valid = reader.Read();
        if (!is_file_valid)
            throw std::runtime_error("Invalid DICOM file is given to treatment_session.");

        // Check media storage modality
        // Declare mtype based on media storage name of DICOM file
        gdcm::MediaStorage ms;
        ms.SetFromFile(reader.GetFile());

        switch (ms) {
        case gdcm::MediaStorage::RTPlanStorage:
            mtype_ = RTPLAN;
            break;
        case gdcm::MediaStorage::RTIonPlanStorage:
            mtype_ = IONPLAN;
            break;
        case gdcm::MediaStorage::RTIonBeamsTreatmentRecordStorage:
            mtype_ = IONRECORD;
            break;
        default:
            throw std::runtime_error("treatment_session does not supports given RTMODALITY");
        }

        // Get dataset and modality
        mqi_ds_   = new mqi::dataset(reader.GetFile().GetDataSet(), true);
        seq_tags_ = &mqi::seqtags_per_modality.at(mtype_);

        ///< machine name is set
        // If there is not beam name in function, use DICOM tag (beam name) in the file.
        if (!m_name.compare("")) {
            ///< we assume machine name is in 1-st beam sequence
            ///< This part doesn't work for RTIBTR
            const mqi::dataset* beam_ds;
            if (mtype_ == IONPLAN) {
                beam_ds = (*mqi_ds_)(seq_tags_->at("beam"))[0];
            } else if (mtype_ == IONRECORD) {
                beam_ds = (*mqi_ds_)(seq_tags_->at("machine"))[0];
            } else {
                throw std::runtime_error(
                  "treatment_session can't find machine name and institute.");
            }

            std::vector<std::string> in;
            beam_ds->get_values("InstitutionName", in);
            std::vector<std::string> bn;
            beam_ds->get_values("TreatmentMachineName", bn);
            assert(bn.size() > 0);

            machine_name_ = (in.size() == 0) ? "" : mqi::trim_copy(in[0]);
            machine_name_ += ":" + mqi::trim_copy(bn[0]);

        } else {
            machine_name_ = m_name;
        }

        if (!this->create_machine(machine_name_, mc_code, gantryNum)) {
            std::runtime_error("No MC machine is registered for " + machine_name_);
        }
    }

    /// @brief Creates and configures the treatment machine model.
    ///
    /// This method instantiates the correct machine model (e.g., `mqi::gtr1`, `mqi::gtr2`)
    /// based on the provided machine name and gantry number.
    /// @param machine_name The name of the machine (e.g., "SMC:GTR1").
    /// @param mc_code The MC engine being used.
    /// @param gantryNum The gantry number (1 or 2) to select the specific model.
    /// @return `true` if the machine was created successfully, `false` otherwise.
    bool
    create_machine(std::string machine_name, std::string mc_code, int gantryNum) {
        if (tx_machine_) throw std::runtime_error("Preexisting machine.");
        const size_t deli = machine_name.find(":");

        std::string site = machine_name.substr(0, deli);
        std::transform(site.begin(), site.end(), site.begin(), ::tolower);

        std::string model = machine_name.substr(deli + 1, machine_name.size());

        std::cout << "Creating treatment machine model.. : Machine name --> " << model << std::endl;
        std::cout << "Creating treatment machine model.. : MC code version --> " << mc_code << std::endl;
        std::cout << "Creating treatment machine model.. : Creating SMC treatment machine model.." << std::endl;

        if (gantryNum == 1)
        {
            tx_machine_ = new mqi::gtr1<T>();
            material_   = mqi::gtr1_material_t<T>();
        }
        else if (gantryNum == 2)
        {
            tx_machine_ = new mqi::gtr2<T>();
            material_   = mqi::gtr2_material_t<T>();
        }
        
        // OLD spot scanning PBS version
        // if (site.compare("pbs"))
        //     std::transform(model.begin(), model.end(), model.begin(), ::tolower);

        // SMC don't have machine name as pbs
        // If pbs
        // if (!site.compare("pbs")) {
        //     //Generic PBS beam model
        //     //expecting file
        //     std::cout << "Creating a generic PBS machine from : " << model << "\n";
        //     tx_machine_ = new mqi::pbs<T>(model);
        //     material_   = mqi::pbs_material_t<T>();
        //     return true;
        // } else {
        //     throw std::runtime_error("Valid machine is not available.");
        // }

        return true;
    }

    /// @brief Destructor. Cleans up the treatment machine and dataset objects.
    ~treatment_session() {
        delete tx_machine_;
        delete mqi_ds_;
    }

    /// @brief Retrieves a list of all beam names from the treatment plan.
    /// @return A `std::vector<std::string>` containing the names of all beams.
    std::vector<std::string>
    get_beam_names() {
        auto                     beam_sequence = (*mqi_ds_)(seq_tags_->at("beam"));
        std::vector<std::string> beam_names;
        for (const auto& beam : beam_sequence) {
            std::vector<std::string> tmp;
            beam->get_values("BeamName", tmp);
            beam_names.push_back(mqi::trim_copy(tmp[0]));
        }
        return beam_names;
    }

    /// @brief Gets the total number of beams in the treatment plan.
    /// @return The number of beams.
    /// @note This currently counts all beams, including setup beams.
    int
    get_num_beams() {
        auto beam_sequence = (*mqi_ds_)(seq_tags_->at("beam"));
        /// TODO: we need to count 'treatment' beam only. drop or not to use 'setup' beam.
        int count = 0;
        for (const auto& beam : beam_sequence) {
            count += 1;
        }
        return count;
    }

    /// @brief Gets the number of fractions planned.
    /// @return The total number of fractions.
    int
    get_fractions() {
        auto             fraction_sequence = (*mqi_ds_)(gdcm::Tag(0x300a, 0x0070));
        std::vector<int> n_fractions;
        for (auto fraction : fraction_sequence) {
            fraction->get_values("NumberOfFractionsPlanned", n_fractions);
        }
        return n_fractions[0];
    }

    /// @brief Retrieves the name of a beam given its number.
    /// @param bnb The beam number.
    /// @return The name of the corresponding beam.
    std::string
    get_beam_name(int bnb) {
        std::string beam_name;
        auto        bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {

            std::vector<int>         bn;
            std::vector<std::string> bname;
            i->get_values("BeamNumber", bn);   //works only for RTIP
            assert(bn.size() > 0);
            if (bnb == bn[0]) 
            {
                i->get_values("BeamName", bname);
                beam_name = mqi::trim_copy(bname[0]);
                return beam_name;
            }
        }
        throw std::runtime_error("Invalid beam number.");
    }

    /// @brief Retrieves the DICOM dataset for a beam given its name.
    /// @param bnm The name of the beam.
    /// @return A constant pointer to the `mqi::dataset` for the specified beam.
    const mqi::dataset*
    get_beam_dataset(std::string bnm) {
        auto bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {
            std::vector<std::string> bn;
            i->get_values("BeamName", bn);   //works for RTIP & RTIBTR
            if (bn.size() == 0) continue;
            if (bnm == mqi::trim_copy(bn[0])) { return i; }
        }
        throw std::runtime_error("Invalid beam name.");
    }

    /// @brief Retrieves the DICOM dataset for a beam given its number.
    /// @param bnb The number of the beam.
    /// @return A constant pointer to the `mqi::dataset` for the specified beam.
    const mqi::dataset*
    get_beam_dataset(int bnb) {
        auto bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {
            //i->dump();
            std::vector<int> bn;
            i->get_values("BeamNumber", bn);   //works only for RTIP
            assert(bn.size() > 0);
            if (bnb == bn[0]) { return i; }
        }
        throw std::runtime_error("Invalid beam number.");
    }

    /// @brief Gets the beamline object for a specific beam.
    /// @tparam S The type of the beam identifier (e.g., `std::string` for name, `int` for number).
    /// @param beam_id The identifier of the beam.
    /// @return The `mqi::beamline<T>` object for the specified beam.
    template<typename S>
    mqi::beamline<T>
    get_beamline(S beam_id) {
        if (mtype_ == 1) std::cout << "Creating beam line object.. : RT Plan type --> RT ION" << std::endl;
        std::cout << "Creating beam line object.. : Beam ID --> " << beam_id << std::endl;
        return tx_machine_->create_beamline(this->get_beam_dataset(beam_id), mtype_);
    }

    /// @brief Gets the beam source object for a specific beam.
    /// @tparam S The type of the beam identifier (e.g., `std::string` for name, `int` for number).
    /// @param beam_id The identifier of the beam.
    /// @param coord The coordinate transformation to apply.
    /// @param scale The scaling factor for calculating the number of histories.
    /// @param sid The source-to-isocenter distance in mm.
    /// @return The `mqi::beamsource<T>` object for the specified beam.
    template<typename S>
    mqi::beamsource<T>
    get_beamsource(S beam_id, const mqi::coordinate_transform<T> coord, float scale, T sid) {
        return tx_machine_->create_beamsource(
          this->get_beam_dataset(beam_id), mtype_, coord, scale, sid);
    }

    /// @brief Gets the beam source object for a given beam dataset.
    /// @param beam A pointer to the beam's `mqi::dataset`.
    /// @param coord The coordinate transformation to apply.
    /// @param scale The scaling factor for calculating the number of histories.
    /// @param sid The source-to-isocenter distance in mm.
    /// @return The `mqi::beamsource<T>` object for the specified beam.
    mqi::beamsource<T>
    get_beamsource(mqi::dataset*                      beam,
                   const mqi::coordinate_transform<T> coord,
                   float                              scale,
                   T                                  sid) {
        return tx_machine_->create_beamsource(beam, mtype_, coord, scale, sid);
    }

    /// @brief Gets the beam source object from log file data.
    /// @param logfileData A struct containing data parsed from log files.
    /// @param coord The coordinate transformation to apply.
    /// @param sid The source-to-isocenter distance in mm.
    /// @param rsuse A boolean indicating whether a range shifter is used.
    /// @return The `mqi::beamsource<T>` object.
    /// @note Added by Chanil Jeon in 2023.
    mqi::beamsource<T>
    get_beamsource(mqi::logfiles_t&                   logfileData,
                   const mqi::coordinate_transform<T> coord,
                   T                                  sid,
                   const bool rsuse) {
        return tx_machine_->create_beamsource(logfileData, coord, sid, rsuse);
    }

    /// @brief Gets the timeline object for a specific beam.
    /// @tparam S The type of the beam identifier.
    /// @param beam_id The identifier of the beam.
    /// @return A map representing the beam's timeline.
    template<typename S>
    std::map<T, int32_t>
    get_timeline(S beam_id) {
        return tx_machine_->create_timeline(this->get_beam_dataset(beam_id), mtype_);
    }

    /// @brief Gets the coordinate transformation object for a specific beam.
    /// @tparam S The type of the beam identifier.
    /// @param beam_id The identifier of the beam.
    /// @return The `mqi::coordinate_transform<T>` object for the specified beam.
    template<typename S>
    mqi::coordinate_transform<T>
    get_coordinate(S beam_id) {
        return tx_machine_->create_coordinate_transform(this->get_beam_dataset(beam_id), mtype_);
    }

    /// @brief Dumps a summary of the plan to the console.
    void
    summary(void) {
        //plan_ds
        mqi_ds_->dump();
    }

    /// @brief Returns the modality type of the current session.
    /// @return The `mqi::modality_type` enum value.
    mqi::modality_type
    get_modality_type(void) {
        return mtype_;
    }

    /// @brief Retrieves a specific data element from the top-level dataset.
    /// @param tag The `gdcm::Tag` of the data element to retrieve.
    /// @return A constant reference to the `gdcm::DataElement`.
    const gdcm::DataElement&
    get_dataelement(gdcm::Tag tag) {
        return mqi_ds_[0][tag];
    }
};

}   // namespace mqi

#endif
