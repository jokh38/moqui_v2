#ifndef MQI_BEAM_MODULE_RT_H
#define MQI_BEAM_MODULE_RT_H


/// \file mqi_beam_module_rt.hpp
///
/// \brief Interprets DICOM-RT (photon and electron) beam modules.
///
/// This file defines the `beam_module_rt` class, which is intended to handle
/// beam data from conventional radiotherapy plans (e.g., photons and electrons).
///
/// \see http://dicom.nema.org/dicom/2013/output/chtml/part03/sect_C.8.html#sect_C.8.8.14 for RT Plan
/// \note As of July 2019, there are no plans to fully implement this class. It exists
///       as a placeholder for future development.

#include <moqui/base/mqi_beam_module.hpp>

namespace mqi
{

/// \class beam_module_rt
/// \brief A class for handling photon and electron beams from DICOM-RT plans.
///
/// This class is a placeholder for interpreting beam data for conventional
/// radiotherapy modalities. It inherits from `beam_module` but does not yet
/// provide any specific implementation.
class beam_module_rt : public beam_module
{
public:
    /// \brief Constructs a beam module for conventional radiotherapy.
    ///
    /// \param d A pointer to the DICOM dataset.
    /// \param m The treatment modality type (e.g., PHOTON, ELECTRON).
    beam_module_rt(const mqi::dataset* d, mqi::modality_type m) : beam_module(d, m) {
        ;
    }
    /// \brief Destroys the beam_module_rt object.
    ~beam_module_rt() {
        ;
    }
};

}   // namespace mqi

#endif
