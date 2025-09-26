#ifndef MQI_BEAM_MODULE_H
#define MQI_BEAM_MODULE_H

/// \file mqi_beam_module.hpp
///
/// \brief Defines the top-level abstraction for interpreting DICOM beam modules.
///
/// This file contains the declaration of the `beam_module` class, which serves as a
/// base class for handling different types of DICOM beam modules, such as those for
/// conventional radiotherapy (RT) and ion therapy (RT-Ion).

#include <moqui/base/mqi_dataset.hpp>

namespace mqi
{

/// \class beam_module
/// \brief A base class for interpreting DICOM beam data.
///
/// This class provides a common interface for interpreting beam-related data from a
/// DICOM dataset. It is designed to be subclassed for specific treatment modalities,
/// such as `beam_module_rt` for photons/electrons and `beam_module_rtion` for protons/ions.
/// The primary role of this class and its derivatives is to process control point
/// information and fluence maps to prepare them for use in a Monte Carlo simulation engine.
class beam_module
{
protected:
    /// \brief A pointer to the DICOM dataset containing the beam data.
    const mqi::dataset* ds_;
    /// \brief The treatment modality (e.g., PHOTON, ELECTRON, PROTON).
    const mqi::modality_type modality_;

public:
    /// \brief Constructs a new beam_module object.
    ///
    /// \param d A pointer to the DICOM dataset to be interpreted.
    /// \param m The treatment modality type.
    beam_module(const mqi::dataset* d, mqi::modality_type m) : ds_(d), modality_(m) {
        ;
    }

    /// \brief Destroys the beam_module object.
    virtual ~beam_module() {
        ;
    }

    /// \brief Dumps the contents of the beam module for debugging purposes.
    ///
    /// This is a virtual function that should be overridden by derived classes to
    /// print out the specific details of the beam module.
    virtual void
    dump() const {
        ;
    }
};

}   // namespace mqi

#endif
