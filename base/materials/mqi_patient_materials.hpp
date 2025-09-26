/// \file
///
/// \brief Defines the `mqi::patient_material_t` class for converting Hounsfield Units (HU) to material properties.
///
/// In medical imaging, particularly in CT scans, Hounsfield Units are used to represent
/// the radiodensity of tissues. This class provides the functionality to convert these
/// HU values into physical properties like mass density, which are essential for
#ifndef MQI_PATIENT_MATERIAL_HPP
#define MQI_PATIENT_MATERIAL_HPP

#include <cassert>
#include <moqui/base/materials/mqi_material.hpp>
#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_physics_constants.hpp>

namespace mqi
{

/// \class patient_material_t
/// \brief Represents patient-specific materials based on CT Hounsfield Units (HU).
///
/// \tparam R The floating-point type (e.g., `float` or `double`) used for calculations.
///           This is a C++ feature called a "template," which allows the class to be
///           flexible and work with different levels of precision. It's similar to
///           how you might use type hints for generics in Python.
///
/// This class models patient tissues by converting HU values from a CT scan into
/// mass density. This is crucial for accurately simulating how radiation interacts
/// with different tissues in the body. It inherits from `h2o_t`, meaning it uses
/// water as a base material and adjusts its properties based on the HU value.
template<typename R>
class patient_material_t : public h2o_t<R>
{

public:
    /// \brief Default constructor.
    ///
    /// \note The `CUDA_HOST_DEVICE` macro indicates that this function can be
    ///       executed on both the CPU (the "host") and a CUDA-enabled GPU (the "device").
    ///       This is a key feature for high-performance computing in Moqui.
    CUDA_HOST_DEVICE
    patient_material_t() : h2o_t<R>() {
        ;
    }

    /// \brief Constructs a patient material from a Hounsfield Unit (HU) value.
    ///
    /// When a patient material is created with an HU value, this constructor
    /// immediately calculates the corresponding mass density and radiation length.
    ///
    /// \param[in] hu The Hounsfield Unit value of the tissue.
    ///
    /// \note The `CUDA_HOST_DEVICE` macro allows this constructor to be called
    ///       on both the CPU and GPU.
    CUDA_HOST_DEVICE
    patient_material_t(int16_t hu) : h2o_t<R>() {
        this->rho_mass = hu_to_density(hu);
        this->X0       = radiation_length(this->rho_mass);
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~patient_material_t() {
        ;
    }

    /// \brief Converts a Hounsfield Unit (HU) value to mass density.
    ///
    /// This method implements a piecewise-linear conversion curve to map HU values
    /// to mass densities, based on the method described by Schneider et al.
    /// This is a standard technique in radiotherapy to estimate tissue density
    //  from CT images.
    ///
    /// The function first clamps the HU value to a valid range [-1000, 6000].
    /// It then applies a specific linear formula depending on the HU value to
    /// calculate the mass density in g/cm^3, which is finally converted to g/mm^3.
    ///
    /// \param[in] hu The Hounsfield Unit value.
    /// \return The corresponding mass density in g/mm^3.
    CUDA_HOST_DEVICE
    virtual R
    hu_to_density(int16_t hu) {
        // Clamp the HU value to the expected range.
        if (hu < -1000) 
        {
            hu = -1000;
        } 
        else if (hu > 6000) 
        {
            hu = 6000;
        }
        R rho_mass = 0.0;

        // This block implements a piecewise-linear curve to map HU to density.
        // This is based on the Schneider conversion method, a common standard.
        if (hu >= -1000 && hu < -723) rho_mass = 0.97509 + 0.00097388 * hu;
        else if (hu >= -723 && hu < -553) rho_mass = 0.95141 + 0.00094118 * hu;
        else if (hu >= -553 && hu < -89) rho_mass = 1.0425 + 0.0011056 * hu;
        else if (hu >= -89 && hu < -38) rho_mass = 1.0171 + 0.00082353 * hu;
        else if (hu >= -38 && hu < 4) rho_mass = 0.99893 + 0.00035714 * hu;
        else if (hu >= 4 && hu < 6) rho_mass = 0.9745 + 0.0085 * hu;
        else if (hu >= 6 && hu < 29) rho_mass = 1.0092 + 0.0015652 * hu;
        else if (hu >= 29 && hu < 81) rho_mass = 1.0293 + 0.00084615 * hu;
        else if (hu >= 81 && hu < 229) rho_mass = 1.0711 + 0.00032432 * hu;
        else if (hu >= 229 && hu < 244) rho_mass = 0.9322 + 0.00093334 * hu;
        else if (hu >= 244 && hu < 461) rho_mass = 0.96191 + 0.00081106 * hu;
        else if (hu >= 461 && hu < 829) rho_mass = 1.0538 + 0.00061141 * hu;
        else if (hu >= 829 && hu < 1248) rho_mass = 1.0363 + 0.00063246 * hu;
        else if (hu >= 1248 && hu <= 6000) rho_mass = 1.1206 + 0.00056491 * hu;
        else if (hu > 6000) rho_mass = 4.60;

        // Convert from g/cm^3 to g/mm^3 for consistency with other units in the simulation.
        rho_mass /= 1000.0;
        return rho_mass;
    }
};

}   // namespace mqi

#endif
