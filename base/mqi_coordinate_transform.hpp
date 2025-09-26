/// \file mqi_coordinate_transform.hpp
///
/// \brief Defines a class for mapping points and vectors between different coordinate systems.
///
/// In radiotherapy, it's essential to accurately track particle positions and
/// trajectories relative to both the treatment machine and the patient's anatomy.
/// This requires transforming coordinates between different reference frames, such as
/// the beam source, gantry, patient couch, and the patient's DICOM image data.
/// This class encapsulates the necessary rotation and translation operations to
/// perform these transformations.
#ifndef MQI_COORDINATE_TRANSFORM_HPP
#define MQI_COORDINATE_TRANSFORM_HPP

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \class coordinate_transform
/// \brief Manages 3D coordinate transformations using a series of rotations and a translation.
///
/// \tparam T The floating-point type (e.g., `float` or `double`) used for calculations.
///           This is a C++ "template," which allows the class to be generic and work
///           with different levels of numerical precision.
///
/// This class models the geometric transformations of a radiotherapy treatment machine.
/// It combines rotations from the collimator, gantry, and patient couch, along with
/// a final transformation to the patient's DICOM coordinate system. The final result
/// is a single rotation matrix and a translation vector that can be used to map
/// coordinates from the beam's reference frame to the patient's reference frame.
///
/// The order of rotations is critical and follows the sequence:
/// 1. Collimator rotation
/// 2. Gantry rotation
/// 3. Patient support (couch) rotation
/// 4. IEC to DICOM transformation
template<typename T>
class coordinate_transform
{
public:
    ///< A constant to convert degrees to radians for trigonometric functions.
    const float deg2rad = M_PI / 180.0;

    ///< The translation vector, typically representing the isocenter position in the DICOM coordinate system.
    vec3<T> translation;

    ///< An array storing the rotation angles in degrees for [collimator, gantry, couch, iec].
    std::array<T, 4> angles = { 0.0, 0.0, 0.0, 0.0 };

    ///< Rotation matrix for the collimator (rotates around the beam axis).
    mat3x3<T> collimator;
    ///< Rotation matrix for the gantry (rotates around the patient).
    mat3x3<T> gantry;
    ///< Rotation matrix for the patient support system, or "couch".
    mat3x3<T> patient_support;
    ///< Rotation matrix to transform from the standard IEC gantry coordinate system to the patient's DICOM coordinate system.
    mat3x3<T> iec2dicom;

    ///< The final, combined rotation matrix, resulting from multiplying the individual rotation matrices in the correct order.
    mat3x3<T> rotation;

    /// \brief Constructs a transformation from given angles and a position vector.
    ///
    /// \param[in] ang A reference to an array of 4 angles [collimator, gantry, couch, iec] in degrees.
    /// \param[in] pos A reference to the translation vector (isocenter).
    ///
    /// \note `CUDA_HOST_DEVICE` allows this function to be run on both the CPU (host) and GPU (device).
    CUDA_HOST_DEVICE
    coordinate_transform(std::array<T, 4>& ang, vec3<T>& pos) : angles(ang), translation(pos) {
        collimator      = mat3x3<T>(0, 0, angles[0] * deg2rad);
        gantry          = mat3x3<T>(0, angles[1] * deg2rad, 0);
        patient_support = mat3x3<T>(0, 0, angles[2] * deg2rad);
        iec2dicom       = mat3x3<T>(angles[3] * deg2rad, 0, 0);
        rotation        = iec2dicom * patient_support * gantry * collimator;
    }

    /// \brief Constructs a transformation from constant (non-modifiable) angles and position.
    ///
    /// This is an "overload" of the constructor for cases where the input data is constant.
    /// \param[in] ang A constant reference to an array of 4 angles.
    /// \param[in] pos A constant reference to the translation vector.
    CUDA_HOST_DEVICE
    coordinate_transform(const std::array<T, 4>& ang, const vec3<T>& pos) :
        angles(ang), translation(pos) {
        collimator      = mat3x3<T>(0, 0, angles[0] * deg2rad);
        gantry          = mat3x3<T>(0, angles[1] * deg2rad, 0);
        patient_support = mat3x3<T>(0, 0, angles[2] * deg2rad);
        iec2dicom       = mat3x3<T>(angles[3] * deg2rad, 0, 0);
        rotation        = iec2dicom * patient_support * gantry * collimator;
    }

    /// \brief Copy constructor. Creates a new object as a copy of an existing one.
    /// \param[in] ref A constant reference to the `coordinate_transform` object to copy.
    CUDA_HOST_DEVICE
    coordinate_transform(const coordinate_transform<T>& ref) {
        angles          = ref.angles;
        collimator      = ref.collimator;
        gantry          = ref.gantry;
        patient_support = ref.patient_support;
        iec2dicom       = ref.iec2dicom;
        rotation        = iec2dicom * patient_support * gantry * collimator;
        translation     = ref.translation;
    }

    /// \brief Default constructor. Initializes an identity transformation (no rotation or translation).
    CUDA_HOST_DEVICE
    coordinate_transform() {
        ;
    }

    /// \brief Assignment operator. Copies the content of one transformation object into another.
    /// \param[in] ref A constant reference to the `coordinate_transform` object to assign from.
    /// \return A reference to the updated object (`*this`).
    CUDA_HOST_DEVICE
    coordinate_transform<T>&
    operator=(const coordinate_transform<T>& ref) {
        collimator      = ref.collimator;
        gantry          = ref.gantry;
        patient_support = ref.patient_support;
        iec2dicom       = ref.iec2dicom;
        rotation        = iec2dicom * patient_support * gantry * collimator;
        translation     = ref.translation;
        angles          = ref.angles;
        return *this;
    }

    /// \brief Prints the internal state of the transformation to the console.
    ///
    /// This is a utility function for debugging purposes, allowing developers to inspect
    /// the values of the translation vector and all rotation matrices.
    /// \note `CUDA_HOST` means this function can only be called from the CPU (host) side.
    CUDA_HOST
    void
    dump() {
        std::cout << "--- coordinate transform---" << std::endl;
        std::cout << "    translation ---" << std::endl;
        translation.dump();
        std::cout << "    rotation ---" << std::endl;
        rotation.dump();
        std::cout << "    gantry ---" << std::endl;
        gantry.dump();
        std::cout << "    patient_support ---" << std::endl;
        patient_support.dump();
        std::cout << "    collimator ---" << std::endl;
        collimator.dump();
        std::cout << "    IEC2DICOM ---" << std::endl;
        iec2dicom.dump();
    }
};

}   // namespace mqi

#endif
