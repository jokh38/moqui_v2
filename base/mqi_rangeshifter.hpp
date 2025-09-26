#ifndef MQI_RANGESHIFTER_H
#define MQI_RANGESHIFTER_H

/// \file
///
/// \brief Defines a range shifter, a geometric component used to modify the depth of a radiation beam.
/// \details A range shifter is typically a slab of plastic or other material of a specific thickness
/// placed in the beam path. Its purpose is to reduce the energy of the particle beam, thereby
/// decreasing its penetration depth into the patient. This allows for precise control over the
/// location of the Bragg peak, which is essential for conforming the dose to the tumor.

#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/**
 * @class rangeshifter
 * @brief Represents a range shifter geometry used in radiotherapy, which can be rectangular or cylindrical.
 * @details This class defines the geometry of a range shifter. It inherits from the base `geometry`
 * class, gaining standard properties like position and orientation. It then adds its own specific
 * attributes, such as its shape (rectangle or cylinder) and dimensions. It is assumed to be
 * constructed of a single, uniform material.
 */
class rangeshifter : public geometry
{

public:
    /// Flag indicating the shape of the volume. `true` for a rectangle, `false` for a cylinder.
    const bool is_rectangle;
    /// The dimensions of the volume.
    /// - For a rectangle: vec3(width, height, thickness) in x, y, and z.
    /// - For a cylinder: vec3(radius, 0, thickness). The second component is unused.
    const mqi::vec3<float> volume;

    /// \brief Constructs a rangeshifter object.
    /// \param[in] v The dimensions of the volume. For a rectangle: (width, height, thickness). For a cylinder: (radius, 0, thickness).
    /// \param[in] p The position of the rangeshifter's center, relative to its parent's coordinate system.
    /// \param[in] r The rotation matrix defining the orientation of the rangeshifter.
    /// \param[in] is_rect A boolean flag, `true` if the rangeshifter is rectangular (default), `false` if cylindrical.
    rangeshifter(mqi::vec3<float>&   v,
                 mqi::vec3<float>&   p,
                 mqi::mat3x3<float>& r,
                 bool                is_rect = true) :
        volume(v),
        is_rectangle(is_rect), geometry(p, r, mqi::geometry_type::RANGESHIFTER) {
        ;
    }

    /**
     * @brief Copy constructor.
     * @param[in] rhs The rangeshifter object to copy.
     */
    rangeshifter(const mqi::rangeshifter& rhs) :
        volume(rhs.volume), is_rectangle(rhs.is_rectangle),
        geometry(rhs.pos, rhs.rot, rhs.geotype) {
        ;
    }

    /**
     * @brief Destructor.
     */
    ~rangeshifter() {
        ;
    }

    /**
     * @brief Assignment operator.
     * @param[in] rhs The rangeshifter object to assign from.
     * @return A const reference to the assigned object.
     * @note This operator is unconventional. As all members are `const`, no actual assignment occurs.
     * It effectively returns the object on the right-hand side.
     */
    const rangeshifter&
    operator=(const mqi::rangeshifter& rhs) {
        return rhs;
    }
};

}   // namespace mqi

#endif