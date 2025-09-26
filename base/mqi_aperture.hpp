#ifndef MQI_APERTURE_H
#define MQI_APERTURE_H

/// \file mqi_aperture.hpp
///
/// \brief Geometry model for an aperture.
///
/// An aperture is a device that shapes the beam. It is defined by a set of polygons
/// that form one or more openings. The aperture itself can be either a rectangle
/// or a cylinder.

#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/// \class aperture
/// \brief Represents a beam-shaping aperture.
///
/// This class defines the geometry of an aperture, which can be either rectangular
/// or cylindrical. It is defined by a set of 2D points that form the opening(s).
/// The aperture is a passive beam modifier, meaning it does not change the energy
/// of the beam, only its shape.
class aperture : public geometry
{
public:
    /// \brief Determines if the aperture shape is a rectangle or a cylinder.
    /// If true, the aperture is rectangular. If false, it is cylindrical.
    const bool is_rectangle;

    /// \brief The dimensions of the aperture.
    /// For a rectangular aperture, this represents the (length, width, height).
    /// For a cylindrical aperture, this represents the (radius, height, 0).
    const mqi::vec3<float> volume;

    /// \brief A list of polygons defining the openings in the aperture.
    /// Each polygon is a list of 2D points in the x-y plane. Divergence is not considered.
    const std::vector<std::vector<std::array<float, 2>>> block_data;

public:
    /// \brief Constructs a new aperture object.
    ///
    /// \param xypts A vector of polygons, where each polygon is a vector of 2D points (x, y)
    ///              defining an opening in the aperture.
    /// \param v The dimensions of the aperture volume. For a box, this is (Lx, Ly, Lz).
    ///          For a cylinder, this is (R, thickness, 0).
    /// \param p The center position of the aperture in 3D space.
    /// \param r The rotation of the aperture in 3D space.
    /// \param is_rect A boolean indicating if the aperture is rectangular (true) or cylindrical (false).
    aperture(std::vector<std::vector<std::array<float, 2>>> xypts,
             mqi::vec3<float>&                              v,
             mqi::vec3<float>&                              p,
             mqi::mat3x3<float>&                            r,
             bool                                           is_rect = true) :
        block_data(xypts),
        volume(v), is_rectangle(is_rect), geometry(p, r, mqi::geometry_type::BLOCK) {
        ;
    }

    /// \brief Constructs a copy of an existing aperture object.
    ///
    /// \param rhs The aperture object to be copied.
    aperture(const mqi::aperture& rhs) :
        volume(rhs.volume), block_data(rhs.block_data), is_rectangle(rhs.is_rectangle),
        geometry(rhs.pos, rhs.rot, rhs.geotype) {
        ;
    }

    /// \brief Destroys the aperture object.
    ~aperture() {
        ;
    }

    /// \brief Assigns the values of another aperture object to this one.
    ///
    /// \param rhs The aperture object to be assigned from.
    /// \return A constant reference to the modified aperture object.
    const aperture&
    operator=(const mqi::aperture& rhs) {
        return rhs;
    }
};

}   // namespace mqi
#endif