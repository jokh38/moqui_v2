#ifndef MQI_APERTURE3D_H
#define MQI_APERTURE3D_H

/// \file mqi_aperture3d.hpp
///
/// \brief 3D rectilinear grid geometry for Monte Carlo transport through an aperture.
///
/// This file defines a 3D aperture model based on a rectilinear grid, which is
/// used for particle transport simulations. It supports both CPU and GPU execution
/// via CUDA.

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \class aperture3d
/// \brief A 3D aperture model represented by a rectilinear grid.
///
/// This class extends `grid3d` to model a 3D aperture with one or more openings.
/// It is a template class that can be used with different data types for grid
/// values and coordinates. It is designed to be used in Monte Carlo particle
/// transport simulations and supports CUDA for GPU acceleration.
///
/// \tparam T The data type of the grid values (e.g., dose, HU).
/// \tparam R The data type of the grid coordinates (e.g., float, double).
template<typename T, typename R>
class aperture3d : public grid3d<T, R>
{
public:
    /// \brief The number of openings in the aperture.
    uint16_t num_opening;
    /// \brief An array containing the number of segments for each opening.
    uint16_t* num_segments;
    /// \brief A pointer to an array of 2D vectors defining the segments of each opening.
    mqi::vec2<R>** block_segment;

    /// \brief Default constructor.
    ///
    /// This constructor is intended for use by derived classes and may not be
    /// directly useful in most cases.
    CUDA_HOST_DEVICE
    aperture3d() : grid3d<T, R>() {
        ;
    }

    /// \brief Constructs a rectilinear grid from arrays of coordinates.
    ///
    /// \param xe A 1D array of voxel center coordinates along the x-axis.
    /// \param n_xe The number of elements in the `xe` array.
    /// \param ye A 1D array of voxel center coordinates along the y-axis.
    /// \param n_ye The number of elements in the `ye` array.
    /// \param ze A 1D array of voxel center coordinates along the z-axis.
    /// \param n_ze The number of elements in the `ze` array.
    CUDA_HOST_DEVICE
    aperture3d(const R     xe[],
               const ijk_t n_xe,
               const R     ye[],
               const ijk_t n_ye,
               const R     ze[],
               const ijk_t n_ze) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze) {}

    /// \brief Constructs a rectilinear grid from min/max coordinates and number of steps.
    ///
    /// \param xe_min The minimum x-coordinate.
    /// \param xe_max The maximum x-coordinate.
    /// \param n_xe The number of steps along the x-axis.
    /// \param ye_min The minimum y-coordinate.
    /// \param ye_max The maximum y-coordinate.
    /// \param n_ye The number of steps along the y-axis.
    /// \param ze_min The minimum z-coordinate.
    /// \param ze_max The maximum z-coordinate.
    /// \param n_ze The number of steps along the z-axis.
    CUDA_HOST_DEVICE
    aperture3d(const R     xe_min,
               const R     xe_max,
               const ijk_t n_xe,   //n_xe : steps + 1
               const R     ye_min,
               const R     ye_max,
               const ijk_t n_ye,
               const R     ze_min,
               const R     ze_max,
               const ijk_t n_ze) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze) {}

    /// \brief Constructs an oriented rectilinear grid.
    ///
    /// \param xe_min The minimum x-coordinate.
    /// \param xe_max The maximum x-coordinate.
    /// \param n_xe The number of steps along the x-axis.
    /// \param ye_min The minimum y-coordinate.
    /// \param ye_max The maximum y-coordinate.
    /// \param n_ye The number of steps along the y-axis.
    /// \param ze_min The minimum z-coordinate.
    /// \param ze_max The maximum z-coordinate.
    /// \param n_ze The number of steps along the z-axis.
    /// \param angles A 3-element array representing the rotation angles (in degrees) for each axis.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe_min,
               const R           xe_max,
               const ijk_t       n_xe,   //n_xe : steps + 1
               const R           ye_min,
               const R           ye_max,
               const ijk_t       n_ye,
               const R           ze_min,
               const R           ze_max,
               const ijk_t       n_ze,
               std::array<R, 3>& angles) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, angles) {}

    /// \brief Constructs an oriented rectilinear grid with a rotation matrix.
    ///
    /// \param xe_min The minimum x-coordinate.
    /// \param xe_max The maximum x-coordinate.
    /// \param n_xe The number of steps along the x-axis.
    /// \param ye_min The minimum y-coordinate.
    /// \param ye_max The maximum y-coordinate.
    /// \param n_ye The number of steps along the y-axis.
    /// \param ze_min The minimum z-coordinate.
    /// \param ze_max The maximum z-coordinate.
    /// \param n_ze The number of steps along the z-axis.
    /// \param rxyz The 3x3 rotation matrix.
    CUDA_HOST_DEVICE
    aperture3d(const R        xe_min,
               const R        xe_max,
               const ijk_t    n_xe,   //n_xe : steps + 1
               const R        ye_min,
               const R        ye_max,
               const ijk_t    n_ye,
               const R        ze_min,
               const R        ze_max,
               const ijk_t    n_ze,
               mqi::mat3x3<R> rxyz) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, rxyz) {}

    /// \brief Constructs an oriented rectilinear grid from coordinate arrays and a rotation matrix.
    ///
    /// \param xe A 1D array of voxel center coordinates along the x-axis.
    /// \param n_xe The number of elements in the `xe` array.
    /// \param ye A 1D array of voxel center coordinates along the y-axis.
    /// \param n_ye The number of elements in the `ye` array.
    /// \param ze A 1D array of voxel center coordinates along the z-axis.
    /// \param n_ze The number of elements in the `ze` array.
    /// \param rxyz The 3x3 rotation matrix.
    CUDA_HOST_DEVICE
    aperture3d(const R        xe[],
               const ijk_t    n_xe,
               const R        ye[],
               const ijk_t    n_ye,
               const R        ze[],
               const ijk_t    n_ze,
               mqi::mat3x3<R> rxyz) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze, rxyz) {}

    /// \brief Constructs an oriented rectilinear grid from coordinate arrays and rotation angles.
    ///
    /// \param xe A 1D array of voxel center coordinates along the x-axis.
    /// \param n_xe The number of elements in the `xe` array.
    /// \param ye A 1D array of voxel center coordinates along the y-axis.
    /// \param n_ye The number of elements in the `ye` array.
    /// \param ze A 1D array of voxel center coordinates along the z-axis.
    /// \param n_ze The number of elements in the `ze` array.
    /// \param angles A 3-element array representing the rotation angles (in degrees) for each axis.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe[],
               const ijk_t       n_xe,
               const R           ye[],
               const ijk_t       n_ye,
               const R           ze[],
               const ijk_t       n_ze,
               std::array<R, 3>& angles) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze) {}

    /// \brief Constructs an oriented rectilinear grid with aperture information.
    ///
    /// \param xe_min The minimum x-coordinate.
    /// \param xe_max The maximum x-coordinate.
    /// \param n_xe The number of steps along the x-axis.
    /// \param ye_min The minimum y-coordinate.
    /// \param ye_max The maximum y-coordinate.
    /// \param n_ye The number of steps along the y-axis.
    /// \param ze_min The minimum z-coordinate.
    /// \param ze_max The maximum z-coordinate.
    /// \param n_ze The number of steps along the z-axis.
    /// \param angles A 3-element array representing the rotation angles (in degrees) for each axis.
    /// \param num_opening The number of openings in the aperture.
    /// \param num_segment An array containing the number of segments for each opening.
    /// \param block_segment A pointer to an array of 2D vectors defining the segments of each opening.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe_min,
               const R           xe_max,
               const ijk_t       n_xe,   //n_xe : steps + 1
               const R           ye_min,
               const R           ye_max,
               const ijk_t       n_ye,
               const R           ze_min,
               const R           ze_max,
               const ijk_t       n_ze,
               std::array<R, 3>& angles,
               int16_t           num_opening,
               uint16_t*         num_segment,
               mqi::vec2<R>**    block_segment) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, angles) {
        this->num_opening   = num_opening;
        this->num_segments  = num_segment;
        this->block_segment = block_segment;
    }

    /// \brief Destructor for the aperture3d object.
    ///
    /// Releases dynamically allocated memory for x, y, and z coordinates.
    CUDA_HOST_DEVICE
    ~aperture3d() {
        /*
            delete[] xe_;
            delete[] ye_;
            delete[] ze_;
            */
    }

    /// \brief Determines if a point is inside a polygon using the ray casting algorithm.
    ///
    /// \param pos The 3D position to check (z-coordinate is ignored).
    /// \param segment An array of 2D points defining the polygon.
    /// \param num_segment The number of vertices in the polygon.
    /// \return True if the point is inside the polygon, false otherwise.
    CUDA_HOST_DEVICE
    bool
    sol1_1(mqi::vec3<R> pos, mqi::vec2<R>* segment, uint16_t num_segment) {
        mqi::vec2<R> pos0 = segment[0];
        mqi::vec2<R> pos1;
        float        min_y, max_y, max_x, intersect_x;
        int          count = 0;
        int          i, j, c = 0;
        for (i = 0, j = num_segment - 1; i < num_segment; j = i++) {
            pos0 = segment[i];
            pos1 = segment[j];
            if ((((pos0.y <= pos.y) && (pos.y < pos1.y)) ||
                 ((pos1.y <= pos.y) && (pos.y < pos0.y))) &&
                (pos.x < (pos1.x - pos0.x) * (pos.y - pos0.y) / (pos1.y - pos0.y) + pos0.x)) {
                c = !c;
            }
        }
        return c;
    }

    /// \brief Checks if a 3D point is inside any of the aperture openings.
    ///
    /// \param pos The 3D position to check.
    /// \return True if the point is inside an opening, false otherwise.
    CUDA_HOST_DEVICE
    bool
    is_inside(mqi::vec3<R> pos) {
        bool inside = false;
        for (int i = 0; i < this->num_opening; i++) {
            mqi::vec2<R>* segment = this->block_segment[i];
            inside = sol1_1(pos, segment, this->num_segments[i]);
            if (inside) break;
        }
        return inside;
    }

    /// \brief Calculates the intersection of a ray with the aperture grid.
    ///
    /// This method determines the distance to the next intersection of a ray with the
    /// boundaries of the aperture grid.
    ///
    /// \param p The starting point of the ray.
    /// \param d The direction vector of the ray.
    /// \param idx The current grid index of the ray's starting point.
    /// \return An `intersect_t` object containing information about the intersection.
    CUDA_HOST_DEVICE
    virtual intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d, mqi::vec3<ijk_t> idx) {
        mqi::intersect_t<R> its;
        its.cell = idx;
        its.side = mqi::NONE_XYZ_PLANE;
        mqi::intersect_t<R> its_out;
        mqi::intersect_t<R> its_in;
        its_out.dist   = 0;
        its_out.cell.x = -5;
        its_out.cell.y = -5;
        its_out.cell.z = -5;
        its_in.dist    = 0;
        its_in.cell.x  = -10;
        its_in.cell.y  = -10;
        its_in.cell.z  = -10;

        if (!is_inside(p)) {
            its_out.type = mqi::APERTURE_CLOSE;
            return its_out;
        } else {
            if (d.z < 0) {
                if (mqi::mqi_abs(p.z - this->ze_[0]) < 1e-3) {
                    its_in.dist = 0.0;
                } else {
                    its_in.dist = -(p.z - this->ze_[0]) / d.z;
                }

            } else if (d.z > 0) {
                if (mqi::mqi_abs(p.z - this->ze_[1]) < 1e-3) {
                    its_in.dist = 0;
                } else {
                    its_in.dist = (this->ze_[1] - p.z) / d.z;
                }
            } else {
                its_in.dist = 0;
            }
            assert(its_in.dist >= 0);
            its_in.type = mqi::APERTURE_OPEN;
            return its_in;
        }
    }
};

}   // namespace mqi

#endif
