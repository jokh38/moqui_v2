/// \file mqi_grid3d.hpp
///
/// \brief Defines a generic 3D rectilinear grid for Monte Carlo transport simulations.
///
/// This file contains the definition of the `mqi::grid3d` class, a versatile
/// template class for representing 3D spatial data. A "rectilinear" grid means
/// that the grid lines are straight and orthogonal, but the spacing between them
/// (i.e., the voxel size) can be non-uniform. This makes it suitable for representing
/// data like CT volumes (which can have variable slice thickness) and dose grids.
#ifndef MQI_GRID3D_H
#define MQI_GRID3D_H

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \struct intersect_t
/// \brief Describes the result of a ray-tracing intersection with a grid voxel.
///
/// This structure is used to return the results of calculating the intersection
/// of a particle's path (a ray) with the boundaries of a voxel.
///
/// \tparam R The floating-point type (e.g., `float` or `double`) for coordinate calculations.
template<typename R>
struct intersect_t {
    R              dist; ///< The distance from the ray's origin to the intersection plane. Negative if no intersection.
    cell_side      side; ///< Which side of the cell was entered (e.g., X-plus, Y-minus).
    vec3<ijk_t>    cell; ///< The (i,j,k) index of the intersected cell.
    transport_type type; ///< The type of geometry of the current node (e.g., patient, dosegrid).
};

/// \class grid3d
/// \brief A template class for a 3D rectilinear grid.
///
/// This class is a fundamental data structure used to represent any 3D voxelized
/// data, such as a patient's CT scan or a dose calculation grid.
///
/// \tparam T The data type of the value stored in each grid cell (e.g., `int16_t` for CT data, `float` for dose).
/// \tparam R The floating-point type for the grid's spatial coordinates (e.g., `float` or `double`).
template<typename T, typename R>
class grid3d
{

protected:
    ///< The dimensions of the grid, i.e., the number of voxels in each direction.
    mqi::vec3<ijk_t> dim_;

    ///< Pointers to arrays defining the grid-edge coordinates along each axis.
    ///< Note: The number of edges is one greater than the number of voxels in that dimension.
    R* xe_ = nullptr;
    R* ye_ = nullptr;
    R* ze_ = nullptr;

    ///< The corners of the grid's axis-aligned bounding box.
    mqi::vec3<R> V000_; ///< The corner with the minimum coordinates (x_min, y_min, z_min).
    mqi::vec3<R> V111_; ///< The corner with the maximum coordinates (x_max, y_max, z_max).
    mqi::vec3<R> C_;    ///< The geometric center of the bounding box.

    ///< Normal vectors for each axis of the grid, used in intersection calculations.
    mqi::vec3<R> n100_;
    mqi::vec3<R> n010_;
    mqi::vec3<R> n001_;

    ///< A pointer to the grid's data array, which stores the value for each voxel.
    T* data_ = nullptr;

    /// \brief A helper function to calculate the bounding box corners and center of the grid.
    /// \note `CUDA_HOST_DEVICE` allows this function to be run on both the CPU and GPU.
    CUDA_HOST_DEVICE
    void
    calculate_bounding_box(void) {
        V000_.x = xe_[0];
        V000_.y = ye_[0];
        V000_.z = ze_[0];
        V111_.x = xe_[dim_.x];
        V111_.y = ye_[dim_.y];
        V111_.z = ze_[dim_.z];

        C_ = (V000_ + V111_) * 0.5;

        n100_.x = V111_.x - V000_.x;
        n100_.y = 0.0;
        n100_.z = 0.0;

        n010_.x = 0.0;
        n010_.y = V111_.y - V000_.y;
        n010_.z = 0.0;

        n001_.x = 0.0;
        n001_.y = 0.0;
        n001_.z = V111_.z - V000_.z;
        n100_.normalize();
        n010_.normalize();
        n001_.normalize();
    }

public:
    ///< A rotation matrix to transform points from the grid's local coordinate system to the world system.
    mqi::mat3x3<R> rotation_matrix_fwd;
    ///< The inverse rotation matrix, to transform points from the world system to the grid's local system.
    mqi::mat3x3<R> rotation_matrix_inv;
    ///< A translation vector for positioning the grid in the world coordinate system.
    mqi::vec3<R> translation_vector;

    /// \brief Default constructor.
    CUDA_HOST_DEVICE
    grid3d() {
        ;
    }

    /// \brief Constructs a non-uniform grid from arrays of edge coordinates.
    /// \param[in] xe Array of x-edge coordinates.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye Array of y-edge coordinates.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze Array of z-edge coordinates.
    /// \param[in] n_ze Number of z-edges.
    CUDA_HOST_DEVICE
    grid3d(const R     xe[],
           const ijk_t n_xe,
           const R     ye[],
           const ijk_t n_ye,
           const R     ze[],
           const ijk_t n_ze) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();
    }

    /// \brief Constructs a uniform grid from min/max coordinates and the number of voxels.
    /// \param[in] xe_min Minimum x-coordinate of the grid.
    /// \param[in] xe_max Maximum x-coordinate of the grid.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye_min Minimum y-coordinate.
    /// \param[in] ye_max Maximum y-coordinate.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze_min Minimum z-coordinate.
    /// \param[in] ze_max Maximum z-coordinate.
    /// \param[in] n_ze Number of z-edges.
    CUDA_HOST_DEVICE
    grid3d(const R     xe_min,
           const R     xe_max,
           const ijk_t n_xe,
           const R     ye_min,
           const R     ye_max,
           const ijk_t n_ye,
           const R     ze_min,
           const R     ze_max,
           const ijk_t n_ze) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;

        this->calculate_bounding_box();
    }

    /// \brief Constructs an oriented uniform grid with rotation specified by angles.
    /// \param[in] xe_min Minimum x-coordinate of the grid.
    /// \param[in] xe_max Maximum x-coordinate of the grid.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye_min Minimum y-coordinate.
    /// \param[in] ye_max Maximum y-coordinate.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze_min Minimum z-coordinate.
    /// \param[in] ze_max Maximum z-coordinate.
    /// \param[in] n_ze Number of z-edges.
    /// \param[in] angles Rotation angles (in degrees) for each axis.
    CUDA_HOST_DEVICE
    grid3d(const R           xe_min,
           const R           xe_max,
           const ijk_t       n_xe,
           const R           ye_min,
           const R           ye_max,
           const ijk_t       n_ye,
           const R           ze_min,
           const R           ze_max,
           const ijk_t       n_ze,
           std::array<R, 3>& angles) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;
        this->calculate_bounding_box();

        rotation_matrix_fwd.rotate(static_cast<R>(angles[0] * M_PI / 180.0),
                                   static_cast<R>(angles[1] * M_PI / 180.0),
                                   static_cast<R>(angles[2] * M_PI / 180.0));
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// \brief Constructs an oriented uniform grid with a specified rotation matrix.
    /// \param[in] xe_min Minimum x-coordinate of the grid.
    /// \param[in] xe_max Maximum x-coordinate of the grid.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye_min Minimum y-coordinate.
    /// \param[in] ye_max Maximum y-coordinate.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze_min Minimum z-coordinate.
    /// \param[in] ze_max Maximum z-coordinate.
    /// \param[in] n_ze Number of z-edges.
    /// \param[in] rxyz The 3x3 rotation matrix.
    CUDA_HOST_DEVICE
    grid3d(const R        xe_min,
           const R        xe_max,
           const ijk_t    n_xe,
           const R        ye_min,
           const R        ye_max,
           const ijk_t    n_ye,
           const R        ze_min,
           const R        ze_max,
           const ijk_t    n_ze,
           mqi::mat3x3<R> rxyz) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;
        this->calculate_bounding_box();

        rotation_matrix_fwd = rxyz;
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// \brief Constructs an oriented non-uniform grid from edge arrays and a rotation matrix.
    /// \param[in] xe Array of x-edge coordinates.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye Array of y-edge coordinates.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze Array of z-edge coordinates.
    /// \param[in] n_ze Number of z-edges.
    /// \param[in] rxyz The 3x3 rotation matrix.
    CUDA_HOST_DEVICE
    grid3d(const R        xe[],
           const ijk_t    n_xe,
           const R        ye[],
           const ijk_t    n_ye,
           const R        ze[],
           const ijk_t    n_ze,
           mqi::mat3x3<R> rxyz) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();

        rotation_matrix_fwd = rxyz;
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// \brief Constructs an oriented non-uniform grid from edge arrays and rotation angles.
    /// \param[in] xe Array of x-edge coordinates.
    /// \param[in] n_xe Number of x-edges (number of voxels + 1).
    /// \param[in] ye Array of y-edge coordinates.
    /// \param[in] n_ye Number of y-edges.
    /// \param[in] ze Array of z-edge coordinates.
    /// \param[in] n_ze Number of z-edges.
    /// \param[in] angles Rotation angles (in degrees) for each axis.
    CUDA_HOST_DEVICE
    grid3d(const R           xe[],
           const ijk_t       n_xe,
           const R           ye[],
           const ijk_t       n_ye,
           const R           ze[],
           const ijk_t       n_ze,
           std::array<R, 3>& angles) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();

        rotation_matrix_fwd.rotate(static_cast<R>(angles[0] * M_PI / 180.0),
                                   static_cast<R>(angles[1] * M_PI / 180.0),
                                   static_cast<R>(angles[2] * M_PI / 180.0));
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~grid3d() {}

    /// \brief Sets the grid edges using externally managed arrays.
    CUDA_HOST_DEVICE
    virtual void
    set_edges(R* xe, ijk_t nx, R* ye, ijk_t ny, R* ze, ijk_t nz) {
        xe_    = xe;
        ye_    = ye;
        ze_    = ze;
        dim_.x = nx - 1;
        dim_.y = ny - 1;
        dim_.z = nz - 1;
        this->calculate_bounding_box();
    }

    /// \brief Gets the array of x-edge coordinates.
    /// \return A pointer to the x-edge array.
    CUDA_HOST_DEVICE
    virtual R*
    get_x_edges() {
        return xe_;
    }

    /// \brief Gets the array of y-edge coordinates.
    /// \return A pointer to the y-edge array.
    CUDA_HOST_DEVICE
    virtual R*
    get_y_edges() {
        return ye_;
    }

    /// \brief Gets the array of z-edge coordinates.
    /// \return A pointer to the z-edge array.
    CUDA_HOST_DEVICE
    virtual R*
    get_z_edges() {
        return ze_;
    }

    /// \brief Gets the dimensions of the grid (number of voxels).
    /// \return A `vec3` containing the number of voxels in x, y, and z.
    CUDA_HOST_DEVICE
    mqi::vec3<ijk_t>
    get_nxyz() {
        return dim_;
    }

    /// \brief Accesses the data value at a given 3D index using the `[]` operator.
    /// \param[in] p A `vec3` containing the (i, j, k) index.
    /// \return The data value at the specified index.
    CUDA_HOST_DEVICE
    virtual const T
    operator[](const mqi::vec3<ijk_t> p) {
        return data_[ijk2cnb(p.x, p.y, p.z)];
    }

    /// \brief Accesses the data value at a given 1D flattened index using the `[]` operator.
    /// \param[in] p The 1D index (often called a "copy number" or "cnb").
    /// \return The data value at the specified index.
    CUDA_HOST_DEVICE
    virtual const T
    operator[](const mqi::cnb_t p) {
        return data_[p];
    }

    /// \brief Prints the grid's edge coordinates to the console for debugging.
    CUDA_HOST_DEVICE
    virtual void
    dump_edges() {
        printf("X edges: ");
        for (ijk_t i = 0; i <= dim_.x; ++i) {
            printf(" %f", xe_[i]);
        }
        printf("\n");

        printf("Y edges: ");
        for (ijk_t i = 0; i <= dim_.y; ++i) {
            printf(" %f", ye_[i]);
        }
        printf("\n");

        printf("Z edges: ");
        for (ijk_t i = 0; i <= dim_.z; ++i) {
            printf(" %f", ze_[i]);
        }
        printf("\n");
    }

    /// \brief Converts a 3D index (i,j,k) to a 1D flattened index ("copy number").
    ///
    /// This is a common optimization. Storing 3D data in a 1D array is often more
    /// efficient for memory access patterns, especially on GPUs. This function
    /// provides the mapping from a logical 3D index to the physical 1D array index.
    ///
    /// \param[in] i The x-index.
    /// \param[in] j The y-index.
    /// \param[in] k The z-index.
    /// \return The 1D flattened index.
    CUDA_HOST_DEVICE
    virtual inline cnb_t
    ijk2cnb(ijk_t i, ijk_t j, ijk_t k) {
        return k * dim_.x * dim_.y + j * dim_.x + i;
    }

    /// \brief Converts a 3D index vector to a 1D flattened index.
    /// \param[in] idx A `vec3` containing the (i,j,k) index.
    /// \return The 1D flattened index.
    CUDA_HOST_DEVICE
    cnb_t
    ijk2cnb(vec3<ijk_t> idx) {
        return idx.z * dim_.x * dim_.y + idx.y * dim_.x + idx.x;
    }

    /// \brief Converts a 1D flattened index back to a 3D index (i,j,k).
    /// \param[in] c The 1D flattened index.
    /// \return A `vec3` containing the (i,j,k) index.
    CUDA_HOST_DEVICE
    virtual inline vec3<ijk_t>
    cnb2ijk(cnb_t c) {
        const cnb_t nxy = dim_.x * dim_.y;
        vec3<ijk_t> ijk;
        ijk.z = c / nxy;
        ijk.y = (c % (nxy)) / dim_.x;
        ijk.x = (c % (nxy)) % dim_.x;
        return ijk;
    }

    /// \brief Deletes the internal data array if it has been allocated.
    CUDA_HOST_DEVICE
    void
    delete_data_if_used(void) {
        if (data_ != nullptr) delete[] data_;
    }

    /// \brief A virtual method to load data into the grid.
    ///
    /// This is intended to be overridden by derived classes (like `mqi::ct`)
    /// that have specific ways of loading their data.
    CUDA_HOST
    virtual void
    load_data() {
    }

    /// \brief Sets the grid's data from an externally managed data source.
    /// \param[in] src A pointer to the source data array.
    CUDA_HOST_DEVICE
    virtual void
    set_data(T* src) {
        this->delete_data_if_used();
        data_ = src;
    }

    /// \brief Allocates memory for the data array and fills the entire grid with a single value.
    /// \param[in] a The value to fill the grid with.
    CUDA_HOST_DEVICE
    virtual void
    fill_data(T a) {
        data_ = new T[dim_.x * dim_.y * dim_.z];
        for (uint32_t i = 0; i < dim_.x * dim_.y * dim_.z; ++i)
            data_[i] = a;
    }

    /// \brief Gets a pointer to the grid's internal data array.
    /// \return A pointer to the data.
    CUDA_HOST_DEVICE
    T*
    get_data() const {
        return data_;
    }

    /// \brief Calculates the volume of a voxel at a given 1D flattened index.
    /// \param[in] p The 1D index of the voxel.
    /// \return The volume of the voxel.
    CUDA_HOST_DEVICE
    R
    get_volume(const mqi::cnb_t p) {
        vec3<ijk_t> vox    = cnb2ijk(p);
        R           volume = xe_[vox.x + 1] - xe_[vox.x];
        volume *= ye_[vox.y + 1] - ye_[vox.y];
        volume *= ze_[vox.z + 1] - ze_[vox.z];
        return volume;
    }

    /// \brief Calculates the volume of a voxel at a given 3D index.
    /// \param[in] vox A `vec3` containing the (i,j,k) index of the voxel.
    /// \return The volume of the voxel.
    CUDA_HOST_DEVICE
    R
    get_volume(const mqi::vec3<ijk_t> vox) {
        R volume = xe_[vox.x + 1] - xe_[vox.x];
        volume *= ye_[vox.y + 1] - ye_[vox.y];
        volume *= ze_[vox.z + 1] - ze_[vox.z];
        return volume;
    }

    /// \brief Calculates the intersection of a ray with the boundaries of the current voxel.
    ///
    /// This is a core ray-tracing function. Given a particle's position `p` and
    /// direction `d` inside a voxel `idx`, it calculates the shortest distance
    /// to exit that voxel.
    ///
    /// \param[in] p The starting point of the ray (particle's current position).
    /// \param[in] d The direction vector of the ray.
    /// \param[in] idx The (i,j,k) index of the current voxel.
    /// \return An `intersect_t` struct with the intersection details.
    CUDA_HOST_DEVICE
    intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d, mqi::vec3<ijk_t>& idx) {
        mqi::intersect_t<R> its;
        its.cell = idx;
        its.side = mqi::NONE_XYZ_PLANE;
        its.dist = -1.0;

        R t_max_x = (d.x > 0) ? (xe_[idx.x + 1] - p.x) / d.x : (xe_[idx.x] - p.x) / d.x;
        R t_max_y = (d.y > 0) ? (ye_[idx.y + 1] - p.y) / d.y : (ye_[idx.y] - p.y) / d.y;
        R t_max_z = (d.z > 0) ? (ze_[idx.z + 1] - p.z) / d.z : (ze_[idx.z] - p.z) / d.z;

        R min_dist = mqi::p_inf;
        if (t_max_x > 0 && t_max_x < min_dist) min_dist = t_max_x;
        if (t_max_y > 0 && t_max_y < min_dist) min_dist = t_max_y;
        if (t_max_z > 0 && t_max_z < min_dist) min_dist = t_max_z;

        if (min_dist < mqi::p_inf) { its.dist = min_dist; }
        return its;
    }

    /// \brief Calculates the first intersection of a ray with the entire grid from outside.
    /// \param[in] p The starting point of the ray.
    /// \param[in] d The direction vector of the ray.
    /// \return An `intersect_t` struct with the intersection details.
    CUDA_HOST_DEVICE
    intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d) {
        mqi::intersect_t<R> its;
        its.dist   = -1.0;
        its.side   = mqi::NONE_XYZ_PLANE;
        its.cell.x = -1;
        its.cell.y = -1;
        its.cell.z = -1;

        if (p.x >= xe_[0] && p.x <= xe_[dim_.x] && p.y >= ye_[0] && p.y <= ye_[dim_.y] &&
            p.z >= ze_[0] && p.z <= ze_[dim_.z]) {
            its.cell = this->index(p);
            its.dist = 0;
            return its;
        }

        R t_min_x = (xe_[0] - p.x) / d.x;
        R t_max_x = (xe_[dim_.x] - p.x) / d.x;
        if (t_min_x > t_max_x) std::swap(t_min_x, t_max_x);

        R t_min_y = (ye_[0] - p.y) / d.y;
        R t_max_y = (ye_[dim_.y] - p.y) / d.y;
        if (t_min_y > t_max_y) std::swap(t_min_y, t_max_y);

        if ((t_min_x > t_max_y) || (t_min_y > t_max_x)) return its;

        R t_min = std::max(t_min_x, t_min_y);
        R t_max = std::min(t_max_x, t_max_y);

        R t_min_z = (ze_[0] - p.z) / d.z;
        R t_max_z = (ze_[dim_.z] - p.z) / d.z;
        if (t_min_z > t_max_z) std::swap(t_min_z, t_max_z);

        if ((t_min > t_max_z) || (t_min_z > t_max)) return its;

        t_min = std::max(t_min, t_min_z);
        t_max = std::min(t_max, t_max_z);

        if (t_min > 0) {
            its.dist = t_min;
            mqi::vec3<R> p_on = p + d * its.dist;
            its.cell          = this->index(p_on);
        }

        return its;
    }

    /// \brief Finds the 3D index of the voxel containing a given point.
    ///
    /// This function uses a binary search (`lower_bound`) for efficient lookup,
    /// which is suitable for both uniform and non-uniform grids.
    ///
    /// \param[in] p The physical point (x,y,z) to locate.
    /// \return A `vec3` containing the (i,j,k) index of the voxel.
    CUDA_HOST_DEVICE
    inline mqi::vec3<ijk_t>
    index(const mqi::vec3<R>& p) {
        mqi::vec3<ijk_t> idx;
        idx.x = std::lower_bound(xe_, xe_ + dim_.x + 1, p.x) - xe_ - 1;
        idx.y = std::lower_bound(ye_, ye_ + dim_.y + 1, p.y) - ye_ - 1;
        idx.z = std::lower_bound(ze_, ze_ + dim_.z + 1, p.z) - ze_ - 1;

        if (p.x >= xe_[dim_.x]) idx.x = dim_.x - 1;
        if (p.y >= ye_[dim_.y]) idx.y = dim_.y - 1;
        if (p.z >= ze_[dim_.z]) idx.z = dim_.z - 1;

        return idx;
    }

    /// \brief Updates the voxel index after a particle has crossed a boundary.
    ///
    /// After a particle takes a step that ends exactly on a voxel boundary, this
    /// function updates the index to the adjacent voxel based on the particle's direction.
    ///
    /// \param[in] vtx1 The new vertex position after the step (should be on a boundary).
    /// \param[in] dir1 The direction vector of the particle.
    /// \param[in,out] idx The 3D index to be updated.
    CUDA_HOST_DEVICE
    inline void
    index(mqi::vec3<R>& vtx1, mqi::vec3<R>& dir1, mqi::vec3<ijk_t>& idx) {
        if (dir1.x < 0 && mqi::mqi_abs(vtx1.x - xe_[idx.x]) < mqi::geometry_tolerance) {
            idx.x--;
        } else if (dir1.x > 0 && mqi::mqi_abs(vtx1.x - xe_[idx.x + 1]) < mqi::geometry_tolerance) {
            idx.x++;
        }
        if (dir1.y < 0 && mqi::mqi_abs(vtx1.y - ye_[idx.y]) < mqi::geometry_tolerance) {
            idx.y--;
        } else if (dir1.y > 0 && mqi::mqi_abs(vtx1.y - ye_[idx.y + 1]) < mqi::geometry_tolerance) {
            idx.y++;
        }
        if (dir1.z < 0 && mqi::mqi_abs(vtx1.z - ze_[idx.z]) < mqi::geometry_tolerance) {
            idx.z--;
        } else if (dir1.z > 0 && mqi::mqi_abs(vtx1.z - ze_[idx.z + 1]) < mqi::geometry_tolerance) {
            idx.z++;
        }
    }

    /// \brief Checks if a given 3D index is within the valid grid boundaries.
    /// \param[in] c A `vec3` containing the (i,j,k) index to check.
    /// \return True if the index is valid, false otherwise.
    CUDA_HOST_DEVICE
    inline bool
    is_valid(mqi::vec3<ijk_t>& c) {
        if (c.x < 0 || c.y < 0 || c.z < 0) return false;
        if (c.x >= dim_.x || c.y >= dim_.y || c.z >= dim_.z) return false;
        return true;
    }
};

}   // namespace mqi

#endif
