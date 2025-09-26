#ifndef MQI_RECT3D_H
#define MQI_RECT3D_H

/// \file
///
/// \brief Defines a generic, 3D rectilinear grid for storing data like CT images, dose, or vector fields.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <valarray>

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/**
 * @class rect3d
 * @brief A template class for a 3D rectilinear grid, supporting non-uniform spacing.
 * @details
 * This class is a fundamental data structure for representing any data that exists on a 3D
 * grid where the spacing between points on an axis is not necessarily uniform. It is designed
 * to store data such as CT Hounsfield Units, calculated dose values, or deformation vector fields (DVFs).
 *
 * It provides methods for construction, data access, trilinear interpolation, and various
 * geometric calculations. The class uses raw pointers for its coordinate arrays (`x_`, `y_`, `z_`)
 * and manages their memory manually with `new[]` in the constructors and `delete[]` in the destructor.
 *
 * For a Python developer, this is analogous to having three 1D NumPy arrays for the coordinates
 * and a 3D NumPy array for the data, but with C++'s strong typing and manual memory management.
 *
 * @tparam T The data type of the grid values (e.g., `int16_t` for CT, `float` for dose, `mqi::vec3<float>` for a DVF).
 * @tparam R The data type for the grid coordinates (e.g., `float` or `double` for precision).
 */
template<typename T, typename R>
class rect3d
{

public:
    /**
     * @enum cell_corner
     * @brief Enumerates the 8 corners of a grid cell (voxel) for interpolation.
     * @details The naming convention cXYZ indicates the corner's relative position, where 0 is the lower bound
     * and 1 is the upper bound on each axis. For example, `c000` is the corner with the minimum
     * (x,y,z) coordinates in the cell, and `c111` is the corner with the maximum.
     */
    typedef enum
    {
        //cXYZ, 0 is less, 1 is greater
        c000 = 0, ///< Corner at (x_i,   y_j,   z_k)
        c100,     ///< Corner at (x_i+1, y_j,   z_k)
        c110,     ///< Corner at (x_i+1, y_j+1, z_k)
        c010,     ///< Corner at (x_i,   y_j+1, z_k)
        c001,     ///< Corner at (x_i,   y_j,   z_k+1)
        c101,     ///< Corner at (x_i+1, y_j,   z_k+1)
        c111,     ///< Corner at (x_i+1, y_j+1, z_k+1)
        c011,     ///< Corner at (x_i,   y_j+1, z_k+1)
        cXXX      ///< Invalid or uninitialized corner
    } cell_corner;

protected:
    /// Pointer to a dynamically allocated array of x-coordinates for the center of each voxel along the x-axis.
    R* x_;
    /// Pointer to a dynamically allocated array of y-coordinates for the center of each voxel along the y-axis.
    R* y_;
    /// Pointer to a dynamically allocated array of z-coordinates for the center of each voxel along the z-axis.
    R* z_;

    /// Flag to indicate if an axis is flipped (i.e., coordinates are in descending order).
    bool flip_[3] = { false, false, false };

    /// The number of voxels in each dimension (dim_.x, dim_.y, dim_.z).
    mqi::vec3<ijk_t> dim_;

    /// A 1D array storing the flattened 3D grid data. `std::valarray` is a C++ standard library
    /// class optimized for numerical operations, similar in concept to a NumPy array.
    std::valarray<T> data_;

public:
    /**
     * @brief Default constructor. Intended for use by derived classes which will manually initialize members.
     */
    rect3d() : x_(nullptr), y_(nullptr), z_(nullptr) {
        ;
    }

    /**
     * @brief Constructs a rectilinear grid from std::vector coordinates.
     * @details This constructor allocates new memory for the coordinate arrays and copies the values from the input vectors.
     * @param[in] x A vector of voxel center coordinates along the x-axis.
     * @param[in] y A vector of voxel center coordinates along the y-axis.
     * @param[in] z A vector of voxel center coordinates along the z-axis.
     */
    CUDA_HOST_DEVICE
    rect3d(std::vector<R>& x, std::vector<R>& y, std::vector<R>& z) {
        x_ = new R[x.size()];
        y_ = new R[y.size()];
        z_ = new R[z.size()];

        for (size_t i = 0; i < x.size(); ++i)
            x_[i] = x[i];
        for (size_t i = 0; i < y.size(); ++i)
            y_[i] = y[i];
        for (size_t i = 0; i < z.size(); ++i)
            z_[i] = z[i];

        dim_.x = x.size();
        dim_.y = y.size();
        dim_.z = z.size();
    }

    /**
     * @brief Constructs a rectilinear grid from C-style array coordinates.
     * @param[in] x A C-style array of voxel center coordinates along the x-axis.
     * @param[in] xn The number of elements in the x array.
     * @param[in] y A C-style array of voxel center coordinates along the y-axis.
     * @param[in] yn The number of elements in the y array.
     * @param[in] z A C-style array of voxel center coordinates along the z-axis.
     * @param[in] zn The number of elements in the z array.
     */
    CUDA_HOST_DEVICE
    rect3d(R x[], int xn, R y[], int yn, R z[], int zn) {
        x_ = new R[xn];
        y_ = new R[yn];
        z_ = new R[zn];

        for (size_t i = 0; i < xn; ++i)
            x_[i] = x[i];
        for (size_t i = 0; i < yn; ++i)
            y_[i] = y[i];
        for (size_t i = 0; i < zn; ++i)
            z_[i] = z[i];

        dim_.x = xn;
        dim_.y = yn;
        dim_.z = zn;
    }

    /**
     * @brief Copy constructor.
     * @details Performs a deep copy of the grid, allocating new memory for the coordinate arrays.
     * @param[in] c The rect3d object to copy.
     */
    CUDA_HOST_DEVICE
    rect3d(rect3d& c) {
        dim_ = c.dim_;

        x_ = new R[dim_.x];
        y_ = new R[dim_.y];
        z_ = new R[dim_.z];

        for (size_t i = 0; i < dim_.x; ++i)
            x_[i] = c.x_[i];
        for (size_t i = 0; i < dim_.y; ++i)
            y_[i] = c.y_[i];
        for (size_t i = 0; i < dim_.z; ++i)
            z_[i] = c.z_[i];
    }

    /**
     * @brief Destructor.
     * @details Releases the dynamically allocated memory for the coordinate arrays.
     * In C++, memory allocated with `new[]` must be freed with `delete[]` to prevent memory leaks.
     */
    CUDA_HOST_DEVICE
    ~rect3d() {
        delete[] x_;
        delete[] y_;
        delete[] z_;
    }

    /**
     * @brief Performs trilinear interpolation to find the value at a given point.
     * @param p A std::array representing the 3D point {x, y, z}.
     * @return The interpolated value of type T.
     */
    virtual T
    operator()(const std::array<R, 3> p) {
        return operator()(mqi::vec3<R>(p[0], p[1], p[2]));
    }

    /**
     * @brief Performs trilinear interpolation to find the value at a given point.
     * @param x The x-coordinate of the point.
     * @param y The y-coordinate of the point.
     * @param z The z-coordinate of the point.
     * @return The interpolated value of type T.
     */
    virtual T
    operator()(const R x, const R y, const R z) {
        return operator()(mqi::vec3<R>(x, y, z));
    }

    /**
     * @brief Gets the underlying data storage.
     * @return A const reference to the std::valarray containing the grid data.
     */
    const std::valarray<T>&
    get_data() const {
        return data_;
    }

    /**
     * @brief Gets the x-coordinates of the grid.
     * @return A const pointer to the array of x-coordinates.
     */
    const R*
    get_x() const {
        return x_;
    }

    /**
     * @brief Gets the y-coordinates of the grid.
     * @return A const pointer to the array of y-coordinates.
     */
    const R*
    get_y() const {
        return y_;
    }

    /**
     * @brief Gets the z-coordinates of the grid.
     * @return A const pointer to the array of z-coordinates.
     */
    const R*
    get_z() const {
        return z_;
    }

    /**
     * @brief Performs trilinear interpolation to find the value at a given point.
     * @param pos An mqi::vec3<R> representing the 3D point.
     * @return The interpolated value of type T.
     */
    virtual T
    operator()(const mqi::vec3<R>& pos) {
        std::array<size_t, 3>  c000_idx = this->find_c000_index(pos);
        const std::array<R, 6> cell_pts = this->cell_position(c000_idx);
        const std::array<T, 8> coner    = this->cell_data(c000_idx);

        R xd = (pos.x - cell_pts[0]) / (cell_pts[1] - cell_pts[0]);
        R yd = (pos.y - cell_pts[2]) / (cell_pts[3] - cell_pts[2]);
        R zd = (pos.z - cell_pts[4]) / (cell_pts[5] - cell_pts[4]);

        T c00 = coner[c000] * (1.0 - xd) + coner[c100] * xd;
        T c10 = coner[c010] * (1.0 - xd) + coner[c110] * xd;

        T c01 = coner[c001] * (1.0 - xd) + coner[c101] * xd;
        T c11 = coner[c011] * (1.0 - xd) + coner[c111] * xd;

        T c0 = c00 * (1.0 - yd) + c10 * yd;
        T c1 = c01 * (1.0 - yd) + c11 * yd;

        return c0 * (1.0 - zd) + c1 * zd;
    }

    /**
     * @brief Accesses the data value at a specific grid index.
     * @param p A std::array of size_t representing the {i, j, k} index.
     * @return The value at the specified index.
     */
    virtual T
    operator[](const std::array<size_t, 3> p)   //&
    {
        return data_[ijk2data(p[0], p[1], p[2])];
    }

    /**
     * @brief Accesses the data value at a specific grid index.
     * @param p A std::array of int representing the {i, j, k} index.
     * @return The value at the specified index.
     */
    virtual T
    operator[](const std::array<int, 3> p)   //&
    {
        return data_[ijk2data(p[0], p[1], p[2])];
    }

    /**
     * @brief Retrieves the data values at the 8 corners of a specified grid cell.
     * @param c000_idx The {i, j, k} index of the cell's lower-left-front corner (c000).
     * @return A std::array containing the 8 corner values.
     */
    virtual std::array<T, 8>
    cell_data(const std::array<size_t, 3>& c000_idx) {
        size_t x0 = c000_idx[0];
        size_t x1 = x0 + 1;
        size_t y0 = c000_idx[1];
        size_t y1 = y0 + 1;
        size_t z0 = c000_idx[2];
        size_t z1 = z0 + 1;

        std::array<T, 8> coner = { data_[ijk2data(x0, y0, z0)], data_[ijk2data(x1, y0, z0)],
                                   data_[ijk2data(x1, y1, z0)], data_[ijk2data(x0, y1, z0)],
                                   data_[ijk2data(x0, y0, z1)], data_[ijk2data(x1, y0, z1)],
                                   data_[ijk2data(x1, y1, z1)], data_[ijk2data(x0, y1, z1)] };

        return coner;
    }

    /**
     * @brief Retrieves the coordinates of the bounding box of a specified grid cell.
     * @param c000_idx The {i, j, k} index of the cell's lower-left-front corner (c000).
     * @return A std::array containing the {x0, x1, y0, y1, z0, z1} coordinates of the cell.
     */
    virtual std::array<R, 6>
    cell_position(const std::array<size_t, 3>& c000_idx) {

        size_t xpi = c000_idx[0];
        size_t ypi = c000_idx[1];
        size_t zpi = c000_idx[2];

        std::array<R, 6> pts;
        pts[0] = x_[xpi];
        pts[1] = x_[xpi + 1];
        pts[2] = y_[ypi];
        pts[3] = y_[ypi + 1];
        pts[4] = z_[zpi];
        pts[5] = z_[zpi + 1];

        return pts;
    }

    /**
     * @brief Finds the index of the cell's lower-left-front corner (c000) containing a given point.
     * @param pos The Cartesian coordinates {x, y, z} of the point.
     * @return A std::array containing the {i, j, k} index of the cell.
     */
    CUDA_HOST
    virtual std::array<size_t, 3>
    find_c000_index(const mqi::vec3<R>& pos) {
        std::array<size_t, 3> c000_idx;
        c000_idx[0] = this->find_c000_x_index(pos.x);
        c000_idx[1] = this->find_c000_y_index(pos.y);
        c000_idx[2] = this->find_c000_z_index(pos.z);

        return c000_idx;
    }

    /**
     * @brief Finds the x-index of the cell containing a given x-coordinate.
     * @param x The x-coordinate.
     * @return The x-index (i) of the cell.
     */
    CUDA_HOST
    inline virtual size_t
    find_c000_x_index(const R& x) {
        //In case this code runs on GPU, consider to implement binary_search algorithm or use thrust
        //but thrust performance is not so good from
        //https://groups.google.com/forum/#!topic/thrust-users/kTX6lgntOAc
        R* i = std::lower_bound(x_, x_ + dim_.x, x, std::less_equal<R>());
        return i - x_ - 1;
    }

    /**
     * @brief Finds the y-index of the cell containing a given y-coordinate.
     * @param y The y-coordinate.
     * @return The y-index (j) of the cell.
     */
    CUDA_HOST
    inline virtual size_t
    find_c000_y_index(const R& y) {
        R* j = std::lower_bound(y_, y_ + dim_.y, y, std::less_equal<R>());
        return j - y_ - 1;
    }

    /**
     * @brief Finds the z-index of the cell containing a given z-coordinate.
     * @param z The z-coordinate.
     * @return The z-index (k) of the cell.
     */
    inline virtual size_t
    find_c000_z_index(const R& z) {
        R* k = std::lower_bound(z_, z_ + dim_.z, z, std::less_equal<R>());
        return k - z_ - 1;
    }

    /**
     * @brief Checks if a point is within the grid boundaries (exclusive of the upper boundary).
     * @param p The point to check.
     * @return `true` if the point is inside the grid, `false` otherwise.
     */
    CUDA_HOST_DEVICE
    inline virtual bool
    is_in_point(const vec3<R>& p) {
        ///< check p is inside of pixel grid not entire volume
        ///< Min/Max of src

        if (p.x < x_[0] || p.x >= x_[dim_.x - 1]) return false;
        if (p.y < y_[0] || p.y >= y_[dim_.y - 1]) return false;
        if (p.z < z_[0] || p.z >= z_[dim_.z - 1]) return false;

        return true;
    }

    /**
     * @brief Checks if a point is within the rectangle including the edge.
     * @param p The point to check.
     * @return Always returns true (implementation seems incomplete).
     */
    CUDA_HOST_DEVICE
    inline virtual bool
    is_in_rect(const vec3<R>& p) {
        //if( p.x < xedge_[0] || p.x >= xedge_[1]) return false;
        return true;
    }

    /**
     * @brief Calculates the center position of the grid.
     * @note This is the center based on the first and last voxel coordinates, not the geometric center of the entire volume.
     * @return An mqi::vec3<R> representing the center point.
     */
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_center() {
        return mqi::vec3<R>(0.5 * (x_[0] + x_[dim_.x - 1]),
                            0.5 * (y_[0] + y_[dim_.y - 1]),
                            0.5 * (z_[0] + z_[dim_.z - 1]));
    }

    /**
     * @brief Calculates the total size of the grid volume.
     * @note The size is calculated as the distance between the centers of the first and last voxels, plus half the spacing at each end.
     * @return An mqi::vec3<R> representing the size (Lx, Ly, Lz).
     */
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_size() {
        R Lx = x_[dim_.x - 1] - x_[0];
        Lx += 0.5 * (x_[dim_.x - 1] - x_[dim_.x - 2]);
        Lx += 0.5 * (x_[1] - x_[0]);

        R Ly = y_[dim_.y - 1] - y_[0];
        Ly += 0.5 * (y_[dim_.y - 1] - y_[dim_.y - 2]);
        Ly += 0.5 * (y_[1] - y_[0]);

        R Lz = z_[dim_.z - 1] - z_[0];
        Lz += 0.5 * (z_[dim_.z - 1] - z_[dim_.z - 2]);
        Lz += 0.5 * (z_[1] - z_[0]);

        return mqi::vec3<R>(Lx, Ly, Lz);
    }

    /**
     * @brief Gets the dimensions (number of voxels) of the grid.
     * @return An mqi::vec3<ijk_t> with the number of voxels in x, y, and z.
     */
    CUDA_HOST_DEVICE
    mqi::vec3<ijk_t>
    get_nxyz() {
        return dim_;
    }

    /**
     * @brief Sets the dimensions of the grid.
     * @param dim An mqi::vec3<ijk_t> with the new dimensions.
     */
    CUDA_HOST_DEVICE
    void
    set_nxyz(mqi::vec3<ijk_t> dim) {
        dim_ = dim;
    }
    /**
     * @brief Gets the origin of the grid (center of the first voxel).
     * @return An mqi::vec3<R> representing the origin coordinates.
     */
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_origin() {
        return mqi::vec3<R>(x_[0], y_[0], z_[0]);
    }

    /**
     * @brief Prints the coordinate values for each axis to the console.
     */
    CUDA_HOST
    virtual void
    dump_pts() {
        std::cout << "X: ";
        for (size_t i = 0; i < dim_.x; ++i) {
            std::cout << " " << x_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Y: ";
        for (size_t i = 0; i < dim_.y; ++i) {
            std::cout << " " << y_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Z: ";
        for (size_t i = 0; i < dim_.z; ++i) {
            std::cout << " " << z_[i] << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief Converts a 3D grid index {i, j, k} to a 1D flat array index.
     * @param i The x-index.
     * @param j The y-index.
     * @param k The z-index.
     * @return The corresponding index in the 1D data array.
     */
    CUDA_HOST_DEVICE
    virtual inline size_t
    ijk2data(size_t i, size_t j, size_t k) {
        return k * dim_.x * dim_.y + j * dim_.x + i;
    }

    /**
     * @brief Writes the grid data to a binary file.
     * @param filename The name of the file to write to.
     */
    CUDA_HOST
    virtual void
    write_data(const std::string filename) {
        std::ofstream file1(filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char*>(&data_[0]), data_.size() * sizeof(T));
        file1.close();
    }

    /**
     * @brief Writes a given std::valarray to a binary file.
     * @tparam S The data type of the valarray.
     * @param output The valarray to write.
     * @param filename The name of the file to write to.
     */
    template<class S>
    void
    write_data(std::valarray<S>& output, const std::string filename) {
        std::ofstream file1(filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char*>(&output[0]), output.size() * sizeof(S));
        file1.close();
    }

    /*
    /// this is not yet implemented
    #include <vtkImageReader2.h>
    #include <vtkMetaImageWriter.h>
    #include <vtkSmartPointer.h>

    CUDA_HOST
    virtual void
    write_mha(const std::string filename){

        vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
        writer->SetInputConnection(reader->GetOutputPort());
        writer->SetInpt
        writer->SetCompression(false);
        writer->SetFileName( cl_opts["--output1"][0].c_str() );
        writer->Write();

        std::ofstream file1( filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char *>(&data_[0]), data_.size() * sizeof(T));r
        file1.close();

    }
    */

    /**
     * @brief Allocates memory for the data array based on the grid dimensions.
     */
    CUDA_HOST
    virtual void
    load_data() {
        data_.resize(dim_.x * dim_.y * dim_.z);
    }

    /**
     * @brief Reads data from a raw pointer into the grid's data array.
     * @param src A pointer to the source data.
     * @param total The total number of elements to copy.
     */
    void
    read_data(T* src, size_t total) {
        data_(src, total);
    }

    /**
     * @brief Reads data from a std::valarray into the grid's data array.
     * @param src The source std::valarray.
     */
    void
    read_data(std::valarray<T> src) {
        data_ = src;
    }

    /**
     * @brief Fills the entire grid with a specified value.
     * @param a The value to fill the grid with.
     */
    CUDA_HOST
    virtual void
    fill_data(T a) {
        data_.resize(dim_.x * dim_.y * dim_.z);
        data_ = a;
    }

    /**
     * @brief Checks if any coordinate axis is in descending order and reverses it if necessary.
     * @details This ensures that the internal representation of coordinates is always ascending, which simplifies calculations.
     */
    CUDA_HOST
    void
    flip_xyz_if_any(void) {
        flip_[0] = (x_[1] < x_[0]) ? true : false;
        flip_[1] = (y_[1] < y_[0]) ? true : false;
        flip_[2] = (z_[1] < z_[0]) ? true : false;
        if (flip_[0]) std::reverse(x_, x_ + dim_.x);
        if (flip_[1]) std::reverse(y_, y_ + dim_.y);
        if (flip_[2]) std::reverse(z_, z_ + dim_.z);
    }

    /**
     * @brief Flips the grid data to match the flipped coordinate axes.
     * @details This should be called after `flip_xyz_if_any` to ensure data consistency.
     */
    CUDA_HOST
    void
    flip_data(void) {
        if (flip_[0] == false && flip_[1] == false && flip_[2] == false) { return; }

        std::valarray<T> tmp0(data_.size());   //temporal copy object
        tmp0             = data_;
        long int id_from = 0;
        long int id_to   = 0;

        long int idx = 0;
        long int idy = 0;
        long int idz = 0;

        for (int k = 0; k < dim_.z; ++k) {
            for (int j = 0; j < dim_.y; ++j) {
                for (int i = 0; i < dim_.x; ++i) {
                    idx = (flip_[0]) ? (dim_.x - 1 - i) : i;
                    idy = (flip_[1]) ? (dim_.y - 1 - j) * dim_.x : j * dim_.x;
                    idz = (flip_[2]) ? (dim_.z - 1 - k) * dim_.x * dim_.y : k * dim_.x * dim_.y;

                    id_to        = this->ijk2data(i, j, k);
                    id_from      = idz + idy + idx;
                    data_[id_to] = tmp0[id_from];
                }   //x
            }       //y
        }           //z
    }

    /// A friend function to copy grid information of src to dest
    template<typename T0, typename R0, typename T1, typename R1>
    friend void
    clone_structure(rect3d<T0, R0>& src, rect3d<T1, R1>& dest);

    /// A friend function to interpolate new rect3d from a source
    /// interpolate(ct, dose) : possible
    /// interpolate(dose, dose)
    /// interpolate(dvf, dose) : is not possible
    template<typename T0, typename R0, typename T1, typename R1>
    friend void
    interpolate(rect3d<T0, R0>& src, rect3d<T1, R1>& dest, T1& fill_value);

    /// A friend function to warp source data (rect3d) to destination (rect3d) using dvf.
    /// It will pull src data from ref thus DVF of ref->src is neccessary.
    /// In MIM, when we calculate DIR, ref should be chosen first.
    /// It is recommended that src, dest, dvf have all same resolution.
    template<typename T0, typename R0, typename S0>
    friend void
    warp_linear(rect3d<T0, R0>&            src,
                rect3d<T0, R0>&            dest,
                rect3d<mqi::vec3<S0>, R0>& dvf,
                T0                         fill_value);
};

/**
 * @brief Clones the geometric structure (dimensions and coordinates) from a source grid to a destination grid.
 * @details The data itself is not copied. This is useful for creating a new grid with the same geometry but different data.
 * @tparam T0 Data type of the source grid.
 * @tparam R0 Coordinate type of the source grid.
 * @tparam T1 Data type of the destination grid.
 * @tparam R1 Coordinate type of the destination grid.
 * @param src The source rect3d object.
 * @param dest The destination rect3d object.
 */
template<typename T0, typename R0, typename T1, typename R1>
void
clone_structure(rect3d<T0, R0>& src, rect3d<T1, R1>& dest) {
    dest.dim_ = src.dim_;

    dest.x_ = new R1[dest.dim_.x];
    dest.y_ = new R1[dest.dim_.y];
    dest.z_ = new R1[dest.dim_.z];

    for (size_t i = 0; i < dest.dim_.x; ++i)
        dest.x_[i] = src.x_[i];
    for (size_t i = 0; i < dest.dim_.y; ++i)
        dest.y_[i] = src.y_[i];
    for (size_t i = 0; i < dest.dim_.z; ++i)
        dest.z_[i] = src.z_[i];
}

/**
 * @brief Resamples a source grid onto the geometry of a destination grid using trilinear interpolation.
 * @details For each voxel center in the destination grid, it finds the corresponding value in the source grid
 * via interpolation. If a point is outside the source grid, a specified fill value is used.
 * @tparam T0 Data type of the source grid.
 * @tparam R0 Coordinate type of the source grid.
 * @tparam T1 Data type of the destination grid.
 * @tparam R1 Coordinate type of the destination grid.
 * @param src The source grid to sample from.
 * @param dest The destination grid to fill with interpolated values.
 * @param fill_value The value to use for points outside the source grid.
 */
template<typename T0, typename R0, typename T1, typename R1>
void
interpolate(rect3d<T0, R0>& src, rect3d<T1, R1>& dest, T1& fill_value) {

    std::cout << src.x_[0] << ", " << src.y_[0] << ", " << src.z_[0] << std::endl;
    std::cout << dest.x_[0] << ", " << dest.y_[0] << ", " << dest.z_[0] << std::endl;

    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    ///< center point of destination
    dest.data_.resize(nX * nY * nZ);

    mqi::vec3<R1> p       = { dest.x_[0], dest.y_[0], dest.z_[0] };
    size_t        counter = 0;
    for (size_t k = 0; k < nZ; ++k) {
        p.z = dest.z_[k];
        for (size_t j = 0; j < nY; ++j) {
            p.y = dest.y_[j];
            for (size_t i = 0; i < nX; ++i) {
                p.x                                = dest.x_[i];
                dest.data_[dest.ijk2data(i, j, k)] = src.is_in_point(p) ? src(p) : fill_value;

            }   //x
        }       //y
    }           //z
}

/**
 * @brief Warps a source grid to a destination grid's space using a deformation vector field (DVF).
 * @details This function implements a "pull" or "backward" warping. For each voxel in the destination grid,
 * it uses the DVF to find the corresponding point in the source grid's space and then interpolates the source
 * data at that new point.
 * @tparam T0 Data type of the source and destination grids.
 * @tparam R0 Coordinate type of all grids.
 * @tparam S0 Scalar type of the DVF vectors.
 * @param src The source grid containing the data to be warped.
 * @param dest The destination grid to be filled with the warped data.
 * @param dvf The deformation vector field, which maps points from the destination space to the source space.
 * @param fill_value The value to use if the warped point falls outside the source grid.
 */
template<typename T0, typename R0, typename S0>
void
warp_linear(rect3d<T0, R0>&            src,
            rect3d<T0, R0>&            dest,
            rect3d<mqi::vec3<S0>, R0>& dvf,
            T0                         fill_value) {
    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    dest.data_.resize(nX * nY * nZ);

    ///< Looping reference
    mqi::vec3<R0> p_dest(dest.x_[0], dest.y_[0], dest.z_[0]);

    for (size_t k = 0; k < nZ; ++k) {
        p_dest.z = dest.z_[k];
        for (size_t j = 0; j < nY; ++j) {
            p_dest.y = dest.y_[j];
            for (size_t i = 0; i < nX; ++i) {
                p_dest.x = dest.x_[i];

                T0 value;
                if (dvf.is_in_point(p_dest)) {
                    ///
                    /// If destination point is in DVF grid points,
                    /// apply translation and then check new position is in source
                    /// Then, assign value by interpolating values at 8 coners surrounding new position.
                    ///
                    mqi::vec3<R0> p_new = p_dest + dvf(p_dest);
                    value               = src.is_in_point(p_new) ? src(p_new) : fill_value;
                } else {
                    value = src.is_in_point(p_dest) ? src(p_dest) : fill_value;

                }   //vf.is_in_point
                dest.data_[dest.ijk2data(i, j, k)] = value;

            }   //x
        }       //y
    }           //z
}

}   // namespace mqi

#endif
