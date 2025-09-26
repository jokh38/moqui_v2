/**
 * @file
 * @brief Defines 3x3 and 4x4 matrix classes for 3D transformations.
 */
#ifndef MQI_MATRIX_H
#define MQI_MATRIX_H

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/**
 * @class mat3x3
 * @brief A class for 3x3 matrix operations, primarily for 3D rotations.
 * @tparam T The data type of the matrix elements (e.g., float or double).
 */
template<typename T>
class mat3x3
{
public:
    T x;    ///< Rotation angle around the x-axis in radians.
    T y;    ///< Rotation angle around the y-axis in radians.
    T z;    ///< Rotation angle around the z-axis in radians.
    T xx;   ///< Element at row 1, column 1.
    T xy;   ///< Element at row 1, column 2.
    T xz;   ///< Element at row 1, column 3.
    T yx;   ///< Element at row 2, column 1.
    T yy;   ///< Element at row 2, column 2.
    T yz;   ///< Element at row 2, column 3.
    T zx;   ///< Element at row 3, column 1.
    T zy;   ///< Element at row 3, column 2.
    T zz;   ///< Element at row 3, column 3.

    /**
     * @brief Default constructor. Initializes to an identity matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3() :
        x(0), y(0), z(0), xx(1.0), xy(0), xz(0), yx(0), yy(1.0), yz(0), zx(0), zy(0), zz(1.0) {
        ;
    }

    /**
     * @brief Constructs a matrix from 9 specified elements.
     * @param[in] xx Element at row 1, column 1.
     * @param[in] xy Element at row 1, column 2.
     * @param[in] xz Element at row 1, column 3.
     * @param[in] yx Element at row 2, column 1.
     * @param[in] yy Element at row 2, column 2.
     * @param[in] yz Element at row 2, column 3.
     * @param[in] zx Element at row 3, column 1.
     * @param[in] zy Element at row 3, column 2.
     * @param[in] zz Element at row 3, column 3.
     */
    CUDA_HOST_DEVICE
    mat3x3(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
        x(0), y(0), z(0), xx(xx), xy(xy), xz(xz), yx(yx), yy(yy), yz(yz), zx(zx), zy(zy), zz(zz) {
        ;
    }

    /**
     * @brief Copy constructor.
     * @param[in] ref The matrix to copy from.
     */
    CUDA_HOST_DEVICE
    mat3x3(const mat3x3& ref) {
        x  = ref.x;
        y  = ref.y;
        z  = ref.z;
        xx = ref.xx;
        xy = ref.xy;
        xz = ref.xz;
        yx = ref.yx;
        yy = ref.yy;
        yz = ref.yz;
        zx = ref.zx;
        zy = ref.zy;
        zz = ref.zz;
    }

    /**
     * @brief Constructs a rotation matrix from Euler angles.
     * @param[in] a Rotation angle around the x-axis in radians.
     * @param[in] b Rotation angle around the y-axis in radians.
     * @param[in] c Rotation angle around the z-axis in radians.
     * @details Rotations are applied in x, then y, then z order.
     */
    CUDA_HOST_DEVICE
    mat3x3(T a, T b, T c) :
        x(a), y(b), z(c), xx(1.0), xy(0), xz(0), yx(0), yy(1.0), yz(0), zx(0), zy(0), zz(1.0) {
        //this routine may need a sequence of rotation,
        //e.g., x->y->z, y->x->z, etc. a total of 12
        if (x != 0) this->rotate_x(x);
        if (y != 0) this->rotate_y(y);
        if (z != 0) this->rotate_z(z);
    }

    /**
     * @brief Constructs a rotation matrix from an array of Euler angles.
     * @param[in] abc An array containing rotation angles {x, y, z} in radians.
     */
    CUDA_HOST_DEVICE
    mat3x3(std::array<T, 3>& abc) :
        x(abc[0]), y(abc[1]), z(abc[2]), xx(1.0), xy(0), xz(0), yx(0), yy(1.0), yz(0), zx(0), zy(0),
        zz(1.0) {
        if (x != 0) this->rotate_x(x);
        if (y != 0) this->rotate_y(y);
        if (z != 0) this->rotate_z(z);
    }

    /**
     * @brief Constructs a rotation matrix that aligns one vector to another.
     * @param[in] f The source vector (from).
     * @param[in] t The target vector (to).
     * @details Uses the method from "An efficient method for aligning a 3D vector with a target vector" by Moller & Hughes (1999). Expects normalized vectors.
     */
    CUDA_HOST_DEVICE
    mat3x3(const vec3<T>& f, const vec3<T>& t) {
        ///< a matrix aligns vector (f) to vector (t)
        ///< by Moller & Hughe, 1999
        vec3<T> v = f.cross(t);
        T       c = f.dot(t) / (f.norm() * t.norm());
        T       h = 0;
        if (mqi::mqi_abs(c - 1) < mqi::geometry_tolerance ||
            mqi::mqi_abs(c + 1) < mqi::geometry_tolerance) {
            //// If f and t are completely opposit
            //            c   = f.dot(t);
            //            printf("vector are aligned\n");
            //            c = 1;
            vec3<T> x(1, 0, 0);
            vec3<T> uu = x - f;
            vec3<T> vv = x - t;
            uu.normalize();
            vv.normalize();
            T dot_u  = uu.dot(uu);
            T dot_v  = vv.dot(vv);
            T dot_uv = vv.dot(uu);
            xx       = 1 - 2 / dot_u * uu.x * uu.x - 2 / dot_v * vv.x * vv.x +
                 4 * dot_uv / (dot_u * dot_v) * vv.x * uu.x;
            xy = 0 - 2 / dot_u * uu.x * uu.y - 2 / dot_v * vv.x * vv.y +
                 4 * dot_uv / (dot_u * dot_v) * vv.x * uu.y;
            xz = 0 - 2 / dot_u * uu.x * uu.z - 2 / dot_v * vv.x * vv.z +
                 4 * dot_uv / (dot_u * dot_v) * vv.x * uu.z;

            yx = 0 - 2 / dot_u * uu.y * uu.x - 2 / dot_v * vv.y * vv.x +
                 4 * dot_uv / (dot_u * dot_v) * vv.y * uu.x;
            yy = 1 - 2 / dot_u * uu.y * uu.y - 2 / dot_v * vv.y * vv.y +
                 4 * dot_uv / (dot_u * dot_v) * vv.y * uu.y;
            yz = 0 - 2 / dot_u * uu.y * uu.z - 2 / dot_v * vv.y * vv.z +
                 4 * dot_uv / (dot_u * dot_v) * vv.y * uu.z;

            zx = 0 - 2 / dot_u * uu.z * uu.x - 2 / dot_v * vv.z * vv.x +
                 4 * dot_uv / (dot_u * dot_v) * vv.z * uu.x;
            zy = 0 - 2 / dot_u * uu.z * uu.y - 2 / dot_v * vv.z * vv.y +
                 4 * dot_uv / (dot_u * dot_v) * vv.z * uu.y;
            zz = 1 - 2 / dot_u * uu.z * uu.z - 2 / dot_v * vv.z * vv.z +
                 4 * dot_uv / (dot_u * dot_v) * vv.z * uu.z;
        } else {
            //// Below coded is needed for debugging
#ifdef DEBUG
            printf("they are not aligned\n");
            printf("f ");
            f.dump();
            printf("t ");
            t.dump();
            c = f.dot(t) / (f.norm() * t.norm());
#endif
            h  = (1.0 - c) / (1.0 - c * c);
            xx = c + h * v.x * v.x;
            xy = h * v.x * v.y - v.z;
            xz = h * v.x * v.z + v.y;
            yx = h * v.x * v.y + v.z;
            yy = c + h * v.y * v.y;
            yz = h * v.y * v.z - v.x;
            zx = h * v.x * v.z - v.y;
            zy = h * v.y * v.z + v.x;
            zz = c + h * v.z * v.z;
        }
    }

    /**
     * @brief Destructor.
     */
    CUDA_HOST_DEVICE ~mat3x3() {
        ;
    }

    /**
     * @brief Applies a rotation to the current matrix using Euler angles.
     * @param[in] a Rotation angle around the x-axis in radians.
     * @param[in] b Rotation angle around the y-axis in radians.
     * @param[in] c Rotation angle around the z-axis in radians.
     */
    CUDA_HOST_DEVICE
    void
    rotate(T a, T b, T c) {
        x = a;
        y = b;
        z = c;
        if (x != 0) this->rotate_x(x);
        if (y != 0) this->rotate_y(y);
        if (z != 0) this->rotate_z(z);
    }

    /**
     * @brief Calculates the Euler angles (psi, theta, phi) for the rotation matrix.
     * @param[in] y_is_2nd_quad A flag to resolve ambiguity, not currently used.
     * @return A vec3<T> containing the Euler angles (x: psi, y: theta, z: phi).
     */
    CUDA_HOST
    vec3<T>
    euler_xyz(bool y_is_2nd_quad = false) {

        vec3<T> psi_th_phi;

        if (zx > -1.0 && zx < 1.0) {   //zx neq -1 or 1
            T psi[2], th[2], phi[2];
            th[0]  = -1.0 * std::asin(zx);
            th[1]  = M_PI - th[0];
            psi[0] = std::atan2(zy / std::cos(th[0]), zz / std::cos(th[0]));
            psi[1] = std::atan2(zy / std::cos(th[1]), zz / std::cos(th[1]));
            phi[0] = std::atan2(yx / std::cos(th[0]), xx / std::cos(th[0]));
            phi[1] = std::atan2(yx / std::cos(th[1]), xx / std::cos(th[1]));

            if (th[0] > 0 && th[0] < M_PI * 0.5) {
                psi_th_phi.x = psi[0];
                psi_th_phi.y = th[0];
                psi_th_phi.z = phi[0];
            } else {
                psi_th_phi.x = psi[1];
                psi_th_phi.y = th[1];
                psi_th_phi.z = phi[1];
            }

        } else {
            std::cout << "here\n";   //never get here
            psi_th_phi.z = 0.0;
            if (zx == -1.0) {   //zx = -1
                psi_th_phi.x = psi_th_phi.z + std::atan2(xy, xz);
                psi_th_phi.y = M_PI * 0.5;
            } else {
                psi_th_phi.x = -1.0 * psi_th_phi.z + std::atan2(-1.0 * xy, -1.0 * xz);
                psi_th_phi.y = -1.0 * M_PI * 0.5;
            }
        }
        return psi_th_phi;
    }

    /**
     * @brief Post-multiplies the matrix by a rotation around the x-axis.
     * @param[in] a The rotation angle in radians.
     * @return A reference to the modified matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3&
    rotate_x(T a) {
        x = a;
#if defined(__CUDACC__)
        T c1 = cosf(x);
        T s1 = sinf(x);
#else
        T c1 = std::cos(x);
        T s1 = std::sin(x);
#endif
        T x1 = yx, y1 = yy, z1 = yz;
        yx = c1 * x1 - s1 * zx;
        yy = c1 * y1 - s1 * zy;
        yz = c1 * z1 - s1 * zz;
        zx = s1 * x1 + c1 * zx;
        zy = s1 * y1 + c1 * zy;
        zz = s1 * z1 + c1 * zz;
        return *this;
    }

    /**
     * @brief Post-multiplies the matrix by a rotation around the y-axis.
     * @param[in] a The rotation angle in radians.
     * @return A reference to the modified matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3&
    rotate_y(T a) {
        y = a;
#if defined(__CUDACC__)
        T c1 = cosf(y);
        T s1 = sinf(y);
#else
        T c1 = std::cos(y);
        T s1 = std::sin(y);
#endif
        T x1 = zx, y1 = zy, z1 = zz;
        zx = c1 * x1 - s1 * xx;
        zy = c1 * y1 - s1 * xy;
        zz = c1 * z1 - s1 * xz;
        xx = s1 * x1 + c1 * xx;
        xy = s1 * y1 + c1 * xy;
        xz = s1 * z1 + c1 * xz;
        return *this;
    }

    /**
     * @brief Post-multiplies the matrix by a rotation around the z-axis.
     * @param[in] a The rotation angle in radians.
     * @return A reference to the modified matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3&
    rotate_z(T a) {
        z = a;
#if defined(__CUDACC__)
        T c1 = cosf(z);
        T s1 = sinf(z);
#else
        T c1 = std::cos(z);
        T s1 = std::sin(z);
#endif
        T x1 = xx, y1 = xy, z1 = xz;
        xx = c1 * x1 - s1 * yx;
        xy = c1 * y1 - s1 * yy;
        xz = c1 * z1 - s1 * yz;
        yx = s1 * x1 + c1 * yx;
        yy = s1 * y1 + c1 * yy;
        yz = s1 * z1 + c1 * yz;
        return *this;
    }

    /**
     * @brief Multiplies the matrix by a 3-element std::array.
     * @param[in] r The array to multiply.
     * @return The transformed array.
     */
    CUDA_HOST_DEVICE
    std::array<T, 3>
    operator*(const std::array<T, 3>& r) const {
        return std::array<T, 3>({ xx * r[0] + xy * r[1] + xz * r[2],
                                  yx * r[0] + yy * r[1] + yz * r[2],
                                  zx * r[0] + zy * r[1] + zz * r[2] });
    }

    /**
     * @brief Multiplies the matrix by a 3D vector.
     * @param[in] r The vector to multiply.
     * @return The transformed vector.
     */
    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const vec3<T>& r) const {
        return vec3<T>(xx * r.x + xy * r.y + xz * r.z,
                       yx * r.x + yy * r.y + yz * r.z,
                       zx * r.x + zy * r.y + zz * r.z);
    }

    /**
     * @brief Computes the inverse of the rotation matrix (which is its transpose).
     * @return The inverted matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3
    inverse() const {
        return mat3x3<T>(xx, yx, zx, xy, yy, zy, xz, yz, zz);
    }

    /**
     * @brief Multiplies this matrix by another 3x3 matrix.
     * @param[in] r The matrix to multiply by.
     * @return The resulting matrix.
     */
    CUDA_HOST_DEVICE
    mat3x3<T>
    operator*(const mat3x3<T>& r) const {
        return mat3x3<T>(xx * r.xx + xy * r.yx + xz * r.zx,
                         xx * r.xy + xy * r.yy + xz * r.zy,
                         xx * r.xz + xy * r.yz + xz * r.zz,
                         yx * r.xx + yy * r.yx + yz * r.zx,
                         yx * r.xy + yy * r.yy + yz * r.zy,
                         yx * r.xz + yy * r.yz + yz * r.zz,
                         zx * r.xx + zy * r.yx + zz * r.zx,
                         zx * r.xy + zy * r.yy + zz * r.zy,
                         zx * r.xz + zy * r.yz + zz * r.zz);
    }

    /**
     * @brief Dumps the matrix elements to the console for debugging.
     */
    CUDA_HOST_DEVICE
    void
    dump() const {
#if defined(__CUDACC__)
        printf("(xx, xy, xz) -> (%f, %f, %f)\n", xx, xy, xz);
        printf("(yx, yy, yz) -> (%f, %f, %f)\n", yx, yy, yz);
        printf("(zx, zy, zz) -> (%f, %f, %f)\n", zx, zy, zz);
#else
        std::cout << "(xx, xy, xz) -> (" << xx << " " << xy << " " << xz << ")" << std::endl;
        std::cout << "(yx, yy, yz) -> (" << yx << " " << yy << " " << yz << ")" << std::endl;
        std::cout << "(zx, zy, zz) -> (" << zx << " " << zy << " " << zz << ")" << std::endl;
#endif
    }
};

/**
 * @class mat4x4
 * @brief A class for 4x4 matrix operations, for 3D transformations (rotation and translation).
 * @tparam T The data type of the matrix elements (e.g., float or double).
 */
template<typename T>
class mat4x4
{
public:
    //matrix elements
    T xx;   ///< Element at row 1, column 1.
    T xy;   ///< Element at row 1, column 2.
    T xz;   ///< Element at row 1, column 3.
    T xs;   ///< Element at row 1, column 4 (x-translation).
    T yx;   ///< Element at row 2, column 1.
    T yy;   ///< Element at row 2, column 2.
    T yz;   ///< Element at row 2, column 3.
    T ys;   ///< Element at row 2, column 4 (y-translation).
    T zx;   ///< Element at row 3, column 1.
    T zy;   ///< Element at row 3, column 2.
    T zz;   ///< Element at row 3, column 3.
    T zs;   ///< Element at row 3, column 4 (z-translation).
    T sx;   ///< Element at row 4, column 1.
    T sy;   ///< Element at row 4, column 2.
    T sz;   ///< Element at row 4, column 3.
    T ss;   ///< Element at row 4, column 4.

    /**
     * @brief Default constructor. Initializes to an identity matrix.
     */
    CUDA_HOST_DEVICE
    mat4x4() :
        xx(1.0), xy(0), xz(0), xs(0), yx(0), yy(1.0), yz(0), ys(0), zx(0), zy(0), zz(1.0), zs(0),
        sx(0), sy(0), sz(0.0), ss(1.0) {
        ;
    }

    /**
     * @brief Constructs a matrix from 16 specified elements.
     */
    CUDA_HOST_DEVICE
    mat4x4(T xx,
           T xy,
           T xz,
           T xs,
           T yx,
           T yy,
           T yz,
           T ys,
           T zx,
           T zy,
           T zz,
           T zs,
           T sx,
           T sy,
           T sz,
           T ss) :
        xx(xx),
        xy(xy), xz(xz), xs(xs), yx(yx), yy(yy), yz(yz), ys(ys), zx(zx), zy(zy), zz(zz), zs(zs),
        sx(sx), sy(sy), sz(sz), ss(ss) {
        ;
    }

    /**
     * @brief Copy constructor.
     * @param[in] ref The matrix to copy from.
     */
    CUDA_HOST_DEVICE
    mat4x4(const mat4x4& ref) {
        xx = ref.xx;
        xy = ref.xy;
        xz = ref.xz;
        xs = ref.xs;

        yx = ref.yx;
        yy = ref.yy;
        yz = ref.yz;
        ys = ref.ys;

        zx = ref.zx;
        zy = ref.zy;
        zz = ref.zz;
        zs = ref.zs;

        sx = ref.sx;
        sy = ref.sy;
        sz = ref.sz;
        ss = ref.ss;
    }

    /**
     * @brief Constructs a matrix from a raw array of 16 elements.
     * @param[in] a Pointer to an array of 16 elements in row-major order.
     */
    CUDA_HOST_DEVICE
    mat4x4(const T* a) {
        xx = a[0];
        xy = a[1];
        xz = a[2];
        xs = a[3];
        yx = a[4];
        yy = a[5];
        yz = a[6];
        ys = a[7];
        zx = a[8];
        zy = a[9];
        zz = a[10];
        zs = a[11];
        sx = a[12];
        sy = a[13];
        sz = a[14];
        ss = a[15];
    }

    /**
     * @brief Constructs a transformation matrix from a rotation matrix and a translation vector.
     * @param[in] rot The 3x3 rotation matrix.
     * @param[in] tra The 3D translation vector.
     */
    CUDA_HOST_DEVICE
    mat4x4(const mat3x3<T>& rot, const vec3<T>& tra) {
        xx = rot.xx;
        xy = rot.xy;
        xz = rot.xz;
        xs = tra.x;

        yx = rot.yx;
        yy = rot.yy;
        yz = rot.yz;
        ys = tra.y;

        zx = rot.zx;
        zy = rot.zy;
        zz = rot.zz;
        zs = tra.z;

        sx = 0.0;
        sy = 0.0;
        sz = 0.0;
        ss = 1.0;
    }

    /**
     * @brief Constructs a transformation matrix from a rotation matrix (no translation).
     * @param[in] rot The 3x3 rotation matrix.
     */
    CUDA_HOST_DEVICE
    mat4x4(const mat3x3<T>& rot) {
        xx = rot.xx;
        xy = rot.xy;
        xz = rot.xz;
        xs = 0.0;

        yx = rot.yx;
        yy = rot.yy;
        yz = rot.yz;
        ys = 0.0;

        zx = rot.zx;
        zy = rot.zy;
        zz = rot.zz;
        zs = 0.0;

        sx = 0.0;
        sy = 0.0;
        sz = 0.0;
        ss = 1.0;
    }

    /**
     * @brief Constructs a transformation matrix from a translation vector (no rotation).
     * @param[in] ref The 3D translation vector.
     */
    CUDA_HOST_DEVICE
    mat4x4(const vec3<T>& ref) {
        xx = 1.0;
        xy = 0.0;
        xz = 0.0;
        xs = ref.x;

        yx = 0.0;
        yy = 1.0;
        yz = 0.0;
        ys = ref.y;

        zx = 0.0;
        zy = 0.0;
        zz = 1.0;
        zs = ref.z;

        sx = 0.0;
        sy = 0.0;
        sz = 0.0;
        ss = 1.0;
    }

    /**
     * @brief Destructor.
     */
    CUDA_HOST_DEVICE
    ~mat4x4() {
        ;
    }

    /**
     * @brief Multiplies the matrix by a 4-element std::array.
     * @param[in] r The array to multiply.
     * @return The transformed array.
     */
    CUDA_HOST_DEVICE
    std::array<T, 4>
    operator*(const std::array<T, 4>& r) const {
        return std::array<T, 4>({ xx * r[0] + xy * r[1] + xz * r[2] + xs * r[3],
                                  yx * r[0] + yy * r[1] + yz * r[2] + ys * r[3],
                                  zx * r[0] + zy * r[1] + zz * r[2] + zs * r[3],
                                  sx * r[0] + sy * r[1] + sz * r[2] + ss * r[3] });
    }

    /**
     * @brief Multiplies the matrix by a 4D vector.
     * @param[in] r The vector to multiply.
     * @return The transformed vector.
     */
    CUDA_HOST_DEVICE
    vec4<T>
    operator*(const vec4<T>& r) const {
        return vec4<T>(xx * r.x + xy * r.y + xz * r.z + xs * r.s,
                       yx * r.x + yy * r.y + yz * r.z + ys * r.s,
                       zx * r.x + zy * r.y + zz * r.z + zs * r.s,
                       sx * r.x + sy * r.y + sz * r.z + ss * r.s);
    }

    /**
     * @brief Multiplies the matrix by a 3D vector (point).
     * @param[in] r The vector to multiply.
     * @return The transformed vector.
     * @details Assumes the vector `r` has a homogeneous coordinate of 1.
     */
    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const vec3<T>& r) const {
        return vec3<T>(xx * r.x + xy * r.y + xz * r.z + xs,
                       yx * r.x + yy * r.y + yz * r.z + ys,
                       zx * r.x + zy * r.y + zz * r.z + zs);
    }

    /**
     * @brief Multiplies this matrix by another 4x4 matrix.
     * @param[in] r The matrix to multiply by.
     * @return The resulting matrix.
     */
    CUDA_HOST_DEVICE
    mat4x4<T>
    operator*(const mat4x4<T>& r) const {
        return mat4x4<T>(xx * r.xx + xy * r.yx + xz * r.zx + xs * r.sx,
                         xx * r.xy + xy * r.yy + xz * r.zy + xs * r.sy,
                         xx * r.xz + xy * r.yz + xz * r.zz + xs * r.sz,
                         xx * r.xs + xy * r.ys + xz * r.zs + xs * r.ss,

                         yx * r.xx + yy * r.yx + yz * r.zx + ys * r.sx,
                         yx * r.xy + yy * r.yy + yz * r.zy + ys * r.sy,
                         yx * r.xz + yy * r.yz + yz * r.zz + ys * r.sz,
                         yx * r.xs + yy * r.ys + yz * r.zs + ys * r.ss,

                         zx * r.xx + zy * r.yx + zz * r.zx + zs * r.sx,
                         zx * r.xy + zy * r.yy + zz * r.zy + zs * r.sy,
                         zx * r.xz + zy * r.yz + zz * r.zz + zs * r.sz,
                         zx * r.xs + zy * r.ys + zz * r.zs + zs * r.ss,

                         sx * r.xx + sy * r.yx + sz * r.zx + ss * r.sx,
                         sx * r.xy + sy * r.yy + sz * r.zy + ss * r.sy,
                         sx * r.xz + sy * r.yz + sz * r.zz + ss * r.sz,
                         sx * r.xs + sy * r.ys + sz * r.zs + ss * r.ss);
    }

    /**
     * @brief Dumps the matrix elements to the console for debugging.
     */
    CUDA_HOST_DEVICE
    void
    dump() const {
#if defined(__CUDACC__)
        printf("(xx, xy, xz, xs) -> (%f, %f, %f, %f)\n", xx, xy, xz, xs);
        printf("(yx, yy, yz, ys) -> (%f, %f, %f, %f)\n", yx, yy, yz, ys);
        printf("(zx, zy, zz, zs) -> (%f, %f, %f, %f)\n", zx, zy, zz, zs);
        printf("(sx, sy, sz, ss) -> (%f, %f, %f, %f)\n", sx, sy, sz, ss);
#else
        std::cout << "(xx, xy, xz, xs) -> (" << xx << ", " << xy << ", " << xz << ", " << xs << ")"
                  << std::endl;
        std::cout << "(yx, yy, yz, ys) -> (" << yx << ", " << yy << ", " << yz << ", " << ys << ")"
                  << std::endl;
        std::cout << "(zx, zy, zz, zs) -> (" << zx << ", " << zy << ", " << zz << ", " << zs << ")"
                  << std::endl;
        std::cout << "(zx, zy, zz, ss) -> (" << sx << ", " << sy << ", " << sz << ", " << ss << ")"
                  << std::endl;
#endif
    }
};

}   // namespace mqi
#endif
