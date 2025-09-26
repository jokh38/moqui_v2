#ifndef MQI_VEC_H
#define MQI_VEC_H

/// \file
///
/// RT-Ion vector for 2D, 3D, and 4D.

#include <array>
#include <cmath>
#include <iostream>
#include <moqui/base/mqi_common.hpp>

namespace mqi
{

/// @class vec2
/// @brief A 2D vector class for position and direction representation.
///
/// This class provides basic vector operations for 2D coordinates. It is designed
/// to be compatible with both CPU and CUDA device code.
///
/// @tparam T The numeric type of the vector elements (e.g., `float` or `double`).
template<typename T>
class vec2
{
public:
    T x; ///< The x-component of the vector.
    T y; ///< The y-component of the vector.

    /// @brief Copy constructor (from non-const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec2(vec2& ref) {
        x = ref.x;
        y = ref.y;
    }

    /// @brief Default constructor, initializes to (0, 0).
    CUDA_HOST_DEVICE
    vec2() : x(0), y(0) {
        ;
    }

    /// @brief Constructor with initial values.
    /// @param a The initial x-component.
    /// @param b The initial y-component.
    CUDA_HOST_DEVICE
    vec2(T a, T b) : x(a), y(b) {
        ;
    }

    /// @brief Copy constructor (from const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec2(const vec2& ref) : x(ref.x), y(ref.y) {
        ;
    }

    /// @brief Constructor from a `std::array`.
    /// @param ref An array of size 2 containing the x and y components.
    CUDA_HOST
    vec2(const std::array<T, 2>& ref) : x(ref[0]), y(ref[1]) {
        ;
    }

    /// @brief Destructor.
    CUDA_HOST_DEVICE
    ~vec2() {
        ;
    }

    /// @brief Calculates the Euclidean norm (magnitude) of the vector.
    /// @return The norm of the vector.
    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    T
    norm() const {
        return sqrtf(x * x + y * y);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y);
    }
#endif

    /// @brief Calculates the dot product with another vector.
    /// @param v The other vector.
    /// @return The dot product.
    CUDA_HOST_DEVICE
    T
    dot(const vec2<T>& v) const {
        return x * v.x + y * v.y;
    }

    /// @brief Calculates the 2D cross product (a scalar).
    /// @param v The other vector.
    /// @return The scalar result of the 2D cross product.
    CUDA_HOST_DEVICE
    T
    cross(const vec2<T>& v) const {
        return x * v.y - y * v.x;
    }

    /// @brief Assignment operator.
    /// @param r The vector to assign from.
    /// @return A reference to this vector.
    CUDA_HOST_DEVICE
    vec2<T>&
    operator=(const vec2<T>& r) {
        x = r.x;
        y = r.y;
        return *this;
    }

    /// @brief Vector addition.
    /// @param r The vector to add.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec2<T>
    operator+(const vec2<T>& r) const {
        return vec2<T>(x + r.x, y + r.y);
    }

    /// @brief Vector subtraction.
    /// @param r The vector to subtract.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec2<T>
    operator-(const vec2<T>& r) const {
        return vec2<T>(x - r.x, y - r.y);
    }

    /// @brief Scalar multiplication.
    /// @param r The scalar value.
    /// @return The scaled vector.
    CUDA_HOST_DEVICE
    vec2<T>
    operator*(const T& r) const {
        return vec2<T>(x * r, y * r);
    }

    /// @brief Scalar multiplication with a different numeric type.
    /// @tparam R The type of the scalar.
    /// @param r The scalar value.
    /// @return The scaled vector.
    template<typename R>
    vec2<T>
    operator*(const R& r) const {
        return vec2<T>(x * r, y * r);
    }

    /// @brief Component-wise vector multiplication.
    /// @param r The other vector.
    /// @return The component-wise product vector.
    CUDA_HOST_DEVICE
    vec2<T>
    operator*(const vec2<T>& r) const {
        return vec2<T>(x * r.x, y * r.y);
    }

    /// @brief Dumps the vector's components to the console.
    CUDA_HOST_DEVICE
    void
    dump() const {
        printf("(x,y): (%f, %f)\n", x, y);
    }
};

/// @class vec3
/// @brief A 3D vector class for position and direction representation.
///
/// This class provides basic vector operations for 3D coordinates. It is designed
/// to be compatible with both CPU and CUDA device code.
///
/// @tparam T The numeric type of the vector elements (e.g., `float` or `double`).
template<typename T>
class vec3
{
public:
    T x; ///< The x-component of the vector.
    T y; ///< The y-component of the vector.
    T z; ///< The z-component of the vector.

    /// @brief Copy constructor (from non-const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec3(vec3& ref) {
        x = ref.x;
        y = ref.y;
        z = ref.z;
    }

    /// @brief Default constructor, initializes to (0, 0, 0).
    CUDA_HOST_DEVICE
    vec3() : x(0), y(0), z(0) {
        ;
    }

    /// @brief Constructor with initial values.
    /// @param a The initial x-component.
    /// @param b The initial y-component.
    /// @param c The initial z-component.
    CUDA_HOST_DEVICE
    vec3(T a, T b, T c) : x(a), y(b), z(c) {
        ;
    }

    /// @brief Copy constructor (from const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec3(const vec3& ref) : x(ref.x), y(ref.y), z(ref.z) {
        ;
    }

    /// @brief Constructor from a `std::array`.
    /// @param ref An array of size 3 containing the x, y, and z components.
    CUDA_HOST
    vec3(const std::array<T, 3>& ref) : x(ref[0]), y(ref[1]), z(ref[2]) {
        ;
    }

    /// @brief Constructor from a pointer to an array.
    /// @param ref A pointer to an array with at least 3 elements.
    CUDA_HOST_DEVICE
    vec3(const T* ref) : x(ref[0]), y(ref[1]), z(ref[2]) {
        ;
    }

    /// @brief Destructor.
    CUDA_HOST_DEVICE
    ~vec3() {
        ;
    }

    /// @brief Calculates the Euclidean norm (magnitude) of the vector.
    /// @return The norm of the vector.
    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    T
    norm() const {
        return sqrtf(x * x + y * y + z * z);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
#endif

    /// @brief Normalizes the vector to unit length.
    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    void
    normalize() {
        T n = sqrtf(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
    }
#else
    void
    normalize() {
        T n = std::sqrt(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
    }
#endif

    /// @brief Vector addition.
    /// @param r The vector to add.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec3<T>
    operator+(const vec3<T>& r) const {
        return vec3<T>(x + r.x, y + r.y, z + r.z);
    }

    /// @brief Vector subtraction.
    /// @param r The vector to subtract.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec3<T>
    operator-(const vec3<T>& r) const {
        return vec3<T>(x - r.x, y - r.y, z - r.z);
    }

    /// @brief Scalar multiplication.
    /// @param r The scalar value.
    /// @return The scaled vector.
    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const T& r) const {
        return vec3<T>(x * r, y * r, z * r);
    }

    /// @brief Scalar multiplication (non-const version).
    /// @param r The scalar value.
    /// @return The scaled vector.
    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const T& r) {
        return vec3<T>(x * r, y * r, z * r);
    }

    /// @brief Scalar division.
    /// @param r The scalar value.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec3<T>
    operator/(const T& r) const {
        return vec3<T>(x / r, y / r, z / r);
    }

    /// @brief Calculates the dot product with another vector.
    /// @param v The other vector.
    /// @return The dot product.
    CUDA_HOST_DEVICE
    T
    dot(const vec3<T>& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    /// @brief Calculates the cross product with another vector.
    /// @param r The other vector.
    /// @return The resulting vector (perpendicular to both).
    CUDA_HOST_DEVICE
    vec3<T>
    cross(const vec3<T>& r) const {
        return vec3<T>(y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x);
    }

    /// @brief Assignment operator.
    /// @param r The vector to assign from.
    /// @return A reference to this vector.
    CUDA_HOST_DEVICE
    vec3<T>&
    operator=(const vec3<T>& r) {
        x = r.x;
        y = r.y;
        z = r.z;
        return *this;
    }

    /// @brief Addition-assignment operator.
    /// @param r The vector to add.
    /// @return A reference to this vector.
    CUDA_HOST_DEVICE
    vec3<T>&
    operator+=(const vec3<T>& r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }

    /// @brief Dumps the vector's components to the console.
    CUDA_HOST_DEVICE
    void
    dump() const {
#if defined(__CUDACC__)
        printf("(x, y, z) -> (%f, %f, %f)\n", x, y, z);
#else
        std::cout << "(x, y, z) -> (" << x << ", " << y << ", " << z << ") " << std::endl;
#endif
    }
};

/// @class vec4
/// @brief A 4D vector class.
///
/// This class provides basic vector operations for 4D coordinates. The fourth component `s`
/// can be used for homogeneous coordinates or other purposes. It is designed to be
/// compatible with both CPU and CUDA device code.
///
/// @tparam T The numeric type of the vector elements (e.g., `float` or `double`).
template<typename T>
class vec4
{
public:
    T x; ///< The x-component of the vector.
    T y; ///< The y-component of the vector.
    T z; ///< The z-component of the vector.
    T s; ///< The fourth component of the vector (e.g., scale or w).

    /// @brief Copy constructor (from non-const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec4(vec4& ref) {
        x = ref.x;
        y = ref.y;
        z = ref.z;
        s = ref.s;
    }

    /// @brief Default constructor, initializes to (0, 0, 0, 0).
    CUDA_HOST_DEVICE
    vec4() : x(0), y(0), z(0), s(0) {
        ;
    }

    /// @brief Constructor with initial values.
    /// @param a The initial x-component.
    /// @param b The initial y-component.
    /// @param c The initial z-component.
    /// @param d The initial s-component.
    CUDA_HOST_DEVICE
    vec4(T a, T b, T c, T d) : x(a), y(b), z(c), s(d) {
        ;
    }

    /// @brief Copy constructor (from const reference).
    /// @param ref The vector to copy from.
    CUDA_HOST_DEVICE
    vec4(const vec4& ref) : x(ref.x), y(ref.y), z(ref.z), s(ref.s) {
        ;
    }

    /// @brief Constructor from a pointer to an array.
    /// @param ref A pointer to an array with at least 4 elements.
    CUDA_HOST_DEVICE
    vec4(const T* ref) : x(ref[0]), y(ref[1]), z(ref[2]), s(ref[3]) {
        ;
    }

    /// @brief Constructor from a `std::array`.
    /// @param ref An array of size 4 containing the x, y, z, and s components.
    CUDA_HOST
    vec4(const std::array<T, 4>& ref) : x(ref[0]), y(ref[1]), z(ref[2]), s(ref[3]) {
        ;
    }

    /// @brief Destructor.
    CUDA_HOST_DEVICE
    ~vec4() {
        ;
    }

    /// @brief Calculates the Euclidean norm (magnitude) of the vector.
    /// @return The norm of the vector.
    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    T
    norm() const {
        return sqrtf(x * x + y * y + z * z + s * s);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y + z * z + s * s);
    }
#endif

    /// @brief Vector addition.
    /// @param r The vector to add.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec4<T>
    operator+(const vec4<T>& r) const {
        return vec4<T>(x + r.x, y + r.y, z + r.z, s + r.s);
    }

    /// @brief Vector subtraction.
    /// @param r The vector to subtract.
    /// @return The resulting vector.
    CUDA_HOST_DEVICE
    vec4<T>
    operator-(const vec4<T>& r) const {
        return vec4<T>(x - r.x, y - r.y, z - r.z, s - r.s);
    }

    /// @brief Scalar multiplication.
    /// @param r The scalar value.
    /// @return The scaled vector.
    CUDA_HOST_DEVICE
    vec4<T>
    operator*(const T& r) const {
        return vec4<T>(x * r, y * r, z * r, s * r);
    }

    /// @brief Dumps the vector's components to the console.
    CUDA_HOST_DEVICE
    void
    dump() const {
        printf("(x,y,z,s): (%f, %f, %f)\n", x, y, z, s);
    }
};

}   // namespace mqi

#endif
