#ifndef MQI_UTILS_H
#define MQI_UTILS_H

/// \file
/// Header file containing general functions useful for several classes
/// \see http://www.martinbroadhurst.com/how-to-trim-a-stdstring.html
/// \see https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string

#include <array>
#include <chrono>
#include <map>
#include <moqui/base/mqi_common.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace mqi
{

/// @brief Removes trailing whitespace from a string.
/// @param s The input string.
/// @param delimiters A string containing the characters to trim.
/// @return A new string with trailing delimiters removed.
inline std::string
trim_right_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

/// @brief Removes leading whitespace from a string.
/// @param s The input string.
/// @param delimiters A string containing the characters to trim.
/// @return A new string with leading delimiters removed.
inline std::string
trim_left_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return s.substr(s.find_first_not_of(delimiters));
}

/// @brief Removes leading and trailing whitespace from a string.
/// @param s The input string.
/// @param delimiters A string containing the characters to trim.
/// @return A new string with leading and trailing delimiters removed.
inline std::string
trim_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return trim_left_copy(trim_right_copy(s, delimiters), delimiters);
}

/// @brief Performs linear interpolation on data stored in a map.
/// @tparam T The data type of the values.
/// @tparam S The size of the value arrays in the map.
/// @param db A map where the key is the x-coordinate and the value is an array of y-coordinates.
/// @param x The x-value at which to interpolate.
/// @param y_col The column index of the y-value to use for interpolation.
/// @return The interpolated y-value.
template<typename T, size_t S>
inline T
interp_linear(const std::map<T, std::array<T, S>>& db, const T x, const size_t y_col = 0) {
    auto it_up = db.upper_bound(x);
    if (it_up == db.end()) { return ((--it_up)->second)[y_col]; }
    if (it_up == db.begin()) { return (it_up->second)[y_col]; }
    T    x1      = it_up->first;
    T    y1      = (it_up->second)[y_col];
    auto it_down = --it_up;
    T    x0      = it_down->first;
    T    y0      = (it_down->second)[y_col];
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

/// @brief Performs linear interpolation on tabular data stored in a vector of arrays.
/// @tparam T The data type of the values.
/// @tparam S The size of the column arrays.
/// @param db A vector of arrays representing the data table.
/// @param x The x-value at which to interpolate.
/// @param x_col The column index for the x-values.
/// @param y_col The column index for the y-values.
/// @return The interpolated y-value.
template<typename T, size_t S>
inline T
interp_linear(const std::vector<std::array<T, S>>& db,
              const T                              x,
              const size_t                         x_col = 0,
              const size_t                         y_col = 1) {
    if (x <= db[0][x_col]) return db[0][y_col];
    for (size_t i = 1; i < db.size() - 1; ++i) {
        if (x <= db[i][x_col]) {

            T x0 = db[i - 1][x_col];
            T x1 = db[i][x_col];
            T y0 = db[i - 1][y_col];
            T y1 = db[i][y_col];

            return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }
    }

    return db[db.size() - 1][y_col];
}

/// @brief Performs polynomial interpolation on a table of x-y coordinates.
/// @param vector_X An array of x-coordinates.
/// @param vector_Y An array of y-coordinates.
/// @param x The x-ordinate at which to evaluate the interpolation.
/// @param npoints The number of points in the coordinate tables.
/// @param order The order of the interpolation polynomial.
/// @return The interpolated y-value corresponding to the x-ordinate.
/// @note Adapted from gpmc code.
inline float
TableInterpolation(float* const vector_X,
                   float* const vector_Y,
                   const float  x,
                   const int    npoints,
                   int          order = 4) {
    float result;
    // check order of interpolation
    if (order > npoints) order = npoints;
    // if x is ouside the vector_X[] interval
    if (x <= vector_X[0]) return result = vector_Y[0];
    if (x >= vector_X[npoints - 1]) return result = vector_Y[npoints - 1];
    // loop to find j so that x[j-1] < x < x[j]
    int j = 0;
    while (j < npoints) {
        if (vector_X[j] >= x) break;
        j++;
    }
    // shift j to correspond to (npoint-1)th interpolation
    j = j - order / 2;
    // if j is ouside of the range [0, ... npoints-1]
    if (j < 0) j = 0;
    if (j + order > npoints) j = npoints - order;
    result = 0.0;
    // Allocate enough space for any table we'd like to read.
    float* lambda = new float[npoints];
    for (int is = j; is < j + order; is++) {
        lambda[is] = 1.0;
        for (int il = j; il < j + order; il++) {
            if (il != is)
                lambda[is] = lambda[is] * (x - vector_X[il]) / (vector_X[is] - vector_X[il]);
        }
        result += vector_Y[is] * lambda[is];
    }
    delete[] lambda;
    return result;
}

/// @brief Calculates the starting index and number of jobs for a specific thread.
///
/// This function distributes a total number of jobs among a set of threads as evenly as possible.
/// For example, with 11 jobs and 2 threads, thread 0 gets jobs 0-5 (6 jobs) and thread 1 gets jobs 6-10 (5 jobs).
/// @param n_threads The total number of threads.
/// @param n_jobs The total number of jobs to distribute.
/// @param thread_id The ID of the current thread.
/// @return A `vec2<uint32_t>` where `x` is the starting job index and `y` is the number of jobs for this thread.
CUDA_DEVICE
vec2<uint32_t>
start_and_length(const uint32_t& n_threads, const uint32_t& n_jobs, const uint32_t& thread_id) {

    uint32_t quotient  = n_jobs / n_threads;
    uint32_t remainder = n_jobs % n_threads;

    mqi::vec2<uint32_t> ret;
    ret.x = quotient * thread_id + ((thread_id >= remainder) ? remainder : thread_id);
    ret.y = quotient + 1 * (thread_id < remainder);
    return ret;
}

/// @brief A custom implementation of `std::lower_bound` for sorted arrays.
///
/// This function can be compiled for both host and device code.
/// @param arr A pointer to the sorted array.
/// @param len The length of the array.
/// @param value The value to find the lower bound for.
/// @return The index of the first element not less than `value`.
CUDA_HOST_DEVICE
int32_t
lower_bound_cpp(const int32_t* arr, const int32_t& len, const int32_t& value) {
    int32_t first = 0;
    int32_t count = len;
    int32_t step;

    int32_t it;
    while (count > 0) {

        it   = first;
        step = count / 2;
        it += step;

        if (arr[it] <= value) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return first;
}

}   // namespace mqi

#endif
