/// \file
///
/// \brief This file defines the uni_1d class, a 1-dimensional uniform distribution.
///
/// The uni_1d class generates random samples from a 1-dimensional uniform distribution
/// over a specified range.

#ifndef MQI_UNI_1D_H
#define MQI_UNI_1D_H

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi {

/// \class uni_1d
/// \brief A 1-dimensional uniform probability distribution function.
/// \tparam T The floating-point type of the distribution (e.g., float or double).
///
/// This class provides a 1D uniform distribution. It samples random values
/// from a specified range [min, max].
template<typename T>
class uni_1d : public pdf_Md<T, 1> {
public:
    ///< C++ distribution function for uniform distribution.
    std::uniform_real_distribution<T> func_;

    /// \brief Constructs a new uni_1d object.
    /// \param[in] m An array where m[0] is the minimum value of the range.
    /// \param[in] s An array where s[0] is the maximum value of the range.
    CUDA_HOST_DEVICE
    uni_1d(
        std::array<T, 1>& m,
        std::array<T, 1>& s)
    : pdf_Md<T, 1>(m, s) {
#if !defined(__CUDACC__)
        func_ = std::uniform_real_distribution<T>(m[0], s[0]);
#endif
    }

    /// \brief Constructs a new uni_1d object with const parameters.
    /// \param[in] m An array where m[0] is the minimum value of the range.
    /// \param[in] s An array where s[0] is the maximum value of the range.
    CUDA_HOST_DEVICE
    uni_1d(
        const std::array<T, 1>& m,
        const std::array<T, 1>& s)
    : pdf_Md<T, 1>(m, s) {
#if !defined(__CUDACC__)
        func_ = std::uniform_real_distribution<T>(m[0], s[0]);
#endif
    }

    /// \brief Samples the 1D uniform distribution.
    /// \param[in, out] rng A pointer to a random number engine.
    /// \return An array containing a single random value sampled from the uniform distribution.
    CUDA_HOST_DEVICE
    virtual
    std::array<T, 1>
    operator()(std::default_random_engine* rng) {
        return { func_(*rng) };
    };
};

}
#endif