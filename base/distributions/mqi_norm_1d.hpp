#ifndef MQI_NORM_1D_H
#define MQI_NORM_1D_H

/// \file mqi_norm_1d.hpp
///
/// \brief Defines a 1-dimensional normal (Gaussian) probability distribution function.
///
/// This file contains the `norm_1d` class, which models a 1D normal distribution
/// using the C++ standard library's random number generation facilities.

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi{
/// \class norm_1d
/// \brief A 1-dimensional normal (Gaussian) probability distribution function (PDF).
///
/// This class inherits from `pdf_Md` and models a 1D Gaussian distribution.
/// It uses a `std::normal_distribution` object to generate random numbers
/// according to the specified mean and standard deviation.
///
/// \tparam T The data type of the return value (e.g., float, double).
template<typename T>
class norm_1d : public pdf_Md<T,1> {
public:
    /// \brief The C++ standard library normal distribution function object.
    std::normal_distribution<T> func_;

    /// \brief Constructs a new 1D normal distribution.
    ///
    /// Initializes the base class with the mean and standard deviation, and
    /// creates a `std::normal_distribution` object with these parameters.
    ///
    /// \param m An array containing the mean of the distribution.
    /// \param s An array containing the standard deviation of the distribution.
    CUDA_HOST_DEVICE
    norm_1d(
        std::array<T,1>& m,
        std::array<T,1>& s)
    : pdf_Md<T,1>(m,s)
    {
        func_ = std::normal_distribution<T>(m[0], s[0]);
    }

    /// \brief Constructs a new 1D normal distribution from constant references.
    ///
    /// Initializes the base class with the mean and standard deviation, and
    /// creates a `std::normal_distribution` object with these parameters.
    ///
    /// \param m A const reference to an array containing the mean of the distribution.
    /// \param s A const reference to an array containing the standard deviation of the distribution.
    CUDA_HOST_DEVICE
    norm_1d(
        const std::array<T,1>& m,
        const std::array<T,1> &s)
    : pdf_Md<T,1>(m,s)
    {
        func_ = std::normal_distribution<T>(m[0], s[0]);
    }

    /// \brief Samples the distribution.
    ///
    /// Generates a random number from the normal distribution using the provided
    /// random number engine.
    ///
    /// \param rng A pointer to a random number engine.
    /// \return An array containing the sampled value.
    CUDA_HOST_DEVICE
    virtual
    std::array<T,1>
    operator()(std::default_random_engine* rng){
        return {func_(*rng)};
    };

};

}

#endif
