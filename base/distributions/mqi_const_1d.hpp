#ifndef MQI_CONST1D_H
#define MQI_CONST1D_H

/// \file mqi_const_1d.hpp
///
/// \brief Defines a 1-dimensional constant probability distribution function.
///
/// This file contains the `const_1d` class, which represents a Dirac delta
/// distribution. When sampled, it always returns the same constant value.

#include <random>
#include <functional>
#include <queue>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <functional>
#include <array>

#include <moqui/base/mqi_vec.hpp>
#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/distributions/mqi_pdfMd.hpp>


namespace mqi{

/// \class const_1d
/// \brief A 1-dimensional constant probability distribution function (PDF).
///
/// This class inherits from `pdf_Md` and models a 1D distribution that always
/// returns a constant value (the mean). The standard deviation parameter is ignored.
/// It is useful for representing fixed-value parameters, such as a monoenergetic beam.
///
/// \tparam T The data type of the return value (e.g., float, double).
template<typename T>
class const_1d : public pdf_Md<T,1> {

public:

    /// \brief Constructs a new 1D constant distribution.
    ///
    /// \param m An array containing the mean value of the distribution.
    /// \param s An array containing the standard deviation (ignored).
    CUDA_HOST_DEVICE
    const_1d(
        std::array<T,1>& m,
        std::array<T,1>& s)
        : pdf_Md<T,1>(m,s)
    {;}

    /// \brief Constructs a new 1D constant distribution from constant references.
    ///
    /// \param m A const reference to an array containing the mean value.
    /// \param s A const reference to an array containing the standard deviation (ignored).
    CUDA_HOST_DEVICE
    const_1d(
        const std::array<T,1>& m,
        const std::array<T,1> &s)
        : pdf_Md<T,1>(m,s)
    {;}

    /// \brief Samples the distribution.
    ///
    /// This function always returns the mean value of the distribution, effectively
    /// sampling from a Dirac delta function.
    ///
    /// \param rng A pointer to a random number engine (unused).
    /// \return An array containing the mean value.
    CUDA_HOST_DEVICE
    virtual
    std::array<T,1>
    operator()(std::default_random_engine* rng){
        return pdf_Md<T,1>::mean_;
    };

};

}
#endif