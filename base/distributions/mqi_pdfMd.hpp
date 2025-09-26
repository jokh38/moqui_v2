#ifndef MQI_PDFMD_H
#define MQI_PDFMD_H

/// \file mqi_pdfMd.hpp
///
/// \brief Defines the base class for M-dimensional probability distribution functions.
///
/// This file contains the declaration of the `pdf_Md` class, which is a pure virtual
/// base class that defines the common interface for all probability distribution
/// functions used in the Moqui toolkit.

#include <random>
#include <functional>
#include <queue>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <functional>
#include <array>

#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>
#include <moqui/base/mqi_matrix.hpp>


namespace mqi{

/// \class pdf_Md
/// \brief A pure virtual base class for M-dimensional probability distribution functions (PDFs).
///
/// This class serves as the foundation for all other distribution classes. It stores
/// the mean and standard deviation for each dimension and defines the interface for
/// sampling the distribution via the `operator()` method.
///
/// \tparam T The data type of the distributed values (e.g., float, double).
/// \tparam M The number of dimensions of the distribution.
template<typename T, std::size_t M>
class pdf_Md{
protected:
    /// \brief An array storing the mean value for each of the M dimensions.
    std::array<T,M> mean_  ;
    /// \brief An array storing the standard deviation for each of the M dimensions.
    std::array<T,M> sigma_ ;
public:

    /// \brief Constructs a new M-dimensional PDF.
    ///
    /// \param m An array containing the mean values for each dimension.
    /// \param s An array containing the standard deviation values for each dimension.
    CUDA_HOST_DEVICE
    pdf_Md(
        std::array<T,M>& m, 
        std::array<T,M>& s)
    {
        for(std::size_t i=0; i < M ; ++i){
            mean_[i] = m[i];
            sigma_[i] = s[i];
        }
    }

    /// \brief Constructs a new M-dimensional PDF from constant references.
    ///
    /// \param m A const reference to an array containing the mean values.
    /// \param s A const reference to an array containing the standard deviation values.
    CUDA_HOST_DEVICE
    pdf_Md(
        const std::array<T,M>& m, 
        const std::array<T,M> &s)
    {
        for(std::size_t i=0; i < M ; ++i){
            mean_[i] = m[i];
            sigma_[i] = s[i];
        }
    }
    
    /// \brief Prints the means and standard deviations of the distribution to the console.
    ///
    /// This method is useful for debugging purposes.
    CUDA_HOST_DEVICE
    void 
    dump(){
        for(size_t i=0 ; i < M ; ++i)
        std::cout<< "mean: " << mean_[i] << ", sigma: " << sigma_[i] << std::endl;
    }
    
    /// \brief Virtual destructor for the base class.
    CUDA_HOST_DEVICE
    virtual ~pdf_Md(){;}


    /// \brief Pure virtual function for sampling the distribution.
    ///
    /// Derived classes must implement this operator to provide the specific
    /// sampling logic for their distribution.
    ///
    /// \param rng A pointer to a random number engine.
    /// \return An array of size M containing the sampled values.
    CUDA_HOST_DEVICE
    virtual 
    std::array<T,M>
    operator()(std::default_random_engine* rng) = 0;

};


}

#endif

