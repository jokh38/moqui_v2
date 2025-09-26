/// \file
///
/// \brief This file defines the phsp_6d_uniform class, a 6-dimensional uniform phase-space distribution.
///
/// The phsp_6d_uniform class generates random samples of phase-space variables (position and direction)
/// for particles using a uniform distribution. It also accounts for correlations between position and direction.

#ifndef MQI_PHSP6D_UNIFORM_H
#define MQI_PHSP6D_UNIFORM_H

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi
{

/// \class phsp_6d_uniform
/// \brief A 6-dimensional uniform phase-space distribution.
/// \tparam T The floating-point type of the distribution (e.g., float or double).
///
/// This class models a 6D uniform distribution for phase-space variables (x, y, z, x', y', z').
/// It is used to simulate particle beams where the phase-space can be described by a uniform distribution,
/// while also considering the correlations between position and direction (divergence).
template<typename T>
class phsp_6d_uniform : public pdf_Md<T, 6>
{

private:
    ///< Correlations for x-x' and y-y', respectively.
    std::array<T, 2> rho_;   ///< For X,Y

public:
    ///< Uniform real distribution function for generating random numbers.
    std::uniform_real_distribution<T> func_;

    /// \brief Constructs a new phsp_6d_uniform object.
    /// \param[in] m An array representing the mean values for the 6 phase-space variables {x, y, z, x', y', z'}.
    /// \param[in] s An array representing the standard deviations (or ranges) for the 6 phase-space variables.
    /// \param[in] r An array representing the correlation coefficients for (x, x') and (y, y').
    CUDA_HOST_DEVICE
    phsp_6d_uniform(std::array<T, 6>& m, std::array<T, 6>& s, std::array<T, 2>& r) :
        pdf_Md<T, 6>(m, s), rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        //        func_ = std::normal_distribution<T>(0, 1);
        func_ = std::uniform_real_distribution<T>(-1.0, 1.0);
        //#endif
    }

    /// \brief Constructs a new phsp_6d_uniform object with const parameters.
    /// \param[in] m An array representing the mean values for the 6 phase-space variables {x, y, z, x', y', z'}.
    /// \param[in] s An array representing the standard deviations (or ranges) for the 6 phase-space variables.
    /// \param[in] r An array representing the correlation coefficients for (x, x') and (y, y').
    CUDA_HOST_DEVICE
    phsp_6d_uniform(const std::array<T, 6>& m,
                    const std::array<T, 6>& s,
                    const std::array<T, 2>& r) :
        pdf_Md<T, 6>(m, s),
        rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        //        func_ = std::normal_distribution<T>(0, 1);
        func_ = std::uniform_real_distribution<T>(-1.0, 1.0);
        //#endif
    }

    /// \brief Samples the 6D uniform phase-space distribution.
    /// \param[in, out] rng A pointer to a random number engine.
    /// \return An array containing a random sample of the 6 phase-space variables {x, y, z, x', y', z'}.
    CUDA_HOST_DEVICE
    virtual std::array<T, 6>
    operator()(std::default_random_engine* rng) {
        std::array<T, 6> phsp = pdf_Md<T, 6>::mean_;
        T                Ux   = func_(*rng);
        T                Vx   = func_(*rng);
        T                Uy   = func_(*rng);
        T                Vy   = func_(*rng);
        T                Uz   = func_(*rng);   //T Vz = func_(rng);
        phsp[0] += pdf_Md<T, 6>::sigma_[0] * Ux;
        phsp[1] += pdf_Md<T, 6>::sigma_[1] * Uy;
        phsp[2] += pdf_Md<T, 6>::sigma_[2] * Uz;

        phsp[3] +=
          pdf_Md<T, 6>::sigma_[3] * (rho_[0] * Ux + Vx * std::sqrt(1.0 - rho_[0] * rho_[0]));
        phsp[4] +=
          pdf_Md<T, 6>::sigma_[4] * (rho_[1] * Uy + Vy * std::sqrt(1.0 - rho_[1] * rho_[1]));
        phsp[5] = -1.0 * std::sqrt(1.0 - phsp[3] * phsp[3] - phsp[4] * phsp[4]);
        return phsp;
    };
};

}   // namespace mqi
#endif
