#ifndef MQI_PHSP6D_H
#define MQI_PHSP6D_H

/// \file mqi_phsp6d.hpp
///
/// \brief Defines a 6-dimensional phase-space probability distribution function.
///
/// This file contains the `phsp_6d` class, which models the 6D phase space
/// (position and direction) of a particle beam, typically for a single spot.
/// It assumes a Gaussian distribution for the phase-space variables and allows
/// for correlation between position and divergence.

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi
{
/// \class phsp_6d
/// \brief A 6-dimensional PDF for phase-space variables (x, y, z, x', y', z').
///
/// This class models a 6D Gaussian distribution for a beamlet or spot. It takes into
/// account the mean and standard deviation for each of the 6 phase-space variables,
/// as well as the correlation coefficients (rho) between the spatial (x, y) and
/// angular (x', y') components.
///
/// \tparam T The data type of the phase-space variables (e.g., float, double).
template<typename T>
class phsp_6d : public pdf_Md<T, 6>
{
private:
    /// \brief Correlation coefficients for (x, x') and (y, y').
    std::array<T, 2> rho_;

public:
    /// \brief The C++ standard library normal distribution function object (mean=0, std=1).
    std::normal_distribution<T> func_;

    /// \brief Constructs a new 6D phase-space distribution.
    ///
    /// \param m An array containing the mean values for (x, y, z, x', y', z').
    /// \param s An array containing the standard deviations for (x, y, z, x', y', z').
    /// \param r An array containing the correlation coefficients rho(x,x') and rho(y,y').
    CUDA_HOST_DEVICE
    phsp_6d(std::array<T, 6>& m, std::array<T, 6>& s, std::array<T, 2>& r) :
        pdf_Md<T, 6>(m, s), rho_(r) {
        func_ = std::normal_distribution<T>(0, 1);
    }

    /// \brief Constructs a new 6D phase-space distribution from constant references.
    ///
    /// \param m A const reference to an array containing the mean values.
    /// \param s A const reference to an array containing the standard deviations.
    /// \param r A const reference to an array containing the correlation coefficients.
    CUDA_HOST_DEVICE
    phsp_6d(const std::array<T, 6>& m, const std::array<T, 6>& s, const std::array<T, 2>& r) :
        pdf_Md<T, 6>(m, s), rho_(r) {
        func_ = std::normal_distribution<T>(0, 1);
    }

    /// \brief Samples the 6D phase-space distribution.
    ///
    /// This method generates a random 6D phase-space vector. It first samples
    /// independent standard normal variables and then transforms them using the
    /// specified means, standard deviations, and correlation coefficients to produce
    /// the final correlated phase-space coordinates. The z-direction cosine (phsp[5])
    /// is calculated to ensure the direction vector is normalized.
    ///
    /// \param rng A pointer to a random number engine.
    /// \return An array containing the sampled (x, y, z, x', y', z') values.
    CUDA_HOST_DEVICE
    virtual std::array<T, 6>
    operator()(std::default_random_engine* rng) {
        std::array<T, 6> phsp = pdf_Md<T, 6>::mean_;
        T                Ux   = func_(*rng);
        T                Vx   = func_(*rng);
        T                Uy   = func_(*rng);
        T                Vy   = func_(*rng);
        T                Uz   = func_(*rng);
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
