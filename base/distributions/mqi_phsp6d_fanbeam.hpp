#ifndef MQI_PHSP_6D_FANBEAM_H
#define MQI_PHSP_6D_FANBEAM_H

/// \file mqi_phsp6d_fanbeam.hpp
///
/// \brief Defines a 6-dimensional phase-space PDF for a fan beam.
///
/// This file contains the `phsp_6d_fanbeam` class, which models the 6D phase
/// space for a wide or fan-shaped beam. It uniformly distributes particles over
/// a rectangular area and sets their initial direction based on a source-to-axis
/// distance (SAD).

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi{

/// \class phsp_6d_fanbeam
/// \brief A 6D phase-space PDF for a fan beam.
///
/// This class models a fan beam by first sampling a particle's initial (x, y)
/// position uniformly within a given rectangular range. The initial direction
/// is then determined based on this position and the source-to-axis distance (SAD),
/// creating a diverging fan of particles. Additional Gaussian fluctuations are
/// then added to the position and direction, including correlations.
///
/// \tparam T The data type of the phase-space variables (e.g., float, double).
template<typename T>
class phsp_6d_fanbeam : public pdf_Md<T,6> {
private:
    /// \brief Correlation coefficients for (x, x') and (y, y').
    std::array<T,2> rho_;
    /// \brief Source-to-Axis Distance (SAD) for the x and y directions.
    std::array<T,2> SAD_;

public:
    /// \brief A standard normal distribution (mean=0, std=1) for sampling.
    std::normal_distribution<T>       func_;

    /// \brief A uniform distribution for sampling the initial x-position.
    std::uniform_real_distribution<T> unifx_;
    /// \brief A uniform distribution for sampling the initial y-position.
    std::uniform_real_distribution<T> unify_;

    /// \brief Constructs a new 6D fan beam phase-space distribution.
    ///
    /// \param[in] m An array defining the sampling range: `[x_min, x_max, y_min, y_max, z_min, z_max]`.
    /// \param[in] s An array containing the standard deviations for `[sig_x, sig_y, sig_z, sig_x', sig_y', sig_z']`.
    /// \param[in] r An array containing the correlation coefficients `rho(x,x')` and `rho(y,y')`.
    /// \param[in] o An array containing the source-to-axis distances `[SADx, SADy]`.
    CUDA_HOST_DEVICE
    phsp_6d_fanbeam(
        std::array<T,6>& m,
        std::array<T,6>& s,
        std::array<T,2>& r,
        std::array<T,2>& o
    ) : pdf_Md<T,6>(m,s),
	    rho_(r),
	    SAD_(o)
    {
        unifx_ = std::uniform_real_distribution<T>(m[0], m[1]);
        unify_ = std::uniform_real_distribution<T>(m[2], m[3]);
        func_ = std::normal_distribution<T>(0, 1);
    }
    
    /// \brief Constructs a new 6D fan beam phase-space distribution from constant references.
    ///
    /// \param[in] m A const reference to an array defining the sampling range.
    /// \param[in] s A const reference to an array containing the standard deviations.
    /// \param[in] r A const reference to an array containing the correlation coefficients.
    /// \param[in] o A const reference to an array containing the source-to-axis distances.
    CUDA_HOST_DEVICE
    phsp_6d_fanbeam(
	    const std::array<T,6>& m,
        const std::array<T,6>& s,
        const std::array<T,2>& r,
        const std::array<T,2>& o
    ) : pdf_Md<T,6>(m,s),
	    rho_(r),
	    SAD_(o)
    {
        unifx_ = std::uniform_real_distribution<T>(m[0], m[1]);
        unify_ = std::uniform_real_distribution<T>(m[2], m[3]);
        func_ = std::normal_distribution<T>(0, 1);
    }

    /// \brief Samples the 6D fan beam phase-space distribution.
    ///
    /// This method generates a random 6D phase-space vector by first sampling a
    /// uniform (x,y) position, calculating the divergent direction, and then
    /// applying correlated Gaussian fluctuations to both position and direction.
    ///
    /// \param rng A pointer to a random number engine.
    /// \return An array containing the sampled (x, y, z, x', y', z') values.
    CUDA_HOST_DEVICE
    virtual
    std::array<T,6>
    operator()(std::default_random_engine* rng)
    {
	    auto x = unifx_(*rng) ;
	    auto y = unify_(*rng) ;
	
	    mqi::vec3<T> dir(std::atan(x/SAD_[0]), std::atan(y/SAD_[1]), -1.0);

        std::array<T,6> phsp ;
        T Ux = func_(*rng); T Vx = func_(*rng);
        T Uy = func_(*rng); T Vy = func_(*rng);

        phsp[0] = x + pdf_Md<T,6>::sigma_[0]* Ux ;
        phsp[1] = y + pdf_Md<T,6>::sigma_[1]* Uy ;
        phsp[2] = pdf_Md<T,6>::mean_[5] ;

        phsp[3] = dir.x + pdf_Md<T,6>::sigma_[3] * (rho_[0] * Ux + Vx * std::sqrt(1.0-rho_[0]*rho_[0]));
        phsp[4] = dir.y + pdf_Md<T,6>::sigma_[4] * (rho_[1] * Uy + Vy * std::sqrt(1.0-rho_[1]*rho_[1]));
        phsp[5] =  -1.0*std::sqrt( 1.0 - phsp[3]*phsp[3]- phsp[4]*phsp[4] );

        return phsp;
    };

};



}

#endif

