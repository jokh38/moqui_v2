/// \file
///
/// \brief This file defines the phsp_6d_ray class, a 6-dimensional phase-space distribution for a ray-like source.
///
/// The phsp_6d_ray class generates random samples of phase-space variables (position and direction)
/// for particles originating from a ray-like source. It accounts for correlations between position and direction.

#ifndef MQI_PHSP6D_RAY_H
#define MQI_PHSP6D_RAY_H

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi
{

/// \class phsp_6d_ray
/// \brief A 6-dimensional phase-space distribution for a ray-like source.
/// \tparam T The floating-point type of the distribution (e.g., float or double).
///
/// This class models a 6D Gaussian distribution for phase-space variables (x, y, z, x', y', z').
/// It is designed to simulate particle beams that can be approximated as rays,
/// taking into account the correlations between position and direction (divergence).
template<typename T>
class phsp_6d_ray : public pdf_Md<T, 6>
{
private:
    ///< Correlations for x-x' and y-y', respectively.
    std::array<T, 2> rho_;

    ///< The z-position of the source.
    float source_position;

public:
    ///< Normal distribution function for generating random numbers.
    std::normal_distribution<T> func_;

    /// \brief Constructs a new phsp_6d_ray object.
    /// \param[in] m An array representing the mean values for the 6 phase-space variables {x, y, z, x', y', z'}.
    /// \param[in] s An array representing the standard deviations for the 6 phase-space variables.
    /// \param[in] r An array representing the correlation coefficients for (x, x') and (y, y').
    /// \param[in] source_position The z-position of the particle source.
    CUDA_HOST_DEVICE
    phsp_6d_ray(std::array<T, 6>& m,
                std::array<T, 6>& s,
                std::array<T, 2>& r,
                float             source_position) :
        pdf_Md<T, 6>(m, s),
        rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        func_                 = std::normal_distribution<T>(0, 1);
        this->source_position = source_position;
        //#endif
    }

    /// \brief Constructs a new phsp_6d_ray object with const parameters.
    /// \param[in] m An array representing the mean values for the 6 phase-space variables {x, y, z, x', y', z'}.
    /// \param[in] s An array representing the standard deviations for the 6 phase-space variables.
    /// \param[in] r An array representing the correlation coefficients for (x, x') and (y, y').
    /// \param[in] source_position The z-position of the particle source.
    CUDA_HOST_DEVICE
    phsp_6d_ray(const std::array<T, 6>& m,
                const std::array<T, 6>& s,
                const std::array<T, 6>& r,
                const float             source_position) :
        pdf_Md<T, 6>(m, s),
        rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        func_                 = std::normal_distribution<T>(0, 1);
        this->source_position = source_position;
        //#endif
    }

    /// \brief Samples the 6D phase-space distribution.
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
        T                z    = this->source_position;
        T                A0_x = rho_[0] * rho_[0];
        T                A1_x = pdf_Md<T, 6>::sigma_[3];
        T                A2_x = pdf_Md<T, 6>::sigma_[0] * pdf_Md<T, 6>::sigma_[0];
        T                A0_y = rho_[1] * rho_[1];
        T                A1_y = pdf_Md<T, 6>::sigma_[4];
        T                A2_y = pdf_Md<T, 6>::sigma_[1] * pdf_Md<T, 6>::sigma_[1];
        A2_x                  = A2_x + 2 * A1_x * z + A0_x * z * z;
        A1_x                  = A1_x + A0_x * z;
        A2_y                  = A2_y + 2 * A1_y * z + A0_y * z * z;
        A1_y                  = A1_y + A0_y * z;
        //        printf("A0 %f A1 %f A2 %f\n", A0, A1, A2);
        phsp[0] += std::sqrt(A2_x) * Ux;
        phsp[1] += std::sqrt(A2_y) * Uy;
        phsp[2] += pdf_Md<T, 6>::sigma_[2] * Uz;
        T th20_x = 2 * A0_x;
        if (A2_x > 0.0) {
            phsp[3] += std::sqrt(A2_x) * Ux * A1_x / A2_x;
            th20_x -= 2.0 * A1_x * A1_x / A2_x;
        }
        T th20_y = 2 * A0_y;
        if (A2_y > 0.0) {
            phsp[4] += std::sqrt(A2_y) * Uy * A1_y / A2_y;
            th20_y -= 2.0 * A1_y * A1_y / A2_y;
        }
        T norm = std::sqrt(phsp[3] * phsp[3] + phsp[4] * phsp[4] + phsp[5] * phsp[5]);
        //        printf(
        //          "dir.x %f dir.y %f dir.z %f norm %f th20 %f\n", phsp[3], phsp[4], phsp[5], norm, th20);
        phsp[3] /= norm;
        phsp[4] /= norm;
        phsp[5] /= norm;
        //        printf("Vx %f cos %f sin %f %f\n",
        //               Vx,
        //               std::cos(Vx * std::sqrt(th20 / 2)),
        //               std::sqrt(1 - phsp[3] * phsp[3]),
        //               std::sin(Vx * std::sqrt(th20 / 2)));
        //        printf("dir1 %f %f %f\n",phsp[3]*phsp[3],phsp[4]*phsp[4],phsp[5]*phsp[5]);
        phsp[3] = phsp[3] * std::cos(Vx * std::sqrt(th20_x / 2)) +
                  std::sqrt(1 - phsp[3] * phsp[3]) * std::sin(Vx * std::sqrt(th20_x / 2));
        phsp[4] = phsp[4] * std::cos(Vy * std::sqrt(th20_y / 2)) +
                  std::sqrt(1 - phsp[4] * phsp[4]) * std::sin(Vy * std::sqrt(th20_y / 2));
        //        printf("dir %f %f\n",phsp[3]*phsp[3],phsp[4]*phsp[4]);
        phsp[5] = -1.0 * std::sqrt(1.0 - phsp[3] * phsp[3] - phsp[4] * phsp[4]);
        return phsp;
    };

};

}   // namespace mqi
#endif
