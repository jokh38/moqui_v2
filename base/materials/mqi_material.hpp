/// \file mqi_material.hpp
/// \brief This file defines the base class for materials and specific material implementations.
///
/// \details The accuracy of a Monte Carlo simulation is highly dependent on the correct definition
/// of the physical properties for each material a particle might encounter. This file provides
/// a base class, `material_t`, that defines a common interface for all materials, and then
/// implements specific materials (like water, air) as derived classes.
///
/// Key properties like density, mean excitation energy (I-value), and radiation length are
/// defined here, as they are fundamental inputs to the physics models for stopping power
/// and multiple scattering calculations.
#ifndef MQI_MATERIAL_HPP
#define MQI_MATERIAL_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include "moqui/base/materials/material_table_data.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace mqi
{

///< A type alias for material IDs, providing a clear, descriptive name for the underlying type.
typedef uint16_t material_id;

/// \class material_t
/// \brief A base class representing the physical properties of a material in the simulation.
///
/// \details This class acts as an interface (similar to an Abstract Base Class in Python),
/// defining the essential properties and methods that any material must have to be used in the
/// simulation. Specific materials, like water or air, are implemented as derived classes that
/// inherit from `material_t` and provide concrete values for these properties. This polymorphic
/// design allows the transport engine to handle different materials through a common `material_t*` pointer.
///
/// \tparam R The floating-point type (e.g., `float` or `double`) used for calculations.
template<typename R>
class material_t
{
public:
    ///< A struct holding physical constants and conversion factors.
    physics_constants<R> units;

    ///< A pre-calculated term used in the Bethe-Bloch stopping power formula (2 * pi * r_e^2 * m_e * c^2 * n_el).
    R two_pi_re2_mc2_nel;
    ///< Mass density in g/mm^3.
    R rho_mass;
    ///< Electron density in electrons/mm^3.
    R rho_elec;
    ///< Effective atomic number of the material.
    R Z;
    ///< Molecular weight in g/mol.
    R weight;
    ///< Number of electrons per molecule.
    R electrons;
    ///< Mean excitation energy (ionization potential) in eV. A key parameter for stopping power calculations.
    R Iev;
    ///< Square of the mean excitation energy, pre-calculated for efficiency.
    R Iev_sq;
    ///< Radiation length in mm. A measure of the distance over which a high-energy electron loses most of its energy by bremsstrahlung.
    R X0;
    ///< A unique identifier for the material.
    uint16_t id;

public:
    /// \brief Default constructor.
    /// \note `CUDA_HOST_DEVICE` allows this function to be called from both CPU and GPU code.
    CUDA_HOST_DEVICE
    material_t() {
        ;
    }

    /// \brief Virtual destructor to ensure proper cleanup of derived classes.
    CUDA_HOST_DEVICE
    virtual ~material_t() {
        ;
    }

    /// \brief Assignment operator.
    /// \param[in] r The `material_t` object to be assigned from.
    /// \return A reference to the updated object.
    CUDA_HOST_DEVICE
    material_t<R>&
    operator=(const material_t<R>& r) {
        two_pi_re2_mc2_nel = r.two_pi_re2_mc2_nel;
        rho_mass           = r.rho_mass;
        rho_elec           = r.rho_elec;
        Z                  = r.Z;
        weight             = r.weight;
        electrons          = r.electrons;
        Iev                = r.Iev;
        X0                 = r.X0;
        return *this;
    }

    /// \brief Returns the mass density of the material.
    /// \param[in] scale An optional scaling factor for the density. Defaults to 1.0.
    /// \return The scaled mass density in g/mm^3.
    CUDA_HOST_DEVICE
    inline virtual R
    mass_density(R scale = 1.0) const {
        return scale * rho_mass;
    }

    /// \brief Returns a pre-calculated term used in the stopping power calculation.
    /// \return The pre-calculated stopping power term (2 * pi * r_e^2 * m_e * c^2 * n_el).
    CUDA_HOST_DEVICE
    inline virtual R
    dedx_term0() const {
        return two_pi_re2_mc2_nel;
    }

    /// \brief Returns the effective atomic number of the material.
    /// \return The effective atomic number (Z).
    CUDA_HOST_DEVICE
    inline virtual R
    atomic_number() const {
        return Z;
    }

    /**
     * @brief Calculates the stopping power ratio of the material relative to water.
     * @details The stopping power ratio is used to convert dose-in-medium to the standard
     * clinical quantity of dose-in-water. This `virtual` function provides a default
     * implementation using an empirical formula based on the material's density and the
     * particle's kinetic energy. Derived classes can override this if a more specific
     * model is available.
     *
     * @param[in] Ek The kinetic energy of the particle in MeV.
     * @param[in] id An optional material ID (currently unused).
     * @return The stopping power ratio (dimensionless).
     */
    CUDA_DEVICE
    inline virtual R
    stopping_power_ratio(R Ek, int8_t id = -1) {
        // Convert density from g/mm^3 to g/cm^3 for the formula
        R density_tmp = this->rho_mass * 1000.0;

        if (density_tmp > 0.26) {
            // Empirical formula for materials denser than 0.26 g/cm^3
            R rsp = 1.0123 - 3.386e-5 * Ek;
            rsp += 0.291 * (1.0 + mqi::mqi_pow(Ek, static_cast<R>(-0.3421))) *
                   (mqi::mqi_pow(density_tmp, static_cast<R>(-0.7)) - 1.0);

            if (density_tmp > 0.9)
                return 1.0;   // For high-density materials, approximate as 1.0
            else
                // For intermediate densities, interpolate between a known point and the calculated rsp.
                return mqi::intpl1d<R>(density_tmp, 0.26, 0.9, 0.9925, rsp);
        } else {
            // For low-density materials
            if (density_tmp < 0.0012) {
                return 0.0;   // For near-vacuum, ratio is zero.
            } else {
                // Interpolate for other low-density materials like lung.
                return mqi::intpl1d<R>(density_tmp, 0.0012, 0.26, 0.8815, 0.9925);
            }
        }
    }

    /**
     * @brief Calculates the radiation length of the material.
     * @details This `virtual` function provides a default implementation using an empirical
     * formula from Fippel and Soukup (2004) to calculate the radiation length based on the
     * material's density, relative to the known radiation length of water.
     *
     * @return The radiation length in mm.
     */
    CUDA_HOST_DEVICE
    virtual R
    radiation_length() {
        R radiation_length_mat = 0.0;
        R f                    = 0.0;
        R density_tmp = this->rho_mass * 1000.0;   // Convert to kg/m^3 for formula
        // Fippel's empirical formula based on density regimes
        if (density_tmp <= 0.26) {
            f = 0.9857 + 0.0085 * density_tmp;
        } else if (density_tmp > 0.26 && density_tmp <= 0.9) {
            f = 1.0446 - 0.2180 * density_tmp;
        } else if (density_tmp > 0.9) {
            f = 1.19 + 0.44 * logf(density_tmp - 0.44);
        }
        radiation_length_mat = (this->units.water_density * this->units.radiation_length_water) /
                               (density_tmp * 1e-3 * f);
        return radiation_length_mat;
    }
};

/// \class h2o_t
/// \brief A class representing water, derived from `material_t`.
/// \tparam R The floating-point type used for calculations.
template<typename R>
class h2o_t : public material_t<R>
{
public:
    /// \brief Constructs a new `h2o_t` object and initializes it with the physical properties of water.
    CUDA_HOST_DEVICE
    h2o_t() : material_t<R>() {
        this->rho_mass           = 1.0 / material_t<R>::units.cm3;   // 1.0 g/cm^3 in g/mm^3
        this->rho_elec = 3.3428e+23 / material_t<R>::units.cm3;   // electrons/mm^3
        this->two_pi_re2_mc2_nel = material_t<R>::units.two_pi_re2_mc2_h2o;
        this->Iev                = 78.0;   // Mean excitation energy for water in eV
        this->Iev_sq             = 78.0 * 78.0;
        this->Z = 18;   // Water is H2O, so 2*1 + 16 = 18 nucleons. Z_eff is different but this might be A_eff.
        this->X0                 = 36.0863 * this->units.cm;   // Radiation length of water in mm
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~h2o_t() {
        ;
    }
};

/// \class air_t
/// \brief A class representing air, derived from `material_t`.
/// \tparam R The floating-point type used for calculations.
template<typename R>
class air_t : public material_t<R>
{
public:
    /// \brief Constructs a new `air_t` object and initializes it with the physical properties of air.
    CUDA_HOST_DEVICE
    air_t() : material_t<R>() {
        ///< \todo Some of these values may need to be verified or completed.
        this->rho_mass = 0.0012047 / material_t<R>::units.cm3;   // Checked with NIST
        this->rho_elec =
            3.3428e+23 / material_t<R>::units.cm3;   // TODO: This is water's electron density, should be corrected for air.
        this->two_pi_re2_mc2_nel =
            material_t<R>::units.two_pi_re2_mc2_h2o * this->rho_mass;   // TODO: This is also based on water.
        this->Iev    = 85.7;   // Mean excitation energy for air in eV
        this->Iev_sq = 85.7 * 85.7;
        this->Z = 18;   // TODO: Effective Z for air is ~7.6, A is ~14.5. This seems incorrect.
        this->X0 =
            (36.62 * this->units.cm) *
            this->rho_mass;   // TODO: This seems to be a scaling from water's radiation length per gram.
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~air_t() {
        ;
    }
};

/// \class brass_t
/// \brief A class representing brass, derived from `material_t`.
/// \tparam R The floating-point type used for calculations.
template<typename R>
class brass_t : public material_t<R>
{
public:
    /// \brief Constructs a new `brass_t` object and initializes it with the properties of brass.
    CUDA_HOST_DEVICE
    brass_t() : material_t<R>() {
        ///< \todo Some of these values appear to be placeholders and should be updated with accurate data for brass.
        this->rho_mass = 8.52 / material_t<R>::units.cm3;   // Density of brass (g/cm^3) in g/mm^3
        this->rho_elec =
            3.3428e+23 / material_t<R>::units.cm3;   // TODO: Placeholder (water's value)
        this->two_pi_re2_mc2_nel =
            material_t<R>::units.two_pi_re2_mc2_h2o * this->rho_mass;   // TODO: Placeholder
        this->Iev    = 0;   // eV; TODO: Placeholder, should be ~330 eV for brass.
        this->Iev_sq = 0;
        this->Z      = 18;   // TODO: Placeholder, Z_eff for brass is ~29.6.
        this->X0     = 36.0863 * this->units.cm;   // TODO: Placeholder (water's value)
    }

    /// \brief Destructor.
    CUDA_HOST_DEVICE
    ~brass_t() {
        ;
    }
};

}   // namespace mqi

#endif
