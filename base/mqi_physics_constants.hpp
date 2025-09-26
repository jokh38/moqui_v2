/**
 * @file
 * @brief Defines a struct containing fundamental physical constants and unit conversions.
 * @details This header provides a centralized and consistent source for physical constants
 * required throughout the simulation. Using a single struct for these values prevents
 * "magic numbers" from being scattered across the codebase, which improves readability,
 * reduces the risk of typos, and makes it easy to update the values if needed.
 * The constants are based on well-established sources like the Particle Data Group (PDG).
 */
#ifndef MQI_PHYSICS_CONSTANTS_HPP
#define MQI_PHYSICS_CONSTANTS_HPP

#include <moqui/base/mqi_common.hpp>

namespace mqi
{

/**
 * @struct physics_constants
 * @brief A collection of fundamental physical constants and unit conversions.
 * @tparam R The floating-point type (e.g., float or double) used for the constants.
 * @details This struct is a container for constants used in physics calculations.
 * By using a template, all constants can be defined with a specific numerical precision
 * (`float` or `double`) at compile time. All members are `const`, making instances of this
 * struct immutable, which is a safe practice for handling constants.
 *
 * The base units used are millimeters (mm) for length and Mega-electron Volts (MeV) for energy.
 */
template<typename R>
struct physics_constants
{
    // BASE UNITS
    const R mm  = 1.0;     ///< Default length unit is millimeters.
    const R MeV = 1.0;     ///< Default energy unit is Mega-electron Volts.

    // DERIVED AND CONVERSION UNITS
    const R cm = 10.0;   ///< Centimeters in terms of millimeters.
    const R cm3 =
        cm * cm * cm;   ///< Cubic centimeters in terms of cubic millimeters.
    const R mm3 = mm * mm * mm;     ///< Cubic millimeters.
    const R eV  = 1e-6;             ///< Electron Volts in terms of MeV.

    // PARTICLE MASSES (as rest mass energy, E=m*c^2)
    const R Mp    = 938.272046 * MeV;      ///< Proton rest mass energy in MeV.
    const R Mp_sq = Mp * Mp;               ///< Square of the proton mass energy.
    const R Me    = 0.510998928 * MeV;     ///< Electron rest mass energy in MeV.
    const R Mo    = 14903.3460795634 * MeV;   ///< Oxygen-16 atom rest mass energy in MeV.
    const R Mo_sq = Mo * Mo;                     ///< Square of the oxygen mass energy.

    // MASS RATIOS (dimensionless)
    const R MoMp    = Mo / Mp;       ///< Ratio of Oxygen mass to Proton mass.
    const R MoMp_sq = MoMp * MoMp;   ///< Square of the Oxygen/Proton mass ratio.
    const R MeMp    = Me / Mp;       ///< Ratio of Electron mass to Proton mass.
    const R MeMp_sq = MeMp * MeMp;   ///< Square of the Electron/Proton mass ratio.

    // FUNDAMENTAL CONSTANTS
    const R re =
        2.8179403262e-12 * mm;   ///< Classical electron radius in mm.
    const R re_sq = re * re;     ///< Square of the classical electron radius.

    // PRE-CALCULATED CONSTANTS FOR PERFORMANCE
    // These are common combinations of constants that are pre-calculated to avoid repeated multiplication at runtime.
    const R two_pi_re2_mc2 = 2.0 * M_PI * re_sq * Me;   ///< A pre-calculated constant: 2 * pi * r_e^2 * m_e * c^2.
    const R two_pi_re2_mc2_h2o =
        two_pi_re2_mc2 * 3.3428e+23 /
        cm3;   ///< The `two_pi_re2_mc2` constant scaled by the electron density of water (electrons/mm^3).

    // MATERIAL-SPECIFIC CONSTANTS (for water)
    const R water_density = 1.0 / cm3;   ///< Density of water (1 g/cm^3) expressed in base units (g/mm^3).
    const R radiation_length_water =
        36.0863 * cm;   ///< Radiation length of water in mm.

    // UNIT CONVERSION FACTORS
    const R mev_to_joule = 1.60218e-13;   ///< Conversion factor from MeV to Joules.
};

}   // namespace mqi
#endif
