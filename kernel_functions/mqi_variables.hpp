#ifndef MQI_VARIABLES_HPP
#define MQI_VARIABLES_HPP
#include <moqui/kernel_functions/mqi_kernel_functions.hpp>

/*!
 * @file mqi_variables.hpp
 * @brief Defines global variables for the Monte Carlo simulation.
 * @note This file is marked for future deletion. These variables are used to hold global state for the simulation, such as materials, geometry, and particle data.
*/

///< This file will be deleted soon. (JW: Jan 20, 2021)

namespace mc
{

///< The floating-point type used for phase-space data.
typedef float phsp_t;

///< Pointer to the global array of materials used in the simulation.
mqi::material_t<phsp_t>* mc_materials = nullptr;

///< Pointer to the top-level node (world) of the simulation geometry.
mqi::node_t<phsp_t>* mc_world = nullptr;

///< Pointer to the array of initial particle vertices.
mqi::vertex_t<phsp_t>* mc_vertices = nullptr;

///< A flag to control whether variance is scored in the simulation.
bool mc_score_variance = true;

}   // namespace mc
#endif   //MQI_VARIABLES_CPP
