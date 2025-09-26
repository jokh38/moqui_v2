#ifndef MQI_KERNEL_FUNCTIONS_HPP
#define MQI_KERNEL_FUNCTIONS_HPP

/*!
 * @file mqi_kernel_functions.hpp
 * @brief A convenience header that includes all CUDA kernel-related functions for the Moqui simulation engine.
 * @details This file aggregates the headers for data transfer (upload/download), particle transport, and other GPU-specific operations, simplifying the inclusion of these components in other parts of the codebase.
*/

#include <cassert>
#include <valarray>

#include <moqui/kernel_functions/mqi_download_data.hpp>
#include <moqui/kernel_functions/mqi_print_data.hpp>
#include <moqui/kernel_functions/mqi_transport.hpp>
#include <moqui/kernel_functions/mqi_upload_data.hpp>
#include <moqui/kernel_functions/mqi_variables.hpp>

#endif   //MQI_KERNEL_FUNCTIONS_HPP
