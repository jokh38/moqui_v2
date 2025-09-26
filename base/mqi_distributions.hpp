/// \file mqi_distributions.hpp
///
/// \brief A meta-header that includes all particle distribution function headers.
///
/// In Monte Carlo simulations, the initial properties of particles (e.g., position,
/// energy, direction) are not single values but are sampled from statistical
/// distributions. This is done to accurately model the characteristics of a real-world
/// radiation beam, which is composed of billions of particles with slight variations.
///
/// This header file conveniently includes all available distribution models in Moqui,
/// allowing other parts of the code to easily access them.
#ifndef MQI_DISTRIBUTIONS_H
#define MQI_DISTRIBUTIONS_H

// 1D Distributions
#include <moqui/base/distributions/mqi_const_1d.hpp>   //< For a constant, single value.
#include <moqui/base/distributions/mqi_norm_1d.hpp>    //< For a 1D normal (Gaussian) distribution.
#include <moqui/base/distributions/mqi_uni_1d.hpp>     //< For a 1D uniform distribution.

// N-Dimensional Distributions
#include <moqui/base/distributions/mqi_pdfMd.hpp>      //< For a multi-dimensional probability density function.

// 6D Phase-Space Distributions
#include <moqui/base/distributions/mqi_phsp6d.hpp>          //< Base class for 6D phase-space.
#include <moqui/base/distributions/mqi_phsp6d_fanbeam.hpp>  //< For a fan-shaped beam.
#include <moqui/base/distributions/mqi_phsp6d_uniform.hpp>  //< For a uniform 6D phase-space.

#endif
