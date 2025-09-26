#ifndef MQI_BEAMLET_HPP
#define MQI_BEAMLET_HPP

/// \file mqi_beamlet.hpp
///
/// \brief Defines a beamlet, a fundamental component of a beam model.
///
/// \details A beamlet represents a small, idealized component of a larger radiation beam.
/// For example, in intensity-modulated proton therapy (IMPT), the entire treatment field
/// is composed of thousands of these small "pencil beams", each with its own position and intensity.
///
/// A beamlet is defined by a collection of probability distribution functions (PDFs) that model
/// the "phase-space" of its constituent particles. Phase-space is a concept from physics that
/// combines a particle's position (x, y, z) and momentum/direction (x', y', z') into a single
/// state. By sampling from these distributions, we can generate a realistic population of
/// initial particles for the simulation.

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_distributions.hpp>
#include <moqui/base/mqi_vertex.hpp>

namespace mqi
{

/// \class beamlet
/// \brief Represents a single beamlet, capable of generating initial particle states.
///
/// \details A beamlet models a small part of a larger field (like a pencil beam in IMPT)
/// or an entire field (like a double-scattered proton field). It combines an energy
/// distribution (`energy`) and a 6D phase-space distribution (`fluence`) to generate
/// initial particle states (vertices).
///
/// Importantly, this class only holds *pointers* to the distribution objects. This means
/// that many beamlets can share the same underlying distributions, which is a memory-efficient
/// way to model a treatment field where many pencil beams have the same energy spectrum or
/// spot shape, but different positions.
///
/// It also includes a coordinate transformation to map the locally generated particles
/// (which are typically sampled around the origin (0,0,0)) into the global patient or
/// treatment coordinate system.
///
/// \tparam T The data type for numerical values (e.g., float, double).
template<typename T>
class beamlet
{
protected:
    /// A pointer to the 1D probability distribution function for particle energy.
    mqi::pdf_Md<T, 1>* energy = nullptr;

    /// A pointer to the 6D probability distribution function for particle phase-space.
    /// This distribution samples the initial position (x, y, z) and direction (x', y', z') of a particle.
    mqi::pdf_Md<T, 6>* fluence = nullptr;

    /// The coordinate transformation to map from the local beamlet frame
    /// to the global patient or treatment coordinate system.
    coordinate_transform<T> p_coord;

public:
    /// \brief Constructs a beamlet from energy and fluence distributions.
    ///
    /// \param[in] e A pointer to a 1D PDF for energy. The beamlet does not take ownership.
    /// \param[in] f A pointer to a 6D PDF for fluence. The beamlet does not take ownership.
    CUDA_HOST_DEVICE
    beamlet(mqi::pdf_Md<T, 1>* e, mqi::pdf_Md<T, 6>* f) : energy(e), fluence(f) {
        ;
    }

    /// \brief Default constructor.
    CUDA_HOST_DEVICE
    beamlet() {
        ;
    }

    /// \brief Copy constructor.
    ///
    /// \details Creates a new beamlet as a copy of an existing one. Note that this is a shallow
    /// copy; the pointers to the energy and fluence distributions are copied, not the
    /// distributions themselves.
    /// \param[in] rhs The beamlet object to copy.
    CUDA_HOST_DEVICE
    beamlet(const beamlet<T>& rhs) {
        energy  = rhs.energy;
        fluence = rhs.fluence;
        p_coord = rhs.p_coord;
    }

    /// \brief Sets the coordinate transformation for the beamlet.
    ///
    /// \details This function allows setting the rotation and translation that will be applied
    /// to the generated particles. This is how a beamlet, which is defined in a local
    /// coordinate system around (0,0,0), is placed at its correct position and orientation
    /// in the global simulation geometry.
    ///
    /// \param[in] p A `coordinate_transform` object containing the desired translation and rotation.
    CUDA_HOST_DEVICE
    void
    set_coordinate_transform(coordinate_transform<T> p) {
        p_coord = p;
    }

    /// \brief Samples a particle vertex from the beamlet's distributions.
    ///
    /// \details This overloaded function call operator (`operator()`) makes a `beamlet` object behave
    /// like a function. For a Python developer, this is analogous to defining the `__call__`
    /// method on a class.
    ///
    /// It uses the energy and fluence distributions to generate a single particle's
    /// kinematic properties (energy, position, direction), applies the coordinate
    /// transformation, and returns the result as a `vertex_t` object, which represents
    /// the starting point of a new particle track.
    ///
    /// \param[in] rng A pointer to a C++ standard random number engine to be used for sampling.
    /// \return A `mqi::vertex_t<T>` object representing the generated particle in the global coordinate system.
    virtual mqi::vertex_t<T>
    operator()(std::default_random_engine* rng) {
        // Sample 6 phase-space coordinates from the fluence distribution.
        std::array<T, 6> phsp = (*fluence)(rng);
        mqi::vec3<T>     pos(phsp[0], phsp[1], phsp[2]);
        mqi::vec3<T>     dir(phsp[3], phsp[4], phsp[5]);
        mqi::vertex_t<T> vtx;
        // Sample kinetic energy from the energy distribution.
        vtx.ke = (*energy)(rng)[0];
        // Apply rotation and translation to move the particle into the global frame.
        vtx.pos = p_coord.rotation * pos + p_coord.translation;
        vtx.dir = p_coord.rotation * dir;
        return vtx;
    };
};

}   // namespace mqi

#endif
