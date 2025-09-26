#ifndef MQI_VERTEX_HPP
#define MQI_VERTEX_HPP

#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// @struct vertex_t
/// @brief Represents the state of a particle at a specific point in space and time (a vertex).
///
/// This structure holds the essential properties of a particle at a vertex, which is a point
/// along its track. Various physics processes, geometry boundaries, or other simulation
/// constraints can propose the next vertex for a particle.
///
/// @tparam T The numeric type for the vertex properties (e.g., `float` or `double`).
/// @note This structure does not include the particle type itself, which is managed separately.
template<typename T>
struct vertex_t {
    T       ke;  ///< Kinetic energy of the particle at the vertex.
    vec3<T> pos; ///< Position of the particle in 3D space.
    vec3<T> dir; ///< Direction (as a unit vector) of the particle's momentum.

    /// @brief Assignment operator.
    /// @param rhs The vertex to copy from.
    /// @return A reference to this vertex.
    CUDA_HOST_DEVICE
    vertex_t<T>&
    operator=(const vertex_t<T>& rhs) {
        ke  = rhs.ke;
        pos = rhs.pos;
        dir = rhs.dir;
        return *this;
    }
};

}   // namespace mqi
#endif
