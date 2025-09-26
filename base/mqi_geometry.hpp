/// \file mqi_geometry.hpp
///
/// \brief Defines the abstract base class for all geometric components in the simulation.
///
/// \details This file is central to the object-oriented design of the simulation's geometry.
/// It defines the `mqi::geometry` class, which serves as a common interface for all
/// geometric objects. This approach allows for the creation of a "scene graph" or
/// "geometry hierarchy," where complex objects are built from simpler ones. For example,
/// a treatment machine can be represented as a tree of geometry objects like
/// apertures, range shifters, and collimators, each with its own position and orientation
/// relative to its parent. This makes it possible to handle complex, nested geometries
/// in a clean and polymorphic way.
#ifndef MQI_GEOMETRY_HPP
#define MQI_GEOMETRY_HPP

#include <array>
#include <map>
#include <string>
#include <vector>

#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \enum geometry_type
/// \brief Enumerates the different types of physical or logical components in the simulation geometry.
typedef enum {
    SNOUT,        ///< The final collimating assembly of a particle therapy machine that holds other components.
    RANGESHIFTER, ///< A device, often a plastic slab, used to decrease the beam's energy and penetration depth.
    COMPENSATOR,  ///< A custom-milled device used to shape the dose distribution conformally to the target.
    BLOCK,        ///< A simple, solid block of high-density material used to shape the beam aperture.
    BOLI,         ///< Material placed on the patient's skin to modify the dose distribution at the surface.
    WEDGE,        ///< A wedge-shaped filter used to tilt the isodose curves.
    TRANSFORM,    ///< A logical component representing a coordinate system transformation (e.g., gantry rotation).
    MLC,          ///< A Multi-Leaf Collimator, a device with many independently moving metal leaves to create complex beam shapes.
    PATIENT,      ///< The patient geometry, typically derived from a CT scan.
    DOSEGRID,     ///< A grid used for calculating and storing the dose distribution, often aligned with the patient CT.
    UNKNOWN1,     ///< Unspecified geometry type 1.
    UNKNOWN2,     ///< Unspecified geometry type 2.
    UNKNOWN3,     ///< Unspecified geometry type 3.
    UNKNOWN4      ///< Unspecified geometry type 4.
} geometry_type;

/// \class geometry
/// \brief An abstract base class for all geometric objects in the simulation.
///
/// \details
/// In C++, an "abstract base class" is one that cannot be instantiated on its own
/// (because it has `virtual` methods). It defines a common interface (a set of properties
/// and methods) that all derived classes must implement. This is similar to using `abc.ABC`
/// in Python to define an abstract base class.
///
/// This class ensures that every geometric component in the simulation has a
/// defined position, orientation, and type. By making these properties `const`, the class
/// enforces that once a geometry object is created, its fundamental transform cannot be
/// changed, promoting safer and more predictable code.
class geometry
{

public:
    /// The position of the geometric object's origin, relative to its parent's coordinate system.
    /// This member is `const`, meaning it is set in the constructor and cannot be modified later.
    const mqi::vec3<float> pos;
    /// The rotation matrix defining the object's orientation, relative to its parent.
    /// This member is also `const` and immutable after construction.
    const mqi::mat3x3<float> rot;
    /// The type of the geometry, as defined by the `geometry_type` enum. Also `const`.
    const geometry_type geotype;

    /// \brief Constructs a geometry object with a given position, rotation, and type.
    /// \param[in] p_xyz A reference to the position vector.
    /// \param[in] rot_xyz A reference to the 3x3 rotation matrix.
    /// \param[in] t The geometry type.
    geometry(mqi::vec3<float>& p_xyz, mqi::mat3x3<float>& rot_xyz, mqi::geometry_type t) :
        pos(p_xyz), rot(rot_xyz), geotype(t) {
        ;
    }

    /// \brief Constructs a geometry object with constant (non-modifiable) position, rotation, and type.
    /// \param[in] p_xyz A constant reference to the position vector.
    /// \param[in] rot_xyz A constant reference to the 3x3 rotation matrix.
    /// \param[in] t The geometry type.
    geometry(const mqi::vec3<float>&   p_xyz,
             const mqi::mat3x3<float>& rot_xyz,
             const mqi::geometry_type  t) :
        pos(p_xyz),
        rot(rot_xyz), geotype(t) {
        ;
    }

    /// \brief Copy constructor.
    /// \param[in] rhs The geometry object to copy.
    geometry(const geometry& rhs) : geotype(rhs.geotype), pos(rhs.pos), rot(rhs.rot) {
        ;
    }

    /// \brief Assignment operator.
    /// \note This implementation is unconventional as it returns a `const` reference.
    /// A more standard implementation would return `geometry&` and perform a member-wise copy if the members were not const.
    /// Given that all members are `const`, assignment doesn't change state, so this is effectively a no-op.
    /// \param[in] rhs The geometry object to assign from.
    /// \return A constant reference to the assigned object.
    const geometry&
    operator=(const mqi::geometry& rhs) {
        return rhs;
    }

    /// \brief Virtual destructor.
    ///
    /// \details In C++, when dealing with a class hierarchy (inheritance), it's critical that the
    /// destructor of the base class is `virtual`. This ensures that when you `delete` an object
    // through a pointer to the base class (e.g., `geometry* ptr = new mlc(); delete ptr;`),
    /// the correct destructor for the most-derived class (`mlc` in this case) is called first,
    /// followed by the base class destructors. This prevents memory leaks.
    virtual ~geometry() {
        ;
    }

    /// \brief A virtual method to print information about the geometry.
    ///
    /// \details A `virtual` method can be overridden by derived classes. This enables
    /// "polymorphism", a core concept of object-oriented programming. It allows you to have a
    /// collection of different geometry types (e.g., `std::vector<geometry*>`) and call `dump()`
    /// on each element. The program will automatically execute the correct version of `dump()`
    /// for the actual derived class (e.g., the `mlc` version, the `patient` version, etc.) at runtime.
    /// This is similar to how method calls work on objects of different types in a list in Python.
    virtual void
    dump() const {
        ;
    }
};

}   // namespace mqi
#endif
