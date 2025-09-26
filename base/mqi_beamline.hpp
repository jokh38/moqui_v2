#ifndef MQI_BEAMLINE_H
#define MQI_BEAMLINE_H

/// \file mqi_beamline.hpp
///
/// \brief Defines a beamline as a collection of physical components.
///
/// This file contains the `beamline` class, which serves as a container for all
/// the geometric components that make up a treatment beamline, such as collimators,
/// apertures, and range shifters.

#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/// \class beamline
/// \brief Represents a treatment machine beamline.
///
/// This class holds a collection of `mqi::geometry` objects that together
/// define the physical structure of a treatment beamline. It allows for the
/// dynamic addition of components and provides access to the entire collection.
///
/// \tparam T The data type for numerical values (e.g., float, double).
/// \note The necessity of this class as more than a simple container is yet to be determined.
template<typename T>
class beamline
{
protected:
    /// \brief A vector of pointers to the geometric components of the beamline.
    std::vector<mqi::geometry*> geometries_;

public:
    /// \brief Constructs an empty beamline object.
    beamline() {
        ;
    }

    /// \brief Destroys the beamline object.
    ///
    /// Clears the vector of geometry pointers. Note that this does not deallocate
    /// the `mqi::geometry` objects themselves, as the `beamline` does not own them.
    ~beamline() {
        geometries_.clear();
    }

    /// \brief Appends a new geometry component to the beamline.
    ///
    /// \param geo A pointer to the `mqi::geometry` object to be added.
    void
    append_geometry(mqi::geometry* geo) {
        geometries_.push_back(geo);
    }

    /// \brief Returns a constant reference to the geometry container.
    ///
    /// \return A const reference to the vector of geometry pointers.
    const std::vector<mqi::geometry*>&
    get_geometries() const {
        return geometries_;
    }
};

}   // namespace mqi
#endif
