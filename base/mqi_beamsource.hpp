#ifndef MQI_BEAMSOURCE_H
#define MQI_BEAMSOURCE_H

/// \file mqi_beamsource.hpp
///
/// \brief Defines a source of particles for simulation, composed of multiple beamlets.
///
/// This file contains the `beamsource` class, which manages a collection of `beamlet`
/// objects. It provides an interface for sampling particles (histories) from the
/// appropriate beamlet, based on either a history index or a simulation time.

#include <map>
#include <moqui/base/mqi_beamlet.hpp>

namespace mqi
{

/// \class beamsource
/// \brief Manages a collection of beamlets to model a complete radiation source.
///
/// This class aggregates multiple `beamlet` objects and orchestrates particle
/// sampling. It maintains a list of beamlets, each with an associated number of
/// histories, and provides lookup tables to efficiently find the correct beamlet
/// for a given history number or simulation time.
///
/// \tparam T The data type for numerical values (e.g., float, double).
template<typename T>
class beamsource
{
public:
    /// \brief A container for beamlets.
    /// Each element is a tuple containing the beamlet, its number of histories,
    /// and the cumulative number of histories up to that point.
    std::vector<std::tuple<mqi::beamlet<T>, size_t, size_t>> beamlets_;

    /// \brief A lookup table to map a cumulative history number to a beamlet index.
    /// This is used for efficient sampling based on history index.
    /// \note For GPU execution, this `std::map` may need to be replaced with a
    ///       GPU-compatible data structure.
    std::map<size_t, size_t> cdf2beamlet_;

    /// \brief A pointer to a GPU-compatible array of cumulative history counts.
    size_t* array_cdf_;
    /// \brief A pointer to a GPU-compatible array of beamlet indices.
    size_t* array_beamlet_;
    /// \brief The total number of beamlets, for use with GPU arrays.
    size_t beamlet_size_;
    /// \brief A pointer to a GPU-compatible array of beamlet objects.
    mqi::beamlet<T>* array_beamlets;

    /// \brief A lookup table to map simulation time to a beamlet index.
    /// A beamlet index of -1 indicates a period of no beam pulse (dead time).
    std::map<T, int32_t> timeline_;

    /// \brief Default constructor. Initializes all containers to an empty state.
    beamsource() {
        beamlets_.clear();
        timeline_.clear();
        cdf2beamlet_.clear();
    }

    /// \brief Appends a beamlet to the source with a specified coordinate transformation.
    ///
    /// \param b The `beamlet` object to add.
    /// \param h The number of histories associated with this beamlet.
    /// \param p The `coordinate_transform` to apply to the beamlet.
    /// \param time_on The duration of the beamlet's pulse.
    /// \param time_off The duration of dead time following the pulse.
    void
    append_beamlet(mqi::beamlet<T>              b,
                   size_t                       h,
                   mqi::coordinate_transform<T> p,
                   T                            time_on  = 1,
                   T                            time_off = 0) {
        b.set_coordinate_transform(p);
        this->append_beamlet(b, h, time_on, time_off);
    }

    /// \brief Appends a beamlet based on log file data.
    ///
    /// This is a specialized version for MOQUI simulations driven by treatment log files.
    ///
    /// \param b The `beamlet` object to add.
    /// \param h The number of histories associated with this beamlet.
    /// \param p The `coordinate_transform` to apply to the beamlet.
    void
    append_beamlet_log(mqi::beamlet<T>              b,
                       size_t                       h,
                       mqi::coordinate_transform<T> p) {
        double logfileTime = 0.00006;

        b.set_coordinate_transform(p);

        const size_t acc        = total_histories() + h;
        const size_t beamlet_id = this->total_beamlets();
        cdf2beamlet_.insert(std::make_pair(acc, beamlet_id));
        beamlets_.push_back(std::make_tuple(b, h, acc));

        T acc_time = this->total_delivery_time() + logfileTime;
        timeline_.insert(std::make_pair(acc_time, beamlet_id));
    }

    /// \brief Appends a beamlet to the source's internal containers.
    ///
    /// This is the core function for adding a beamlet. It updates both the
    /// history-based CDF and the time-based timeline.
    ///
    /// \param b The `beamlet` object to add.
    /// \param h The number of histories associated with this beamlet.
    /// \param time_on The duration of the beamlet's pulse.
    /// \param time_off The duration of dead time following the pulse.
    void
    append_beamlet(mqi::beamlet<T> b, size_t h, T time_on = 1, T time_off = 0) {
        const size_t acc        = total_histories() + h;
        const size_t beamlet_id = this->total_beamlets();
        cdf2beamlet_.insert(std::make_pair(acc, beamlet_id));
        beamlets_.push_back(std::make_tuple(b, h, acc));

        T acc_time = this->total_delivery_time() + time_on;
        timeline_.insert(std::make_pair(acc_time, beamlet_id));
        if (time_off != 0) {
            acc_time += time_off;
            timeline_.insert(std::make_pair(acc_time, -1));
        }
    }

    /// \brief Calculates the total delivery time.
    ///
    /// \return The time of the last event in the timeline.
    T
    total_delivery_time() const {
        if (timeline_.size() == 0) return 0.0;
        return std::prev(timeline_.end())->first;
    }

    /// \brief Returns the total number of beamlets in the source.
    ///
    /// \return The total number of beamlets.
    const std::size_t
    total_beamlets() const {
        return beamlets_.size();
    }

    /// \brief Returns the total number of histories for all beamlets.
    ///
    /// \return The total number of histories.
    const std::size_t
    total_histories() {
        return this->total_beamlets() == 0 ? 0 : std::get<2>(beamlets_.back());
    }

    /// \brief Provides access to a beamlet's data by its index.
    ///
    /// \param i The index of the beamlet.
    /// \return A constant reference to the tuple containing the beamlet, its history count,
    ///         and its cumulative history count.
    const std::tuple<mqi::beamlet<T>, size_t, size_t>&
    operator[](unsigned int i) const {
        return beamlets_[i];
    }

    /// \brief Returns the appropriate beamlet for a given history number.
    ///
    /// This operator uses the `cdf2beamlet_` map to find the correct beamlet
    /// that corresponds to the given history index `h`.
    ///
    /// \param h The history number.
    /// \return A constant reference to the corresponding `beamlet`.
    const mqi::beamlet<T>&
    operator()(size_t h) {
        size_t beamlet_id = cdf2beamlet_.upper_bound(h)->second;
        return std::get<0>(beamlets_[beamlet_id]);
    }

    /// \brief Calculates the cumulative number of histories up to a given time.
    ///
    /// This function determines how many particles have been generated up to a
    /// specific point `t0` in the simulation time.
    ///
    /// \param t0 The simulation time.
    /// \return The total number of histories generated up to time `t0`.
    size_t
    cumulative_history_at_time(T t0) {
        auto pulse = timeline_.lower_bound(t0);
        if (pulse == timeline_.end()) return this->total_histories();

        if (pulse->second == -1) {
            while (pulse->second == -1)
                --pulse;
            return std::get<2>(beamlets_[pulse->second]);
        }

        size_t history = std::get<2>(beamlets_[pulse->second]);

        T pulse_time[2];
        pulse_time[0] = (pulse == timeline_.begin()) ? 0.0 : std::prev(pulse)->first;
        pulse_time[1] = pulse->first;

        const T ratio = (t0 - pulse_time[0]) / (pulse_time[1] - pulse_time[0]);

        if (ratio >= 0) {
            const size_t h0 = std::get<1>(beamlets_[pulse->second]);
            const size_t h1 = (size_t) std::floor((1.0 - ratio) * h0);
            history -= h1;
        }
        return history;
    }
};

}   // namespace mqi

#endif
