/// \file mqi_dataset.hpp
///
/// \brief Defines a simplified, user-friendly interface for accessing DICOM data.
///
/// \details
/// This file provides the `mqi::dataset` class, which acts as a wrapper around the
/// more complex `gdcm::DataSet` from the GDCM library. GDCM (Grassroots DICOM) is a
/// C++ library for reading and writing DICOM files. DICOM is the international standard
/// for medical images and related information.
///
/// The purpose of this class is to simplify the process of accessing data elements
/// (like Patient Name) and sequences (nested lists of data) within a DICOM file.
/// DICOM files have a complex, hierarchical structure, and this class provides a much
/// cleaner, more Python-like interface for navigating it.
///
/// \note This class depends on the GDCM (Grassroots DICOM) library.
#ifndef MQI_DATASET_H
#define MQI_DATASET_H

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// These are header files from the third-party GDCM library.
#include "gdcmByteValue.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmItem.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmVM.h"
#include "gdcmVR.h"

namespace mqi
{

/// \enum modality_type
/// \brief Enumerates the different types of DICOM files based on their Modality tag (0008,0060).
/// \details This helps in identifying the purpose of a DICOM file, which is crucial because
/// different modalities may store similar information using different structures or tags.
typedef enum {
    RTPLAN,      ///< A radiation therapy plan, defining treatment parameters.
    IONPLAN,     ///< A plan for ion therapy (e.g., protons).
    RTRECORD,    ///< A record of a delivered radiation therapy treatment.
    IONRECORD,   ///< A record of a delivered ion therapy treatment.
    RTIMAGE,     ///< A radiation therapy image, such as a CT or MR scan.
    RTSTRUCT,    ///< A structure set, defining regions of interest (e.g., tumors, organs).
    RTDOSE,      ///< A calculated or measured radiation dose distribution.
    UNKNOWN_MOD  ///< An unknown or unsupported modality.
} modality_type;

/// \struct beam_id_type
/// \brief Represents a beam identifier, which can be either a number or a string name.
///
/// \details
/// In C++, a `union` is a special data structure that allows a value to be interpreted as
/// one of several different types. Here, it allows the beam identifier to be stored as
/// either an integer (`number`) or a character pointer (`name`) in the same memory location.
/// This is a memory-saving technique not commonly seen in Python. The `type` enum is used
/// as a discriminator to track which type is currently being used.
struct beam_id_type
{
    /// \brief The type of the beam identifier.
    enum
    {
        NUM, ///< The identifier is a number.
        STR  ///< The identifier is a string.
    } type;
    /// \brief The value of the beam identifier.
    union {
        int         number; ///< The numeric identifier.
        const char* name;   ///< The string identifier.
    };
};

/// \brief A map to find the correct DICOM tag for a sequence based on the file's modality.
///
/// \details
/// Different DICOM modalities (like RTPLAN vs. IONPLAN) may use different tags for the
/// same logical concept (e.g., the "beam sequence"). This lookup table provides a
/// standardized way to find the correct tag for common sequences, making the code
/// more robust and adaptable to different DICOM file types.
/// The `static const` keywords mean that this map is a single, read-only instance
/// shared across all parts of the program.
static const std::map<const modality_type, const std::map<const std::string, const gdcm::Tag>>
  seqtags_per_modality = {
      //modality, seq_name, tag
      { RTPLAN,
        {
          { "beam", gdcm::Tag(0x300a, 0x00b0) }   // Beam Sequence for conventional RT
        } },
      { IONPLAN,
        { { "beam", gdcm::Tag(0x300a, 0x03a2) },    // Ion Beam Sequence
          { "snout", gdcm::Tag(0x300a, 0x030c) },   // Snout Sequence
          { "rs", gdcm::Tag(0x300a, 0x0314) },      // Range Shifter Sequence
          { "rsss", gdcm::Tag(0x300a, 0x0360) },    // Range Shifter Settings Sequence
          { "blk", gdcm::Tag(0x300a, 0x03a6) },     // Ion Block Sequence
          { "comp", gdcm::Tag(0x300a, 0x0e2a) },    // Compensator Sequence
          { "ctrl", gdcm::Tag(0x300a, 0x03a8) } } },
      { IONRECORD,
        {
          { "beam", gdcm::Tag(0x3008, 0x0021) },    // Referenced Beam Sequence
          { "snout", gdcm::Tag(0x3008, 0x00f0) },   // Referenced Snout Sequence
          { "rs", gdcm::Tag(0x3008, 0x00f2) },      // Referenced Range Shifter Sequence
          { "rsss", gdcm::Tag(0x300a, 0x0360) },    // Range Shifter Settings Sequence
          { "blk", gdcm::Tag(0x3008, 0x00d0) },     // Referenced Block Sequence
          { "comp", gdcm::Tag(0x3008, 0x00c0) },    // Referenced Compensator Sequence
          { "ctrl", gdcm::Tag(0x3008, 0x0041) },    // Referenced Control Point Sequence
          { "machine", gdcm::Tag(0x300a, 0x0206) }   // Treatment Machine Record Sequence
        } }
  };

/// \class dataset
/// \brief A wrapper for `gdcm::DataSet` that simplifies accessing DICOM data elements and sequences.
///
/// \details
/// This class provides an intuitive interface for querying DICOM data by overloading
/// the `[]` and `()` operators. This allows for clean and readable access to data
/// elements and nested sequences by using either their DICOM tag or a human-readable keyword.
/// For a Python developer, this is like having a dictionary where you can access values
/// by key, and some of those values can be lists of other dictionaries.
///
/// It also provides helper methods (`get_values`) to automatically convert raw DICOM
/// byte data into standard C++ types like `std::vector<int>`, `std::vector<float>`, etc.
///
/// \b Example:
/// \code{.cpp}
/// // Assuming 'plan_ds' is a 'mqi::dataset' object for an RTPLAN file.
///
/// // Access a sequence of beams using the () operator and a keyword.
/// // This is like accessing a list of dicts in Python.
/// const std::vector<const dataset*> beam_datasets = plan_ds("IonBeamSequence");
///
/// // For each beam in the sequence...
/// for (const auto& beam_ds : beam_datasets) {
///     // Get a value using the get_values method and a keyword.
///     std::vector<int> num_blocks;
///     beam_ds->get_values("NumberOfBlocks", num_blocks);
/// }
/// \endcode
class dataset
{
protected:
    /// A lookup table to store and quickly access nested datasets (sequences).
    /// It maps a tag and keyword to a vector of child `dataset` objects.
    /// `std::tuple` is a C++ object capable of holding a collection of elements,
    /// each of a different type, similar to a Python tuple.
    std::vector<std::tuple<const gdcm::Tag, const std::string, std::vector<const dataset*>>>
      ds_lut_;

    ///< A constant reference to the underlying GDCM dataset object. This class wraps
    ///< `gdcm_ds_` but does not own it.
    const gdcm::DataSet gdcm_ds_;

public:
    /// \brief Constructs a `dataset` object from a `gdcm::DataSet`.
    ///
    /// \details
    /// When created, the constructor iterates through the GDCM dataset. If it finds
    /// a sequence (a tag with Value Representation 'SQ'), it recursively creates new
    /// `mqi::dataset` objects for each item in that sequence and stores them in the
    /// `ds_lut_` (lookup table) for easy access later.
    ///
    /// \param[in] d The `gdcm::DataSet` to wrap.
    /// \param[in] include_sq If true (the default), recursively processes sequences.
    dataset(const gdcm::DataSet& d, const bool include_sq = true) : gdcm_ds_(d) {
        // Get a reference to the global DICOM dictionary for tag lookups.
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        if (!include_sq) { return; }

        // Iterate over all data elements in the GDCM dataset.
        for (auto el = gdcm_ds_.Begin(); el != gdcm_ds_.End(); ++el) {
            const gdcm::Tag&       tag   = el->GetTag();
            const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);

            // If the data element is not a sequence (VR != SQ), skip it.
            if (!(entry.GetVR() & gdcm::VR::SQ)) continue;

            // It is a sequence. Get the sequence of items.
            // `gdcm::SmartPointer` is a "smart pointer" that manages memory automatically,
            // similar to how Python handles object references.
            gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = el->GetValueAsSQ();
            std::vector<const dataset*>               tmp(0);
            // Loop through each item in the sequence.
            for (size_t i = 1; i <= sqi->GetNumberOfItems(); ++i) {
                const gdcm::Item& itm = sqi->GetItem(i);
                // Recursively create a new mqi::dataset for the nested dataset in the item.
                // `new` allocates memory, which must be freed later in the destructor.
                tmp.push_back(new mqi::dataset(itm.GetNestedDataSet()));
            }
            // Store the newly created vector of datasets in our lookup table.
            ds_lut_.push_back(std::make_tuple(tag, entry.GetKeyword(), tmp));
        }
    }

    /// \brief Accesses a `gdcm::DataElement` by its DICOM tag.
    ///
    /// \details
    /// This is an "overloaded operator" that allows you to use square brackets `[]`
    /// to access data elements, similar to accessing elements in a Python dictionary.
    ///
    /// \param[in] t The `gdcm::Tag` of the data element to find.
    /// \return A constant reference to the `gdcm::DataElement`. Returns a globally shared empty element if not found.
    const gdcm::DataElement&
    operator[](const gdcm::Tag& t) const {
        if (gdcm_ds_.FindDataElement(t) && !gdcm_ds_.GetDataElement(t).IsEmpty()) {
            return gdcm_ds_.GetDataElement(t);
        } else {
            static gdcm::DataElement empty;   // A single, shared empty element to return on failure.
            empty.Clear();
            return empty;
        }
    }

    /// \brief Accesses a `gdcm::DataElement` by its keyword.
    /// \param[in] keyword The keyword of the data element (e.g., "PatientName").
    /// \return A constant reference to the `gdcm::DataElement`. Returns a globally shared empty element if not found.
    const gdcm::DataElement&
    operator[](const char* keyword) const {
        std::string        tmp(keyword);
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        for (auto it = gdcm_ds_.Begin(); it != gdcm_ds_.End(); ++it) {
            const gdcm::DictEntry& entry = dicts.GetDictEntry(it->GetTag());
            if (!tmp.compare(entry.GetKeyword())) return (*it);
        }
        static gdcm::DataElement empty;
        empty.Clear();
        return empty;
    }

    /// \brief Accesses a sequence of nested datasets by its tag.
    ///
    /// \details
    /// This overloaded operator uses parentheses `()` to access sequences, which
    /// contain their own sets of data elements. This is analogous to calling a function
    /// or a callable object in Python.
    ///
    /// \param[in] t The `gdcm::Tag` of the sequence.
    /// \return A vector of pointers to the constant `dataset` objects within the sequence. Returns an empty vector if not found.
    std::vector<const dataset*>
    operator()(const gdcm::Tag& t) const {
        for (auto& i : ds_lut_)
            if (std::get<0>(i) == t) { return std::get<2>(i); }
        std::vector<const dataset*> empty(0);
        return empty;
    }

    /// \brief Accesses a sequence of nested datasets by its keyword.
    /// \param[in] s The keyword of the sequence (e.g., "BeamSequence").
    /// \return A vector of pointers to the constant `dataset` objects within the sequence. Returns an empty vector if not found.
    std::vector<const dataset*>
    operator()(const char* s) const {
        std::string tmp(s);
        for (auto& i : ds_lut_)
            if (!tmp.compare(std::get<1>(i))) return std::get<2>(i);

        std::vector<const dataset*> empty(0);
        return empty;
    }

    /// \brief Destructor.
    ///
    /// \details
    /// In C++, objects created with `new` must be manually deleted to prevent memory leaks.
    /// This destructor is automatically called when a `dataset` object is destroyed. It
    /// iterates through the lookup table and `delete`s all the nested `dataset` objects
    /// that were allocated in the constructor. This contrasts with Python's automatic
    /// garbage collection.
    ~dataset() {
        for (auto& i : ds_lut_) {
            for (auto& j : std::get<2>(i)) {
                delete j;   // Free the memory for each nested dataset.
            }
            std::get<2>(i).clear();
        }
        ds_lut_.clear();
    }

    /// \brief Prints all data elements in the dataset to the console.
    ///
    /// This is a utility function for debugging, providing a quick way to inspect the
    /// contents of a DICOM dataset.
    void
    dump() const {
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        for (auto el = gdcm_ds_.Begin(); el != gdcm_ds_.End(); ++el) {
            const gdcm::DictEntry& entry = dicts.GetDictEntry(el->GetTag());
            std::cout << entry.GetKeyword() << ": " << el->GetTag() << ", VR: " << el->GetVR()
                      << ", length: " << el->GetVL() << std::endl;
        }

        std::cout << " LUT: --- size: " << ds_lut_.size() << std::endl;
        for (auto& i : ds_lut_) {
            std::cout << "     (" << std::get<0>(i) << ", " << std::get<1>(i) << ", "
                      << std::get<2>(i).size() << " )";
        }
        std::cout << std::endl;
    }

    /// \brief Extracts and converts values from a raw DICOM byte stream into a vector of floats.
    /// \param[in] bv Pointer to the `gdcm::ByteValue` containing the raw data.
    /// \param[in] vr The Value Representation (e.g., FL for float, DS for Decimal String).
    /// \param[in] vm The Value Multiplicity (how many values are expected).
    /// \param[in] vl The Value Length (the total length in bytes).
    /// \param[out] res A reference to a vector of floats where the results will be stored.
    void
    get_values(const gdcm::ByteValue* bv,
               const gdcm::VR         vr,
               const gdcm::VM         vm,
               const gdcm::VL         vl,
               std::vector<float>&    res) const {
        res.clear();

        if (vr & gdcm::VR::VRBINARY) {
            // For binary representations like FL (Float), the data is stored in native
            // float format. We can copy the bytes directly.
            assert(vr & gdcm::VR::FL);
            size_t ndim = vl / sizeof(float);
            res.resize(ndim);
            bv->GetBuffer((char*) &res[0], ndim * sizeof(float));
        } else if (vr & gdcm::VR::VRASCII) {
            // For string representations like DS (Decimal String), the numbers are stored
            // as text. Values are often separated by backslashes ('\').
            std::string       s    = std::string(bv->GetPointer(), bv->GetLength());
            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(std::stof(s.substr(beg, next)));   // Convert substring to float
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// \brief Extracts and converts values from a raw DICOM byte stream into a vector of integers.
    /// \param[in] bv Pointer to the `gdcm::ByteValue` containing the raw data.
    /// \param[in] vr The Value Representation (e.g., IS for Integer String, US for Unsigned Short).
    /// \param[in] vm The Value Multiplicity.
    /// \param[in] vl The Value Length.
    /// \param[out] res A reference to a vector of integers where the results will be stored.
    /// \param[in] show (Unused) A boolean flag.
    void
    get_values(const gdcm::ByteValue* bv,
               const gdcm::VR         vr,
               const gdcm::VM         vm,
               const gdcm::VL         vl,
               std::vector<int>&      res,
               bool                   show = false) const {
        res.clear();
        size_t ndim = vm.GetLength();
        if (ndim == 0) {
            if (vr & gdcm::VR::FL) {   // Should be an integer type, but defensive coding
                ndim = vl / sizeof(int);
            }
        }
        assert(ndim >= 1);

        if (vr & gdcm::VR::VRBINARY) {
            // For binary integer types (like US, UL), copy the bytes directly.
            res.resize(ndim);
            bv->GetBuffer((char*) &res[0], ndim * sizeof(int));
        } else if (vr & gdcm::VR::VRASCII) {
            // For string representations like IS (Integer String), parse the string.
            std::string s = std::string(bv->GetPointer(), bv->GetLength());
            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(std::stoi(s.substr(beg, next)));   // Convert substring to integer
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// \brief Extracts values from a raw DICOM byte stream into a vector of strings.
    /// \param[in] bv Pointer to the `gdcm::ByteValue` containing the raw data.
    /// \param[in] vr The Value Representation (e.g., PN for Patient Name, LO for Long String).
    /// \param[in] vm The Value Multiplicity.
    /// \param[in] vl The Value Length.
    /// \param[out] res A reference to a vector of strings where the results will be stored.
    void
    get_values(const gdcm::ByteValue*    bv,
               const gdcm::VR            vr,
               const gdcm::VM            vm,
               const gdcm::VL            vl,
               std::vector<std::string>& res) const {
        res.clear();
        size_t ndim = vm.GetLength();
        assert(ndim >= 1);

        if (vr & gdcm::VR::VRASCII) {
            // For string types, parse the string, splitting by backslashes.
            std::string s = std::string(bv->GetPointer(), bv->GetLength());
            // Trim trailing spaces which are common in DICOM strings
            s.erase(s.find_last_not_of(" \n\r\t") + 1);
            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(s.substr(beg, next));
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// \brief Gets values for a data element specified by its keyword.
    ///
    /// \details
    /// This is a "template method". It can be used to get values of any type (e.g., `int`,
    /// `float`, `std::string`) without needing a separate function for each type. The
    /// C++ compiler generates the appropriate version at compile-time based on the type
    /// you request in your code. This is a powerful C++ feature for writing generic code.
    ///
    /// \tparam T The type of the values to retrieve (e.g., `float`, `int`).
    /// \param[in] keyword The keyword of the data element (e.g., "PatientID").
    /// \param[out] result A reference to a vector that will store the results.
    template<typename T>
    void
    get_values(const char* keyword, std::vector<T>& result) const {
        result.clear();
        const gdcm::DataElement& el = (*this)[keyword];
        if (el.IsEmpty()) {
            result.resize(0);
            return;
        }
        const gdcm::Dicts&     dicts = gdcm::Global::GetInstance().GetDicts();
        const gdcm::Tag&       tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);
        // Call the appropriate non-template get_values overload based on type T.
        this->get_values(el.GetByteValue(), entry.GetVR(), entry.GetVM(), el.GetVL(), result);
    }

    /// \brief Gets values from a given `gdcm::DataElement`.
    ///
    /// This is a template method that provides a convenient way to extract and convert
    /// the value of a data element into a specific C++ type.
    ///
    /// \tparam T The type of the values to retrieve.
    /// \param[in] el The `gdcm::DataElement` from which to extract values.
    /// \param[out] result A reference to a vector that will store the results.
    template<typename T>
    void
    get_values(const gdcm::DataElement& el, std::vector<T>& result) const {
        result.clear();
        if (el.IsEmpty()) {
            result.resize(0);
            return;
        }
        const gdcm::Dicts&     dicts = gdcm::Global::GetInstance().GetDicts();
        const gdcm::Tag&       tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);
        this->get_values(el.GetByteValue(), entry.GetVR(), entry.GetVM(), el.GetVL(), result);
    }

    /// \brief Prints detailed information about a `gdcm::DataElement`.
    ///
    /// This is a utility function for debugging, showing the keyword, Value Multiplicity (VM),
    /// Value Representation (VR), Value Length (VL), and the actual value of a data element.
    /// \param[in] el The data element to print.
    void
    print_dataelement(const gdcm::DataElement& el) const {
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();
        const gdcm::ByteValue* bv = el.GetByteValue();
        const gdcm::Tag        tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);

        std::cout << "---- " << entry.GetKeyword() << " VM:" << entry.GetVM().GetLength()
                  << " VR:" << entry.GetVR() << ", GetVL: " << el.GetVL()
                  << ", GetByteValues: " << el.GetByteValue() << ", VR length"
                  << el.GetVR().GetLength() << ", VR size"
                  << el.GetVR().GetVRString(el.GetVR())
                  << ",    value:";
        if (el.IsEmpty()) {
            std::cout << "no value." << std::endl;
        } else {
            if (entry.GetVR() & gdcm::VR::FL) {
                const size_t ndim = el.GetVL() / sizeof(float);
                float c[ndim];
                bv->GetBuffer((char*) c, sizeof(c));
                std::cout << "FL encoding: ";
                for (size_t i = 0; i < ndim; ++i) {
                    std::cout << "   " << i << ": " << c[i];
                }
                std::cout << std::endl;
            } else {
                // Other types could be printed here if needed.
            }
        }
    }
};

}   // namespace mqi
#endif
