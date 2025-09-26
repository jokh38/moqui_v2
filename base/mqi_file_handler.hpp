/// \file mqi_file_handler.hpp
/// \brief Defines classes for file handling, such as reading mask files and parsing configuration files.
#ifndef MQI_FILE_HANDLER_HPP
#define MQI_FILE_HANDLER_HPP

#include <fstream>
#include <iostream>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_utils.hpp>

namespace mqi
{

/// \class mask_reader
/// \brief A class for reading and processing 3D mask files in the MetaImage (.mha) format.
///
/// \details In this context, a "mask" is a 3D grid where each voxel (pixel) is either 0 (inactive) or 1 (active).
/// It's used to define a specific Region of Interest (ROI) within the larger CT grid,
/// such as a tumor or a critical organ. This allows the simulation to perform calculations
/// (like scoring dose) only within that specific region, which can save significant computation time and memory,
/// especially on the GPU. This class handles reading `.mha` files, combining multiple masks, and
/// converting them into memory-efficient representations for the simulation engine.
class mask_reader
{
public:
    /// A list of filenames for the mask files to be read.
    std::vector<std::string> mask_filenames;
    /// The dimensions (nx, ny, nz) of the CT grid that this mask corresponds to.
    mqi::vec3<ijk_t> ct_dim;
    /// The total number of voxels in the CT grid (nx * ny * nz).
    size_t ct_size;
    /// A pointer to the combined mask data, where each voxel is represented by a `uint8_t`.
    /// This memory is allocated by `read_mask_files` and must be deallocated by the user or this class's destructor.
    uint8_t* mask_total = nullptr;

    /// \brief Default constructor.
    mask_reader() {
        ;
    }

    /// \brief Constructs a `mask_reader` with given CT dimensions.
    /// \param[in] ct_dim The dimensions of the corresponding CT grid.
    mask_reader(mqi::vec3<ijk_t> ct_dim) {
        this->ct_dim  = ct_dim;
        this->ct_size = ct_dim.x * ct_dim.y * ct_dim.z;
    }

    /// \brief Constructs a `mask_reader` with a list of files and CT dimensions.
    /// \param[in] filelist A vector of file paths to the mask files.
    /// \param[in] ct_dim The dimensions of the corresponding CT grid.
    mask_reader(std::vector<std::string> filelist, mqi::vec3<ijk_t> ct_dim) {
        this->mask_filenames = filelist;
        this->ct_dim         = ct_dim;
        this->ct_size        = ct_dim.x * ct_dim.y * ct_dim.z;
    }

    /// \brief Destructor. Frees the memory allocated for the combined mask.
    ~mask_reader() {
        if (mask_total != nullptr) {
            delete[] mask_total;
        }
    }

    /// \brief Reads a single mask file in the MetaImage (.mha) format.
    ///
    /// \details This function parses the header of the .mha file to determine the dimensions
    /// and then reads the binary pixel data into a buffer.
    /// \param[in] filename The path to the .mha file.
    /// \return A pointer to the mask data as a dynamically allocated array of `uint8_t`.
    ///         The caller is responsible for `delete[]`ing this memory.
    /// \note `CUDA_HOST` indicates this function runs on the CPU.
    CUDA_HOST
    uint8_t*
    read_mha_file(std::string filename) {
        std::string   line;
        std::ifstream fid(filename, std::ios::ate | std::ios::binary);   // Open in binary mode
        if (!fid.is_open()) {
            throw std::runtime_error("Cannot open mask file: " + filename);
        }
        fid.seekg(0);
        std::string delimeter = "=";
        size_t      pos, pos_current, pos_prev, total_size;
        std::string parameter, value;
        uint16_t    nx, ny, nz;
        uint8_t*    mask = nullptr;
        // Parse the MHA header
        while (std::getline(fid, line)) {
            line      = trim_copy(line);
            pos       = line.find(delimeter);
            parameter = trim_copy(line.substr(0, pos));
            if (strcasecmp(parameter.c_str(), "DimSize") == 0) {
                value       = trim_copy(line.substr(pos + 1));
                pos_prev    = 0;
                pos_current = value.find(" ");
                nx          = std::atoi(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                ny          = std::atoi(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                nz          = std::atoi(value.substr(pos_prev, pos_current).c_str());
            } else if (strcasecmp(parameter.c_str(), "ElementDataFile") == 0) {
                value = trim_copy(line.substr(pos + 1));
                if (strcasecmp(value.c_str(), "LOCAL") != 0) {
                    throw std::runtime_error("Mask files does not contain data.");
                }
                break;   // Stop parsing after finding the data file entry
            }
        }
        total_size = nx * ny * nz;
        mask       = new uint8_t[total_size];
        // Read the binary data block
        fid.read((char*) (mask), total_size * sizeof(uint8_t));
        fid.close();
        return mask;
    }

    /// \brief Reads and combines multiple mask files into a single mask.
    ///
    /// \details This method iterates through the `mask_filenames` list, reads each file using
    /// `read_mha_file`, and logically ORs its values into the `mask_total` array.
    /// This allows for the combination of several ROIs into one.
    /// It allocates memory for `mask_total` which is freed in the destructor.
    CUDA_HOST
    void
    read_mask_files() {
        if (mask_filenames.empty()) {
            throw std::runtime_error("Mask filelist are required for masking scorers.");
        }
        // Allocate and initialize the total mask buffer.
        this->mask_total = new uint8_t[this->ct_size];
        std::memset(this->mask_total, 0, this->ct_size * sizeof(uint8_t));

        uint8_t* mask_temp = nullptr;
        for (const auto& filename : mask_filenames) {
            printf("Reading maskfile %s\n", filename.c_str());
            try {
                mask_temp = read_mha_file(filename);
                for (size_t i = 0; i < this->ct_size; i++) {
                    assert(mask_temp[i] == 0 || mask_temp[i] == 1);
                    this->mask_total[i] |= mask_temp[i];   // Use logical OR to combine masks
                }
                delete[] mask_temp;   // Free the temporary mask buffer
                mask_temp = nullptr;
            } catch (...) {
                if (mask_temp) delete[] mask_temp;
                throw;   // Re-throw the exception
            }
        }
    }

    /// \brief Creates a mapping from a full voxel index to a compact scorer index.
    ///
    /// \details To save memory and computation, especially on a GPU, we often only want to store
    /// data for the "active" voxels inside a mask. This function creates an indirection map
    /// where each active voxel is assigned a new, compact index (0, 1, 2,...). Voxels outside
    /// the mask are assigned an index of -1 to signify they should be ignored.
    ///
    /// \param[in] mask A pointer to the mask data.
    /// \return A pointer to a dynamically allocated array where each element is the compact scorer index or -1 if masked out.
    ///         The caller is responsible for `delete[]`ing this memory.
    CUDA_HOST
    int32_t*
    mask_mapping(uint8_t* mask) {
        int32_t* scorer_idx = new int32_t[this->ct_size];
        uint32_t count      = 0;
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (mask[ind] > 0) {
                scorer_idx[ind] = count;
                count++;
            } else {
                scorer_idx[ind] = -1;
            }
        }
        return scorer_idx;
    }

    /// \brief Creates a default identity mapping where every voxel is active.
    /// \details This is used when no mask is provided, and the entire CT grid should be scored.
    /// \return A pointer to a dynamically allocated array where each voxel index maps to itself (e.g., `scorer_idx[i] = i`).
    ///         The caller is responsible for `delete[]`ing this memory.
    CUDA_HOST
    int32_t*
    mask_mapping() {
        int32_t* scorer_idx = new int32_t[this->ct_size];
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            scorer_idx[ind] = ind;
        }
        return scorer_idx;
    }

    /// \brief Calculates the number of active (non-masked) voxels in a scorer index map.
    /// \param[in] scorer_idx The scorer index map generated by `mask_mapping`.
    /// \return The number of active voxels.
    CUDA_HOST
    uint32_t
    size(int32_t* scorer_idx) {
        int count = 0;
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (scorer_idx[ind] >= 0) {
                count++;
            }
        }
        return count;
    }

    /// \brief Saves the mapping of original-to-compact indices to a text file for debugging.
    /// \param[in] filename The name of the file to save to.
    /// \param[in] scorer_idx The scorer index map to save.
    CUDA_HOST
    void
    save_map(std::string filename, int32_t* scorer_idx) {
        std::ofstream fid0(filename);
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (scorer_idx[ind] >= 0) {
                fid0 << ind << " " << scorer_idx[ind] << "\n";
            }
        }
        fid0.close();
    }

    /// \brief Sets the total mask using data from an external source.
    /// \param[in] mask A pointer to the external mask data.
    CUDA_HOST
    void
    set_mask(uint8_t* mask) {
        this->mask_total = mask;
    }

    /// \brief Converts 3D grid indices (i, j, k) to a 1D flattened index.
    /// \param[in] i The x-index.
    /// \param[in] j The y-index.
    /// \param[in] k The z-index.
    /// \return The 1D voxel index (cnb - "current voxel number").
    CUDA_HOST
    cnb_t
    ijk2cnb(ijk_t i, ijk_t j, ijk_t k) {
        return k * this->ct_dim.x * this->ct_dim.y + j * this->ct_dim.x + i;
    }

    /// \brief Converts the bitmap mask into a more compressed Region of Interest (ROI) representation.
    ///
    /// \details This function scans the `mask_total` and identifies contiguous segments of active
    /// voxels, storing them as a series of start points and lengths (strides). This
    /// "run-length encoding" can be a more memory-efficient way to represent an ROI
    /// than a full 3D bitmap, especially for large, contiguous regions.
    ///
    /// \return A pointer to the newly created `mqi::roi_t` object. The caller is responsible for deleting this object.
    CUDA_HOST
    mqi::roi_t*
    mask_to_roi() {
        uint32_t              original_size = this->ct_size;
        uint32_t              length        = 0;
        mqi::roi_mapping_t    method        = mqi::CONTOUR;
        std::vector<uint32_t> start_vec;
        std::vector<uint32_t> stride_vec;
        std::vector<uint32_t> acc_stride_vec;
        bool                  ind_started = false;
        int32_t               cnb, start_ind, acc_stride_tmp;
        for (ijk_t z_ind = 0; z_ind < this->ct_dim.z; z_ind++) {
            for (ijk_t y_ind = 0; y_ind < this->ct_dim.y; y_ind++) {
                for (ijk_t x_ind = 0; x_ind < this->ct_dim.x; x_ind++) {
                    if (this->mask_total[ijk2cnb(x_ind, y_ind, z_ind)] == 1 && !ind_started) {
                        ind_started = true;
                        start_vec.push_back(ijk2cnb(x_ind, y_ind, z_ind));
                    }
                    if (this->mask_total[ijk2cnb(x_ind, y_ind, z_ind)] == 0 && ind_started) {
                        ind_started = false;
                        if (acc_stride_vec.size() > 0)
                            acc_stride_tmp = acc_stride_vec.back();
                        else
                            acc_stride_tmp = 0;
                        start_ind = start_vec.back();
                        stride_vec.push_back(ijk2cnb(x_ind, y_ind, z_ind) - start_ind);
                        acc_stride_vec.push_back(acc_stride_tmp + stride_vec.back());
                    }
                }
            }
        }
        uint32_t* start      = new uint32_t[start_vec.size()];
        uint32_t* stride     = new uint32_t[start_vec.size()];
        uint32_t* acc_stride = new uint32_t[acc_stride_vec.size()];

        std::copy(start_vec.begin(), start_vec.end(), start);
        std::copy(stride_vec.begin(), stride_vec.end(), stride);
        std::copy(acc_stride_vec.begin(), acc_stride_vec.end(), acc_stride);
        length = start_vec.size();
        mqi::roi_t* roi = new mqi::roi_t(method, original_size, length, start, stride, acc_stride);
        return roi;
    }
};

/// \class file_parser
/// \brief A simple parser for reading key-value pair configuration files.
///
/// \details This class reads a simulation configuration file (e.g., `moqui_tps.in`),
/// parses the parameters, and provides methods to retrieve values by their key.
/// It supports various data types, default values, and parsing of comma-separated lists,
/// providing a simple, dependency-free way to configure simulations.
class file_parser
{
public:
    /// The path to the configuration file.
    std::string filename;
    /// The delimiter string used to separate keys and values (e.g., "=").
    std::string delimeter;
    /// A vector containing all non-empty, non-comment lines from the configuration file.
    std::vector<std::string> parameters_total;

    /// \brief Constructs a `file_parser` object.
    /// \param[in] filename The path to the configuration file.
    /// \param[in] delimeter The delimiter string to separate keys and values.
    file_parser(std::string filename, std::string delimeter) {
        this->filename         = filename;
        this->delimeter        = delimeter;
        this->parameters_total = read_input_parameters();
    }
    /// \brief Destructor.
    ~file_parser() {
        ;
    }

    /// \brief Reads all lines from the input configuration file, skipping comments and empty lines.
    /// \return A vector of strings, where each string is a line from the file.
    CUDA_HOST
    std::vector<std::string>
    read_input_parameters() {
        std::ifstream            fid(this->filename);
        std::string              line;
        std::vector<std::string> parameters_total;
        size_t                   comment;
        if (fid.is_open()) {
            while (getline(fid, line)) {
                if (line.empty()) continue;
                line = trim_copy(line);
                if (line.at(0) == '#') continue;   // Skip comment lines
                comment = line.find("#");
                if (comment != std::string::npos) {
                    line = line.substr(0, comment);
                }   // Ignore inline comments
                parameters_total.push_back(line);
            }
            fid.close();
        } else {
            throw std::runtime_error("Cannot open input parameter file.");
        }
        return parameters_total;
    }

    /// \brief Extracts the directory path from a full file path string.
    /// \param[in] filepath The full path to the file.
    /// \return The directory part of the path.
    CUDA_HOST
    std::string
    get_path(std::string filepath) {
        std::string delimeter   = "/";
        std::string path        = "";
        size_t      pos_current = 0, pos_prev = 0;
        pos_current = filepath.find(delimeter);
        while (pos_current != std::string::npos) {
            if (path.empty()) {
                path += filepath.substr(pos_prev, pos_current - pos_prev);
            } else {
                path += "/" + filepath.substr(pos_prev, pos_current - pos_prev);
            }

            pos_prev    = pos_current + 1;
            pos_current = filepath.find(delimeter, pos_prev);
        }
        return path;
    }

    /// \brief Gets a string value for a given configuration option (key).
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] default_value The value to return if the key is not found in the file.
    /// \return The parameter value as a string.
    CUDA_HOST
    std::string
    get_string(std::string option, std::string default_value) {
        size_t      pos;
        std::string option_in, value = default_value;
        for (int line = 0; line < parameters_total.size(); line++) {
            pos = parameters_total[line].find(delimeter);
            if (pos != std::string::npos) {
                option_in = trim_copy(parameters_total[line].substr(0, pos));
                if (strcasecmp(option.c_str(), option_in.c_str()) == 0) {
                    value = trim_copy(parameters_total[line].substr(pos + 1, std::string::npos));
                    return value;
                }
            }
        }
        return value;
    }

    /// \brief Gets a vector of strings for a given option key, where values are separated by a delimiter.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] delimeter1 The delimiter used to separate values within the string (e.g., ",").
    /// \return A vector of strings.
    CUDA_HOST
    std::vector<std::string>
    get_string_vector(std::string option, std::string delimeter1) {
        size_t                   pos_current = 0, pos_prev = 0;
        std::string              value = get_string(option, "");
        std::vector<std::string> values;
        if (value.empty()) return values;
        pos_current = value.find(delimeter1);
        std::string temp;
        // Loop to find all occurrences of the delimiter
        while (pos_current != std::string::npos) {
            temp = trim_copy(value.substr(pos_prev, pos_current - pos_prev));
            if (!temp.empty()) {
                values.push_back(temp);
            }
            pos_prev    = pos_current + 1;
            pos_current = value.find(delimeter1, pos_prev);
        }
        // Add the last token after the final delimiter
        temp = trim_copy(value.substr(pos_prev, pos_current - pos_prev));
        if (!temp.empty()) {
            values.push_back(temp);
        }
        return values;
    }

    /// \brief Gets a float value for a given option key.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] default_value The value to return if the key is not found.
    /// \return The parameter value as a float.
    CUDA_HOST
    float
    get_float(std::string option, float default_value) {
        float value = std::atof(get_string(option, std::to_string(default_value)).c_str());
        return value;
    }

    /// \brief Gets a vector of floats for a given option key.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] delimeter The delimiter used to separate values within the string.
    /// \return A vector of floats.
    CUDA_HOST
    std::vector<float>
    get_float_vector(std::string option, std::string delimeter) {
        std::vector<std::string> value_tmp = get_string_vector(option, delimeter);
        std::vector<float>       value;
        if (!value_tmp.empty()) {
            for (int i = 0; i < value_tmp.size(); i++) {
                value.push_back(std::atof(value_tmp[i].c_str()));
            }
        }
        return value;
    }

    /// \brief Gets an integer value for a given option key.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] default_value The value to return if the key is not found.
    /// \return The parameter value as an integer.
    CUDA_HOST
    int
    get_int(std::string option, int default_value) {
        int value = std::atoi(get_string(option, std::to_string(default_value)).c_str());
        return value;
    }

    /// \brief Gets a vector of integers for a given option key.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] delimeter The delimiter used to separate values within the string.
    /// \return A vector of integers.
    CUDA_HOST
    std::vector<int>
    get_int_vector(std::string option, std::string delimeter) {
        std::vector<std::string> value_tmp = get_string_vector(option, delimeter);
        std::vector<int>         value;
        if (!value_tmp.empty()) {
            for (int i = 0; i < value_tmp.size(); i++) {
                value.push_back(std::atoi(value_tmp[i].c_str()));
            }
        }
        return value;
    }

    /// \brief Gets a boolean value for a given option key.
    ///
    /// \details Recognizes "true" (case-insensitive) or any non-zero integer as true.
    /// All other values are considered false.
    /// \param[in] option The key of the parameter to retrieve.
    /// \param[in] default_value The value to return if the key is not found.
    /// \return The parameter value as a boolean.
    CUDA_HOST
    bool
    get_bool(std::string option, bool default_value) {
        std::string temp  = get_string(option, std::to_string(default_value));
        bool        value = strcasecmp(temp.c_str(), "true") == 0 || std::atoi(temp.c_str()) != 0;
        return value;
    }

    /// \brief Converts a scorer name string to its corresponding `scorer_t` enum value.
    /// \param[in] scorer_name The name of the scorer from the config file (e.g., "Dose").
    /// \return The corresponding `scorer_t` enum value.
    CUDA_HOST
    scorer_t
    string_to_scorer_type(std::string scorer_name) {
        scorer_t type = mqi::VIRTUAL;
        if (strcasecmp(scorer_name.c_str(), "EnergyDeposition") == 0) {
            type = mqi::ENERGY_DEPOSITION;
        } else if (strcasecmp(scorer_name.c_str(), "Dose") == 0) {
            type = mqi::DOSE;
        } else if (strcasecmp(scorer_name.c_str(), "LETd") == 0) {
            type = mqi::LETd;
        } else if (strcasecmp(scorer_name.c_str(), "LETt") == 0) {
            type = mqi::LETt;
        } else if (strcasecmp(scorer_name.c_str(), "Dij") == 0) {
            type = mqi::DOSE_Dij;
        } else if (strcasecmp(scorer_name.c_str(), "TrackLength") == 0) {
            type = mqi::TRACK_LENGTH;
        } else {
            throw std::runtime_error("Unrecognized scorer name");
        }
        return type;
    }

    /// \brief Converts an aperture type string to its corresponding `aperture_type_t` enum value.
    /// \param[in] aperture_string The name of the aperture type from the config file (e.g., "MASK").
    /// \return The corresponding `aperture_type_t` enum value.
    CUDA_HOST
    aperture_type_t
    string_to_aperture_type(std::string aperture_string) {
        aperture_type_t type;
        if (strcasecmp(aperture_string.c_str(), "VOLUME") == 0) {
            type = mqi::VOLUME;
        } else if (strcasecmp(aperture_string.c_str(), "MASK") == 0) {
            type = mqi::MASK;
        } else {
            throw std::runtime_error("Undefined aperture type.");
        }
        return type;
    }

    /// \brief Converts a simulation type string to its corresponding `sim_type_t` enum value.
    /// \param[in] sim_name The name of the simulation type from the config file (e.g., "perBeam").
    /// \return The corresponding `sim_type_t` enum value.
    CUDA_HOST
    sim_type_t
    string_to_sim_type(std::string sim_name) {
        sim_type_t type;
        if (strcasecmp(sim_name.c_str(), "perBeam") == 0) {
            type = mqi::PER_BEAM;
        } else if (strcasecmp(sim_name.c_str(), "perSpot") == 0) {
            type = mqi::PER_SPOT;
        }
        return type;
    }
};
}   // namespace mqi

#endif