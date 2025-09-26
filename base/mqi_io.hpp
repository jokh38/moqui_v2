/// \file mqi_io.hpp
///
/// \brief Defines a collection of functions for saving simulation data to various file formats.
///
/// This file provides a set of input/output (I/O) utilities to export data from the
/// simulation's internal structures (like scorers) into standard file formats that can be
/// easily used by external analysis and visualization tools. Supported formats include
/// raw binary, NumPy arrays (`.npz`), and medical imaging formats (`.mhd`, `.mha`).
#ifndef MQI_IO_HPP
#define MQI_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iomanip>   // std::setprecision
#include <iostream>
#include <numeric>   //accumulate
#include <valarray>
#include <zlib.h>

#include <sys/mman.h>   //for io

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_sparse_io.hpp>
#include <moqui/base/mqi_scorer.hpp>

namespace mqi
{
/// \namespace io
/// \brief A namespace for input/output operations, such as saving simulation results.
namespace io
{
/// \brief Saves sparse scorer data to separate binary files.
///
/// This function extracts the non-zero entries from a scorer's hash table and saves
/// them into three separate raw binary files: `_key1.raw`, `_key2.raw`, and `_value.raw`.
/// This format is simple and fast but contains no metadata.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the scorer object containing the data.
/// \param[in] scale A scaling factor to apply to the scorer values before saving.
/// \param[in] filepath The directory path where the files will be saved.
/// \param[in] filename The base name for the output files.
template<typename R>
void
save_to_bin(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename);

/// \brief Saves a dense array to a single binary file.
///
/// This function scales the data in the source array and writes it to a single `.raw` file.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the source data array.
/// \param[in] scale A scaling factor to apply to the data.
/// \param[in] filepath The directory path for the output file.
/// \param[in] filename The name of the output file.
/// \param[in] length The number of elements in the array.
template<typename R>
void
save_to_bin(const R*           src,
            const R            scale,
            const std::string& filepath,
            const std::string& filename,
            const uint32_t     length);

/// \brief Saves scorer data to a compressed NumPy `.npz` file in spot-major CSR format.
///
/// This function converts the sparse scorer data into a Compressed Sparse Row (CSR) matrix
/// and saves it to a `.npz` file. The CSR format is highly efficient for storing sparse
/// matrices and is directly compatible with Python libraries like SciPy. In this "spot-major"
/// version, each row of the matrix corresponds to a beam spot, and each column corresponds
/// to a voxel index.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the scorer object.
/// \param[in] scale A scaling factor to apply to the scorer values.
/// \param[in] filepath The directory path for the output file.
/// \param[in] filename The name of the output file.
/// \param[in] dim The dimensions of the scoring grid.
/// \param[in] num_spots The total number of spots, which determines the number of rows in the matrix.
template<typename R>
void
save_to_npz(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            mqi::vec3<mqi::ijk_t> dim,
            uint32_t              num_spots);

/// \brief Saves scorer data to a compressed NumPy `.npz` file in voxel-major CSR format.
///
/// This function also saves the data in CSR format, but in a "voxel-major" layout.
/// Each row of the matrix corresponds to a voxel within the defined Region of Interest (ROI),
/// and each column corresponds to a beam spot.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the scorer object.
/// \param[in] scale A scaling factor to apply to the scorer values.
/// \param[in] filepath The directory path for the output file.
/// \param[in] filename The name of the output file.
/// \param[in] dim The dimensions of the scoring grid.
/// \param[in] num_spots The total number of spots.
template<typename R>
void
save_to_npz2(const mqi::scorer<R>* src,
             const R               scale,
             const std::string&    filepath,
             const std::string&    filename,
             mqi::vec3<mqi::ijk_t> dim,
             uint32_t              num_spots);

/// \brief Saves scorer data to a compressed NumPy `.npz` file with additional processing.
///
/// This is an extended version of `save_to_npz` that applies time scaling and a
/// threshold to the data before saving it in the spot-major CSR format.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the scorer object.
/// \param[in] scale A scaling factor to apply to the scorer values.
/// \param[in] filepath The directory path for the output file.
/// \param[in] filename The name of the output file.
/// \param[in] dim The dimensions of the scoring grid.
/// \param[in] num_spots The number of spots.
/// \param[in] time_scale An array of time scaling factors to apply to each spot.
/// \param[in] threshold A threshold value to apply to the data.
template<typename R>
void
save_to_npz(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            mqi::vec3<mqi::ijk_t> dim,
            uint32_t              num_spots,
            R*                    time_scale,
            R                     threshold);

/// \brief Saves sparse key-value data to separate binary files.
///
/// This function extracts non-empty entries from a `key_value` array and saves them
/// into three separate raw binary files: `_key1.raw`, `_key2.raw`, and `_value.raw`.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] src A pointer to the array of `key_value` structs.
/// \param[in] scale A scaling factor to apply to the values.
/// \param[in] max_capacity The total capacity of the source array.
/// \param[in] filepath The directory path for the output files.
/// \param[in] filename The base name for the output files.
template<typename R>
void
save_to_bin(const mqi::key_value* src,
            const R               scale,
            uint32_t              max_capacity,
            const std::string&    filepath,
            const std::string&    filename);

/// \brief Saves volumetric data to a MetaImage header/raw file pair (`.mhd`/`.raw`).
///
/// This format is common in medical imaging and is used by toolkits like ITK and
/// visualization software like 3D Slicer. It consists of a text header file (`.mhd`)
/// containing metadata and a separate binary file (`.raw`) containing the grid data.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] children A pointer to the node structure that defines the grid geometry.
/// \param[in] src A pointer to the source data array.
/// \param[in] scale A scaling factor to apply to the data.
/// \param[in] filepath The directory path for the output files.
/// \param[in] filename The base name for the output files.
/// \param[in] length The number of elements in the source array.
template<typename R>
void
save_to_mhd(const mqi::node_t<R>* children,
            const double*         src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            const uint32_t        length);

/// \brief Saves volumetric data to a single MetaImage file (`.mha`).
///
/// The `.mha` format is similar to `.mhd`, but it combines both the header metadata
/// and the binary grid data into a single file for convenience.
///
/// \tparam R The floating-point type (e.g., `float` or `double`).
/// \param[in] children A pointer to the node structure that defines the grid geometry.
/// \param[in] src A pointer to the source data array.
/// \param[in] scale A scaling factor to apply to the data.
/// \param[in] filepath The directory path for the output file.
/// \param[in] filename The name of the output file.
/// \param[in] length The number of elements in the source array.
template<typename R>
void
save_to_mha(const mqi::node_t<R>* children,
            const double*         src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            const uint32_t        length);
}   // namespace io
}   // namespace mqi

// Template function implementations are included in the header file.

template<typename R>
void
mqi::io::save_to_bin(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename) {
    // Extract non-zero key-value pairs from the scorer's hash table.
    std::vector<mqi::key_t> key1;
    std::vector<mqi::key_t> key2;
    std::vector<double>     value;
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair &&
            src->data_[ind].value > 0) {
            key1.push_back(src->data_[ind].key1);
            key2.push_back(src->data_[ind].key2);
            value.push_back(src->data_[ind].value * scale);
        }
    }
    printf("Non-zero elements: %lu\n", value.size());

    // Write the key and value vectors to separate binary files.
    std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key1)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw" << std::endl;
    fid_key1.write(reinterpret_cast<const char*>(key1.data()), key1.size() * sizeof(mqi::key_t));
    fid_key1.close();

    std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key2)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw" << std::endl;
    fid_key2.write(reinterpret_cast<const char*>(key2.data()), key2.size() * sizeof(mqi::key_t));
    fid_key2.close();

    std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                          std::ios::out | std::ios::binary);
    if (!fid_bin)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw" << std::endl;
    fid_bin.write(reinterpret_cast<const char*>(value.data()), value.size() * sizeof(double));
    fid_bin.close();
}

template<typename R>
void
mqi::io::save_to_bin(const R*           src,
                     const R            scale,
                     const std::string& filepath,
                     const std::string& filename,
                     const uint32_t     length) {
    // Create a copy of the data to apply scaling without modifying the original.
    std::valarray<R> dest(src, length);
    dest *= scale;
    // Open the output file in binary mode and write the data.
    std::ofstream fid_bin(filepath + "/" + filename + ".raw", std::ios::out | std::ios::binary);
    if (!fid_bin) {
        std::cout << "Cannot write :" << filepath + "/" + filename + ".raw" << std::endl;
        return;
    }
    fid_bin.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(R));
    fid_bin.close();
}

template<typename R>
void
mqi::io::save_to_bin(const mqi::key_value* src,
                     const R               scale,
                     uint32_t              max_capacity,
                     const std::string&    filepath,
                     const std::string&    filename) {
    // Extract non-zero key-value pairs from the source array.
    std::vector<mqi::key_t> key1;
    std::vector<mqi::key_t> key2;
    std::vector<R>          value;
    for (int ind = 0; ind < max_capacity; ind++) {
        if (src[ind].key1 != mqi::empty_pair && src[ind].key2 != mqi::empty_pair &&
            src[ind].value > 0) {
            key1.push_back(src[ind].key1);
            key2.push_back(src[ind].key2);
            value.push_back(src[ind].value * scale);
        }
    }
    printf("Non-zero elements: %lu\n", value.size());

    // Write the key and value vectors to separate binary files.
    std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key1)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw" << std::endl;
    fid_key1.write(reinterpret_cast<const char*>(key1.data()), key1.size() * sizeof(mqi::key_t));
    fid_key1.close();

    std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key2)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw" << std::endl;
    fid_key2.write(reinterpret_cast<const char*>(key2.data()), key2.size() * sizeof(mqi::key_t));
    fid_key2.close();

    std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                          std::ios::out | std::ios::binary);
    if (!fid_bin)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw" << std::endl;
    fid_bin.write(reinterpret_cast<const char*>(value.data()), value.size() * sizeof(R));
    fid_bin.close();
}

template<typename R>
void
mqi::io::save_to_npz(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     mqi::vec3<mqi::ijk_t> dim,
                     uint32_t              num_spots) {
    uint32_t vol_size = dim.x * dim.y * dim.z;
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";

    // Create temporary vectors to group voxel indices and values by spot index.
    auto* value_vec = new std::vector<double>[num_spots];
    auto* vox_vec   = new std::vector<mqi::key_t>[num_spots];

    // Iterate through the hash table and populate the temporary vectors.
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            mqi::key_t vox_ind  = src->data_[ind].key1;
            mqi::key_t spot_ind = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            value_vec[spot_ind].push_back(src->data_[ind].value * scale);
            vox_vec[spot_ind].push_back(vox_ind);
        }
    }

    // --- Convert the grouped data into the three CSR arrays: data, indices, and indptr ---
    std::vector<double>   data_vec;
    std::vector<uint32_t> indices_vec;
    std::vector<uint32_t> indptr_vec;

    int nnz_count = 0;
    indptr_vec.push_back(nnz_count); // The first entry of indptr is always 0.
    for (int i = 0; i < num_spots; i++) {
        data_vec.insert(data_vec.end(), value_vec[i].begin(), value_vec[i].end());
        indices_vec.insert(indices_vec.end(), vox_vec[i].begin(), vox_vec[i].end());
        nnz_count += vox_vec[i].size();
        indptr_vec.push_back(nnz_count);
    }

    // --- Prepare data for saving with the cnpy library ---
    uint32_t shape[2] = { num_spots, vol_size };
    std::string format = "csr";

    // Save the five arrays that constitute a SciPy CSR matrix to a single .npz file.
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indices", indices_vec, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indptr", indptr_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "shape", shape, 2, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "data", data_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "format", format, 3, "a");

    delete[] value_vec;
    delete[] vox_vec;
}

template<typename R>
void
mqi::io::save_to_npz2(const mqi::scorer<R>* src,
                      const R               scale,
                      const std::string&    filepath,
                      const std::string&    filename,
                      mqi::vec3<mqi::ijk_t> dim,
                      uint32_t              num_spots) {
    // In this voxel-major version, the number of rows is the number of active voxels in the ROI.
    uint32_t vol_size = src->roi_->get_mask_size();
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";

    // Create temporary vectors to group spot indices and values by voxel index.
    auto* value_vec = new std::vector<double>[vol_size];
    auto* spot_vec  = new std::vector<mqi::key_t>[vol_size];

    // Iterate through the hash table and populate the temporary vectors.
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            // Map the original voxel index to the compact ROI index.
            mqi::key_t vox_ind = src->roi_->get_mask_idx(src->data_[ind].key1);
            if (vox_ind < 0) continue; // Skip if the voxel is not in the ROI.

            mqi::key_t spot_ind = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            value_vec[vox_ind].push_back(src->data_[ind].value * scale);
            spot_vec[vox_ind].push_back(spot_ind);
        }
    }

    // Sort the spot indices for each voxel, as required by the CSR format.
    for (int ind = 0; ind < vol_size; ind++) {
        if (spot_vec[ind].size() > 1) {
            std::vector<int> sort_indices(spot_vec[ind].size());
            std::iota(sort_indices.begin(), sort_indices.end(), 0);
            sort(sort_indices.begin(), sort_indices.end(), [&](int i, int j) {
                return spot_vec[ind][i] < spot_vec[ind][j];
            });
            // Reorder the spot and value vectors according to the sorted indices.
            std::vector<double>     sorted_value(spot_vec[ind].size());
            std::vector<mqi::key_t> sorted_spot(spot_vec[ind].size());
            for (int sorted_ind = 0; sorted_ind < spot_vec[ind].size(); sorted_ind++) {
                sorted_value[sorted_ind] = value_vec[ind][sort_indices[sorted_ind]];
                sorted_spot[sorted_ind]  = spot_vec[ind][sort_indices[sorted_ind]];
            }
            spot_vec[ind]  = sorted_spot;
            value_vec[ind] = sorted_value;
        }
    }

    // --- Convert the grouped data into the three CSR arrays ---
    std::vector<double>   data_vec;
    std::vector<uint32_t> indices_vec;
    std::vector<uint32_t> indptr_vec;
    int spot_count = 0;
    indptr_vec.push_back(spot_count);
    for (int i = 0; i < vol_size; i++) {
        data_vec.insert(data_vec.end(), value_vec[i].begin(), value_vec[i].end());
        indices_vec.insert(indices_vec.end(), spot_vec[i].begin(), spot_vec[i].end());
        spot_count += spot_vec[i].size();
        indptr_vec.push_back(spot_count);
    }

    // --- Prepare data for saving with the cnpy library ---
    uint32_t    shape[2] = { vol_size, num_spots };
    std::string format   = "csr";

    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indices", indices_vec, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indptr", indptr_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "shape", shape, 2, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "data", data_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "format", format, 3, "a");

    delete[] value_vec;
    delete[] spot_vec;
}

template<typename R>
void
mqi::io::save_to_npz(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     mqi::vec3<mqi::ijk_t> dim,
                     uint32_t              num_spots,
                     R*                    time_scale,
                     R                     threshold) {
    uint32_t vol_size = dim.x * dim.y * dim.z;
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";

    auto* value_vec = new std::vector<double>[num_spots];
    auto* vox_vec   = new std::vector<mqi::key_t>[num_spots];

    // Iterate through the hash table, apply scaling and thresholding, and populate temporary vectors.
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            mqi::key_t vox_ind  = src->data_[ind].key1;
            mqi::key_t spot_ind = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            double value = src->data_[ind].value;
            value *= scale;
            value -= 2 * threshold;
            if (value < 0) value = 0;
            value /= time_scale[spot_ind];
            value_vec[spot_ind].push_back(value);
            vox_vec[spot_ind].push_back(vox_ind);
        }
    }

    // Convert the grouped data into the three CSR arrays.
    std::vector<double>   data_vec;
    std::vector<uint32_t> indices_vec;
    std::vector<uint32_t> indptr_vec;
    int vox_count = 0;
    indptr_vec.push_back(vox_count);
    for (int i = 0; i < num_spots; i++) {
        data_vec.insert(data_vec.end(), value_vec[i].begin(), value_vec[i].end());
        indices_vec.insert(indices_vec.end(), vox_vec[i].begin(), vox_vec[i].end());
        vox_count += vox_vec[i].size();
        indptr_vec.push_back(vox_count);
    }

    // Prepare and save the arrays to an .npz file.
    uint32_t    shape[2] = { num_spots, vol_size };
    std::string format   = "csr";
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indices", indices_vec, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "indptr", indptr_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "shape", shape, 2, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "data", data_vec, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", "format", format, 3, "a");

    delete[] value_vec;
    delete[] vox_vec;
}

template<typename R>
void
mqi::io::save_to_mhd(const mqi::node_t<R>* children,
                     const double*         src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     const uint32_t        length) {
    // Extract geometry information from the node structure.
    float dx = children->geo[0].get_x_edges()[1] - children->geo[0].get_x_edges()[0];
    float dy = children->geo[0].get_y_edges()[1] - children->geo[0].get_y_edges()[0];
    float dz = children->geo[0].get_z_edges()[1] - children->geo[0].get_z_edges()[0];
    float x0 = children->geo[0].get_x_edges()[0] + dx * 0.5;
    float y0 = children->geo[0].get_y_edges()[0] + dy * 0.5;
    float z0 = children->geo[0].get_z_edges()[0] + dz * 0.5;

    // Write the .mhd header file with all necessary metadata.
    std::ofstream fid_header(filepath + "/" + filename + ".mhd", std::ios::out);
    if (!fid_header) { std::cout << "Cannot open file!" << std::endl; return; }
    fid_header << "ObjectType = Image\n";
    fid_header << "NDims = 3\n";
    fid_header << "BinaryData = True\n";
    fid_header << "BinaryDataByteOrderMSB = False\n";
    fid_header << "CompressedData = False\n";
    fid_header << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
    fid_header << "Offset = " << x0 << " " << y0 << " " << z0 << std::endl;
    fid_header << "CenterOfRotation = 0 0 0\n";
    fid_header << "AnatomicOrientation = RAI\n";
    fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
               << children->geo[0].get_nxyz().y << " " << children->geo[0].get_nxyz().z << "\n";
    fid_header << "ElementType = MET_DOUBLE\n";
    fid_header << "ElementSpacing = " << dx << " " << dy << " " << dz << "\n";
    fid_header << "ElementDataFile = " << filename << ".raw\n";
    fid_header.close();
    if (!fid_header.good()) { std::cout << "Error occurred at writing time!" << std::endl; }

    // Scale the data and write it to the separate .raw file.
    std::valarray<double> dest(src, length);
    dest *= scale;
    std::ofstream fid_raw(filepath + "/" + filename + ".raw", std::ios::out | std::ios::binary);
    if (!fid_raw) { std::cout << "Cannot open file!" << std::endl; return; }
    fid_raw.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(double));
    fid_raw.close();
    if (!fid_raw.good()) { std::cout << "Error occurred at writing time!" << std::endl; }
}

template<typename R>
void
mqi::io::save_to_mha(const mqi::node_t<R>* children,
                     const double*         src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     const uint32_t        length) {
    // Extract geometry information.
    float dx = children->geo[0].get_x_edges()[1] - children->geo[0].get_x_edges()[0];
    float dy = children->geo[0].get_y_edges()[1] - children->geo[0].get_y_edges()[0];
    float dz = children->geo[0].get_z_edges()[1] - children->geo[0].get_z_edges()[0];
    float x0 = children->geo[0].get_x_edges()[0] + dx * 0.5;
    float y0 = children->geo[0].get_y_edges()[0] + dy * 0.5;
    float z0 = children->geo[0].get_z_edges()[0] + dz * 0.5;

    // Scale the source data.
    std::valarray<double> dest(src, length);
    dest *= scale;

    // Open the .mha file and write the header.
    std::ofstream fid_header(filepath + "/" + filename + ".mha", std::ios::out | std::ios::binary);
    if (!fid_header) { std::cout << "Cannot open file!" << std::endl; return; }
    fid_header << "ObjectType = Image\n";
    fid_header << "NDims = 3\n";
    fid_header << "BinaryData = True\n";
    fid_header << "BinaryDataByteOrderMSB = False\n";
    fid_header << "CompressedData = False\n";
    fid_header << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
    fid_header << "Offset = " << std::setprecision(9) << x0 << " " << y0 << " " << z0 << "\n";
    fid_header << "CenterOfRotation = 0 0 0\n";
    fid_header << "AnatomicOrientation = RAI\n";
    fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
               << children->geo[0].get_nxyz().y << " " << children->geo[0].get_nxyz().z << "\n";
    fid_header << "ElementType = MET_DOUBLE\n";
    fid_header << "HeaderSize = -1\n";
    fid_header << "ElementSpacing = " << std::setprecision(9) << dx << " " << dy << " " << dz << "\n";
    fid_header << "ElementDataFile = LOCAL\n";

    // Write the binary data block directly after the header.
    fid_header.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(double));
    fid_header.close();
    if (!fid_header.good()) { std::cout << "Error occurred at writing time!" << std::endl; }
}

#endif
