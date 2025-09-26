/// \file mqi_ct.hpp
///
/// \brief Defines a class for handling 3D Computed Tomography (CT) image data.
///
/// \details This file provides the `mqi::ct` class, which is responsible for reading a
/// series of 2D DICOM CT slices from a directory and assembling them into a
/// coherent 3D grid. This 3D grid, representing the patient's anatomy in terms of
/// radiodensity, is fundamental for setting up the simulation geometry and assigning
/// materials. The pixel values in the final grid are in Hounsfield Units (HU).
///
/// \note This class depends on the GDCM (Grassroots DICOM) library for parsing
///       DICOM files.
#ifndef MQI_CT_H
#define MQI_CT_H

#include <sys/stat.h>

// GDCM headers for DICOM file processing
#include "gdcmAttribute.h"
#include "gdcmDirectory.h"
#include "gdcmIPPSorter.h"
#include "gdcmImageReader.h"
#include "gdcmScanner.h"

#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/mqi_rect3d.hpp>

namespace mqi
{

/// \class ct
/// \brief Represents a 3D CT image, loading and managing data from DICOM files.
///
/// \tparam R The floating-point type (e.g., `float` or `double`) for grid coordinates.
///           This is a C++ template, making the class adaptable to different numerical precisions,
///           similar to how a Python function can operate on different data types.
///
/// \details This class inherits from `rect3d<int16_t, R>` (like `class ct(rect3d):` in Python),
/// which provides the basic structure and functionality of a 3D grid. The `ct` class extends
/// this by adding specialized functionality to read a directory of DICOM files, sort them into the
/// correct spatial order, extract critical metadata (like image dimensions and pixel spacing),
/// and load the pixel data (in Hounsfield Units) into the grid.
template<typename R>
class ct : public rect3d<int16_t, R>
{
protected:
    /// A vector of file paths to the DICOM CT images, sorted by spatial location (Z-position).
    std::vector<std::string> files_;

    /// A map to quickly find a file path using its unique SOP Instance UID.
    std::map<std::string, std::string> uid2file_;

    /// The path to the directory containing the CT files.
    /// Note: This is a C-style string, manually allocated.
    char* ct_dir;

    /// The physical distance between the centers of adjacent pixels in the X-dimension (in mm).
    R dx_;
    /// The physical distance between the centers of adjacent pixels in the Y-dimension (in mm).
    R dy_;
    /// An array of physical distances between the centers of adjacent slices in the Z-dimension (in mm).
    /// This is an array to handle cases with variable slice thickness. Manually allocated.
    R* dz_;

public:
    /// \brief Default constructor.
    /// \note `CUDA_HOST` indicates this function can only be called from CPU (host) code.
    CUDA_HOST
    ct() {
        ;
    }

    /// \brief Constructs a `ct` object and initializes its structure from a DICOM directory.
    ///
    /// \details This constructor performs the following steps:
    /// 1. Uses `gdcm::Directory` to load all filenames from the specified directory.
    /// 2. Uses `gdcm::Scanner` to filter the files, keeping only those with the "CT" modality tag.
    /// 3. Uses `gdcm::IPPSorter` to sort the CT files based on their Image Position (Patient) tag.
    ///    This is a critical step, as filenames do not guarantee correct anatomical order.
    /// 4. Scans the sorted files again to extract essential metadata from DICOM tags, such as:
    ///    - Image dimensions (rows, columns)
    ///    - Pixel spacing (dx, dy)
    ///    - Slice thickness and position (for dz and z-coordinates)
    /// 5. Initializes the grid structure (dimensions and coordinate arrays) based on the metadata.
    ///    This involves manually allocating memory for the coordinate arrays with `new`.
    ///
    /// \param[in] f The path to the directory containing the CT DICOM files.
    /// \param[in] is_print If true, prints the detected file information to the console.
    /// \note This constructor sets up the grid's properties but does not load the actual pixel data.
    ///       Call `load_data()` separately to populate the grid with Hounsfield Units.
    CUDA_HOST
    ct(std::string f, bool is_print = false) {
        // Manually allocate and copy the directory path string.
        ct_dir = new char[f.length() + 1];
        strcpy(ct_dir, f.c_str());

        // Use GDCM to load all files from the specified directory (non-recursively).
        gdcm::Directory dir;
        dir.Load(this->ct_dir, false);
        const gdcm::Directory::FilenamesType& all_files = dir.GetFilenames();

        // Scan files to find those with Modality (0008,0060) set to "CT".
        gdcm::Scanner   scanner;
        const gdcm::Tag ct_tag = gdcm::Tag(0x08, 0x60);
        scanner.AddTag(ct_tag);
        scanner.Scan(all_files);
        auto ct_files = scanner.GetAllFilenamesFromTagToValue(ct_tag, "CT");

        // Sort the CT files by their spatial position to ensure correct slice order.
        gdcm::IPPSorter ippsorter;
        ippsorter.SetComputeZSpacing(false);
        ippsorter.Sort(ct_files);
        files_ = ippsorter.GetFilenames();

        // Scan the sorted files to extract necessary metadata tags.
        gdcm::Scanner s;
        s.AddTag(gdcm::Tag(0x0008, 0x0018));   ///< SOP instance UID
        s.AddTag(gdcm::Tag(0x0020, 0x0032));   ///< Image Position (Patient)
        s.AddTag(gdcm::Tag(0x0028, 0x0010));   ///< Rows
        s.AddTag(gdcm::Tag(0x0028, 0x0011));   ///< Columns
        s.AddTag(gdcm::Tag(0x0028, 0x0030));   ///< Pixel spacing
        s.AddTag(gdcm::Tag(0x0018, 0x0050));   ///< Slice thickness
        if (!s.Scan(files_)) assert("scan fail.");

        size_t nx;                   ///< columns
        size_t ny;                   ///< rows
        size_t nz = files_.size();   ///< number of images.

        // Initialize grid properties and manually allocate coordinate arrays.
        this->z_     = new R[nz];
        this->dim_.z = nz;
        double x0, y0;
        dz_ = new R[nz];
        float dz;
        for (size_t i = 0; i < nz; ++i) {
            gdcm::Scanner::TagToValue const& m0 = s.GetMapping(files_[i].c_str());

            // Parse the Image Position (Patient) tag, which is a string like "x\y\z".
            std::string  img_position(m0.find(gdcm::Tag(0x0020, 0x0032))->second);
            unsigned int deli0 = img_position.find_first_of('\\');
            unsigned int deli1 = img_position.find_last_of('\\');
            this->z_[i]        = (R) (std::stod(img_position.substr(deli1 + 1)));
            dz_[i]             = std::stod(m0.find(gdcm::Tag(0x0018, 0x0050))->second);

            // Determine grid parameters from the first image slice.
            if (i == 0) {
                x0 = std::stod(img_position.substr(0, deli0));
                y0 = std::stod(img_position.substr(deli0 + 1, deli1 - deli0));

                ny            = std::stoi(m0.find(gdcm::Tag(0x0028, 0x0010))->second);   // Rows
                nx            = std::stoi(m0.find(gdcm::Tag(0x0028, 0x0011))->second);   // Columns
                this->dim_.x = nx;
                this->dim_.y = ny;
                // Parse Pixel Spacing tag, which is a string like "dx\dy".
                std::string  pixel_spacing(m0.find(gdcm::Tag(0x0028, 0x0030))->second);
                unsigned int deli = pixel_spacing.find_first_of('\\');
                dx_               = std::stod(pixel_spacing.substr(0, deli));
                dy_               = std::stod(pixel_spacing.substr(deli + 1));
            }

            // Map SOP Instance UID to file path for quick lookup.
            uid2file_.insert(std::make_pair(m0.find(gdcm::Tag(0x0008, 0x0018))->second, files_[i]));
        }

        // Populate the x and y coordinate arrays.
        this->x_ = new R[nx];
        for (size_t i = 0; i < nx; ++i) {
            this->x_[i] = x0 + dx_ * i;
        }
        this->y_ = new R[ny];
        for (size_t i = 0; i < ny; ++i) {
            this->y_[i] = y0 + dy_ * i;
        }

        // Print summary if requested.
        if (is_print) {
            std::cout << "Reading DICOM directory.. : Getting patient CT info.." << std::endl;
            std::cout << "Reading DICOM directory.. : Detected CT info is as below." << std::endl;
            std::cout << "CT (nx, ny, nz) -> (" << this->dim_.x << ", " << this->dim_.y << ", "
                      << this->dim_.z << ")" << std::endl;
            std::cout << "CT (dx, dy, dz) -> (" << dx_ << ", " << dy_ << ", "
                      << this->z_[1] - this->z_[0] << ")" << std::endl;
            std::cout << "CT (x, y, z) -> (" << this->x_[0] << ", " << this->y_[0] << ", "
                      << this->z_[0] << ")" << std::endl;
        }
    }

    /// \brief Destructor to clean up manually allocated memory.
    /// \details In C++, memory allocated with `new[]` must be explicitly freed with `delete[]`
    /// to prevent memory leaks. This destructor is called automatically when a `ct` object
    /// goes out of scope or is deleted. The base class destructor is called automatically.
    CUDA_HOST
    ~ct() {
        delete[] ct_dir;
        delete[] dz_;
    }

    /// \brief Loads the pixel data from the DICOM files into the grid's data buffer.
    ///
    /// \details This method iterates through the sorted DICOM files, reads the raw pixel data
    /// for each slice using `gdcm::ImageReader`, and populates the `data_` member of the base `rect3d` class.
    /// It then applies the "Rescale Slope" and "Rescale Intercept" values found in
    /// the DICOM metadata. This is a mandatory step to convert the stored pixel values
    /// into meaningful Hounsfield Units (HU), where water is 0 HU and air is -1000 HU.
    /// The formula is: `HU = slope * raw_pixel_value + intercept`.
    CUDA_HOST
    virtual void
    load_data() {
        std::cout << "Reading DICOM directory.. : Loading patient CT pixel data.." << std::endl;
        size_t nb_voxels_2d = this->dim_.x * this->dim_.y;
        size_t nb_voxels_3d = nb_voxels_2d * this->dim_.z;

        this->data_.resize(nb_voxels_3d);
        float intercept = 0;
        float slope     = 1;

        for (size_t i = 0; i < this->dim_.z; ++i) {
            gdcm::ImageReader reader;
            reader.SetFileName(files_[i].c_str());
            reader.Read();
            const gdcm::Image& img = reader.GetImage();

            // Extract slope and intercept for HU conversion.
            intercept = float(img.GetIntercept());
            slope     = float(img.GetSlope());

            // Read the raw pixel data into the correct slice of the 3D grid.
            gdcm::PixelFormat pixeltype = img.GetPixelFormat();
            if (pixeltype == gdcm::PixelFormat::INT16) {
                img.GetBuffer((char*) &this->data_[i * nb_voxels_2d]);
            } else {
                // Production code should handle other pixel types if they are expected.
                assert(0);
            }
        }
        // Apply rescale slope and intercept to the entire dataset to get Hounsfield Units.
        this->data_ = this->data_ * int16_t(slope) + intercept;
        std::cout << "Reading DICOM directory.. : Patient CT pixel data successfully loaded." << std::endl;
    }

    /// \brief Finds the grid index in the x-dimension for a given physical x-coordinate.
    /// \param[in] x The x-coordinate (in mm).
    /// \return The corresponding index in the x-dimension.
    inline virtual size_t
    find_c000_x_index(const R& x) {
        assert((x >= rect3d<int16_t, R>::x_[0]) &&
               (x < rect3d<int16_t, R>::x_[rect3d<int16_t, R>::dim_.x - 1]));
        assert(dx_ > 0);
        return floor((x - rect3d<int16_t, R>::x_[0]) / dx_);
    }

    /// \brief Finds the grid index in the y-dimension for a given physical y-coordinate.
    /// \param[in] y The y-coordinate (in mm).
    /// \return The corresponding index in the y-dimension.
    inline virtual size_t
    find_c000_y_index(const R& y) {
        assert((y >= rect3d<int16_t, R>::y_[0]) &&
               (y < rect3d<int16_t, R>::y_[rect3d<int16_t, R>::dim_.y - 1]));
        assert(dy_ > 0);
        return floor((y - rect3d<int16_t, R>::y_[0]) / dy_);
    }

    /// \brief Gets the pixel spacing in the x-dimension.
    /// \return The pixel spacing in x (in mm).
    inline virtual R
    get_dx() {
        return dx_;
    }

    /// \brief Gets the pixel spacing in the y-dimension.
    /// \return The pixel spacing in y (in mm).
    inline virtual R
    get_dy() {
        return dy_;
    }

    /// \brief Gets the array of pixel spacings in the z-dimension.
    /// \return A pointer to the array of z-spacings (in mm).
    inline virtual R*
    get_dz() {
        return dz_;
    }

    /// \brief Sets the pixel spacing in the x-dimension.
    /// \param[in] dx The new pixel spacing in x.
    inline virtual void
    set_dx(R dx) {
        dx_ = dx;
    }

    /// \brief Sets the pixel spacing in the y-dimension.
    /// \param[in] dy The new pixel spacing in y.
    inline virtual void
    set_dy(R dy) {
        dy_ = dy;
    }

    /// \brief Sets the array of pixel spacings in the z-dimension.
    /// \param[in] dz A pointer to the new array of z-spacings.
    inline virtual void
    set_dz(R* dz) {
        dz_ = dz;
    }

    /// \brief A utility function to create a new grid with the same structure as this CT grid.
    ///
    /// A `friend` function is a function defined outside the class that is granted
    /// access to the class's private and protected members. This is used here to
    /// allow `clone_ct_structure` to directly copy the grid parameters from a `ct`
    /// object to another `rect3d` object.
    ///
    /// \tparam R0 The coordinate type of the source `ct` grid.
    /// \tparam T1 The pixel type of the destination `rect3d` grid.
    /// \tparam R1 The coordinate type of the destination `rect3d` grid.
    /// \param[in] src The source `ct` object whose structure is to be copied.
    /// \param[out] dest The destination `rect3d` object that will receive the structure.
    template<typename R0, typename T1, typename R1>
    friend void
    clone_ct_structure(ct<R0>& src, rect3d<T1, R1>& dest);
};

}   // namespace mqi

#endif
