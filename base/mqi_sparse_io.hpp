#ifndef MQI_SPARSE_IO_HPP
#define MQI_SPARSE_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>   //accumulate
#include <valarray>
#include <zlib.h>

#include <sys/mman.h>   //for io

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_scorer.hpp>

namespace mqi
{
/**
 * @namespace io
 * @brief Provides functions for low-level input/output operations, specifically for creating NumPy NPZ files.
 */
namespace io
{
/**
 * @brief Saves a std::string as a variable in a .npz file.
 * @details This function manually constructs the necessary ZIP and NPY headers to save string data.
 * It supports both writing to a new file ('w') and appending to an existing one ('a').
 * @param filename The path to the output .npz file.
 * @param var_name The name of the variable to be saved within the .npz archive.
 * @param data The string data to save.
 * @param shape The shape of the data (should be 1 for a single string).
 * @param mode The file mode: "w" for write (creates a new file) or "a" for append (adds to an existing file).
 */
void
save_npz(std::string filename,
         std::string var_name,
         std::string data,
         size_t      shape,
         std::string mode);

/**
 * @brief Saves a C-style array as a variable in a .npz file.
 * @details This function template manually constructs the necessary ZIP and NPY headers to save a numeric array.
 * It supports both writing to a new file ('w') and appending to an existing one ('a').
 * @tparam T The data type of the array elements.
 * @param filename The path to the output .npz file.
 * @param var_name The name of the variable to be saved within the .npz archive.
 * @param data A pointer to the beginning of the data array.
 * @param shape The number of elements in the array.
 * @param mode The file mode: "w" for write (creates a new file) or "a" for append (adds to an existing file).
 */
template<typename T>
void
save_npz(std::string filename, std::string var_name, T* data, size_t shape, std::string mode);

/**
 * @brief Appends the bytes of a std::string to a character vector.
 * @param vec The character vector to append to.
 * @param str The string to append.
 */
void
push_value(std::vector<char>& vec, const std::string str);

/**
 * @brief Appends the bytes of a uint16_t to a character vector.
 * @param vec The character vector to append to.
 * @param str The uint16_t value to append.
 */
void
push_value(std::vector<char>& vec, const uint16_t str);

/**
 * @brief Appends the bytes of a uint32_t to a character vector.
 * @param vec The character vector to append to.
 * @param str The uint32_t value to append.
 */
void
push_value(std::vector<char>& vec, const uint32_t str);

/**
 * @brief Parses the End of Central Directory Record (EOCD) of a ZIP file.
 * @details This is used in append mode to find the location of the existing central directory,
 * allowing new data to be appended correctly.
 * @param filename The path to the ZIP (.npz) file.
 * @param nrecs Output parameter for the number of records in the central directory.
 * @param global_header_size Output parameter for the total size of the central directory.
 * @param global_header_offset Output parameter for the starting offset of the central directory.
 */
void
parse_zip_footer(std::string filename,
                 uint16_t&   nrecs,
                 size_t&     global_header_size,
                 size_t&     global_header_offset);

/**
 * @brief Maps a C++ typeid to a NumPy type character code.
 * @param t The std::type_info of the C++ type.
 * @return A character representing the NumPy data type (e.g., 'f' for float, 'i' for int).
 */
char
map_type(const std::type_info& t);
}   // namespace io
}   // namespace mqi

void
mqi::io::push_value(std::vector<char>& vec, const std::string str) {
    for (int i = 0; i < str.length(); i++) {
        vec.push_back(str[i]);
    }
}

void
mqi::io::push_value(std::vector<char>& vec, const uint16_t str) {
    for (size_t byte = 0; byte < sizeof(str); byte++) {
        char val = *((char*) &str + byte);
        vec.push_back(val);
    }
}

void
mqi::io::push_value(std::vector<char>& vec, const uint32_t str) {
    for (size_t byte = 0; byte < sizeof(str); byte++) {
        char val = *((char*) &str + byte);
        vec.push_back(val);
    }
}

void
mqi::io::parse_zip_footer(std::string filename,
                          uint16_t&   nrecs,
                          size_t&     global_header_size,
                          size_t&     global_header_offset) {
    std::ifstream     fp(filename.c_str(), std::ios::in | std::ios::binary);
    std::vector<char> footer(22);
    fp.seekg(-22, fp.end);
    fp.read(&footer[0], 22);
    if (!fp) {
        std::cout << "error: only " << fp.gcount() << " could be read" << std::endl;
        exit(-1);
    }
    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no              = *(uint16_t*) &footer[4];
    disk_start           = *(uint16_t*) &footer[6];
    nrecs_on_disk        = *(uint16_t*) &footer[8];
    nrecs                = *(uint16_t*) &footer[10];
    global_header_size   = *(uint32_t*) &footer[12];
    global_header_offset = *(uint32_t*) &footer[16];
    comment_len          = *(uint16_t*) &footer[20];
    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
    fp.seekg(global_header_offset, fp.beg);

    fp.close();
}

char
mqi::io::map_type(const std::type_info& t) {
    if (t == typeid(float))
        return 'f';
    else if (t == typeid(double))
        return 'f';
    else if (t == typeid(long double))
        return 'f';

    else if (t == typeid(int))
        return 'i';
    else if (t == typeid(char))
        return 'i';
    else if (t == typeid(short))
        return 'i';
    else if (t == typeid(long))
        return 'i';
    else if (t == typeid(long long))
        return 'i';

    else if (t == typeid(unsigned char))
        return 'u';
    else if (t == typeid(unsigned short))
        return 'u';
    else if (t == typeid(unsigned long))
        return 'u';
    else if (t == typeid(unsigned long long))
        return 'u';
    else if (t == typeid(unsigned int))
        return 'u';

    else if (t == typeid(bool))
        return 'b';

    else if (t == typeid(std::complex<float>))
        return 'c';
    else if (t == typeid(std::complex<double>))
        return 'c';
    else if (t == typeid(std::complex<long double>))
        return 'c';
    else if (t == typeid(std::string))
        return 'S';
    else
        return '?';
}

void
mqi::io::save_npz(std::string filename,
                  std::string var_name,
                  std::string data,
                  size_t      shape,
                  std::string mode) {
    std::fstream      fid_out;
    uint16_t          nrecs                = 0;
    size_t            global_header_offset = 0;
    std::vector<char> global_header;
    if (mode == "a") {
        size_t global_header_size;
        mqi::io::parse_zip_footer(filename, nrecs, global_header_size, global_header_offset);
        fid_out.open(filename, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
        fid_out.seekg(0, fid_out.beg);
        if (fid_out.eof()) { printf("end of file reached\n"); }
        fid_out.seekg(global_header_offset, fid_out.beg);
        global_header.resize(global_header_size);
        if (fid_out.eof()) { printf("end of file reached\n"); }

        fid_out.read(&global_header[0], global_header_size);

        if (!fid_out) {
            std::cout << "error: only " << fid_out.gcount() << " could be read" << std::endl;
            exit(-1);
        }
        fid_out.seekg(global_header_offset, fid_out.beg);

    } else if (mode == "w") {
        fid_out.open(filename, std::ios::out | std::ios::binary);
    } else {
        throw std::runtime_error("Undefined mode\n");
    }
    std::vector<char> dict;
    mqi::io::push_value(dict, "{'descr': '");
    if (mqi::io::map_type(typeid(std::string)) == 'S') {
        mqi::io::push_value(dict, "|");
        dict.push_back(mqi::io::map_type(typeid(std::string)));
        mqi::io::push_value(dict, std::to_string(data.length()));
    } else {
        printf("wrong call\n");
        exit(-1);
    }

    mqi::io::push_value(dict, "', 'fortran_order': False, 'shape': (");
    mqi::io::push_value(dict, "), }");
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.push_back('\n');
    std::vector<char> header;
    header.push_back((char) 0x93);
    mqi::io::push_value(header, "NUMPY");
    header.push_back((char) 0x01);
    header.push_back((char) 0x00);
    mqi::io::push_value(header, (uint16_t) dict.size());
    header.insert(header.end(), dict.begin(), dict.end());

    size_t   nbytes = sizeof(std::string) * data.length() + header.size();
    uint32_t crc    = crc32(0L, (uint8_t*) &header[0], header.size());
    crc             = crc32(crc, (uint8_t*) data.c_str(), sizeof(std::string) * data.length());

    //build the local header
    std::vector<char> local_header;
    mqi::io::push_value(local_header, "PK");                         //first part of sig
    mqi::io::push_value(local_header, (uint16_t) 0x0403);            //second part of sig
    mqi::io::push_value(local_header, (uint16_t) 20);                //min version to extract
    mqi::io::push_value(local_header, (uint16_t) 0);                 //general purpose bit flag
    mqi::io::push_value(local_header, (uint16_t) 0);                 //compression method
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod time
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod date
    mqi::io::push_value(local_header, (uint32_t) crc);               //crc
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //compressed size
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //uncompressed size
    mqi::io::push_value(local_header, (uint16_t) var_name.size());   //fname length
    mqi::io::push_value(local_header, (uint16_t) 0);                 //extra field length
    mqi::io::push_value(local_header, var_name);
    //build global header
    mqi::io::push_value(global_header, "PK");                //first part of sig
    mqi::io::push_value(global_header, (uint16_t) 0x0201);   //second part of sig
    mqi::io::push_value(global_header, (uint16_t) 20);       //version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    mqi::io::push_value(global_header, (uint16_t) 0);   //file comment length
    mqi::io::push_value(global_header, (uint16_t) 0);   //disk number where file starts
    mqi::io::push_value(global_header, (uint16_t) 0);   //internal file attributes
    mqi::io::push_value(global_header, (uint32_t) 0);   //external file attributes
    mqi::io::push_value(
      global_header,
      (uint32_t)
        global_header_offset);   //relative offset of local file header, since it begins where the global header used to begin
    mqi::io::push_value(global_header, var_name);
    std::vector<char> footer;
    mqi::io::push_value(footer, "PK");                              //first part of sig
    mqi::io::push_value(footer, (uint16_t) 0x0605);                 //second part of sig
    mqi::io::push_value(footer, (uint16_t) 0);                      //number of this disk
    mqi::io::push_value(footer, (uint16_t) 0);                      //disk where footer starts
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //number of records on this disk
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //total number of records
    mqi::io::push_value(footer, (uint32_t) global_header.size());   //nbytes of global headers
    mqi::io::push_value(
      footer,
      (uint32_t) (global_header_offset + nbytes +
                  local_header
                    .size()));   //offset of start of global headers, since global header now starts after newly written array
    mqi::io::push_value(footer, (uint16_t) 0);   //zip file comment length

    fid_out.write(reinterpret_cast<const char*>(&local_header[0]),
                  sizeof(char) * local_header.size());
    fid_out.write(reinterpret_cast<const char*>(&header[0]), sizeof(char) * header.size());
    fid_out.write(reinterpret_cast<const char*>(&data[0]), sizeof(std::string) * data.length());
    fid_out.write(reinterpret_cast<const char*>(&global_header[0]),
                  sizeof(char) * global_header.size());
    fid_out.write(reinterpret_cast<const char*>(&footer[0]), sizeof(char) * footer.size());
    fid_out.close();
}

template<typename T>
void
mqi::io::save_npz(std::string filename,
                  std::string var_name,
                  T*          data,
                  size_t      shape,
                  std::string mode) {
    std::fstream      fid_out;
    uint16_t          nrecs                = 0;
    size_t            global_header_offset = 0;
    std::vector<char> global_header;
    if (mode == "a") {
        size_t global_header_size;
        mqi::io::parse_zip_footer(filename, nrecs, global_header_size, global_header_offset);
        fid_out.open(filename, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
        fid_out.seekg(0, fid_out.beg);
        if (fid_out.eof()) { printf("end of file reached\n"); }
        fid_out.seekg(global_header_offset, fid_out.beg);

        global_header.resize(global_header_size);
        if (fid_out.eof()) { printf("end of file reached\n"); }

        fid_out.read(&global_header[0], global_header_size);

        if (!fid_out) {
            std::cout << "error: only " << fid_out.gcount() << " could be read" << std::endl;
            exit(-1);
        }
        fid_out.seekg(global_header_offset, fid_out.beg);

    } else if (mode == "w") {
        fid_out.open(filename, std::ios::out | std::ios::binary);
    } else {
        throw std::runtime_error("Undefined mode\n");
    }
    std::vector<char> dict;
    mqi::io::push_value(dict, "{'descr': '");
    if (mqi::io::map_type(typeid(T)) == 'S') {
        mqi::io::push_value(dict, "|");
        dict.push_back(mqi::io::map_type(typeid(T)));
        mqi::io::push_value(dict, "4");
    } else {
        mqi::io::push_value(dict, "<");
        dict.push_back(mqi::io::map_type(typeid(T)));
        mqi::io::push_value(dict, std::to_string(sizeof(T)));
    }

    mqi::io::push_value(dict, "', 'fortran_order': False, 'shape': (");
    mqi::io::push_value(dict, std::to_string(shape));
    mqi::io::push_value(dict, ",), }");
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.push_back('\n');
    std::vector<char> header;
    header.push_back((char) 0x93);
    mqi::io::push_value(header, "NUMPY");
    header.push_back((char) 0x01);
    header.push_back((char) 0x00);
    mqi::io::push_value(header, (uint16_t) dict.size());
    header.insert(header.end(), dict.begin(), dict.end());

    size_t   nbytes = shape * sizeof(T) + header.size();
    uint32_t crc    = crc32(0L, (uint8_t*) &header[0], header.size());
    crc             = crc32(crc, (uint8_t*) data, shape * sizeof(T));

    //build the local header
    std::vector<char> local_header;
    mqi::io::push_value(local_header, "PK");                         //first part of sig
    mqi::io::push_value(local_header, (uint16_t) 0x0403);            //second part of sig
    mqi::io::push_value(local_header, (uint16_t) 20);                //min version to extract
    mqi::io::push_value(local_header, (uint16_t) 0);                 //general purpose bit flag
    mqi::io::push_value(local_header, (uint16_t) 0);                 //compression method
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod time
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod date
    mqi::io::push_value(local_header, (uint32_t) crc);               //crc
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //compressed size
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //uncompressed size
    mqi::io::push_value(local_header, (uint16_t) var_name.size());   //fname length
    mqi::io::push_value(local_header, (uint16_t) 0);                 //extra field length
    mqi::io::push_value(local_header, var_name);
    //build global header
    mqi::io::push_value(global_header, "PK");                //first part of sig
    mqi::io::push_value(global_header, (uint16_t) 0x0201);   //second part of sig
    mqi::io::push_value(global_header, (uint16_t) 20);       //version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    mqi::io::push_value(global_header, (uint16_t) 0);   //file comment length
    mqi::io::push_value(global_header, (uint16_t) 0);   //disk number where file starts
    mqi::io::push_value(global_header, (uint16_t) 0);   //internal file attributes
    mqi::io::push_value(global_header, (uint32_t) 0);   //external file attributes
    mqi::io::push_value(
      global_header,
      (uint32_t)
        global_header_offset);   //relative offset of local file header, since it begins where the global header used to begin
    mqi::io::push_value(global_header, var_name);
    //build footer
    std::vector<char> footer;
    mqi::io::push_value(footer, "PK");                              //first part of sig
    mqi::io::push_value(footer, (uint16_t) 0x0605);                 //second part of sig
    mqi::io::push_value(footer, (uint16_t) 0);                      //number of this disk
    mqi::io::push_value(footer, (uint16_t) 0);                      //disk where footer starts
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //number of records on this disk
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //total number of records
    mqi::io::push_value(footer, (uint32_t) global_header.size());   //nbytes of global headers
    mqi::io::push_value(
      footer,
      (uint32_t) (global_header_offset + nbytes +
                  local_header
                    .size()));   //offset of start of global headers, since global header now starts after newly written array
    mqi::io::push_value(footer, (uint16_t) 0);   //zip file comment length

    fid_out.write(reinterpret_cast<const char*>(&local_header[0]),
                  sizeof(char) * local_header.size());
    fid_out.write(reinterpret_cast<const char*>(&header[0]), sizeof(char) * header.size());
    fid_out.write(reinterpret_cast<const char*>(&data[0]), sizeof(T) * shape);
    fid_out.write(reinterpret_cast<const char*>(&global_header[0]),
                  sizeof(char) * global_header.size());
    fid_out.write(reinterpret_cast<const char*>(&footer[0]), sizeof(char) * footer.size());
    fid_out.close();
}

#endif