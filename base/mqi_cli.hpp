#ifndef MQI_CLI_HPP
#define MQI_CLI_HPP

/// \file mqi_cli.hpp
///
/// \brief Defines a command-line interface (CLI) helper class.
///
/// \details This file provides the `cli` class, which is designed to parse and manage
/// command-line arguments for executables like unit tests and simulation applications.
/// It allows a user to specify settings like input file paths, simulation parameters,
/// and output options from the command line, making the programs flexible and configurable
/// without needing to recompile the code.

#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>

#include <sys/stat.h>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mqi
{

/// \class cli
/// \brief A simple command-line interface parser for tests and applications.
///
/// \details This class provides a basic framework for parsing command-line arguments.
/// It predefines a set of expected options (e.g., "--dicom_path") and stores any
/// provided values for them. For a Python developer, this is similar in purpose to
/// Python's `argparse` module, but much simpler.
///
/// Its main purposes are:
/// 1. Reading input files (e.g., RT-Ion plans).
/// 2. Configuring geometries, patients, dose grids, and beamline components.
/// 3. Setting up beam sources and simulation parameters (e.g., number of histories).
class cli
{

protected:
    /// A map to store the command-line options and their arguments.
    /// The key is the option name (e.g., "--dicom_path"), and the value is a
    /// vector of string arguments provided for that option.
    /// For a Python developer, this is analogous to a dictionary where keys are strings
    /// and values are lists of strings, e.g., `{'--nhistory': ['10000']}`.
    std::map<const std::string, std::vector<std::string>> parameters;

public:
    /// \brief Constructs the `cli` object and initializes the list of expected parameters.
    /// \details This pre-populates the `parameters` map with all valid option keys. This allows
    /// the `read` method to easily check if an option seen on the command line is a known one.
    cli() {
        parameters = std::map<const std::string, std::vector<std::string>>({
          { "--dicom_path", {} },
          { "--bname", {} },
          { "--bnumber", {} },
          { "--spots", {} },
          { "--beamlets", {} },
          { "--pph", {} },
          { "--sid", {} },
          { "--output_prefix", {} },
          { "--nhistory", {} },
          { "--pxyz", {} },
          { "--source_energy", {} },
          { "--energy_variance", {} },
          { "--rxyz", {} },
          { "--lxyz", {} },
          { "--nxyz", {} },
          { "--spot_position", {} },
          { "--spot_size", {} },
          { "--spot_angles", {} },
          { "--spot_energy", {} },
          { "--histories", {} },
          { "--threads", {} },
          { "--score_variance", {} },
          { "--gpu_id", {} },
          { "--output_format", {} },
          { "--random_seed", {} },
          { "--phantom_path", {} }
        });
    }

    /// \brief Destroys the `cli` object.
    ~cli() {
        ;
    }

    /// \brief Parses the command-line arguments and populates the parameters map.
    ///
    /// \details This function iterates through the command-line arguments (`argv`). When it
    /// encounters a known option (a key that exists in the `parameters` map),
    /// it consumes all subsequent arguments as values for that option, stopping when it
    /// finds the next option (a string starting with "--") or runs out of arguments.
    ///
    /// \param[in] argc The argument count, automatically supplied to `main`.
    /// \param[in] argv The argument vector (an array of C-style strings), automatically supplied to `main`.
    void
    read(int argc, char** argv) {
        std::cout << "# of arguments: " << argc << std::endl;

        // Start from 1 to skip the program name (argv[0])
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto        it  = parameters.find(arg);   // Check if the argument is a known option

            if (it != parameters.end()) {
                // It is a known option. Now, consume all following arguments as its values.
                int j = i + 1;
                // Keep consuming until we reach the end or find the next option flag.
                while (j < argc && std::string(argv[j]).rfind("--", 0) != 0) {
                    it->second.push_back(argv[j]);
                    j++;
                }
                i = j - 1;   // Move the outer loop's counter past the consumed arguments.
            } else {
                std::cerr << "Warning: Unknown option '" << arg << "' ignored." << std::endl;
            }
        }

        // Print out the parsed parameters for verification
        for (const auto& pair : parameters) {
            if (!pair.second.empty()) {
                std::cout << pair.first << " : ";
                for (const auto& param : pair.second) {
                    std::cout << param << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    /// \brief Prints a help message describing the available command-line options.
    ///
    /// \param[in] s The name of the executable (`argv[0]`).
    virtual void
    print_help(char* s) {
        std::cout << "Usage:   " << s << " [-option] [argument]" << std::endl;
        std::cout << "options:  " << std::endl;
        for (const auto& pair : parameters) {
            std::cout << "          " << pair.first << std::endl;
        }
    }

    /// \brief Provides read-only access to the arguments for a specific option.
    /// \details This "overloaded operator" allows you to use square brackets `[]` to access
    /// the parsed arguments in a clean, dictionary-like syntax.
    /// Example: `std::vector<std::string> energies = my_cli["--source_energy"];`
    ///
    /// \param[in] t The name of the command-line option (e.g., "--dicom_path").
    /// \return A const vector of strings containing the arguments for the specified option.
    ///         Returns an empty vector if the option was not provided on the command line.
    const std::vector<std::string>
    operator[](const std::string& t) {
        return parameters[t];
    }
};
}   // namespace mqi

#endif
