# Specifies the minimum version of CMake required to build the project.
cmake_minimum_required(VERSION 3.10)

# Defines the project name.
project(env)

# A toggle to enable or disable GPU (CUDA) support.
set(GPU ON)

# The extension of the main code should be cpp to compile it using g++
# for CPU version and using nvcc for GPU version.
# It will not be compiled using g++ if the extension is cu

# Hard-coded path to a specific CUDA compiler and toolkit.
# This ensures a consistent build environment.
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc CACHE FILEPATH "CUDA 11.7 nvcc" FORCE)
set(CUDATOOLkit_ROOT /usr/local/cuda-11.7 CACHE PATH "CUDA 11.7 root" FORCE)

# Defines the main executable target and its source file.
add_executable(tps_env tps_env.cpp)

# Conditional compilation block based on the GPU variable.
if (GPU)
    # GPU (CUDA) build configuration.
    message("Compile using nvcc")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    # Set the CUDA architecture (e.g., 75 for Turing).
    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_COMPILER nvcc)
    # Set CUDA compiler flags for performance and to suppress warnings.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    # Tell CMake to compile the .cpp file with the CUDA compiler (nvcc).
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)
else ()
    # CPU-only build configuration.
    message("Compile using g++")
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CXX_COMPILER g++)
endif ()

# Find the GDCM library, which is required for DICOM file handling.
find_package(GDCM REQUIRED)
include(${GDCM_USE_FILE})

# On Apple platforms, find the CoreFoundation library.
if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

# Add include directories for the project, GDCM, and CUDA.
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../../../
        ${CUDAToolkit_INCLUDE_DIRS}
        #/usr/local/cuda/include
)

# Add the GDCM library directory to the linker path.
link_directories(${GDCM_DIR})

# Link the executable against the required libraries.
target_link_libraries(tps_env
        ${COREFOUNDATION_LIBRARY}
        gdcmCommon
        gdcmDSED
        gdcmMEXD
        gdcmjpeg12
        gdcmjpeg8
        gdcmDICT
        gdcmIOD
        gdcmMSFF
        gdcmjpeg16
        )
