cmake_minimum_required(VERSION 3.10)
project(env LANGUAGES CXX)

option(GPU "Enable GPU support" OFF)

add_executable(tps_env tps_env.cpp)

if (GPU)
    message(STATUS "Compiling with CUDA support")
    # RTX A5000 = sm_86
    set(CMAKE_CUDA_ARCHITECTURES 86)
    # 런타임을 동적 링크로 (ldd에 libcudart가 보이도록)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    # setting Cuda 11.7
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc CACHE FILEPATH "CUDA 11.7 nvcc" FORCE)
    set(CUDATOOLkit_ROOT /usr/local/cuda-11.7 CACHE PATH "CUDA 11.7 root" FORCE)

    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)

    include_directories(${CUDATOOLkit_INCLUDE_DIRS} /usr/local/cuda/include)
    target_link_libraries(tps_env PRIVATE CUDA::cudart)
else ()
    message(STATUS "Compiling without CUDA support (CPU only)")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CXX_COMPILER g++)
endif ()

find_package(GDCM REQUIRED)
include(${GDCM_USE_FILE})

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../
)

link_directories(${GDCM_DIR})

target_link_libraries(tps_env PRIVATE
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