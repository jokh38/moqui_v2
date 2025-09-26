/// \file mqi_error_check.hpp
/// \brief Defines error-checking utilities for CUDA operations.
///
/// This file provides essential functions and macros for robust error handling,
/// which is critical in C++ and CUDA programming. Unlike Python, where exceptions
/// are common, C/C++ often relies on checking return codes from functions.
/// This is especially important for CUDA, as many GPU operations are asynchronous,
/// meaning an error might not be reported until much later. These utilities
/// help to catch and report errors as soon as they occur.
#ifndef MQI_ERROR_CHECK_HPP
#define MQI_ERROR_CHECK_HPP

namespace mqi
{
/// \brief Checks for any asynchronous errors returned by the CUDA runtime.
///
/// Many CUDA operations, especially kernel launches, are asynchronous. This means the
/// CPU code continues to execute without waiting for the GPU to finish. If an error
-/// occurs on the GPU during this time, it won't be caught by checking the return
+/// occurs on the GPU during this time, it won't be caught by checking the return
/// value of the kernel launch itself. `cudaGetLastError()` is used to retrieve such
/// asynchronous errors.
///
/// This function should be called after a kernel launch or other asynchronous
/// operations to ensure they completed successfully.
///
/// \param[in] msg A custom message to print, helping to identify where the check was performed.
inline void
check_cuda_last_error(const char* msg) {
    // This code is only compiled when using a CUDA compiler (like NVCC).
    // The __CUDACC__ macro is defined by the CUDA compiler.
#if defined(__CUDACC__)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        printf(
          "current total %f MB free %f MB\n", total / (1024.0 * 1024.0), free / (1024.0 * 1024.0));
        printf("CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

}   // namespace mqi


/// \note The following code is only included if the file is being compiled by a
///       CUDA compiler. This is a common technique called "conditional compilation"
///       used to separate platform-specific or library-specific code.
#if defined(__CUDACC__)

/// \def gpu_err_chk(ans)
/// \brief A macro to check the return code of a synchronous CUDA API call.
///
/// In C++, a macro is a rule that replaces a piece of text with another before
/// the code is compiled. This one, `gpu_err_chk`, is designed to wrap a CUDA
/// function call. It takes the function call as an argument (`ans`), executes it,
/// and then passes the return value to the `cuda_error_checker` function.
///
/// The advantage of using a macro here is that it automatically captures the file name
/// (`__FILE__`) and line number (`__LINE__`) where the error occurred, which is
/// extremely helpful for debugging.
///
/// \b Example:
/// \code{.cpp}
/// // This will call cudaMalloc and immediately check if it returned cudaSuccess.
/// gpu_err_chk(cudaMalloc(&d_data, size));
/// \endcode
/// \param ans The CUDA API call to be executed and checked.
#define gpu_err_chk(ans)                                                                           \
    { cuda_error_checker((ans), __FILE__, __LINE__); }

/// \brief A helper function that checks a CUDA error code and reports it if needed.
///
/// This function is called by the `gpu_err_chk` macro. It checks if the given
/// error code is `cudaSuccess`. If not, it prints a detailed error message to the
/// standard error stream (`stderr`), including the error string, file, and line number,
/// and then exits the program.
///
/// \param[in] code The `cudaError_t` code returned by a CUDA API call.
/// \param[in] file The name of the source file where the error occurred (provided by `__FILE__`).
/// \param[in] line The line number where the error occurred (provided by `__LINE__`).
/// \param[in] abort If true (the default), the program will exit upon encountering an error.
inline void
cuda_error_checker(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif

#endif