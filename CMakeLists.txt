cmake_minimum_required(VERSION 3.15)
project(MyProject)

# Set compiler standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find OpenBLAS and LAPACK
find_package(OpenBLAS REQUIRED)
find_package(LAPACK REQUIRED)

# --- Optional CUDA Setup ---
# Check for CUDA without making it "REQUIRED" initially
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}. Enabling CUDA targets.")
else()
    message(WARNING "CUDA not found. Skipping CUDA-based library targets.")
endif()

# --- Include directory discovery ---
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${OpenBLAS_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Target Configuration ---
function(add_pybind11_library target_name src_file use_cuda)
    # Check if we are trying to build CUDA but it's not available
    if(use_cuda AND NOT CUDAToolkit_FOUND)
        message(STATUS "Skipping target ${target_name} because CUDA was not found.")
        return()
    endif()

    add_library(${target_name} SHARED ${src_file})
    
    # Modern CMake: Use target_link_libraries for everything instead of global include_directories if possible
    if(use_cuda)
        target_link_libraries(${target_name} PRIVATE
            CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cudart
            pybind11::module
        )
        target_compile_options(${target_name} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
        )
    else()
        target_link_libraries(${target_name} PRIVATE
            OpenBLAS::OpenBLAS
            ${LAPACK_LIBRARIES}
            pybind11::module
        )
    endif()

    target_compile_definitions(${target_name} PRIVATE NDEBUG)
    target_compile_options(${target_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
    )
endfunction()

# --- Build targets ---

# CPU Targets (always built)
add_pybind11_library(randSvd randSvd.cpp OFF)
add_pybind11_library(localization localization.cpp OFF)

# CUDA Targets (conditionally built inside the function)
add_pybind11_library(randSvdCuda randSvdCuda.cu ON)
add_pybind11_library(localizationCuda localizationCuda.cu ON)

# Output .so files to build directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})