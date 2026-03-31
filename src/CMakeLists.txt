cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Remove CUDA from project languages to prevent immediate failure if nvcc is missing
project(osvmp2 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========== DIRECTORY STRUCTURE ==========
# Current file is at ./src/CMakeLists.txt
# BASE_DIR is parent of src (./)
# Libraries go to ./osvmp2/lib/

get_filename_component(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)  # /path/to/src
get_filename_component(BASE_DIR ${SRC_DIR} DIRECTORY)                  # /path/to (parent of src)

set(CPP_DIR ${SRC_DIR}/cpp)
set(CUDA_DIR ${SRC_DIR}/cuda)
set(INCLUDE_CPP_DIR ${SRC_DIR}/include/cpp)
set(INCLUDE_CUDA_DIR ${SRC_DIR}/include/cuda)
set(LIB_DIR ${BASE_DIR}/osvmp2/lib)
set(MDJ_DIR ${CUDA_DIR}/int4cMdJCuda)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIB_DIR})
file(MAKE_DIRECTORY ${LIB_DIR})

message(STATUS "Source dir: ${SRC_DIR}")
message(STATUS "Base dir:   ${BASE_DIR}")
message(STATUS "Library output: ${LIB_DIR}")

# ========== PACKAGES ==========
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# ========== OPTIONAL CUDA DETECTION ==========
set(USE_CUDA FALSE)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(USE_CUDA TRUE)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION} - Enabling CUDA modules")
    
    # CUDA Specific Config
    set(COMMON_CUDA_FLAGS -O3 -arch=sm_75 --ptxas-options=-v)
    set(CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures (e.g., 75;80;86)")
else()
    message(STATUS "CUDA not found - Skipping CUDA modules (CPU-only build)")
endif()

# ========== AUTOMATIC BLAS DETECTION ==========
set(BLAS_FOUND FALSE)

if(DEFINED BLAS_ROOT AND EXISTS ${BLAS_ROOT})
    message(STATUS "Using user-provided BLAS_ROOT: ${BLAS_ROOT}")
    set(BLAS_INCLUDE_DIR ${BLAS_ROOT})
    set(BLAS_FOUND TRUE)
else()
    # Try CMake's FindBLAS first
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        get_filename_component(BLAS_LIB_DIR ${BLAS_LIBRARIES} DIRECTORY)
        find_path(BLAS_INCLUDE_DIR cblas.h
            HINTS 
                ${BLAS_LIB_DIR}/../include
                ${BLAS_LIB_DIR}/../../include
            PATHS
                /usr/include
                /usr/local/include
                /opt/homebrew/opt/openblas/include
                $ENV{CONDA_PREFIX}/include
                $ENV{CONDA_PREFIX}/pkgs/openblas-*/include
        )
    endif()
    
    # Search common paths
    if(NOT BLAS_INCLUDE_DIR)
        find_path(BLAS_INCLUDE_DIR cblas.h
            PATHS
                /usr/include/openblas
                /usr/local/include/openblas
                /opt/homebrew/opt/openblas/include
                /opt/openblas/include
                $ENV{HOME}/miniforge3/include
                $ENV{HOME}/miniconda3/include
                $ENV{HOME}/anaconda3/include
                $ENV{CONDA_PREFIX}/include
                $ENV{CONDA_PREFIX}/pkgs/openblas-*/include
                /usr/include/x86_64-linux-gnu
        )
    endif()
    
    # Try pkg-config
    if(NOT BLAS_INCLUDE_DIR)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENBLAS QUIET openblas)
            if(OPENBLAS_FOUND)
                set(BLAS_INCLUDE_DIR ${OPENBLAS_INCLUDE_DIRS})
            endif()
        endif()
    endif()
endif()

if(NOT BLAS_INCLUDE_DIR)
    message(FATAL_ERROR 
        "Could not find BLAS/OpenBLAS.\n"
        "Install OpenBLAS or specify: cmake -DBLAS_ROOT=/path/to/openblas/include .."
    )
endif()
message(STATUS "BLAS include: ${BLAS_INCLUDE_DIR}")

# ========== BUILD CONFIGURATION ==========
set(COMMON_CXX_FLAGS -O3 -fPIC -Wall)

# ========== C++ MODULES ==========
# These build regardless of CUDA availability
set(CPP_MODULES
    linalgUtils
    localization
    randSvd
)

foreach(module ${CPP_MODULES})
    add_library(${module} SHARED ${CPP_DIR}/${module}.cpp)
    
    target_compile_options(${module} PRIVATE ${COMMON_CXX_FLAGS})
    
    target_include_directories(${module} PRIVATE 
        ${INCLUDE_CPP_DIR}
        ${BLAS_INCLUDE_DIR}
    )
    
    # Add linalgUtils.cpp to dependent modules
    if(NOT module STREQUAL "linalgUtils")
        target_sources(${module} PRIVATE ${CPP_DIR}/linalgUtils.cpp)
    endif()
    
    target_link_libraries(${module} PRIVATE
        cblas
        lapack
        pybind11::module
    )
    
    set_target_properties(${module} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
    )
endforeach()

# ========== CUDA MODULES ==========
# Only build if CUDA was found
if(USE_CUDA)
    set(CUDA_MODULES
        osvMp2Cuda
        pbcIntCuda
        localizationCuda
        linalgUtilsCuda
        randSvdCuda
        int3c2eCuda
        int4cJKCuda
    )

    foreach(module ${CUDA_MODULES})
        add_library(${module} SHARED ${CUDA_DIR}/${module}.cu)
        
        set_target_properties(${module} PROPERTIES
            CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
            PREFIX ""
            SUFFIX ".so"
            LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
        )
        
        target_compile_options(${module} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        )
        
        target_include_directories(${module} PRIVATE
            ${INCLUDE_CUDA_DIR}
        )
        
        # Add linalgUtilsCuda.cu to dependent modules
        if(NOT module STREQUAL "linalgUtilsCuda")
            target_sources(${module} PRIVATE ${CUDA_DIR}/linalgUtilsCuda.cu)
        endif()
        
        target_link_libraries(${module} PRIVATE
            CUDA::cublas
            CUDA::cudart
            CUDA::cusolver
            CUDA::curand
            pybind11::module
        )
    endforeach()

    # ========== int4cMdJCuda (Multi-file module) ==========
    add_library(int4cMdJCuda SHARED
        ${MDJ_DIR}/md_contract_j.cu
        ${MDJ_DIR}/unrolled_md_j.cu
        ${MDJ_DIR}/unrolled_md_j_4dm.cu
        ${MDJ_DIR}/md_j_driver.cu
        ${MDJ_DIR}/md_pairdata.c
    )

    set_target_properties(int4cMdJCuda PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
    )

    target_compile_options(int4cMdJCuda PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    )

    target_include_directories(int4cMdJCuda PRIVATE
        ${MDJ_DIR}
        ${INCLUDE_CUDA_DIR}
    )

    target_link_libraries(int4cMdJCuda PRIVATE
        CUDA::cublas
        CUDA::cudart
        CUDA::cusolver
        CUDA::curand
        pybind11::module
    )
endif()