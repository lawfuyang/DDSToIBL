cmake_minimum_required(VERSION 3.18)
project(DDSToIBL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Speed up MSVC builds by enabling multi-processor compilation.
if(MSVC)
    add_compile_options(/MP)
endif()

# Download stb_image.h if not present
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/stb_image.h")
    message(STATUS "Downloading stb_image.h...")
    file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" "${CMAKE_SOURCE_DIR}/stb_image.h" SHOW_PROGRESS)
endif()

# Configure output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Disable DirectXTex options for minimal DDS read/write functionality
set(BUILD_TOOLS OFF CACHE BOOL "Build tex command-line tools" FORCE)
set(BUILD_SAMPLE OFF CACHE BOOL "Build DDSView sample" FORCE)
set(BUILD_DX11 OFF CACHE BOOL "Build with DirectX11 Runtime support" FORCE)
set(BUILD_DX12 OFF CACHE BOOL "Build with DirectX12 Runtime support" FORCE)
set(BC_USE_OPENMP ON CACHE BOOL "Build with OpenMP support" FORCE)
set(ENABLE_OPENEXR_SUPPORT OFF CACHE BOOL "Build with OpenEXR support" FORCE)
set(ENABLE_LIBJPEG_SUPPORT OFF CACHE BOOL "Build with libjpeg support" FORCE)
set(ENABLE_LIBPNG_SUPPORT OFF CACHE BOOL "Build with libpng support" FORCE)

add_subdirectory(DirectXTex)

# Legacy find
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Bypass VS version check for NVCC
list(APPEND CUDA_NVCC_FLAGS "-allow-unsupported-compiler")

# Ensure CUDA uses the same Runtime Library (MD/MDd) as the rest of the project
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "/MD$<$<CONFIG:Debug>:d>")

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_75,code=sm_75")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_80,code=sm_80")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_87,code=sm_87")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_90,code=sm_90")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_100,code=sm_100")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_120,code=sm_120")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_120,code=compute_120")

cuda_add_executable(${PROJECT_NAME} main.cpp bake_ibl.cu)

target_link_libraries(${PROJECT_NAME} DirectXTex ${CUDA_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE DirectXTex/DirectXTex)

# Fix LNK4098 warning by ignoring LIBCMT
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()
