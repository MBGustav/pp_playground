# src/CMakeLists.txt

# Include directories
enable_language("CUDA")
set(library_name cudaMatrixLib)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    "-O3"
)

set(lib_sources 
    "${PROJECT_SOURCE_DIR}/src/cudaMatrix.cu"
    "${PROJECT_SOURCE_DIR}/src/cudaLinalg.cu"
)

include_directories(
    kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Create cudaMatrixLib
add_library(${library_name} ${lib_sources})

# Join with source
target_sources(${library_name} PRIVATE ${lib_sources})
