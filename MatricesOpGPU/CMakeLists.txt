# CMakeLists.txt

cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")
option(USE_CUDA "Use CUDA" ON)

# export PATH=/path/to/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH

project(matrixCalc CUDA CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/arq)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_CUDA_ARCHITECTURES "sm_30")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA and OpenMP
find_package(CUDA REQUIRED)
find_package(OpenMP)

# Include directories
include_directories(include)

# Add subdirectory for library
add_subdirectory(src)
link_directories(src)


# Create executable
add_executable(testing main.cu)


# Set properties for CUDA separable compilation
set_target_properties(testing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_ARCHITECTURES "sm_40"
)

# Link libraries
target_link_libraries(testing cudaMatrixLib)

# set_target_properties(testing PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
# Set compile options for CUDA
# target_compile_options(testing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
