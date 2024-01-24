cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(matrixCalc)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA and openMP
find_package(CUDA REQUIRED)
find_package (OpenMP)


# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


# Include directories
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# Include source files cuda
add_subdirectory(src)

add_executable(testing main.cpp "src/kernel.cu")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70") 
set_target_properties(testing PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_ARCHITECTURES "sm_40"
)

# Example for creating an executable that uses the library
target_link_libraries(testing 
        PUBLIC cudaMatrix 
                ${CUDA_LIBRARIES}
)

target_compile_options(testing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)