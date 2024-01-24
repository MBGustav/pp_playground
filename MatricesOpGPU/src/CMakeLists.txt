
set(LIB_NAME cudaMatrix)
# Source files
file(GLOB SOURCES 
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp" 
    "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)


# Set CUDA architecture flags
# set_property(TARGET ${LIB_NAME} 
# PROPERTY CUDA_ARCHITECTURES 35 50 60 70 75
# )

# Create library
add_library(${LIB_NAME} ${SOURCES})