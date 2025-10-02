cmake_minimum_required(VERSION 3.15)
project(SchrodingerSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add spdlog submodule
add_subdirectory(extern/spdlog)

# Find LAPACKE
find_library(LAPACKE_LIB NAMES lapacke REQUIRED)
find_path(LAPACKE_INCLUDE NAMES lapacke.h REQUIRED)

# Find OpenBLAS
find_library(OPENBLAS_LIB NAMES openblas REQUIRED)
find_path(OPENBLAS_INCLUDE NAMES cblas.h REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create executable
add_executable(main src/main.cpp)

# Include directories
target_include_directories(main PRIVATE 
    ${LAPACKE_INCLUDE}
    ${OPENBLAS_INCLUDE}
)

# Link libraries (note order!)
target_link_libraries(main PRIVATE
    spdlog::spdlog
    ${LAPACKE_LIB}
    m
    pthread
    ${OPENBLAS_LIB}
    OpenMP::OpenMP_CXX
)
