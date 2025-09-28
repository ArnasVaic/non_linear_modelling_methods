cmake_minimum_required(VERSION 3.15)
project(SchrodingerSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add spdlog submodule
add_subdirectory(extern/spdlog)

# Try to find OpenBLAS
find_package(OpenBLAS REQUIRED)

# Try to find LAPACKE (C interface to LAPACK)
find_library(LAPACKE_LIB NAMES lapacke REQUIRED)
find_path(LAPACKE_INCLUDE NAMES lapacke.h REQUIRED)

# Create executable
add_executable(main src/main.cpp)

# Include directories
target_include_directories(main PRIVATE ${OpenBLAS_INCLUDE_DIR} ${LAPACKE_INCLUDE})

# Link libraries
target_link_libraries(main PRIVATE spdlog::spdlog ${OpenBLAS_LIB} ${LAPACKE_LIB} m)
