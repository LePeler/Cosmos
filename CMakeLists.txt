cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(cosmos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# find Eigen
find_package(Eigen3 REQUIRED)
# find OpenMP
find_package(OpenMP REQUIRED)
# find MPI
find_package(MPI REQUIRED)
# find BLAS
find_package(BLAS REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

# library
add_library(cosmos INTERFACE)
target_include_directories(cosmos INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cosmos INTERFACE ${EIGEN3_INCLUDE_DIR})
set_target_properties(cosmos PROPERTIES POSITION_INDEPENDENT_CODE ON)

# function to add an executable,
# automatically link all needed libraries
# and set the appropriate compile definitions
function(add_cosmos_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        cosmos
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        BLAS::BLAS
        lapacke
    )
    target_compile_definitions(${name} PRIVATE
        EIGEN_USE_BLAS
        EIGEN_USE_LAPACKE
    )
endfunction()


# test executable
add_cosmos_executable(cosmos_test
    test.cxx
)
# LCDM MCMC executable
add_cosmos_executable(run_LCDM
    mcmc/LCDM.cxx
)
# f1 MCMC executable
add_cosmos_executable(run_f1
    mcmc/f1.cxx
)
# f2 MCMC executable
add_cosmos_executable(run_f2
    mcmc/f2.cxx
)
# SFT MCMC executable
add_cosmos_executable(run_SFT
    mcmc/SFT.cxx
)
