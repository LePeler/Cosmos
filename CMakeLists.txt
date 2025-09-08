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


# executable to simulate LCDM
add_executable(LCDM
    models/LCDM.cxx
)
target_link_libraries(LCDM PRIVATE cosmos)

# executable to simulate quintessence
add_executable(quintessence
    models/quintessence.cxx
)
target_link_libraries(quintessence PRIVATE cosmos)


# test executable
add_executable(cosmos_test
    test.cxx
)
target_link_libraries(cosmos_test PRIVATE cosmos)
target_link_libraries(cosmos_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(cosmos_test PRIVATE MPI::MPI_CXX)
target_link_libraries(cosmos_test PRIVATE BLAS::BLAS)
target_link_libraries(cosmos_test PRIVATE lapacke)
target_compile_definitions(cosmos_test PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)

# MCMC executable
add_executable(run_MCMC
    models/run_MCMC.cxx
)
target_link_libraries(run_MCMC PRIVATE cosmos)
target_link_libraries(run_MCMC PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(run_MCMC PRIVATE MPI::MPI_CXX)
target_link_libraries(run_MCMC PRIVATE BLAS::BLAS)
target_link_libraries(run_MCMC PRIVATE lapacke)
target_compile_definitions(run_MCMC PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
