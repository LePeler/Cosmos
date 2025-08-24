cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(cosmos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# find Eigen
find_package(Eigen3 REQUIRED)

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


