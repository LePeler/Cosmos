cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(cosmos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# find pybind11
find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

# library
add_library(cosmos INTERFACE)
target_include_directories(cosmos INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(cosmos PROPERTIES POSITION_INDEPENDENT_CODE ON)

# python module
pybind11_add_module(cosmos-python
    pybindings/solvers.cxx
    pybindings/samplers.cxx
)
target_link_libraries(cosmos-python PRIVATE cosmos)


# copy .so to cosmos folder for Python access
set(target_output_dir "${CMAKE_CURRENT_BINARY_DIR}/cosmos-python")
file(MAKE_DIRECTORY ${target_output_dir})
add_custom_command(TARGET cosmos-python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cosmos-python> ${target_output_dir}/__init__.so
)

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


