cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0022 NEW)
project(ArrayIO)

find_package(MPI REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
add_library(ArrayIO STATIC ArrayIO.cxx)
target_link_libraries(ArrayIO ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_include_directories(ArrayIO
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

install(TARGETS ArrayIO
    EXPORT ArrayIO
    INCLUDES DESTINATION include
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

install(EXPORT ArrayIO
    DESTINATION lib
    FILE ArrayIO.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
