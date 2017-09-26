find_package(MPI REQUIRED)
add_library(mpi INTERFACE)

target_include_directories(mpi SYSTEM INTERFACE ${MPI_C_INCLUDE_PATH}
  ${MPI_CXX_INCLUDE_PATH})

target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

install(TARGETS mpi EXPORT mpi)
install(EXPORT mpi DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
