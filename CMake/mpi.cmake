find_package(MPI)
if (NOT MPI_C_FOUND)
  message(FETAL_ERROR "Failed to locate MPI C libraries and headers")
endif()

add_library(mpi INTERFACE)

target_include_directories(mpi SYSTEM INTERFACE
  ${MPI_C_INCLUDE_PATH} ${MPI_C_INCLUDE_DIRS})

target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES})

install(TARGETS mpi EXPORT mpi)
install(EXPORT mpi DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
