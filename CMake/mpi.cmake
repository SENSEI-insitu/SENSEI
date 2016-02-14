#------------------------------------------------------------------------------
# Since we need MPI left and right, this makes it easier to deal with MPI.
find_package(MPI REQUIRED)
function(mpi_link target)
  target_include_directories(${target}
    SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
  target_compile_definitions(${target}
    PRIVATE ${MPI_C_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  target_link_libraries(${target}
    PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  if(MPI_C_LINK_FLAGS OR MPI_CXX_LINK_FLAGS)
    set_target_properties(${target}
      PROPERTIES LINK_FLAGS ${MPI_C_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})
  endif()
endfunction()
