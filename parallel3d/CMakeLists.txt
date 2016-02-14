#------------------------------------------------------------------------------
set (sources
  parallel_3D.c)

#------------------------------------------------------------------------------
add_executable(3D_Grid ${sources})
target_link_libraries(3D_Grid PRIVATE histogram)
target_compile_definitions(3D_Grid PRIVATE _FILE_OFFSET_BITS=64 _LARGEFILE64_SOURCE)

include(mpi)
mpi_link(3D_Grid)

if(ENABLE_SENSEI)
  target_link_libraries(3D_Grid PRIVATE core)
endif()
