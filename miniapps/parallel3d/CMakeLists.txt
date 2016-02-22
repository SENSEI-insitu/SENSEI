#------------------------------------------------------------------------------
set(sources
  parallel_3D.c
  histogram.cpp)

if(ENABLE_SENSEI)
  list(APPEND sources
    DataAdaptor.h
    DataAdaptor.cxx
    Bridge.h
    Bridge.cxx)
endif()

#------------------------------------------------------------------------------
add_executable(3D_Grid ${sources})
target_compile_definitions(3D_Grid PRIVATE _FILE_OFFSET_BITS=64 _LARGEFILE64_SOURCE)
target_link_libraries(3D_Grid PRIVATE m mpi)

if(ENABLE_SENSEI)
  target_link_libraries(3D_Grid PRIVATE core configurable)
endif()
