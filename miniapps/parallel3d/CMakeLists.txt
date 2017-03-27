set(sources parallel_3D.c histogram.cpp)
set(libs m mpi timer)

if (ENABLE_SENSEI)
  list(APPEND sources DataAdaptor.cxx Bridge.cxx)
  list(APPEND libs sensei)
endif()

add_executable(3D_Grid ${sources})
target_compile_definitions(3D_Grid PRIVATE _FILE_OFFSET_BITS=64 _LARGEFILE64_SOURCE)
target_link_libraries(3D_Grid PRIVATE ${libs})
