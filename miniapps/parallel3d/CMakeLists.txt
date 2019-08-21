set(sources parallel3d.c histogram.cpp)
set(libs m sMPI)

if (ENABLE_SENSEI)
  list(APPEND sources DataAdaptor.cxx Bridge.cxx)
  list(APPEND libs sensei)
endif()

add_executable(parallel3d ${sources})
target_link_libraries(parallel3d PRIVATE ${libs})

target_compile_definitions(parallel3d PRIVATE
  _FILE_OFFSET_BITS=64 _LARGEFILE64_SOURCE)

install(TARGETS parallel3d
  RUNTIME DESTINATION bin)
