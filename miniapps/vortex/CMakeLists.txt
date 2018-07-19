set(sources vortex.cpp simulation_data.cpp patch.cpp)
set(libs m mpi timer)

if (ENABLE_SENSEI)
  list(APPEND sources VortexDataAdaptor.cpp)
  list(APPEND libs sensei)
endif()

add_executable(vortex ${sources})
target_link_libraries(vortex PRIVATE ${libs})
