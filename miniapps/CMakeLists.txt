if(ENABLE_PARALLEL3D)
  message(STATUS "Enabled: Parallel3D miniapp.")
  add_subdirectory(parallel3d)
else()
  message(STATUS "Disabled: Parallel3D miniapp.")
endif()

if(ENABLE_OSCILLATORS)
  message(STATUS "Enabled: Oscillators miniapp.")
  add_subdirectory(oscillators)
else()
  message(STATUS "Disabled: Oscillators miniapp.")
endif()

