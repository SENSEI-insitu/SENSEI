if (ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)

  if(NOT HDF5_IS_PARALLEL)
    message(SEND_ERROR "Failed to locate parallel hdf5 installation")
  endif()

  add_library(sHDF5 INTERFACE)
  target_link_libraries(sHDF5 INTERFACE ${HDF5_LIBRARIES})
  target_include_directories(sHDF5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})

  install(TARGETS sHDF5 EXPORT sHDF5)
  install(EXPORT sHDF5 DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()
