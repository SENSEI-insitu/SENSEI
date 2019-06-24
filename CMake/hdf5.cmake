if(ENABLE_HDF5 STREQUAL AUTO)
  find_package(HDF5 COMPONENTS C)
elseif(ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()

message(STATUS "check: HDF5 found? ${HDF5_FOUND} ${HDF5_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR}")
message(STATUS "check: HDF5 is parallel? ${HDF5_IS_PARALLEL}")

if(HDF5_FOUND AND HDF5_IS_PARALLEL)
  add_library(sHDF5 INTERFACE)
  target_link_libraries(sHDF5 INTERFACE ${HDF5_LIBRARIES})
  target_include_directories(sHDF5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})

  install(TARGETS sHDF5 EXPORT sHDF5)
  install(EXPORT sHDF5 DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
 else()
          message(SEND_ERROR "Failed to locate parallel hdf5 installation")	
endif()






