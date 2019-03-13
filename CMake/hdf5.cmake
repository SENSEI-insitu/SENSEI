if(ENABLE_HDF5 STREQUAL AUTO)
  find_package(HDF5 COMPONENTS C)
elseif(ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()

message(STATUS "HDF5 found? ${HDF5_FOUND} ${HDF5_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR}")

if(HDF5_FOUND)
  add_library(sDataElevator INTERFACE)
  target_link_libraries(sDataElevator INTERFACE ${HDF5_LIBRARIES})
  target_include_directories(sDataElevator SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})

  install(TARGETS sDataElevator EXPORT sDataElevator)
  install(EXPORT sDataElevator DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()






