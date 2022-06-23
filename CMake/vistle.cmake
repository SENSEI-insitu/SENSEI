if(ENABLE_VISTLE)
  find_package(Vistle REQUIRED COMPONENTS sensei sensei_vtk boost_mpi)
  add_library(sVistle INTERFACE)
  target_link_libraries(sVistle
    INTERFACE
      Vistle::vistle_sensei
      Vistle::vistle_sensei_vtk
  )
  get_target_property(vistle_boost_mpi_includes Vistle::vistle_boost_mpi INTERFACE_INCLUDE_DIRECTORIES) 
  target_include_directories(sVistle INTERFACE ${vistle_boost_mpi_includes})      
  target_compile_definitions(sVistle INTERFACE VISTLE_ROOT="${Vistle_DIR}/../../..")
  install(TARGETS sVistle EXPORT sVistle)
  install(EXPORT sVistle DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake EXPORT_LINK_INTERFACE_LIBRARIES)

  add_library(sVTK INTERFACE)

    # use the VTK found by Vistle
    message("VTK_LIBRARIES=" ${VTK_LIBRARIES})
    target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})

  install(TARGETS sVTK EXPORT sVTK)
  install(EXPORT sVTK DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)
  set(ENABLE_VTK_CORE TRUE)
endif()
