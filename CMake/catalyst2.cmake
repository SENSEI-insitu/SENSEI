if (ENABLE_CATALYST2)
  if (${ENABLE_CONDUIT})
    message(FATAL_ERROR "Catalyst2 bring its own conduit, please set ENABLE_CONDUIT to OFF")
  endif()

  find_package(VTK CONFIG COMPONENTS  CommonCore CommonExecutionModel IOCore)
  find_package(catalyst REQUIRED)

  add_library(sCatalyst2 INTERFACE)
  target_link_libraries(sCatalyst2 INTERFACE catalyst::catalyst)
  target_link_libraries(sCatalyst2 INTERFACE ${VTK_LIBRARIES})

  install(TARGETS sCatalyst2 EXPORT sCatalyst2)
  install(EXPORT sCatalyst2 DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()
