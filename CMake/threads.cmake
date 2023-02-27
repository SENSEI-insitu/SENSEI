find_package(Threads)
# IMPORTED_GLOBAL is needed for consistency with SVTK
set_property(TARGET Threads::Threads PROPERTY IMPORTED_GLOBAL TRUE)
add_library(thread INTERFACE)
target_link_libraries(thread INTERFACE ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS thread EXPORT thread)
install(EXPORT thread DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
  EXPORT_LINK_INTERFACE_LIBRARIES)
