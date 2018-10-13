add_library(timer STATIC MemoryProfiler.cxx Timer.cxx)

target_include_directories(timer SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/timer>)

target_link_libraries(timer sVTK sMPI)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION include/timer FILES_MATCHING PATTERN "*.h")

install(TARGETS timer EXPORT timer
  INCLUDES DESTINATION include/timer ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(EXPORT timer DESTINATION lib/cmake
  EXPORT_LINK_INTERFACE_LIBRARIES)
