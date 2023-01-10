configure_file(senseiConfig.h.in senseiConfig.h)

install(FILES ${CMAKE_BINARY_DIR}/senseiConfig.h DESTINATION include)

set(SENSEI_BUILD OFF)

configure_file(CMake/SENSEIConfig.cmake.in
  ${sensei_CMAKE_INSTALL_CMAKEDIR}/SENSEIConfig.cmake @ONLY)

configure_file(CMake/SENSEIConfigVersion.cmake.in
  ${sensei_CMAKE_INSTALL_CMAKEDIR}/SENSEIConfigVersion.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/${sensei_CMAKE_INSTALL_CMAKEDIR}/SENSEIConfig.cmake
  ${CMAKE_BINARY_DIR}/${sensei_CMAKE_INSTALL_CMAKEDIR}/SENSEIConfigVersion.cmake
  DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR})
