configure_file(senseiConfig.h.in senseiConfig.h)

install(FILES ${CMAKE_BINARY_DIR}/senseiConfig.h DESTINATION include)

set(SENSEI_BUILD OFF)

configure_file(CMake/SENSEIConfig.cmake.in
  ${CMAKE_INSTALL_LIBDIR}/cmake/SENSEIConfig.cmake @ONLY)

configure_file(CMake/SENSEIConfigVersion.cmake.in
  ${CMAKE_INSTALL_LIBDIR}/cmake/SENSEIConfigVersion.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/SENSEIConfig.cmake
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/SENSEIConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
