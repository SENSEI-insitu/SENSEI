configure_file(senseiConfig.h.in senseiConfig.h)
install(FILES ${CMAKE_BINARY_DIR}/senseiConfig.h DESTINATION include)

set(SENSEI_BUILD OFF)
configure_file(CMake/SENSEIConfig.cmake.in lib/cmake/SENSEIConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/SENSEIConfig.cmake DESTINATION lib/cmake)
