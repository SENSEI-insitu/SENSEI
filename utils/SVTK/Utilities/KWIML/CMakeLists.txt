if (BUILD_TESTING)
  set(KWIML_TEST_ENABLE 1)
  set(KWIML_TEST_PREFIX svtkkwiml)
endif ()

if (_svtk_build_INSTALL_HEADERS)
  set(KWIML_INSTALL_INCLUDE_DIR "${_svtk_build_HEADERS_DESTINATION}")
  set(KWIML_INCLUDE_PREFIX "svtkkwiml")
  set(KWIML_INSTALL_INCLUDE_OPTIONS
    COMPONENT "${_svtk_build_HEADERS_COMPONENT}")
endif ()

svtk_module_third_party_internal(
  HEADER_ONLY
  LICENSE_FILES "svtkkwiml/Copyright.txt"
  VERSION       "1.0.0")

svtk_module_install_headers(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/svtk_kwiml.h")

