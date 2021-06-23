if (NOT (DEFINED svtk_cmake_dir AND
         DEFINED svtk_cmake_build_dir AND
         DEFINED svtk_cmake_destination AND
         DEFINED svtk_modules))
  message(FATAL_ERROR
    "svtkInstallCMakePackageCompileTools is missing input variables.")
endif ()

configure_file(
  "${svtk_cmake_dir}/svtkcompiletools-config.cmake.in"
  "${svtk_cmake_build_dir}/svtkcompiletools-config.cmake"
  @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${svtk_cmake_build_dir}/svtkcompiletools-config-version.cmake"
  VERSION "${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}.${SVTK_BUILD_VERSION}"
  COMPATIBILITY AnyNewerVersion)

# For convenience, a package is written to the top of the build tree. At some
# point, this should probably be deprecated and warn when it is used.
file(GENERATE
  OUTPUT  "${CMAKE_BINARY_DIR}/svtkcompiletools-config.cmake"
  CONTENT "include(\"${svtk_cmake_build_dir}/svtkcompiletools-config.cmake\")\n")
configure_file(
  "${svtk_cmake_build_dir}/svtkcompiletools-config-version.cmake"
  "${CMAKE_BINARY_DIR}/svtkcompiletools-config-version.cmake"
  COPYONLY)

install(
  FILES       "${svtk_cmake_build_dir}/svtkcompiletools-config.cmake"
              "${svtk_cmake_build_dir}/svtkcompiletools-config-version.cmake"
  DESTINATION "${svtk_cmake_destination}"
  COMPONENT   "development")
