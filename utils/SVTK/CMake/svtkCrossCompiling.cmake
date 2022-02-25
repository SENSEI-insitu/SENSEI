if(CMAKE_CROSSCOMPILING
    AND NOT SVTKCompileTools_FOUND
    AND NOT DEFINED CMAKE_CROSSCOMPILING_EMULATOR
    AND SVTK_ENABLE_WRAPPING)
  # if CMAKE_CROSSCOMPILING is true and crosscompiling emulator is not available, we need
  # to import build-tools targets.
  find_package(SVTKCompileTools REQUIRED)
endif()
