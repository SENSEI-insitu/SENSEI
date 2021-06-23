add_subdirectory(Cxx)

if (SVTK_WRAP_PYTHON)
  svtk_module_test_data(
    Data/earth.ppm
    Data/headsq/,REGEX:.*
    Data/masonry.bmp)

  add_subdirectory(Python)
endif ()
