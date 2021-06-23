# Configure files with settings for use by the build.
option(SVTK_ENABLE_WRAPPING "Whether wrapping is available or not" ON)
mark_as_advanced(SVTK_ENABLE_WRAPPING)

# Add the option for build the Python wrapping to SVTK.
include(CMakeDependentOption)
cmake_dependent_option(SVTK_WRAP_PYTHON "Should SVTK Python wrapping be built?" OFF
  "SVTK_ENABLE_WRAPPING" OFF)
set(SVTK_PYTHON_VERSION 2 CACHE STRING
  "Python version to use")
set_property(CACHE SVTK_PYTHON_VERSION
  PROPERTY
    STRINGS "2;3")

# Force reset of hints file location in cache if it was moved
if(SVTK_WRAP_HINTS AND NOT EXISTS ${SVTK_WRAP_HINTS})
  unset(SVTK_WRAP_HINTS CACHE)
  unset(SVTK_WRAP_HINTS)
endif()

if(BUILD_TESTING OR SVTK_WRAP_PYTHON)
  # SVTK only supports a single Python version at a time, so make artifact
  # finding interactive.
  set("Python${SVTK_PYTHON_VERSION}_ARTIFACTS_INTERACTIVE" ON)
  # Need PYTHON_EXECUTABLE for HeaderTesting or python wrapping
  find_package("Python${SVTK_PYTHON_VERSION}" QUIET COMPONENTS Interpreter)
endif()

if(SVTK_WRAP_PYTHON)
  set(SVTK_WRAP_PYTHON_EXE SVTK::WrapPython)
  set(SVTK_WRAP_PYTHON_INIT_EXE SVTK::WrapPythonInit)
endif()

cmake_dependent_option(SVTK_USE_TK "Build SVTK with Tk support" OFF
  "SVTK_WRAP_PYTHON" OFF)

cmake_dependent_option(SVTK_WRAP_JAVA "Should SVTK Java wrapping be built?" OFF
  "SVTK_ENABLE_WRAPPING;NOT CMAKE_VERSION VERSION_LESS \"3.12\"" OFF)
if(SVTK_WRAP_JAVA)
  set(SVTK_WRAP_JAVA3_INIT_DIR "${SVTK_SOURCE_DIR}/Wrapping/Java")
  # Wrapping executables.
  set(SVTK_WRAP_JAVA_EXE  SVTK::WrapJava)
  set(SVTK_PARSE_JAVA_EXE SVTK::ParseJava)

  # Java package location.
  set(SVTK_JAVA_JAR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/svtk.jar")
  set(SVTK_JAVA_HOME "${SVTK_BINARY_DIR}/java/svtk")
  file(MAKE_DIRECTORY "${SVTK_JAVA_HOME}")
endif()
