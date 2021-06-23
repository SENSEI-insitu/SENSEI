#
# Build the documentation
#
find_package(Doxygen REQUIRED)
find_package(Perl REQUIRED)
find_package(HTMLHelp)

option(DOXYGEN_SHORT_NAMES "Build Doxygen of SVTK using short file names" OFF)
option(DOXYGEN_KEEP_TEMP "Do not delete the temporary Doxygen headers" OFF)
option(DOXYGEN_GENERATE_HTMLHELP "Generate HTML help (CHM) files" ON)
mark_as_advanced(DOXYGEN_SHORT_NAMES DOXYGEN_KEEP_TEMP DOXYGEN_GENERATE_HTMLHELP)
if(DOXYGEN_SHORT_NAMES)
  set(DOXYGEN_SHORT_NAMES YES)
else()
  set(DOXYGEN_SHORT_NAMES NO)
endif()
if(DOXYGEN_GENERATE_HTMLHELP)
  set(DOXYGEN_GENERATE_HTMLHELP YES)
else()
  set(DOXYGEN_GENERATE_HTMLHELP NO)
endif()

#
# Configure the script and the doxyfile, then add target
#
set(HAVE_DOT_YESNO NO)
if(DOT)
  set(HAVE_DOT_YESNO YES)
  if(NOT DOT_PATH)
    get_filename_component(DOT_PATH ${DOT} PATH)
  endif()
endif()

# Build up a list of all module source directories. Note that this should be
# all source directories and so does not use the normal variables.
set(SVTK_MODULE_DIRS_DOXYGEN "")
foreach(module IN LISTS svtk_module_files)
  file(RELATIVE_PATH module_dir "${SVTK_SOURCE_DIR}" "${module}")
  get_filename_component(module_BASE ${module_dir} PATH)
  if(module_BASE MATCHES "^Utilities" OR module_BASE MATCHES "^ThirdParty")
    # Skip the utilities and third parties directories.
    continue()
  endif ()
  # Use both the source and binary directories, this ensures that
  # generated files will be included, but they may then be architecture,
  # build configuration and/or compiler specific. All source directories
  # are included whether the module is enabled or not.
  string(APPEND SVTK_MODULE_DIRS_DOXYGEN
    "\"${SVTK_SOURCE_DIR}/${module_BASE}\"
\"${SVTK_BINARY_DIR}/${module_BASE}\"
")
endforeach()

configure_file(
  ${SVTK_SOURCE_DIR}/Utilities/Doxygen/doxyfile.in
  ${SVTK_BINARY_DIR}/Utilities/Doxygen/doxyfile)

configure_file(
  ${SVTK_SOURCE_DIR}/Utilities/Doxygen/doc_makeall.cmake.in
  ${SVTK_BINARY_DIR}/Utilities/Doxygen/doc_makeall.cmake
  @ONLY)

add_custom_target(DoxygenDoc
  ${CMAKE_COMMAND}
  -P ${SVTK_BINARY_DIR}/Utilities/Doxygen/doc_makeall.cmake
  DEPENDS ${SVTK_BINARY_DIR}/Utilities/Doxygen/doc_makeall.cmake)

if(NOT SVTK_INSTALL_NO_DOCUMENTATION)
  macro(__svtk_install_documentation_files glob)
    file(GLOB __files "${CMAKE_CURRENT_SOURCE_DIR}/${glob}")
    install(FILES ${__files}
      DESTINATION ${SVTK_INSTALL_DOXYGEN_DIR}
      COMPONENT Development)
  endmacro()
  __svtk_install_documentation_files("*.css")
  __svtk_install_documentation_files("*.gif")
  __svtk_install_documentation_files("*.html")
  __svtk_install_documentation_files("*.pl")
  __svtk_install_documentation_files("*.stop")
  install(FILES doc_readme.txt
    DESTINATION ${SVTK_INSTALL_DOXYGEN_DIR}
    COMPONENT Development)
  install(DIRECTORY ${SVTK_BINARY_DIR}/Utilities/Doxygen/doc/html
      DESTINATION ${SVTK_INSTALL_DOXYGEN_DIR}
      COMPONENT Development)
endif()
