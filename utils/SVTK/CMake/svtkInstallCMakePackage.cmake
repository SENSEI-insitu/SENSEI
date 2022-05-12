if (NOT (DEFINED svtk_cmake_dir AND
         DEFINED svtk_cmake_build_dir AND
         DEFINED svtk_cmake_destination AND
         DEFINED svtk_modules))
  message(FATAL_ERROR
    "svtkInstallCMakePackage is missing input variables.")
endif ()

set(svtk_all_components)
foreach (svtk_module IN LISTS svtk_modules)
  string(REPLACE "SVTK::" "" svtk_component "${svtk_module}")
  list(APPEND svtk_all_components
    "${svtk_component}")
endforeach ()

if (TARGET "SVTK::svtkm")
  set(svtk_has_svtkm ON)
else ()
  set(svtk_has_svtkm OFF)
endif ()

_svtk_module_write_import_prefix("${svtk_cmake_build_dir}/svtk-prefix.cmake" "${svtk_cmake_destination}")

set(svtk_python_version "")
if (SVTK_WRAP_PYTHON)
  set(svtk_python_version "${SVTK_PYTHON_VERSION}")
endif ()

configure_file(
  "${svtk_cmake_dir}/svtk-config.cmake.in"
  "${svtk_cmake_build_dir}/svtk-config.cmake"
  @ONLY)

configure_file(
  "${svtk_cmake_dir}/svtk-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/svtk-config.cmake"
  @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${svtk_cmake_build_dir}/svtk-config-version.cmake"
  VERSION "${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}.${SVTK_BUILD_VERSION}"
  COMPATIBILITY AnyNewerVersion)

# For convenience, a package is written to the top of the build tree. At some
# point, this should probably be deprecated and warn when it is used.
file(GENERATE
  OUTPUT  "${CMAKE_BINARY_DIR}/svtk-config.cmake"
  CONTENT "include(\"${svtk_cmake_build_dir}/svtk-config.cmake\")\n")
configure_file(
  "${svtk_cmake_build_dir}/svtk-config-version.cmake"
  "${CMAKE_BINARY_DIR}/svtk-config-version.cmake"
  COPYONLY)

set(svtk_cmake_module_files
  Finddouble-conversion.cmake
  FindEigen3.cmake
  FindEXPAT.cmake
  FindFFMPEG.cmake
  FindFontConfig.cmake
  FindFreetype.cmake
  FindGL2PS.cmake
  FindGLEW.cmake
  FindJOGL.cmake
  FindJsonCpp.cmake
  FindLibHaru.cmake
  FindLibPROJ.cmake
  FindLibXml2.cmake
  FindLZ4.cmake
  FindLZMA.cmake
  Findmpi4py.cmake
  FindMySQL.cmake
  FindNetCDF.cmake
  FindODBC.cmake
  FindOGG.cmake
  FindOpenMP.cmake
  FindOpenSlide.cmake
  FindOpenVR.cmake
  FindOSMesa.cmake
  FindPEGTL.cmake
  FindTBB.cmake
  FindTHEORA.cmake
  Findutf8cpp.cmake

  svtkCMakeBackports.cmake
  svtkDetectLibraryType.cmake
  svtkEncodeString.cmake
  svtkHashSource.cmake
  svtkModule.cmake
  svtkModuleGraphviz.cmake
  svtkModuleJson.cmake
  svtkModuleTesting.cmake
  svtkModuleWrapJava.cmake
  svtkModuleWrapPython.cmake
  svtkObjectFactory.cmake
  svtkObjectFactory.cxx.in
  svtkObjectFactory.h.in
  svtkTestingDriver.cmake
  svtkTestingRenderingDriver.cmake
  svtkTopologicalSort.cmake
  svtk-use-file-compat.cmake
  svtk-use-file-deprecated.cmake
  svtk-use-file-error.cmake)
set(svtk_cmake_patch_files
  patches/3.13/FindZLIB.cmake
  patches/3.16/FindMPI/fortranparam_mpi.f90.in
  patches/3.16/FindMPI/libver_mpi.c
  patches/3.16/FindMPI/libver_mpi.f90.in
  patches/3.16/FindMPI/mpiver.f90.in
  patches/3.16/FindMPI/test_mpi.c
  patches/3.16/FindMPI/test_mpi.f90.in
  patches/3.16/FindMPI.cmake
  patches/3.16/FindPostgreSQL.cmake
  patches/3.18/FindPython/Support.cmake
  patches/3.18/FindPython2.cmake
  patches/3.18/FindPython3.cmake
  patches/99/FindGDAL.cmake
  patches/99/FindHDF5.cmake
  patches/99/FindJPEG.cmake
  patches/99/FindLibArchive.cmake
  patches/99/FindOpenGL.cmake
  patches/99/FindSQLite3.cmake
  patches/99/FindX11.cmake)

set(svtk_cmake_files_to_install)
foreach (svtk_cmake_module_file IN LISTS svtk_cmake_module_files svtk_cmake_patch_files)
  configure_file(
    "${svtk_cmake_dir}/${svtk_cmake_module_file}"
    "${svtk_cmake_build_dir}/${svtk_cmake_module_file}"
    COPYONLY)
  list(APPEND svtk_cmake_files_to_install
    "${svtk_cmake_module_file}")
endforeach ()

include(svtkInstallCMakePackageHelpers)

if (NOT DEFINED SVTK_RELOCATABLE_INSTALL)
  option(SVTK_RELOCATABLE_INSTALL "Do not embed hard-coded paths into the install" ON)
  mark_as_advanced(SVTK_RELOCATABLE_INSTALL)
endif ()
if (NOT SVTK_RELOCATABLE_INSTALL)
  list(APPEND svtk_cmake_files_to_install
    "${svtk_cmake_build_dir}/svtk-find-package-helpers.cmake")
endif ()

foreach (svtk_cmake_file IN LISTS svtk_cmake_files_to_install)
  if (IS_ABSOLUTE "${svtk_cmake_file}")
    file(RELATIVE_PATH svtk_cmake_subdir_root "${svtk_cmake_build_dir}" "${svtk_cmake_file}")
    get_filename_component(svtk_cmake_subdir "${svtk_cmake_subdir_root}" DIRECTORY)
    set(svtk_cmake_original_file "${svtk_cmake_file}")
  else ()
    get_filename_component(svtk_cmake_subdir "${svtk_cmake_file}" DIRECTORY)
    set(svtk_cmake_original_file "${svtk_cmake_dir}/${svtk_cmake_file}")
  endif ()
  install(
    FILES       "${svtk_cmake_original_file}"
    DESTINATION "${svtk_cmake_destination}/${svtk_cmake_subdir}"
    COMPONENT   "development")
endforeach ()

install(
  FILES       "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/svtk-config.cmake"
              "${svtk_cmake_build_dir}/svtk-config-version.cmake"
              "${svtk_cmake_build_dir}/svtk-prefix.cmake"
  DESTINATION "${svtk_cmake_destination}"
  COMPONENT   "development")

svtk_module_export_find_packages(
  CMAKE_DESTINATION "${svtk_cmake_destination}"
  FILE_NAME         "svtk-svtk-module-find-packages.cmake"
  MODULES           ${svtk_modules})
