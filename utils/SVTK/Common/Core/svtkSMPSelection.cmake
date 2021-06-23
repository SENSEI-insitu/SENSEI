set(SVTK_SMP_IMPLEMENTATION_TYPE "Sequential"
  CACHE STRING "Which multi-threaded parallelism implementation to use. Options are Sequential, OpenMP or TBB")
set_property(CACHE SVTK_SMP_IMPLEMENTATION_TYPE
  PROPERTY
    STRINGS Sequential OpenMP TBB)

if (NOT (SVTK_SMP_IMPLEMENTATION_TYPE STREQUAL "OpenMP" OR
         SVTK_SMP_IMPLEMENTATION_TYPE STREQUAL "TBB"))
  set_property(CACHE SVTK_SMP_IMPLEMENTATION_TYPE
    PROPERTY
      VALUE "Sequential")
endif ()

set(svtk_smp_headers_to_configure)
set(svtk_smp_defines)
set(svtk_smp_use_default_atomics ON)

if (SVTK_SMP_IMPLEMENTATION_TYPE STREQUAL "TBB")
  svtk_module_find_package(PACKAGE TBB)
  list(APPEND svtk_smp_libraries
    TBB::tbb)

  set(svtk_smp_use_default_atomics OFF)
  set(svtk_smp_implementation_dir "${CMAKE_CURRENT_SOURCE_DIR}/SMP/TBB")
  list(APPEND svtk_smp_sources
    "${svtk_smp_implementation_dir}/svtkSMPTools.cxx")
  list(APPEND svtk_smp_headers_to_configure
    svtkAtomic.h
    svtkSMPToolsInternal.h
    svtkSMPThreadLocal.h)

elseif (SVTK_SMP_IMPLEMENTATION_TYPE STREQUAL "OpenMP")
  svtk_module_find_package(PACKAGE OpenMP)

  list(APPEND svtk_smp_libraries
    OpenMP::OpenMP_CXX)

  set(svtk_smp_implementation_dir "${CMAKE_CURRENT_SOURCE_DIR}/SMP/OpenMP")
  list(APPEND svtk_smp_sources
    "${svtk_smp_implementation_dir}/svtkSMPTools.cxx"
    "${svtk_smp_implementation_dir}/svtkSMPThreadLocalImpl.cxx")
  list(APPEND svtk_smp_headers_to_configure
    svtkSMPThreadLocal.h
    svtkSMPThreadLocalImpl.h
    svtkSMPToolsInternal.h)

  if (OpenMP_CXX_SPEC_DATE AND NOT "${OpenMP_CXX_SPEC_DATE}" LESS "201107")
    set(svtk_smp_use_default_atomics OFF)
    list(APPEND svtk_smp_sources
      "${svtk_smp_implementation_dir}/svtkAtomic.cxx")
    list(APPEND svtk_smp_headers_to_configure
      svtkAtomic.h)

    set_source_files_properties(svtkAtomic.cxx
      PROPERITES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  else()
    message(WARNING
      "Required OpenMP version (3.1) for atomics not detected. Using default "
      "atomics implementation.")
  endif()

elseif (SVTK_SMP_IMPLEMENTATION_TYPE STREQUAL "Sequential")
  set(svtk_smp_implementation_dir "${CMAKE_CURRENT_SOURCE_DIR}/SMP/Sequential")
  list(APPEND svtk_smp_sources
    "${svtk_smp_implementation_dir}/svtkSMPTools.cxx")
  list(APPEND svtk_smp_headers_to_configure
    svtkSMPThreadLocal.h
    svtkSMPToolsInternal.h)
endif()

if (svtk_smp_use_default_atomics)
  include(CheckSymbolExists)

  include("${CMAKE_CURRENT_SOURCE_DIR}/svtkTestBuiltins.cmake")

  set(svtkAtomic_defines)

  # Check for atomic functions
  if (WIN32)
    check_symbol_exists(InterlockedAdd "windows.h" SVTK_HAS_INTERLOCKEDADD)

    if (SVTK_HAS_INTERLOCKEDADD)
      list(APPEND svtkAtomic_defines "SVTK_HAS_INTERLOCKEDADD")
    endif ()
  endif()

  set_source_files_properties(svtkAtomic.cxx
    PROPERITES
      COMPILE_DEFINITIONS "${svtkAtomic_defines}")

  set(svtk_atomics_default_impl_dir "${CMAKE_CURRENT_SOURCE_DIR}/SMP/Sequential")
  list(APPEND svtk_smp_sources
    "${svtk_atomics_default_impl_dir}/svtkAtomic.cxx")
  configure_file(
    "${svtk_atomics_default_impl_dir}/svtkAtomic.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/svtkAtomic.h")
  list(APPEND svtk_smp_headers
    "${CMAKE_CURRENT_BINARY_DIR}/svtkAtomic.h")
endif()

foreach (svtk_smp_header IN LISTS svtk_smp_headers_to_configure)
  configure_file(
    "${svtk_smp_implementation_dir}/${svtk_smp_header}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${svtk_smp_header}"
    COPYONLY)
  list(APPEND svtk_smp_headers
    "${CMAKE_CURRENT_BINARY_DIR}/${svtk_smp_header}")
endforeach()

list(APPEND svtk_smp_headers
  svtkSMPTools.h
  svtkSMPThreadLocalObject.h)
