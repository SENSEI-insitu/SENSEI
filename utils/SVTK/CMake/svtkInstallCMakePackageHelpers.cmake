function (_svtk_package_append_variables)
  set(_svtk_package_variables)
  foreach (var IN LISTS ARGN)
    if (NOT ${var})
      continue ()
    endif ()

    get_property(type_is_set CACHE "${var}"
      PROPERTY TYPE SET)
    if (type_is_set)
      get_property(type CACHE "${var}"
        PROPERTY TYPE)
    else ()
      set(type UNINITIALIZED)
    endif ()

    string(APPEND _svtk_package_variables
      "if (NOT DEFINED \"${var}\" OR NOT ${var})
  set(\"${var}\" \"${${var}}\" CACHE ${type} \"Third-party helper setting from \${CMAKE_FIND_PACKAGE_NAME}\")
endif ()
")
  endforeach ()

  set(svtk_find_package_code
    "${svtk_find_package_code}${_svtk_package_variables}"
    PARENT_SCOPE)
endfunction ()

get_property(_svtk_packages GLOBAL
  PROPERTY _svtk_module_find_packages_SVTK)
if (_svtk_packages)
  list(REMOVE_DUPLICATES _svtk_packages)
endif ()

# Per-package variable forwarding goes here.
set(Boost_find_package_vars
  Boost_INCLUDE_DIR)
set(OSMesa_find_package_vars
  OSMESA_INCLUDE_DIR
  OSMESA_LIBRARY)

if ("ospray" IN_LIST _svtk_packages)
  # FIXME: ospray depends on embree, but does not help finders at all.
  # https://github.com/ospray/ospray/issues/352
  list(APPEND _svtk_packages
    embree)
endif ()

set(svtk_find_package_code)
foreach (_svtk_package IN LISTS _svtk_packages)
  _svtk_package_append_variables(
    # Standard CMake `find_package` mechanisms.
    "${_svtk_package}_DIR"
    "${_svtk_package}_ROOT"

    # Per-package custom variables.
    ${${_svtk_package}_find_package_vars})
endforeach ()

file(GENERATE
  OUTPUT  "${svtk_cmake_build_dir}/svtk-find-package-helpers.cmake"
  CONTENT "${svtk_find_package_code}")
