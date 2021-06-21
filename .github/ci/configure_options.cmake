# Options that can be overridden based on the
# configuration name.
function (configuration_flag variable configuration)
  if ("$ENV{CMAKE_CONFIGURATION}" MATCHES "${configuration}")
    set("${variable}" ON CACHE BOOL "")
  else ()
    set("${variable}" OFF CACHE BOOL "")
  endif ()
endfunction ()

if (ENABLE_VTK_IO)
  configuration_flag(ENABLE_VTK_RENDERING "rendering")
endif ()

if (ENABLE_VTKM)
  configuration_flag(ENABLE_VTKM_RENDERING "rendering")
endif ()
