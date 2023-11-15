# Options that can be overridden based on the
# configuration name.
function (configuration_flag variable configuration)
  if ("$ENV{CMAKE_CONFIGURATION}" MATCHES "${configuration}")
    set("${variable}" ON CACHE BOOL "")
  else ()
    set("${variable}" OFF CACHE BOOL "")
  endif ()
endfunction ()

if (SENSEI_ENABLE_VTK_IO)
  configuration_flag(SENSEI_ENABLE_VTK_RENDERING "rendering")
endif ()

if (SENSEI_ENABLE_VTKM)
  configuration_flag(SENSEI_ENABLE_VTKM_RENDERING "rendering")
endif ()
