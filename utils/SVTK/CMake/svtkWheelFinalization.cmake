set(svtk_features)

if (TARGET SVTK::mpi)
  list(APPEND svtk_features "mpi")
endif ()

if (TARGET SVTK::RenderingCore)
  list(APPEND svtk_features "rendering")
endif ()

if (TARGET SVTK::RenderingOpenVR)
  list(APPEND svtk_features "openvr")
endif ()

if (TARGET SVTK::RenderingOpenGL2)
  set(has_onscreen OFF)
  set(has_offscreen OFF)
  if (SVTK_USE_X)
    list(APPEND svtk_features "rendering-onscreen-x11")
    set(has_onscreen ON)
  endif ()
  if (SVTK_USE_COCOA)
    list(APPEND svtk_features "rendering-onscreen-cocoa")
    set(has_onscreen ON)
  endif ()
  if (WIN32)
    list(APPEND svtk_features "rendering-onscreen-windows")
    set(has_onscreen ON)
  endif ()
  if (SVTK_OPENGL_HAS_OSMESA)
    list(APPEND svtk_features "rendering-offscreen-osmesa")
    set(has_offscreen ON)
  endif ()

  if (has_onscreen)
    list(APPEND svtk_features "rendering-onscreen")
  endif ()
  if (has_offscreen)
    list(APPEND svtk_features "rendering-offscreen")
  endif ()

  if (SVTK_OPENGL_USE_GLES)
    list(APPEND svtk_features "rendering-backend-gles")
  else ()
    list(APPEND svtk_features "rendering-backend-gl")
  endif ()
  if (SVTK_OPENGL_HAS_EGL)
    list(APPEND svtk_features "rendering-backend-egl")
  endif ()
endif ()

set(svtk_feature_entries "")
foreach (svtk_feature IN LISTS svtk_features)
  string(APPEND svtk_feature_entries
    "    '${svtk_feature}': [],\n")
endforeach ()
file(WRITE "${CMAKE_BINARY_DIR}/svtk_features.py"
  "FEATURES = {\n${svtk_feature_entries}}\n")
