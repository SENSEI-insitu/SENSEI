#[==[.md
# svtkOpenGLOptions

This module provides options that control which OpenGL and Windowing system
libraries are used.

#]==]
include(CMakeDependentOption)

# For each platform specific API, we define SVTK_USE_<API> options.
cmake_dependent_option(SVTK_USE_COCOA "Use Cocoa for SVTK render windows" ON
  "APPLE;NOT APPLE_IOS" OFF)
mark_as_advanced(SVTK_USE_COCOA)

set(default_use_x OFF)
if(UNIX AND NOT ANDROID AND NOT APPLE AND NOT APPLE_IOS AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(default_use_x ON)
endif()
option(SVTK_USE_X "Use X for SVTK render windows" ${default_use_x})
mark_as_advanced(SVTK_USE_X)

# For optional APIs that could be available for the OpenGL implementation
# being used, we define SVTK_OPENGL_HAS_<feature> options. These are not to be
# treated as mutually exclusive.

#-----------------------------------------------------------------------------
# OSMesa variables
#-----------------------------------------------------------------------------
# OpenGL implementation supports OSMesa for creating offscreen context.
option(SVTK_OPENGL_HAS_OSMESA
  "The OpenGL library being used supports offscreen Mesa (OSMesa)" OFF)
mark_as_advanced(SVTK_OPENGL_HAS_OSMESA)

#-----------------------------------------------------------------------------
# GLES variables
#-----------------------------------------------------------------------------

set(default_has_egl OFF)
if (ANDROID)
  set(SVTK_OPENGL_USE_GLES ON CACHE INTERNAL "Use the OpenGL ES API")
  set(default_has_egl ON)
else ()
  # OpenGLES implementation.
  option(SVTK_OPENGL_USE_GLES "Use the OpenGL ES API" OFF)
  mark_as_advanced(SVTK_OPENGL_USE_GLES)
endif ()

#-----------------------------------------------------------------------------
# EGL variables
#-----------------------------------------------------------------------------
# OpenGL implementation supports EGL for creating offscreen context.
option(SVTK_OPENGL_HAS_EGL "The OpenGL library being used supports EGL" "${default_has_egl}")
mark_as_advanced(SVTK_OPENGL_HAS_EGL)

set(SVTK_DEFAULT_EGL_DEVICE_INDEX "0" CACHE STRING
  "EGL device (graphics card) index to use by default for EGL render windows.")
mark_as_advanced(SVTK_DEFAULT_EGL_DEVICE_INDEX)

#-----------------------------------------------------------------------------
# Irrespective of support for offscreen API, SVTK_DEFAULT_RENDER_WINDOW_OFFSCREEN
# lets the user select the default state for the  `Offscreen` flag on the
# svtkRenderWindow when it is instantiated (formerly SVTK_USE_OFFSCREEN).
option(SVTK_DEFAULT_RENDER_WINDOW_OFFSCREEN "Use offscreen render window by default" OFF)
mark_as_advanced(SVTK_DEFAULT_RENDER_WINDOW_OFFSCREEN)

#-----------------------------------------------------------------------------
set(SVTK_CAN_DO_OFFSCREEN FALSE)
set(SVTK_CAN_DO_ONSCREEN FALSE)
set(SVTK_CAN_DO_HEADLESS FALSE)

if(WIN32 OR SVTK_OPENGL_HAS_OSMESA OR SVTK_OPENGL_HAS_EGL)
  set(SVTK_CAN_DO_OFFSCREEN TRUE)
endif()
if(WIN32 OR SVTK_USE_COCOA OR SVTK_USE_X) # XXX: See error message below.
  set(SVTK_CAN_DO_ONSCREEN TRUE)
endif()

if(SVTK_OPENGL_HAS_OSMESA OR SVTK_OPENGL_HAS_EGL)
  set(SVTK_CAN_DO_HEADLESS TRUE)
endif()

if(APPLE_IOS OR ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(SVTK_CAN_DO_HEADLESS FALSE)
endif()

if (SVTK_OPENGL_HAS_OSMESA AND SVTK_CAN_DO_ONSCREEN)
  message(FATAL_ERROR
    "The `SVTK_OPENGL_HAS_OSMESA` is ignored if any of the following is true: "
    "the target platform is Windows, `SVTK_USE_COCOA` is `ON`, or `SVTK_USE_X` "
    "is `ON`. OSMesa does not support on-screen rendering and SVTK's OpenGL "
    "selection is at build time, so the current build configuration is not "
    "satisfiable.")
endif ()

cmake_dependent_option(
  SVTK_USE_OPENGL_DELAYED_LOAD
  "Use delay loading for OpenGL"
  OFF "WIN32;COMMAND target_link_options" OFF)
mark_as_advanced(SVTK_USE_OPENGL_DELAYED_LOAD)

#-----------------------------------------------------------------------------
# For builds where we can support both on-screen and headless rendering, the default
# is to create an on-screen render window. Setting this option to ON will change the default
# to create an headless render window by default instead.
cmake_dependent_option(
  SVTK_DEFAULT_RENDER_WINDOW_HEADLESS
  "Enable to create the headless render window when `svtkRenderWindow` is instantiated."
  OFF "SVTK_CAN_DO_ONSCREEN;SVTK_CAN_DO_HEADLESS" OFF)
