# TODO: Update FindJNI to use imported targets.
find_package(JNI REQUIRED)

set(classes
  svtkJavaUtil)

set(headers
  svtkJavaAwt.h)

svtk_module_add_module(SVTK::Java
  CLASSES ${classes}
  HEADERS ${headers})
svtk_module_include(SVTK::Java SYSTEM
  PUBLIC
    ${JNI_INCLUDE_DIRS})
svtk_module_link(SVTK::Java
  PUBLIC
    ${JNI_LIBRARIES})
