# Forbid downloading resources from the network during a build. This helps
# when building on systems without network connectivity to determine which
# resources much be obtained manually and made available to the build.
option(SVTK_FORBID_DOWNLOADS "Do not download source code or data from the network" OFF)
mark_as_advanced(SVTK_FORBID_DOWNLOADS)
macro(svtk_download_attempt_check _name)
  if(SVTK_FORBID_DOWNLOADS)
    message(SEND_ERROR "Attempted to download ${_name} when SVTK_FORBID_DOWNLOADS is ON")
  endif()
endmacro()
