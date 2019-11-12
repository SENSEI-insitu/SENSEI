###############################################################################
#  Expects ASCENT_DIR to point to a Ascent installation.
#
# This file defines the following CMake variables:
#  ASCENT_FOUND - If Ascent was found
#  ASCENT_INCLUDE_DIRS - The Ascent include directories
#  target_link_libraries(<target> ascent::ascent)
#
###############################################################################

# Check for ASCENT_DIR
if(NOT ASCENT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Ascent. Ascent requires explicit ASCENT_DIR.")
endif()

if(NOT EXISTS ${ASCENT_DIR}/lib/cmake/AscentConfig.cmake)
    MESSAGE(FATAL_ERROR "Could not find Ascent CMake include file (${ASCENT_DIR}/lib/cmake/AscentConfig.cmake)")
endif()

# Use CMake's find_package to import Ascent's targets
find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS ${ASCENT_DIR}/lib/cmake)

