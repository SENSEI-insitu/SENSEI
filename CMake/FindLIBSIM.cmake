# - Find libsim
# Find the native libsim includes and library
#
# set VISIT_DIR to the install/arch directory
# For example on my workstation, I use
# -DVISIT_DIR=/work/SENSEI/visit-install/2.13.0/linux-x86_64/
#
#  LIBSIM_INCLUDE_DIR  - user modifiable choice of where libsim headers are
#  LIBSIM_LIBRARY      - user modifiable choice of where libsim libraries are
#
# This module returns these variables for the rest of the project to use.
#
#  LIBSIM_FOUND          - True if libsim found including required interfaces(see below)
#  LIBSIM_LIBRARIES      - All libsim related libraries.
#  LIBSIM_INCLUDE_DIRS   - All directories to include.
#
if(LIBSIM_INCLUDE_DIR AND LIBSIM_LIBRARY)
  set(LIBSIM_FIND_QUIETLY TRUE)
endif()

if(LIBSIM_DIR AND NOT VISIT_DIR)
  set(VISIT_DIR "${LIBSIM_DIR}/../..")
endif()

# pick up third party
set(LIBSIM_THIRD_PARTY)
list (FIND Libsim_FIND_COMPONENTS VTK tmp)
if (${tmp} GREATER -1)
  find_package(VTK)
  list(APPEND LIBSIM_THIRD_PARTY ${VTK_LIBRARIES})
endif()

list (FIND Libsim_FIND_COMPONENTS IceT tmp)
if (${tmp} GREATER -1)
  find_package(IceT)
  list(APPEND LIBSIM_THIRD_PARTY ${ICET_LIBRARIES})
endif()

# lib
if(VISIT_DIR)
  find_library(LIBSIM_LIBRARY NAMES simV2
    PATHS "${VISIT_DIR}/libsim/V2/lib"
    NO_DEFAULT_PATH)
endif()

find_library(LIBSIM_LIBRARY NAMES simV2
  PATHS ENV LD_LIBRARY_PATH NO_DEFAULT_PATH)

find_library(LIBSIM_LIBRARY NAMES simV2)

mark_as_advanced(LIBSIM_LIBRARY)
set(LIBSIM_LIBRARIES ${LIBSIM_LIBRARY} ${LIBSIM_THIRD_PARTY} dl)

# header
if(VISIT_DIR)
  find_path(LIBSIM_INCLUDE_DIR VisItControlInterface_V2.h
    PATHS "${VISIT_DIR}/libsim/V2/include" NO_DEFAULT_PATH)
endif()

get_filename_component(LIBSIM_LIBRARY_DIR
  ${LIBSIM_LIBRARY} DIRECTORY)

find_path(LIBSIM_INCLUDE_DIR VisItControlInterface_V2.h
  PATHS "${LIBSIM_LIBRARY_DIR}/../include"
  NO_DEFAULT_PATH)

find_path(LIBSIM_INCLUDE_DIR VisItControlInterface_V2.h)

mark_as_advanced(LIBSIM_INCLUDE_DIR)
set(LIBSIM_INCLUDE_DIRS ${LIBSIM_INCLUDE_DIR})

set(err_msg "Failed to locate libsim in VISIT_DIR=\"${VISIT_DIR}\"")

# validate
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBSIM
  ${err_msg} LIBSIM_LIBRARIES LIBSIM_INCLUDE_DIRS)
