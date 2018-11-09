find_package(libIS REQUIRED)

if (NOT LAMMPS_DIR)
  message(FATAL_ERROR "To build LAMMPS driver you must set -DLAMMPS_DIR=<root of lammps dir>")
endif()

include_directories(${LAMMPS_DIR}/src)

set(sources lammpsDriver.cxx)

set(libs 
    ${LAMMPS_DIR}/src/liblammps_mpi.a
    is_sim
)

if(ENABLE_SENSEI)
  list(APPEND sources
    lammpsDataAdaptor.h
    lammpsDataAdaptor.cxx
    lammpsBridge.h
    lammpsBridge.cxx
  )
  list(APPEND libs sensei)
endif()

add_executable(lammpsDriver ${sources})

# fix to build libsim version on Ubuntu
#set_target_properties(lammpsDriver PROPERTIES LINK_FLAGS "-Wl,--no-as-needed -ldl")

target_link_libraries(lammpsDriver ${libs})

install(TARGETS lammpsDriver
  RUNTIME DESTINATION bin
)



