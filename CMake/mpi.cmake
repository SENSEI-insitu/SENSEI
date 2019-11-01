
if (ENABLE_CRAY_MPICH OR (NOT DEFINED ENABLE_CRAY_MPICH AND NOT ("$ENV{CRAY_MPICH_DIR}" STREQUAL "")))
    set(ENV{PKG_CONFIG_PATH} "$ENV{CRAY_MPICH_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(PkgConfig QUIET)
    pkg_check_modules(CRAY_MPICH REQUIRED QUIET mpich)
    set(MPI_C_INCLUDE_PATH ${CRAY_MPICH_INCLUDE_DIRS} CACHE STRING "MPI include directories")
    set(MPI_C_LIBRARIES ${CRAY_MPICH_LDFLAGS} CACHE STRING "MPI link dependencies")
    set(MPIEXEC srun CACHE STRING "Platform MPI run equivalent")
    set(MPI_C_FOUND CACHE BOOL ON "status of MPI config")
else()
    find_package(MPI)
endif()

if (NOT MPI_C_FOUND)
  message(FETAL_ERROR "Failed to locate MPI C libraries and headers")
endif()

# MPI to use extern "C" when including headers
add_definitions(-DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1)

# interface libarary for use elsewhere in the project
add_library(sMPI INTERFACE)

target_include_directories(sMPI SYSTEM INTERFACE
  ${MPI_C_INCLUDE_PATH} ${MPI_C_INCLUDE_DIRS})

target_link_libraries(sMPI INTERFACE ${MPI_C_LIBRARIES})

install(TARGETS sMPI EXPORT sMPI)
install(EXPORT sMPI DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
