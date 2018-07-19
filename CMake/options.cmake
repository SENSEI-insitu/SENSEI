option(ENABLE_SENSEI "Enable Sensei infrastucture" ON)

cmake_dependent_option(ENABLE_PYTHON
  "Enable Python binding to Sensei infrastucture" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_VTK_GENERIC_ARRAYS
  "VTK build has Generic arrays" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_CATALYST
  "Enable analysis methods that use Catalyst" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_CATALYST_PYTHON
  "Enable analysis methods that use Catalyst Python scripts" OFF
  "ENABLE_CATALYST" OFF)

cmake_dependent_option(ENABLE_ADIOS
  "Enable analysis methods that use ADIOS" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_LIBSIM
  "Enable analysis methods that use Libsim" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_VTK_IO
  "Enable use of vtk I/O" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_VTK_MPI
  "Enable use of parallel vtk" OFF
  "ENABLE_SENSEI" OFF)

cmake_dependent_option(ENABLE_VTK_M
  "Enable analysis methods that use VTK-m" OFF
  "ENABLE_VTK_MPI" OFF)


option(ENABLE_PARALLEL3D "Enable Parallel3D miniapp" ON)
option(ENABLE_OSCILLATORS "Enable Oscillators miniapp" ON)
option(ENABLE_MANDELBROT "Enable Mandelbrot miniapp" ON)
option(ENABLE_VORTEX "Enable Vortex miniapp" ON)

message(STATUS "ENABLE_SENSEI=${ENABLE_SENSEI}")
message(STATUS "ENABLE_PYTHON=${ENABLE_PYTHON}")
message(STATUS "ENABLE_VTK_GENERIC_ARRAYS=${ENABLE_VTK_GENERIC_ARRAYS}")
message(STATUS "ENABLE_CATALYST=${ENABLE_CATALYST}")
message(STATUS "ENABLE_CATALYST_PYTHON=${ENABLE_CATALYST}")
message(STATUS "ENABLE_ADIOS=${ENABLE_ADIOS}")
message(STATUS "ENABLE_LIBSIM=${ENABLE_LIBSIM}")
message(STATUS "ENABLE_VTK_IO=${ENABLE_VTK_IO}")
message(STATUS "ENABLE_VTK_MPI=${ENABLE_VTK_MPI}")
message(STATUS "ENABLE_PARALLEL3D=${ENABLE_PARALLEL3D}")
message(STATUS "ENABLE_OSCILLATORS=${ENABLE_OSCILLATORS}")
