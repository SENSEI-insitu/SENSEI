include(CMakeDependentOption)

option(SENSEI_ENABLE_SENSEI "Enable Sensei infrastucture" ON)

option(SENSEI_ENABLE_CUDA "Enable the use of CUDA" OFF)

cmake_dependent_option(SENSEI_ENABLE_CUDA_MPI
  "Enable the use of CUDA aware MPI" OFF
  "SENSEI_ENABLE_CUDA" OFF)

cmake_dependent_option(SENSEI_ENABLE_HIP
  "Enable analysis methods that use HIP" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_OPENMP
  "Enable analysis methods that use OpenMP device offload" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_PYTHON
  "Enable Python binding to Sensei infrastucture" ON
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_CATALYST
  "Enable analysis methods that use Catalyst" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_CATALYST2
  "Enable analysis methods that use Catalyst2" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_CATALYST_PYTHON
  "Enable analysis methods that use Catalyst Python scripts" ON
  "SENSEI_ENABLE_CATALYST" OFF)

cmake_dependent_option(SENSEI_ENABLE_ADIOS1
  "Enable analysis methods that use ADIOS 1" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_ADIOS2
  "Enable analysis methods that use ADIOS 2" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_HDF5
  "Enable analysis methods that use HDF5" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_CONDUIT
  "Enable analysis methods that use Conduit" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_ASCENT
  "Enable analysis methods that use Ascent" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_ASCENT
  "Enable analysis methods that use ASCENT" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_LIBSIM
  "Enable analysis methods that use Libsim" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

# enable VTK by default if Catalyst is present.
set(VTK_DEFAULT OFF)
if (SENSEI_ENABLE_CATALYST)
  set(VTK_DEFAULT ON)
endif()

cmake_dependent_option(SENSEI_ENABLE_OSPRAY
  "Enable analysis methods that use OSPRay" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_IO
  "Enable use of vtk I/O" ${VTK_DEFAULT}
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_MPI
  "Enable use of parallel vtk" ${VTK_DEFAULT}
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_RENDERING
  "Enable use of VTK's rendering libraries" ${VTK_DEFAULT}
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_ACCELERATORS
  "Enable analysis methods that use VTK-m via VTK's Accelerators module" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_FILTERS
  "Enable use of VTK's generic filters library" ${VTK_DEFAULT}
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY
  "Enable use of VTK's parallel geometry filter library" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTKM
  "Enable analysis methods that use VTK-m" OFF
  "SENSEI_ENABLE_SENSEI" OFF)

cmake_dependent_option(SENSEI_ENABLE_VTKM_RENDERING
  "Enable analysis methods that use VTK-m's rendering library" OFF
  "SENSEI_ENABLE_VTKM" OFF)

option(SENSEI_ENABLE_OPTS "A version of the getopt function" ON)
option(SENSEI_ENABLE_PROFILER "Enable the internal profiler" OFF)
option(SENSEI_ENABLE_OSCILLATORS "Enable Oscillators miniapp" ON)
option(SENSEI_ENABLE_MANDELBROT "Enable Mandelbrot AMR miniapp" ON)
option(SENSEI_ENABLE_VORTEX "Enable Vortex miniapp (experimental)" OFF)
option(SENSEI_ENABLE_CONDUITTEST "Enable Conduit miniapp (experimental)" OFF)
option(SENSEI_ENABLE_KRIPKE "Enable Kripke miniapp (experimental)" OFF)
option(SENSEI_USE_EXTERNAL_pugixml "Use external pugixml library" OFF)

message(STATUS "SENSEI: SENSEI_ENABLE_SENSEI=${SENSEI_ENABLE_SENSEI}")
message(STATUS "SENSEI: SENSEI_ENABLE_CUDA=${SENSEI_ENABLE_CUDA}")
message(STATUS "SENSEI: SENSEI_ENABLE_PYTHON=${SENSEI_ENABLE_PYTHON}")
message(STATUS "SENSEI: SENSEI_ENABLE_CATALYST=${SENSEI_ENABLE_CATALYST}")
message(STATUS "SENSEI: SENSEI_ENABLE_CATALYST_PYTHON=${SENSEI_ENABLE_CATALYST_PYTHON}")
message(STATUS "SENSEI: SENSEI_ENABLE_CATALYST2=${SENSEI_ENABLE_CATALYST2}")
message(STATUS "SENSEI: SENSEI_ENABLE_ADIOS1=${SENSEI_ENABLE_ADIOS1}")
message(STATUS "SENSEI: SENSEI_ENABLE_ADIOS2=${SENSEI_ENABLE_ADIOS2}")
message(STATUS "SENSEI: SENSEI_ENABLE_HDF5=${SENSEI_ENABLE_HDF5}")
message(STATUS "SENSEI: SENSEI_ENABLE_CONDUIT=${SENSEI_ENABLE_CONDUIT}")
message(STATUS "SENSEI: SENSEI_ENABLE_ASCENT=${SENSEI_ENABLE_ASCENT}")
message(STATUS "SENSEI: SENSEI_ENABLE_LIBSIM=${SENSEI_ENABLE_LIBSIM}")
message(STATUS "SENSEI: SENSEI_ENABLE_OSPRAY=${SENSEI_ENABLE_OSPRAY}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_IO=${SENSEI_ENABLE_VTK_IO}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_MPI=${SENSEI_ENABLE_VTK_MPI}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_RENDERING=${SENSEI_ENABLE_VTK_RENDERING}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_ACCELERATORS=${SENSEI_ENABLE_VTK_ACCELERATORS}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY=${SENSEI_ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTK_FILTERS=${SENSEI_ENABLE_VTK_FILTERS}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTKM=${SENSEI_ENABLE_VTKM}")
message(STATUS "SENSEI: SENSEI_ENABLE_VTKM_RENDERING=${SENSEI_ENABLE_VTKM_RENDERING}")
message(STATUS "SENSEI: SENSEI_ENABLE_PROFILER=${SENSEI_ENABLE_PROFILER}")
message(STATUS "SENSEI: SENSEI_ENABLE_OPTS=${SENSEI_ENABLE_OPTS}")
message(STATUS "SENSEI: SENSEI_ENABLE_OSCILLATORS=${SENSEI_ENABLE_OSCILLATORS}")
message(STATUS "SENSEI: SENSEI_ENABLE_CONDUITTEST=${SENSEI_ENABLE_CONDUITTEST}")
message(STATUS "SENSEI: SENSEI_ENABLE_KRIPKE=${SENSEI_ENABLE_KRIPKE}")
message(STATUS "SENSEI: SENSEI_USE_EXTERNAL_pugixml=${SENSEI_USE_EXTERNAL_pugixml}")

if (SENSEI_ENABLE_ADIOS1 AND SENSEI_ENABLE_ADIOS2)
  message(FATAL_ERROR "ADIOS1 and ADIOS2 are mutually exclusive build options")
endif()
