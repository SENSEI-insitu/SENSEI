# ![SENSEI](doc/images/sensei_logo_small.png)
The SENSEI project takes aim at a set of research challenges for enabling
scientific knowledge discovery within the context of in situ processing at
extreme-scale concurrency. This work is motivated by a widening gap between
FLOPs and I/O capacity which will make full-resolution, I/O-intensive post hoc
analysis prohibitively expensive, if not impossible.

We focus on new algorithms for analysis, and visualization - topological,
geometric, statistical analysis, flow field analysis, pattern detection and
matching - suitable for use in an in situ context aimed specifically at
enabling scientific knowledge discovery in several exemplar application areas
of importance to DOE.  Complementary to the in situ algorithmic work, we focus
on several leading in situ infrastructures, and tackle research questions
germane to enabling new algorithms to run at scale across a diversity of
existing in situ implementations.

Our intent is to move the field of in situ processing in a direction where it
may ultimately be possible to write an algorithm once, then have it execute in
one of several different in situ software implementations. The combination of
algorithmic and infrastructure work is grounded in direct interactions with
specific application code teams, all of which are engaged in their own R&D
aimed at evolving to the exascale.

|Quick links |
|------------|
| [Project Organization](#project-organization) |
| [Build and Install](#build-and-install) |
| [Using the SENSEI Library](#using-the-sensei-library)

[![Documentation Status](https://readthedocs.org/projects/sensei-insitu/badge/?version=rtd_user_guide)](https://sensei-insitu.readthedocs.io/en/rtd_user_guide/?badge=rtd_user_guide)

## SENSEI library
The SENSEI library contains core base classes that declare the AnalysisAdaptor
API which is used to interface to in situ infrastructures and implement custom
analyses; the DataAdaptor API which AnalysisAdaptors use to access simulation
data in a consistent way; and a number of implementations of both. For more
information see our [SC16 paper](http://dl.acm.org/citation.cfm?id=3015010).

### Source code
SENSEI is open source and freely available on github at https://github.com/SENSEI-insitu/SENSEI

### Data model
SENSEI makes use of a heavily stripped down and mangled version of the
[VTK](https://vtk.org) 9.0.0 data model. The best source of documentation for
SENSEI's data model is VTK itself ([VTK doxygen](https://vtk.org/doc/nightly/html/classvtkDataObject.html)).

### Instrumenting a simulation
Instrumenting a simulation typically involves creating and initializing an
instance of sensei::ConfigurableAnalysis with an XML file and passing a
simulation specific sensei::DataAdaptor when in situ processing is invoked.

| Class | Description |
| ----- | ----------- |
| [sensei::ConfigurableAnalysis](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_analysis.html) | uses a run time provided XML file to slect and confgure one or more library specific data consumers or in transit transports |
| [sensei::DataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_data_adaptor.html) | simulations implement an instance that packages simulation data into SENSEI's data model |
| [sensei::SVTKDataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_svtk_data_adaptor.html) | An adaptor that can manage and serve SVTK data objects. Use this to return data from an analysis  |

### In situ data processing
SENSEI comes with a number of ready to use in situ processing options. These include:

| Class | Description |
| ----- | ----------- |
| [sensei::AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_analysis.html) | base class for in situ data processors |
| [sensei::DataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_data_adaptor.html) | defines the API by which data processors fetch data from the simulation |
| [sensei::Histogram](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_histogram.html) | Computes histograms |
| [sensei::AscentAnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_ascent_analysis_adaptor.html) | Processes simulation data using Ascent |
| [sensei::CatalystAnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_catalyst_analysis_adaptor.html) | Processes simulation data using ParaView Catalyst |
| [sensei::LibsimAnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_libsim_analysis_adaptor.html) | Processes simulation data using VisIt Libsim |
| [sensei::Autocorrelation](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_autocorrelation.html) | Compute autocorrelation of simulation data over time |
| [sensei::VTKPosthocIO](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_vtk_posthoc_io.html) | Writes simulation data to disk in a VTK format |
| [sensei::VTKAmrWriter](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_vtk_amr_writer.html) | Writes simulation data to disk in a VTK format |
| [sensei::PythonAnalysis](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_python_analysis.html) | Invokes user provided Pythons scripts that process simulation data |
| [sensei::SliceExtract](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_slice_extract.html) | Computes planar slices and iso-surfaces on simulation data |

### User defined in situ processing
A unique feature of SENSEI is the ability to invoke user provided code written
in Python or C++ on a SENSEI instrumented simulation. This makes SENSEI much
easier to extend and customize than other solutions.

| Class | Description |
| ----- | ----------- |
| [sensei::AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_analysis_adaptor.html) | used to invoke user defined C++ code. Override the sensei::AnalysisAdaptor::Execute method.  |
| [sensei::PythonAnalysis](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_python_analysis.html) | used to invoke user defined Python code Implement an Execute function in Python |

For more information see our [ISAV 2018](https://doi.org/10.1145/3281464.3281465) paper.

### In transit data processing
It is often advantageous to move data onto a seperate set of compute nodes for
concurrent processing in a job seperate from the simulation. SENSEI supports
this through run time configurable transports and the SENSEIEndPoint an
application written in C++ that can be configured to receive and process data
while simulation is running.

| Class | Description |
| ----- | ----------- |
| [sensei::AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_analysis_adaptor.html) | base class for the write side of the transport |
| [sensei::DataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_data_adaptor.html) | base class for the read side of the transport |
| [sensei::ConfigurableAnalysis](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_analysis.html) | used to select and configure the write side of the transport at run time from XML |
| [sensei::ConfigurableInTransitDataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_intransit_data_adaptor.html) | used to configure the read side of the transport at run time from XML |
| [sensei::ADIOS2AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_adios2_analysis_adaptor.html) | The write side of the ADIOS 2 transport |
| [sensei::ADIOS2DataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_adios2_data_adaptor.html) | The read side of the ADIOS 2 transport |
| [sensei::HDF5AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_hdf5_analysis_adaptor.html) | The write side of the HDF5 transport |
| [sensei::HDF5DataAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_hdf5_data_adaptor.html) | The read side of the HDF5 transport |
| [sensei::Partitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_partitioner.html) | base class for data partitioner which maps data to MPI ranks as it is moved |
| [sensei::ConfigurablePartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_partitioner.html) | Selects and configures one of the partitioners at run time from XML |
| [sensei::BlockPartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_block_partitioner.html) | maps blocks to ranks such that consecutive blocks share a rank |
| [sensei::PlanarPartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_planar_partitioner.html) | Maps blocks to MPI ranks in a round robbin fassion |
| [sensei::MappedPartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_mapped_partitioner.html) | Maps blocks to MPI ranks using a run time user provided mapping |
| [sensei::IsoSurfacePartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_iso_surface_partitioner.html) | Maps blocks to MPI ranks such that blocks not intersecting the iso surface are excluded |
| [sensei::PlanarSlicePartitioner](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_planar_slice_paritioner.html) | Maps blocks to MPI ranks such that blocks not intersecting the slice are excluded |

For more information see our [EGPGV 2020](https://doi.org/10.2312/pgv.20201073)
and [ISAV 2018](https://doi.org/10.1145/3364228.3364237) papers.

### The SENSEI End point
End points are programs that receive and analyze simulation data through transport
layers such as ADIOS and LibIS. The SENSEI end point uses the transport's data adaptor to
read data being serialized by the transport's analysis adaptor and pass it back
into a SENSEI analysis for further processing.

### Mini-apps
SENSEI ships with a number of mini-apps that demonstrate use of the SENSEI
library with custom analyses and the supported in situ infrastructures. When
the SENSEI library is enabled the mini-apps will make use of any of the
supported in situ infrastructures that are enabled. When the SENSEI library is
disabled mini-apps are restricted to the custom analysis such as histogram and
autocorrelation.

More information on each mini-app is provided in the coresponding README in the
mini-app's source directory.

* [Oscillators](miniapps/oscillators/README.md) The miniapp from year II
  generates time varying data on a uniform mesh and demonstrates usage with in
  situ infrasturctures, histogram, and autocorrelation analyses.

* [Newton++](https://github.com/SENSEI-insitu/newtonpp) A complete re-write of
  the original in C++ using OpenMP target offload for platform portable
  acceleration. This mini app demonstrates zero-copy data transfer from GPUs and
  accelerators. 

* [Newton Python](miniapps/newton/README.md) This Python n-body miniapp demonstrates
  usage of in situ infrastructures and custom analyses from Python. This has been replaced by
  the C++ port Newton++.

### Data Model
SENSEI makes use of a fork of VTK 9.0.0 that has been mangled and minified.
Minification has removed all source code from our fork except for the data
model and its dependencies. This substantially reduces the overheads associated
with VTK providing only the features we need for our data model.  VTK filters
and I/O can be used by pointing the build to an install of standard VTK as
released by Kitware.
Mangling changed the character sequences VTK and vtk to SVTK and svtk this
allows for interoberability with VTK as released by Kitware.

### Support for Heterogeneous Architectures
Extensions to SENSEI's execution and data model have been introduced to support
in situ on heterogeneous architectures. 
Extensions to the data model make zero-copy transfer of accelerator backed
memory possible.
Simply use
[svtkHAMRDataArray](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsvtk_h_a_m_r_data_array.html)
when passing array based data in your data adaptor and initialize it using one of
the zero-copy constructors.
Extensions to SENSEI's execution model provide placement controls as well as
control over synchronous or asynchronous execution method.
The controls for our execution model extensions are accessed through the APIs defined in
[sensei::AnalysisAdaptor](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_analysis_adaptor.html) or via XML attributes defined in
[sensei::ConfigurableAnalysis](https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_analysis.html). 

For more details including source code examples, XML, and a demonstration at
scale on NERSC's Cray NVIDIA system Perlmutter please see our [SC23
paper](https://arxiv.org/pdf/2310.02926.pdf).

## Build and Install
The SENSEI project uses CMake 3.0 or later. The CMake build options allow you
to choose which of the mini-apps to build as well as which frameworks to enable.
It is fine to enable multiple infrastructures, however note that Catalyst and
Libsim are currently mutually exclusive options due to their respective use of
different versions of VTK.

### Typical build procedure
```bash
$ mkdir build
$ cd build
$ ccmake .. # set one or more -D options as needed
$ make
$ make install
```

### Build Options
| Build Option | Default | Description                     |
|--------------|---------|---------------------------------|
| `SENSEI_ENABLE_CUDA` | OFF | Enables CUDA accelerated codes. Requires compute capability 7.5 and CUDA 11 or later. |
| `SENSEI_ENABLE_PYTHON` | OFF | Enables Python bindings. Requires VTK, Python, Numpy, mpi4py, and SWIG. |
| `SENSEI_ENABLE_CATALYST` | OFF | Enables the Catalyst analysis adaptor. Depends on ParaView Catalyst. Set `ParaView_DIR`. |
| `SENSEI_ENABLE_CATALYST_PYTHON` | OFF | Enables Python features of the Catalyst analysis adaptor.  |
| `SENSEI_ENABLE_ASCENT` | OFF | Enables the Ascent analysis adaptor. |
| `SENSEI_ENABLE_ADIOS1` | OFF | Enables ADIOS 1 adaptors and endpoints. Set `ADIOS_DIR`. |
| `SENSEI_ENABLE_HDF5` | OFF | Enables HDF5 adaptors and endpoints. Set `HDF5_DIR`. |
| `SENSEI_ENABLE_LIBSIM` | OFF | Enables Libsim data and analysis adaptors. Requires Libsim. Set `VTK_DIR` and `LIBSIM_DIR`. |
| `SENSEI_ENABLE_VTK_IO` | OFF | Enables adaptors to write to VTK XML format. |
| `SENSEI_ENABLE_VTK_MPI` | OFF | Enables MPI parallel VTK filters, such as parallel I/O. |
| `SENSEI_ENABLE_VTKM` | ON | Enables analyses that use VTKm directly instead of via VTK. |
| `SENSEI_ENABLE_OSCILLATORS` | ON | Enables the oscillators mini-app. |
| `VTK_DIR` | | Set to the directory containing VTKConfig.cmake. |
| `ParaView_DIR` | | Set to the directory containing ParaViewConfig.cmake. |
| `ADIOS_DIR` | | Set to the directory containing ADIOSConfig.cmake |
| `LIBSIM_DIR` | | Path to libsim install. |

### For use with Ascent
```bash
cmake -DENABLE_ASCENT=ON -DVTKM_DIR=[your path] -DVTKH_DIR=[your path] \
    -DCONDUIT_DIR=[your path] -DAscent_DIR=[your path] -DVTK_DIR=[your path] \
    ..
```
Note that the VTK build needs to explicitly disable use of VTK-m as this will
conflict with the version required by Ascent.  We used the instructions for
building Ascent and its dependencies (VTK-m, VTK-h, Conduit, etc) manually as
described in the Ascent documentation.

### For use the Libsim
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_LIBSIM=ON -DVTK_DIR=[your path] -DLIBSIM_DIR=[your path] ..
```
`VTK_DIR` should point to the VTK used by Libsim.

### For use with Catalyst
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_CATALYST=ON -DParaView_DIR=[your path] ..
```
Optionally, `-DENABLE_CATALYST_PYTHON=ON` will enable Catalyst Python scripts.
Note that a development version of ParaView is required when building with
both `SENSEI_ENABLE_CATALYST` and `SENSEI_ENABLE_VTKM` are enabled as released versions of
ParaView (5.5.2 and earlier) do not include a modern-enough version of vtk-m.

### For use with ADIOS 1
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_ADIOS1=ON -DVTK_DIR=[your path] -DADIOS_DIR=[your path] ..
```
Can be used with either `ParaView_DIR` when configuring in conjunction with
Catalyst, or `VTK_DIR` otherwise.

### For use with Python
In essence this is as simple as adding `-DENABLE_PYTHON=ON -DSENSEI_PYTHON_VERSION=3`
However, VTK (or ParaView when used with Catalyst) needs to be built with
Python enabled and the SENSEI build needs to use the same version; and NumPy,
mpi4py, and SWIG are required. Note that there are some caveats when used with
Catalyst and Libsim. These are described in more detail in the Newton mini app
[README](miniapps/newton/README.md).


### Enable writing to Visit ".visit" format or ParaView ".pvd" format
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_VTK_IO=ON  -DVTK_DIR=[your path] ..
```
Can be used with either `ParaView_DIR` or `VTK_DIR`.

### For use with experimental VTK-m analyses
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_VTKM=ON -DVTK_DIR=[your path] ..
```
Note that a development version of VTK is required when building with
both `SENSEI_ENABLE_SENSEI` and `SENSEI_ENABLE_VTKM` are enabled as released versions of
VTK (8.1.1 and earlier) do not include a modern-enough version of vtk-m.

## Using the SENSEI library from another project
To use SENSEI from your CMake based project include the SENSEI CMake config in
your CMakeLists.txt.
```cmake
find_package(SENSEI REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec sensei ...)
```
Additionally, your source code may need to include `senseiConfig.h` to capture
compile time configuration.
