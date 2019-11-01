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

## Project Organization
### SENSEI library
The SENSEI library contains core base classes that declare the AnalysisAdaptor
API which is used to interface to in situ infrastructures and implement custom
analyses; the DataAdaptor API which AnalysisAdaptors use to access simulation
data in a consistent way; and a number of implementations of both. For more
information see our [SC16 paper](http://dl.acm.org/citation.cfm?id=3015010).

#### DataAdaptors
| Class             | Description |
|-------------------|-------------|
| DataAdaptor       | Base class declaring data adaptor API |
| VTKDataAdaptor    | Implementation for use with VTK data sets. This adaptor can be used to pass VTK data sets from the simulation to the Analysis. |
| ADIOS1DataAdaptor | Implementation that serves up data from ADIOS 1. For use in an ADIOS 1 End point. |
| HDF5DataAdaptor   | Implementation that serves up data from HDF5. For use in a HDF5  End point. |

#### AnalysisAdaptors
| Class                   | Description |
|-------------------------|-------------|
| AnalysisAdaptor         | Base class declaring analysis adaptor API |
| ADIOS1AnalysisAdaptor   | Implementation for using ADIOS 1 from your simulation. |
| HDF5AnalysisAdaptor     | Implementation for using HDF5 from your simulation. |
| LibsimAnalysisAdaptor   | Implementation for using Libsim from your simulation. |
| CatalystAnalysisAdaptor | Implementation for using Catalyst from your simulaiton. |
| Autocorrelation         | Implementation that computes [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation)  |
| Histogram               | Implementation that computes histograms. |
| PosthocIO               | Implementation that writes uniform meshes using VTK or MPI I/O. This was used in year II miniapp campaign. |
| VTKPosthocIO            | Implementation that writes VTK data sets using VTK XML format to the ".visit" format readable by VisIt,  or ".pvd" format readable by ParaView. |
| ConfigurableAnalysis    | Implementation that reads an XML configuration to select and configure one or more of the other analysis adaptors. This can be used to quickly switch between the analysis adaptors at run time. |

### Mini-apps
SENSEI ships with a number of mini-apps that demonstrate use of the SENSEI
library with custom analyses and the supported in situ infrastructures. When
the SENSEI library is enabled the mini-apps will make use of any of the
supported in situ infrastructures that are enabled. When the SENSEI library is
disabled mini-apps are restricted to the custom analysis such as histogram and
autocorrelation.

More information on each mini-app is provided in the coresponding README in the
mini-app's source directory.

* [Parallel3D](miniapps/parallel3d/README.md) The miniapp from year I generates
  data on a uniform mesh and demonstrates usage with in situ infrasturctures
  and histogram analysis.

* [Oscillators](miniapps/oscillators/README.md) The miniapp from year II
  generates time varying data on a uniform mesh and demonstrates usage with in
  situ infrasturctures, histogram, and autocorrelation analyses.

* [Newton](miniapps/newton/README.md) This Python n-body miniapp demonstrates
  usage of in situ infrastructures and custom analyses from Python.

### End points
End points are programs that receive and analyze simulation data through transport
layers such as ADIOS and LibIS. The end point uses the transport's data adaptor to
reads data being serialized by the transport's analysis adaptor and pass it back
into a SENSEI analysis for further processing.

* [ADIOSAnalysisEndPoint](endpoints/README.md)

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
| `ENABLE_SENSEI` | ON | Enables the core SENSEI library. Requires VTK. When this is disabled, the included mini-apps will run fixed analyses. When enabled, the mini-apps will pass data through SENSEI and the analysis may be configured at run-time. This allows SENSEI overhead to be characterized.|
| `ENABLE_PYTHON` | OFF | Enables Python bindings. Requires VTK, Python, Numpy, mpi4py, and SWIG. |
| `ENABLE_VTK_GENERIC_ARRAYS` | OFF | Enables use of VTK's generic array feature.  |
| `ENABLE_CATALYST` | OFF | Enables the Catalyst analysis adaptor. Depends on ParaView Catalyst. Set `ParaView_DIR`. |
| `ENABLE_CATALYST_PYTHON` | OFF | Enables Python features of the Catalyst analysis adaptor.  |
| `ENABLE_ADIOS1` | OFF | Enables ADIOS 1 adaptors and endpoints. Set `ADIOS_DIR`. |
| `ENABLE_HDF5` | OFF | Enables HDF5 adaptors and endpoints. Set `HDF5_DIR`. |
| `ENABLE_LIBSIM` | OFF | Enables Libsim data and analysis adaptors. Requires Libsim. Set `VTK_DIR` and `LIBSIM_DIR`. |
| `ENABLE_VTK_IO` | OFF | Enables adaptors to write to VTK XML format. |
| `ENABLE_VTK_MPI` | OFF | Enables MPI parallel VTK filters, such as parallel I/O. |
| `ENABLE_VTKM` | ON | Enables analyses that use VTKm directly instead of via VTK. |
| `ENABLE_PARALLEL3D` | ON | Enables the parallel 3D mini-app. |
| `ENABLE_OSCILLATORS` | ON | Enables the oscillators mini-app. |
| `VTK_DIR` | | Set to the directory containing VTKConfig.cmake. |
| `ParaView_DIR` | | Set to the directory containing ParaViewConfig.cmake. |
| `ADIOS_DIR` | | Set to the directory containing ADIOSConfig.cmake |
| `LIBSIM_DIR` | | Path to libsim install. |


### For use with ADIOS 1
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_ADIOS1=ON -DVTK_DIR=[your path] -DADIOS_DIR=[your path] ..
```
Can be used with either `ParaView_DIR` when configuring in conjunction with
Catalyst, or `VTK_DIR` otherwise.

### For use with Libsim
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
both `ENABLE_CATALYST` and `ENABLE_VTKM` are enabled as released versions of
ParaView (5.5.2 and earlier) do not include a modern-enough version of vtk-m.

### Enable writing to Visit ".visit" format or ParaView ".pvd" format
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_VTK_IO=ON  -DVTK_DIR=[your path] ..
```
Can be used with either `ParaView_DIR` or `VTK_DIR`.

### For use with VTK-m
```bash
cmake -DENABLE_SENSEI=ON -DENABLE_VTKM=ON -DVTK_DIR=[your path] ..
```
Note that a development version of VTK is required when building with
both `ENABLE_SENSEI` and `ENABLE_VTKM` are enabled as released versions of
VTK (8.1.1 and earlier) do not include a modern-enough version of vtk-m.

### Enabling Python bindings
In essence this is as simple as adding `-DENABLE_PYTHON=ON`. However, VTK (or
ParaView when used with Catalyst) needs to be built with Python enabled, and
NumPy, mpi4py, and SWIG are required. Note that there are some caveats when
used with Catalyst and Libsim. These are described in more detail in the Newton
mini app [README](miniapps/newton/README.md).

## Using the SENSEI library
To use SENSEI from your CMake based project include the SENSEI CMake config in
your CMakeLists.txt.
```cmake
find_package(SENSEI REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec sensei ...)
```
Additionally, your source code may need to include `senseiConfig.h` to capture
compile time configuration.

Included Software and Software Dependencies
-------------------------------------------
The SENSEI framework includes the following software:

* [DIY2](https://github.com/diatomic/diy), Copyright (c) 2015, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy).
* [{fmt}](https://github.com/fmtlib/fmt), Copyright (c) 2012-2016, Victor Zverovich.
* [pugixml](https://github.com/zeux/pugixml), Copyright (c) 2006-2016 Arseny Kapoulkine.

The SENSEI framework makes use of (links to) the following software:
* [ADIOS 1](https://www.olcf.ornl.gov/center-projects/adios/), Copyright (c) 2008 - 2009.
  UT-BATTELLE, LLC. Copyright (c) 2008 - 2009.  Georgia Institute of Technology.
* [ParaView/Catalyst](https://gitlab.kitware.com/paraview/paraview), Copyright (c) 2005-2008 Sandia Corporation, Kitware Inc.
  Sensei requires ParaView v5.5.1 or later when `ENABLE_CATALYST` is on
  and a development version (v5.6.0 or later) when both `ENABLE_CATALYST` and `ENABLE_VTKM` are on.
* [VisIt/libsim](http://visit.llnl.gov), Copyright (c) 2000 - 2016, Lawrence Livermore National Security, LLC.
* [VTK](https://gitlab.kitware.com/vtk/vtk), Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen.
  Sensei can use VTK provided separately or the VTK included with
  VisIt/libSim (VTK v6.1 when `ENABLE_LIBSIM` is on) or
  ParaView/Catalyst (VTK v8.1 when `ENABLE_CATALYST` is on).
  If VTK is provided separately and both `ENABLE_VTK` and `ENABLE_VTKM` are on,
  Sensei requires a development version (VTK v9.0 or later).
* [VTKm](https://gitlab.kitware.com/vtk/vtkm), Copyright (c) 2014-2018 NTESS, SNL, LANL, UT-Battelle, Kitware, UC Davis.
  A development version is currently required as packaging infrastructure has recently changed.

For full license information regarding included and used software please refer
to the file [THIRDPARTY_SOFTWARE_LICENSES](THIRDPARTY_SOFTWARE_LICENSES).
