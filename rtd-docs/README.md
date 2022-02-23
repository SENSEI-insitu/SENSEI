### SENSEI infrastructure for generic in situ
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

For more
information see our [SC16 paper](http://dl.acm.org/citation.cfm?id=3015010).

### Source code
SENSEI is open source and freely available on github at https://github.com/SENSEI-insitu/SENSEI

### Data model
SENSEI makes use of a heavily stripped down and mangled version of the
[VTK](https://vtk.org) 9.0.0 data model.

### Instrumenting a simulation
Instrumenting a simulation typically involves creating and initializing an
instance of sensei::ConfigurableAnalysis with an XML file and passing a
simulation specific sensei::DataAdaptor when in situ processing is invoked.

| Class | Description |
| ----- | ----------- |
| sensei::ConfigurableAnalysis | uses a run time provided XML file to slect and confgure one or more library specific data consumers or in transit transports |
| sensei::DataAdaptor | simulations implement an instance that packages simulation data into SENSEI's data model |

### In situ data processing
SENSEI comes with a number of ready to use in situ processing options. These include:

| Class | Description |
| ----- | ----------- |
| sensei::AnalsysiAdaptor | base class for in situ data processors |
| sensei::DataAdaptor | defines the API by which data processors fetch data from the simulation |
| sensei::Histogram | Computes histograms |
| sensei::AscentAnalysisAdaptor | Processes simulation data using Ascent |
| sensei::CatalystAnalysisAdaptor | Processes simulation data using ParaView Catalyst |
| sensei::LibsimAnalysisAdaptor | Processes simulation data using VisIt Libsim |
| sensei::Autocorrelation | Compute autocorrelation of simulation data over time |
| sensei::VTKPosthocIO | Writes simulation data to disk in a VTK format |
| sensei::VTKAmrWriter | Writes simulation data to disk in a VTK format |
| sensei::PythonAnalysis | Invokes user provided Pythons scripts that process simulation data |
| sensei::SliceExtract | Computes planar slices and iso-surfaces on simulation data |

### User defined in situ processing
A unique feature of SENSEI is the ability to invoke user provided code written
in Python or C++ on a SENSEI instrumented simulation. This makes SENSEI much
easier to extend and customize than other solutions.

| Class | Description |
| ----- | ----------- |
| sensei::AnalysisAdaptor | used to invoke user defined C++ code. Override the sensei::AnalysisAdaptor::Execute method.  |
| sensei::PythonAnalysis | used to invoke user defined Python code Implement an Execute function in Python |

For more information see our [ISAV 2018](https://doi.org/10.1145/3281464.3281465) paper.

### In transit data processing
It is often advantageous to move data onto a seperate set of compute nodes for
concurrent processing in a job seperate from the simulation. SENSEI supports
this through run time configurable transports and the SENSEIEndPoint an
application written in C++ that can be configured to receive and process data
while simulation is running.

| Class | Description |
| ----- | ----------- |
| sensei::AnalsysiAdaptor | base class for the write side of the transport |
| sensei::DataAdaptor | base class for the read side of the transport |
| sensei::ConfigurableAnalysis | used to select and configure the write side of the transport at run time from XML |
| sensei::ConfigurableInTransitDataAdaptor | used to configure the read side of the transport at run time from XML |
| sensei::ADIOS2AnalysisAdaptor | The write side of the ADIOS 2 transport |
| sensei::ADIOS2DataAdaptor | The read side of the ADIOS 2 transport |
| sensei::HDF5AnalysisAdaptor | The write side of the HDF5 transport |
| sensei::HDF5DataAdaptor | The read side of the HDF5 transport |
| sensei::Partitioner | base class for data partitioner which maps data to MPI ranks as it is moved |
| sensei::ConfigurablePartitioner | Selects and configures one of the partitioners at run time from XML |
| sensei::BlockPartitioner | maps blocks to ranks such that consecutive blocks share a rank |
| sensei::PlanarPartitioner | Maps blocks to MPI ranks in a round robbin fassion |
| sensei::MappedPartitioner | Maps blocks to MPI ranks using a run time user provided mapping |
| sensei::IsoSurfacePartitioner | Maps blocks to MPI ranks such that blocks not intersecting the iso surface are excluded |
| sensei::PlanarSlicePartitioner | Maps blocks to MPI ranks such that blocks not intersecting the slice are excluded |

For more information see our [EGPGV 2020](https://doi.org/10.2312/pgv.20201073)
and [ISAV 2018](https://doi.org/10.1145/3364228.3364237) papers.

### Mini-apps
SENSEI ships with a number of mini-apps that demonstrate use of the SENSEI
library with custom analyses and the supported in situ infrastructures. When
the SENSEI library is enabled the mini-apps will make use of any of the
supported in situ infrastructures that are enabled. When the SENSEI library is
disabled mini-apps are restricted to the custom analysis such as histogram and
autocorrelation.
