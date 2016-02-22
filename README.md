SENSEI
======

This is the `Sensei` repository for the code for all the miniapps and analysis
routines.

Directory Structure
-------------------
    - CMake/
    - core/
    - analysis/
        - autocorrelation/
        - histogram/
        - configurable/
    - infrastuctures/
        - adios/
        - catalyst/
    - miniapps/
        - oscillators/
        - parallel3d/
    - utils/
        - diy/
        - grid/
        - opts/
        - pugixml/
     - configs/

Build instructions
---------------------

The project uses CMake 3.0 or later. The options provided allow you to choose
which of the miniapps to build as well as which frameworks to enable.

When **ENABLE_SENSEI** is off, none of the insitu frameworks are useable.
However, miniapps are trypically instrumented with a prototype analysis code
e.g. **3D_Grid** is set up to compute histograms while **oscillators** is set up to
do an autocorrelation analysis.

When **ENABLE_SENSEI**, the miniapps take in an configuration xml that is used
to configure analysis via Sensei framework. Looks at the
[3dgrid.xml](configs/3dgrid.xml), [oscillator.xml](configs/oscillator.xml) and
[adiosendpoint.xml](configs/adiosendpoint.xml) for examples of these config
files.

Miniapps have
been instrumented to support certain analyses that have been enabled as decribed
later. Let's look at the various CMake flags and how they affect the generated build.

* **ENABLE_SENSEI**: (ON/OFF) Set to ON to enable `Sensei`. Several analysis routines
e.g. histogram, autocorrelation provide an implementation that does not use `Sensei`.
All miniapps were written to work directly with at least one of these raw implementations. Thus,
when **ENABLE_SENSEI** is OFF, miniapps will these raw analysis implementations that they were
coded with. If ON, you will need to set the **VTK_DIR** to point to an existing VTK build since
`Sensei` depends on `vtkCommonDataModel` module for its data model and adaptor classes.

* **VTK_DIR**: Path to `VTKConfig.cmake` file. When building with **libsim** or **Catalyst**,
you should point this to the VTK build used/included by the two frameworks.

You can choose which analysis routines to compile using the following flags.

* **ENABLE_HISTOGRAM**: (ON/OFF) Set to ON to enable the histogram analysis routine. If
**ENABLE_SENSEI** is ON, then this will build an implementation of the histogram analaysis that
uses *Generic arrays* along with a *Analysis Adaptor* for Sensei. Since one would need a special build
of VTK (until generic arrays land in VTK/ParaView repositories), you should disable this if using an older
VTK.

* **ENABLE_AUTOCORRELATION**: (ON/OFF) Set to ON to enable the autocorrelation routine.
If **ENABLE_SENSEI** is on, then this will also build the corresponding *analysis adaptor* for Sensei.

You can choose which miniapp to build using the following flags.

* **ENABLE_PARALLEL3D**: (ON/OFF) Set to ON to build the `parallel_3d` miniapp from miniapp campaign #1.
This miniapp can do histogram analysis is **ENABLE_HISTOGRAM** is ON. If **ENABLE_SENSEI** is ON, this will use the Sensei
bridge along with data and analysis adaptors to do the histogram.

* **ENABLE_OSCILLATORS**: (ON/OFF) Set to ON to build the `oscillators` miniapp from miniapp campaign #2.
If **ENABLE_SENSEI** is OFF, this miniapp can do autocorrelation analysis alone if enabled
(i.e. **ENABLE_AUTOCORRELATION** is ON). If **ENABLE_SENSEI** is ON, this miniapp supports the histogram,
autocorrelation, catalyst-slice via the `Sensei` bridge. Of course, each of the analysis routines need to be
enabled using the corresponding **ENABLE_*** flags.

To use analysis routines from Catalyst, you can use the following flags.

* **ENABLE_CATALYST**: (ON/OFF) Set to ON to enable analysis routines that use Catalyst. This option is
only available when **ENABLE_SENSEI** is ON. This builds an analysis adaptor for Sensei that invokes Catalyst calls
to do the data processing and visualization. When set to ON, you will have to point **ParaView_DIR** to a ParaView (or Catalyst) build
containing ParaViewConfig.cmake file. After enabling, you can explcitly enable/disable individual analysis routines that use Catalyst using the
**ENABLE_CATALYST_*** options.

* **ENABLE_CATALYST_SLICE**: (ON/OFF) Set to ON to enable a simple Catalyst analysis and visualization pipeline that slices the
dataset and then renders it using pseudo coloring. This option is only available when **ENABLE_CATALYST** is ON.

* **ENABLE_ADIOS**: (ON/OFF) Set to ON to enable ADIOS components. When enabled,
this generates a **ADIOSAnalysisEndPoint** that can be used as an endpoint components
that reads data being serialized by the ADIOS analysis adaptor and pass it back
into a `Sensei` bridge for further analysis.

The following table makes it easier to understand the current status of what analyis routines are supported
by which app.

| App | Histogram (B) | Histogram (S) | Autocorrelation (B) | Autocorrelation (S) | Slice (C) | ADIOS Serialization (A) |
|-----|:-------------:|:-------------:|:-------------------:|:-------------------:|:---------:|:-----------------------:|
| 3D_Grid / Parallel3D | x   | x      |                     |                     | x         | x                       |
| Oscillators |       | x             | x                   | x                   | x         |                         |
| ADIOSAnalysisEndPoint | | x         |                     | x                   | x         |                         |

Legend:
* (B) : basic i.e. used when ENABLE_SENSEI is OFF
* (S) : requires ENABLE_SENSEI to be ON
* (C) : requires ENABLE_SENSEI and ENABLE_CATALYST to be ON
* (A) : requires ENABLE_SENSEI and ENABLE_ADIOS to be ON



Typical build usage:

    make build
    cd build
    ccmake .. # set cmake options as needed
    make
    
Miniapp executables will be generated under `bin`.

Miniapps
---------
Details on each of the miniapps are as follows:

* [Parallel3D](miniapps/parallel3d/README.md)
* [Oscillators](miniapps/oscillators/README.md)


