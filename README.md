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
e.g. `3D_Grid` is set up to compute histograms while `oscillators` is set up to
do an autocorrelation analysis.

When **ENABLE_SENSEI**, the miniapps take in an configuration xml that is used
to configure analysis via Sensei infrastructure. Looks at the
[3dgrid.xml](configs/3dgrid.xml), [oscillator.xml](configs/oscillator.xml) and
[adiosendpoint.xml](configs/adiosendpoint.xml) for examples of these config
files.

Let's look at the various CMake flags and how they affect the generated build.

* **ENABLE_SENSEI**: (ON/OFF) Set to ON to enable `Sensei`.  Thus,
when **ENABLE_SENSEI** is OFF, miniapps will only support the one insitu analysis routine
they were hardcoded for. If ON, you will need to set the **VTK_DIR** to point to an existing VTK build since
`Sensei` depends on `vtkCommonDataModel` module for its data model and adaptor classes.

* **VTK_DIR**: Path to `VTKConfig.cmake` file. When building with **libsim** or **Catalyst**,
you should point this to the VTK build used/included by the two frameworks.

* **VTK_HAS_GENERIC_ARRAYS**: Set to ON if you have a custom VTK build with Generic Array support. The
current Sensei Histogram implementation uses generic arrays API and hence is not built unless
this is set to ON.

You can choose which miniapp to build using the following flags.

* **ENABLE_PARALLEL3D**: (ON/OFF) Set to ON to build the `parallel_3d` miniapp from miniapp campaign #1.
This miniapp can do histogram analysis if **ENABLE_SENSEI** is OFF. If **ENABLE_SENSEI** is ON, this will use the Sensei
bridge along with data and analysis adaptors to do the analysis specified in the configuration XML.

* **ENABLE_OSCILLATORS**: (ON/OFF) Set to ON to build the `oscillators` miniapp from miniapp campaign #2.
If **ENABLE_SENSEI** is OFF, this miniapp can do autocorrelation analysis alone. If **ENABLE_SENSEI** is ON, this miniapp supports the histogram,
autocorrelation, catalyst-slice as specified in a configuration xml.

To use analysis routines from Catalyst, you can use the following flags.

* **ENABLE_CATALYST**: (ON/OFF) Set to ON to enable analysis routines that use Catalyst. This option is
only available when **ENABLE_SENSEI** is ON. This builds an analysis adaptor for Sensei that invokes Catalyst calls
to do the data processing and visualization. When set to ON, you will have to point **ParaView_DIR** to a ParaView (or Catalyst) build
containing ParaViewConfig.cmake file.

* **ENABLE_ADIOS**: (ON/OFF) Set to ON to enable ADIOS components. When enabled,
this generates a **ADIOSAnalysisEndPoint** that can be used as an endpoint components
that reads data being serialized by the ADIOS analysis adaptor and pass it back
into a `Sensei` bridge for further analysis. **ADIOSAnalysisEndPoint** itself can be given
configuration XML for select analysis routines to run via Sensei infrastructure.

Typical build usage:

    make build
    cd build
    ccmake .. # set cmake options as needed
    make
    
All executables will be generated under `bin`.

Miniapps
---------
Details on each of the miniapps are as follows:

* [Parallel3D](miniapps/parallel3d/README.md)
* [Oscillators](miniapps/oscillators/README.md)


