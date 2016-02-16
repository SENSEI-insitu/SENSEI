SENSEI
======

This is the Sensei repository for the code for all the miniapps and analysis
routines.

Directory Structure
-------------------

Add notes about directory structure.

CMake Variables
---------------------

 Variable        | Description                                                 
-----------------|-------------------------------------------------------------
 ENABLE_SENSEI   | Enable building Sensei infrastructure. It's possible to build and run all the miniapps without using Sesei infrastructure. If enabled, all the miniapps will use Sensei data adaptor and analysis adaptors for the enabled analyses. Note, when Sensei infrastructure is not enabled miniapps only execute the analyses they were explicitly coded for. Miniapps generally support more analyses when Sensei is enabled.
 ENABLE_HISTOGRAM| Enable histogram analysis for miniapps that support it.
 ENABLE_AUTOCORRELATION | Enable autocorrelation analysis for miniapps that support it.
 ENABLE_OSCILLATORS | Enable oscillators miniapp. When Sensei is not enabled, this miniapp is hardcoded to do autocorrelation analysis, if enabled.
 ENABLE_PARALLEL3D | Enable parallel_3d miniapp.
 VTK_DIR           | When ENABLE_SENSEI is ON, this variable must be set to the directory containig the `VTKConfig.cmake` file. 
 
 

 
 
