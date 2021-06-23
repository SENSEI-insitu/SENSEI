This package includes information about changes in SVTK since last release
and utilities for upgrading to SVTK 4.0.

Two pdf files and two perl scripts are included:

FieldDataChanges.pdf:
 This document discusses changes to svtkDataSetAttributes, svtkPointData,
 svtkCellData and svtkFieldData.

AttributeChanges.pdf:
 This document discusses changes to the way SVTK handles attributes.
 It focuses on the removal of svtkAttributeData and it's subclasses
 (svtkScalars, svtkVectors, svtkNormals, svtkTCoords, svtkTensors).

DiagAttribute.pl :
 This script tries to find deprecated attribute data classes and
 methods and warns the user whenever it finds them. It also suggests
 possible modification to bring code up to date.

UpgradeFrom32.pl:
 This script tries to find deprecated classes and methods and replace
 them with new classes/methods. Please note that it can not fix all
 possible problems. However, it should be relatively easy to trace
 those problems from compilation errors.


Here is the related entry from SVTK FAQ at
http://public.kitware.com/cgi-bin/svtkfaq :

6.7. Changes in SVTK since 3.2

* Changes to svtkDataSetAttributes, svtkFieldData and svtkDataArray: All attributes (scalars, vectors...) are now stored in the field data as svtkDataArray's. svtkDataSetAttributes became a sub-class of svtkFieldData. For backwards compatibility, the interface which allows setting/getting the attributes the old way (by passing in a sub-class of svtkAttributeData such as svtkScalars) is still supported but it will be removed in the future. Therefore, the developers should use the new interface which requires passing in a svtkDataArray to set an attribute. svtkAttributeData and it's sub-classes (svtkScalars, svtkVectors...) will be deprecated in the near future; developers should use svtkDataArray and it's sub-classes instead. We are in the process of removing the use of these classes from svtk filters.

* Subclasses of svtkAttributeData (svtkScalars, svtkVectors, svtkNormals, svtkTCoords, svtkTensors) were removed. As of SVTK 4.0, svtkDataArray and it's sub-classes should be used to represent attributes and fields. Detailed description of the changes and utilities for upgrading from 3.2 to 4.0 can be found in the package http://public.kitware.com/SVTK/files/Upgrading.zip.

* Improved support for parallel visualization: svtkMultiProcessController and it's sub-classes have been re-structured and mostly re-written. The functionality of svtkMultiProcessController have been re-distributed between svtkMultiProcessController and svtkCommunicator. svtkCommunicator is responsible of sending/receiving messages whereas svtkMultiProcessController (and it's subclasses) is responsible of program flow/control (for example processing rmi's). New classes have been added to the Parallel directory. These include svtkCommunicator, svtkMPIGroup, svtkMPICommunicator, svtkSharedMemoryCommunicator, svtkMPIEventLog... Examples for C++ can be found in the examples directories.

* svtkSocketCommunicator and svtkSocketController have been added. These support message passing via BSD sockets. Best used together with input-output ports.

* svtkIterativeClosestPointTransform has been added. This class is an implementation of the ICP algorithm. It matches two surfaces using the iterative closest point (ICP) algorithm. The core of the algorithm is to match each vertex in one surface with the closest surface point on the other, then apply the transformation that modify one surface to best match the other (in a least square sense).

* The SetFileName, SaveImageAsPPM and related methods in svtkRenderWindow have been removed. svtkWindowToImageFilter combined with any of the image writers provides greater functionality.

* Support for reading and writing PGM and JPEG images has been included.



