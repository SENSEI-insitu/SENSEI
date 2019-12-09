.. _data_model:
Data model
==========
The data model is a key piece of the system. It allows data to be packaged and
shared between simulations and analysis back ends.  SENSEI's data model relies
on VTK's `vtkDataObject`_ class hierarchy to provide containers of array based
data, VTK's conventions for mesh based data (i.e. ordering of FEM cells), and
our own metadata object that is used to describe simulation data and it's
mapping onto hardware resources.

Representing mesh based data
----------------------------
SENSEI makes use of VTK data object's to represent simulation data. VTK
supports a diverse set of mesh and non-mesh based data. Figure
`numref`:data_types shows a subset of the types of data supported in the VTK
data model.

.. data_types:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/data_types.png
   :width: 65 %
   :align: center

   A subset of the supported data types.

A key concept in understanding our use of VTK is that we view all data
conceptually as multi-block. By multi-block we mean that each MPI rank has zero
or more blocks of data. When we say blocks we really mean chunks or pieces,
because the blocks can be anything ranging from point sets, to FEM cells, to
hierarchical AMR data. to tables, to arrays. The blocks of a multi-block are
distributed across the simulation's MPI ranks with each rank owning a subset of
the blocks. An example is depicted in figure `numref`:multi_block where the 2
data blocks of a multi-block dataset are partitioned across 2 MPI ranks.

.. multi_block:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/multi_block.png
   :width: 65 %
   :align: center

   Multi-block data. Each rank has zero or more data blocks. In VTK non-local blocks are nullptr's.

A strength of VTK is the diversity of data sets that can be represented. A
challenge that comes with this lies in VTK's complexity. SENSEI's data model
only relies on VTK's common, core and data libraries reducing surface area and
complexity when dealing with VTK. While it is possible to use any class derived
from `vtkDataObject`_ with SENSEI the following data sets are supported
universally by all transports and analysis back-ends.

+-------------------------+--------------------------------------------------------------------+
| VTK Class               | Description                                                        |
+-------------------------+--------------------------------------------------------------------+
| `vtkImageData`_         | Blocks of uniform Cartesian geometry                               |
+-------------------------+--------------------------------------------------------------------+
| `vtkRectilinearGrid`_   | Blocks of stretched Cartesian geometry                             |
+-------------------------+--------------------------------------------------------------------+
| `vtkUnstructuredGrid`_  | Blocks of finite element method cell zoo and particle meshes       |
+-------------------------+--------------------------------------------------------------------+
| `vtkPolyData`_          | Blocks of particle meshes                                          |
+-------------------------+--------------------------------------------------------------------+
| `vtkStructuredGrid`_    | Blocks of logically Cartesian (aka Curvilinear) geometries         |
+-------------------------+--------------------------------------------------------------------+
| `vtkOverlappingAMR`_    | A collection of blocks in a block structured AMR hierarchy         |
+-------------------------+--------------------------------------------------------------------+
| `vtkMultiBlockDataSet`_ | A collection of data blocks distributed across MPI ranks           |
+-------------------------+--------------------------------------------------------------------+

.. _vtkDataObject: https://vtk.org/doc/nightly/html/classvtkDataObject.html
.. _vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html
.. _vtkRectilinearGrid: https://vtk.org/doc/nightly/html/classvtkRectilinearGrid.html
.. _vtkUnstructuredGrid: https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
.. _vtkPolyData: https://vtk.org/doc/nightly/html/classvtkPolyData.html
.. _vtkStructuredGrid: https://vtk.org/doc/nightly/html/classvtkStructuredGrid.html
.. _vtkOverlappingAMR: https://vtk.org/doc/nightly/html/classvtkOverlappingAMR.html
.. _vtkMultiBlockDataSet: https://vtk.org/doc/nightly/html/classvtkMultiBlockDataSet.html

As mentioned VTK's data model is both rich and complex. VTK's capabilities go
well beyond SENSEI's universal support. However, any dataset type derived from
`vtkDataObject`_ can be used with SENSEI including those not listed in the table
above. The successful use of classes not listed in the above table depends on
support implemented by the back end or transport in question.

Representing array based data
-----------------------------
Each block of a simulation mesh is expected to contain one or more data arrays
that hold scalar, vector, and tensor fields generated by the simulation.
VTK's data arrays are used to present array based data. VTK's data arrays are
similar to the STL's std::vector, but optimized for high-performance computing.
One such optimization is the support for zero-copy data transfer.  With
zero-copy data transfer it is possible to pass a pointer to simulation data
directly to an analysis back-end without making a copy of the data.

All of the mesh based types in VTK are derived from `vtkDataSet`_.
`vtkDataSet`_ defines the common API's for accessing collections of VTK data
arrays by geometric centering. SENSEI supports the following two containers in
all back-ends and transports.

+-----------------+------------------------------------+
| Class           | Description                        |
+-----------------+------------------------------------+
| `vtkPointData`_ | Container of node centered arrays  |
+-----------------+------------------------------------+
| `vtkCellData`_  | Container of cell centered arrays  |
+-----------------+------------------------------------+

.. _vtkDataSet: https://vtk.org/doc/nightly/html/classvtkDataSet.html
.. _vtkPointData: https://vtk.org/doc/nightly/html/classvtkPointData.html
.. _vtkCellData: https://vtk.org/doc/nightly/html/classvtkCellData.html

VTK data arrays support use of any C++ POD type. The two main classes of VTK
data arrays of interest here are:

+-----------------------------+--------------------------------------------------------+
| Class                       | Description                                            |
+-----------------------------+--------------------------------------------------------+
| `vtkAOSDataArrayTemplate`_  | Use with scalar, vector and tensor data in AOS layout  |
+-----------------------------+--------------------------------------------------------+
| `vtkSOADataArrayTemplate`_  | Use with vector and tensor data in SOA layout          |
+-----------------------------+--------------------------------------------------------+

These classes define the API for array based data in VTK. Note the AOS layout
is the default in VTK and that classes such as `vtkFloatArray`_,
`vtkDoubleArray`_, `vtkIntArray`_ etc are aliases to vtkAOSDataArrayTemplate.
For simplicity sake one can and should use these aliases anywhere an AOS layout
is needed.

.. _vtkAOSDataArrayTemplate: https://vtk.org/doc/nightly/html/classvtkAOSDataArrayTemplate.html
.. _vtkSOADataArrayTemplate: https://vtk.org/doc/nightly/html/classvtkSOADataArrayTemplate.html
.. _vtkFloatArray: https://vtk.org/doc/nightly/html/classvtkFloatArray.html
.. _vtkDoubleArray: https://vtk.org/doc/nightly/html/classvtkDoubleArray.html
.. _vtkIntArray: https://vtk.org/doc/nightly/html/classvtkIntArray.html

Zero-copy into VTK
^^^^^^^^^^^^^^^^^^
The following snippet of code shows how to pass a 3 component vector field in
the AOS layout from the simulation into VTK using the zero-copy mechanism:

.. code-block:: cpp

    // VTK's default is AOS, no need to use vtkAOSDataArrayTemplate
    vtkDoubleArray *aos = vtkDoubleArray::New();
    aos->SetNumberOfComponents(3);
    aos->SetArray(v, 3*nxy, 0);
    aos->SetName("velocity");

    // add the array as usual
    im->GetPointData()->AddArray(aos);

    // give up our reference
    aos->Delete();

The following snippet of code shows how to pass a 3 component vector field in
the SOA layout from the simulation into VTK using the zero-copy mechanism:

.. code-block:: cpp

    // use the SOA class
    vtkSOADataArrayTemplate<double> *soa = vtkSOADataArrayTemplate<double>::New();
    soa->SetNumberOfComponents(3);

    // pass a pointer for each array
    soa->SetArray(0, vx, nxy, true);
    soa->SetArray(1, vy, nxy);
    soa->SetArray(2, vz, nxy);
    soa->SetName("velocity");

    // add to the image as usual
    im->GetPointData()->AddArray(soa);

    // git rid of our reference
    soa->Delete();

In both these examples 'im' is a dataset for some block in a multiblock data set.

Accessing blocks of data
------------------------
This section pertains to accessing data for analysis. During analysis one may
obtain a mesh from the simulation. With the mesh in hand one can walk the
blocks of data and access the array collections. Arrays in the array collection
are accessed and a pointer to the data is obtained for processing. The
collections of blocks in VTK are derived from `vtkCompositeDataSet`_.
`vtkCompositeDataSet`_ defines the API for generically access blocks via the
`vtkCompositeDataIterator`_ class. The `vtkCompositeDataIterator`_ is used to
visit all data blocks local to the MPI rank.

.. _vtkCompositeDataSet: https://vtk.org/doc/nightly/html/classvtkCompositeDataSet.html
.. _vtkCompositeDataIterator: https://vtk.org/doc/nightly/html/classvtkCompositeDataIterator.html

Getting help with VTK
---------------------
For those new to VTK a good place to start is the `VTK user guide`_  which
contains a chapter devoted to learning VTK data model as well as numerous
examples. On the `VTK community support`_ forums volunteers, and often the VTK
developers them selves, answer questions in an effort to help new users.

.. _VTK User Guide: https://vtk.org/vtk-users-guide/
.. _VTK community support: https://vtk.org/community-support/

Metadata
--------
SENSEI makes use of a custom metadata object to describe simulation data and
its mapping onto hardware resources. This is in large part to support in transit
operation where one must make decisions about how simulation data maps onto
available analysis resources prior to accessing the data.

+-----------------+--------------------+-------------------------------------------------------------------+
| Applies to      | Field name         | Purpose                                                           |
+=================+====================+===================================================================+
| **entire mesh** | GlobalView         | tells if the information describes data on this rank or all ranks |
|                 +--------------------+-------------------------------------------------------------------+
|                 | MeshName           | name of mesh                                                      |
|                 +--------------------+-------------------------------------------------------------------+
|                 | MeshType           | VTK type enum of the container mesh type                          |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockType          | VTK type enum of block mesh type                                  |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumBlocks          | global number of blocks                                           |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumBlocksLocal     | number of blocks on each rank                                     |
|                 +--------------------+-------------------------------------------------------------------+
|                 | Extent             | global index space extent :math:`^{\dagger,\S,*}`                 |
|                 +--------------------+-------------------------------------------------------------------+
|                 | Bounds             | global bounding box :math:`^*`                                    |
|                 +--------------------+-------------------------------------------------------------------+
|                 | CoordinateType     | type enum of point data :math:`^\ddagger`                         |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumPoints          | total number of points in all blocks :math:`^*`                   |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumCells           | total number of cells in all blocks :math:`^*`                    |
|                 +--------------------+-------------------------------------------------------------------+
|                 | CellArraySize      | total cell array size in all blocks :math:`^*`                    |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumArrays          | number of arrays                                                  |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumGhostCells      | number of ghost cell layers                                       |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumGhostNodes      | number of ghost node layers                                       |
|                 +--------------------+-------------------------------------------------------------------+
|                 | NumLevels          | number of AMR levels  (AMR)                                       |
|                 +--------------------+-------------------------------------------------------------------+
|                 | PeriodicBoundary   | indicates presence of a periodic boundary                         |
|                 +--------------------+-------------------------------------------------------------------+
|                 | StaticMesh         |  non zero if the mesh does not change in time                     |
+-----------------+--------------------+-------------------------------------------------------------------+
| **each array**  | ArrayName          |  name of each data array                                          |
|                 +--------------------+-------------------------------------------------------------------+
|                 | ArrayCentering     |  centering of each data array                                     |
|                 +--------------------+-------------------------------------------------------------------+
|                 | ArrayComponents    |  number of components of each array                               |
|                 +--------------------+-------------------------------------------------------------------+
|                 | ArrayType          |  VTK type enum of each data array                                 |
|                 +--------------------+-------------------------------------------------------------------+
|                 | ArrayRange         |  global min,max of each array :math:`^*`                          |
+-----------------+--------------------+-------------------------------------------------------------------+
| **each block**  | BlockOwner         |  rank where each block resides :math:`^*`                         |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockIds           |  global id of each block :math:`^*`                               |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockNumPoints     |  number of points for each block :math:`^*`                       |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockNumCells      |  number of cells for each block :math:`^*`                        |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockCellArraySize |  cell array size for each block :math:`^{\ddagger,*}`             |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockExtents       |  index space extent of each block :math:`^{\dagger,\S,*}`         |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockBounds        |  bounds of each block :math:`^*`                                  |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockLevel         |  AMR level of each block :math:`^\S`                              |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlockArrayRange    |  min max of each array on each block :math:`^*`                   |
+-----------------+--------------------+-------------------------------------------------------------------+
| **each level**  | RefRatio           +  refinement ratio in i,j, and k direction :math:`^\S`             |
|                 +--------------------+-------------------------------------------------------------------+
|                 | BlocksPerLevel     +  number of blocks in each level :math:`^\S`                       |
+-----------------+--------------------+-------------------------------------------------------------------+

The metadata structure is intended to be descriptive and cover all of the
supported scenarios. Some of the fields are potentially expensive to generate
and not always needed. As a result not all fields are used in all scenarios.
Flags are used by the analysis to specify which fields are required. The
following table is used in conjunction with the above table to define under
which circumstances the specific the fields are required.

+--------------------+-----------------------------------+
| symbol             | required ...                      |
+--------------------+-----------------------------------+
|                    | always required                   |
+--------------------+-----------------------------------+
| :math:`*`          | only if requested by the analysis |
+--------------------+-----------------------------------+
| :math:`\dagger`    | with Cartesian meshes             |
+--------------------+-----------------------------------+
| :math:`\ddagger`   | with unstructured meshes          |
+--------------------+-----------------------------------+
| :math:`\S`         | with AMR meshes                   |
+--------------------+-----------------------------------+

Simulations are expected to provide local views of metadata, and can optionally
provide global views of metadata. The GlobalView field is used to indicate
which is provided. SENSEI contains utilities to generate a global view form a
local one.

Ghost zone and AMR mask array conventions
-----------------------------------------
SENSEI uses the conventions defined by VisIt and recently adopted by VTK and
ParaView for masking ghost zones and covered cells in overlapping AMR data.
In accordance with VTK convention these arrays must by named vtkGhostType.

Mask values for cells and cell centered data:

+--------------------------------------+-----+
| Type                                 | Bit |
+--------------------------------------+-----+
| valid cell, not masked               | 0   |
+--------------------------------------+-----+
| Enhanced connectivity zone           | 1   |
+--------------------------------------+-----+
| Reduced connectivity zone            | 2   |
+--------------------------------------+-----+
| Refined zone in AMR grid             | 3   |
+--------------------------------------+-----+
| Zone exterior to the entire problem  | 4   |
+--------------------------------------+-----+
| Zone not applicable to problem       | 5   |
+--------------------------------------+-----+

Mask values for points and point centered data:

+--------------------------------------+-----+
| Type                                 | Bit |
+--------------------------------------+-----+
| Valid node, not masked               | 0   |
+--------------------------------------+-----+
| Node not applicable to problem       | 1   |
+--------------------------------------+-----+

For more information see the `Kitware blog on ghost cells`_ and the
`VisIt ghost data documentation`_.

.. _Kitware blog on ghost cells: http://www.visitusers.org/index.php?title=Representing_ghost_data
.. _VisIt ghost data documentation: https://blog.kitware.com/ghost-and-blanking-visibility-changes/

Overhead due to the SENSEI data model
--------------------------------------
As in any HPC application we are concerned with the overhead associated with
our design choices. To prove that we have minimal impact on a simulation we did
a series of scaling and performance analyses up to 45k cores on a Cray
supercomputer. We then ran a series of common visualization and analysis tasks
up to 1M cores on second system. The results of our experiments that showed the
SENSEI API and data model have negligible impact on both memory use and
run-time of the simulation. A selection of the results are shown in figure
:numref:`perf`.

.. _perf:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/overheads.png
   :width: 100 %
   :align: center

   Run-time (left) and memory use (right) with (orange) and without (blue) SENSEI.

The full details of the performance and scaling studies can be found in our `SC16 paper`_.

.. _SC16 paper: https://dl.acm.org/citation.cfm?id=3015010

