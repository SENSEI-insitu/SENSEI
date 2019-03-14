DataAdaptor API
===============

In SENSEI, the data adaptor provides metadata about which meshes and arrays
a simulation _can_ provide; it also provides those meshes on demand, in
response to requests from analysis endpoints.

Lifecycle
---------

The key event in the lifetime of a `sensei::DataAdaptor` is when it is
passed to an instance of a `sensei::AnalysisAdaptor` at each time step
so that user-configured analyses can be performed.

Many simulations choose to create an instance of the data adaptor and
keep it alive for the duration of the simulation process.
In this case, the bridge simply creates an instance at initialization time,
passes it to an analysis adaptor at each time step, and then destroys it
during finalization.
However, there is no need for the data adaptor to exist between analysis phases.
For simulations that are severely memory-constrained it is possible to
create and destroy an adaptor at each timestep.
In general, it is preferable to simply construct the data adaptor once
at initialization and ensure that the adaptor does not hold significant
resources between time steps.

The next two sections describe the same data adaptor programming interface
in two different contexts:
:ref:`from_sim` (to an analysis), which is written for simulation developers; and
:ref:`from_analysis`, which is written for analysis adaptor developers.

.. _from_sim:

Providing data from a simulation
--------------------------------

SENSEI provides the base `sensei::DataAdaptor` class, which is abstract,
and a concrete subclass named `sensei::VTKDataAdaptor` that you may choose
to use for simple situations.
You may also create a custom subclass of the base class;
this is demonstrated by the oscillators miniapp that comes with SENSEI
and discussed in detail below, in the `Subclassing DataAdaptor`_ section.

Whichever method you choose to create a data adaptor,
you will need to provide your simulation state to the data adaptor
as one or more `vtkDataObject` instances.
The details of how to do this are covered in
:ref:`data_model`, while this section focuses solely on the
data adaptor class.

Using the VTKDataAdaptor
^^^^^^^^^^^^^^^^^^^^^^^^

If your sensei bridge can create fully-populated VTK dataset(s)
representing the simulation state without significant overhead,
then it is probably best to use SENSEI's provided data adaptor.
At each time step, the bridge can simply call

.. code-block:: c++

  vtkDataObject* mesh; // populated with your simulation state
  vtkNew<sensei::VTKDataAdaptor> dataAdaptor;
  dataAdaptor->SetDataObject("mesh", mesh);
  // You may call SetDataObject multiple times if your
  // simulation needs to provide state that is
  // heterogenous in nature.
  analysisAdaptor->Execute(dataAdaptor);

The call to SetDataObject on the adaptor will cause the adaptor to assume
ownership of the mesh by increasing the reference count of the mesh data object.
The analysis adaptor's Execute method instructs the data adaptor
to release all of the data objects it owns.

Subclassing DataAdaptor
^^^^^^^^^^^^^^^^^^^^^^^

If your simulation requires array data to be copied
in order to create a VTK representation of it,
then it is best to subclass the base data adaptor
so that only the portions of data that will
be used by the analysis are copied.

The relevant methods to override are

.. code-block:: c++

  class YourDataAdaptor : public DataAdaptor
  {
  public:
    int GetNumberOfMeshes(
      unsigned int& numMeshes) override;
    virtual int GetMeshMetadata(
      unsigned int id,
      MeshMetadataPtr& metadata) override;
    virtual int GetMesh(
      const std::string& meshName,
      bool structureOnly,
      vtkDataObject*& mesh) override;
    virtual int ReleaseData() override;

    // If your simulation shares information
    // at process boundaries, also override:
    int AddGhostNodesArray(
      vtkDataObject* mesh,
      const std::string& meshName) override;
    int AddGhostCellsArray(
      vtkDataObject* mesh,
      const std::string& meshName) override;
  };

The GetNumberOfMeshes() method simply returns the number of VTK
data objects your simulation requires to represent its state.
Analyses will call this method and loop over the resulting
integer range from 0 up to (but not including) the returned value,
calling GetMeshMetadata() to obtain information about the
structure of and arrays defined on the corresponding simulation state.

The MeshMetaData structure returned by GetMeshMetaData() contains a
description of the simulation state available (see :ref:`data_model` for details).
It also contains a MeshName member that can be used with the GetMesh() method
you must override.
When an analysis determines that it needs access to particular
simulation state (either by explicitly being configured to ask
for a mesh with a given name or by inspecting metadata for
relevant mesh data), it will:

* Call your data adaptor's GetMesh() method to obtain a "bare" VTK data object.
  When you implement this method, return only a minimal object with no data
  arrays (point data, cell data, or field data) provided.
  If the corresponding mesh is a composite data object, you should return an
  object with child objects matching your simulation's structure.
  Ownership of the data object you return is passed to the analysis adaptor;
  you do not need to manage it and it is frequently best not to cache the
  returned object by holding a reference to it yourself.
  The GetMesh() function takes a boolean `structureOnly` argument that,
  when true, indicates your analysis does not need explicit cell connectivity.
  When your simulation is returning an unstructured data object such as
  polydata or an unstructured grid, this flag indicates you need not add cells
  to the object.
* Call your data adaptor's AddGhostNodesArray() and/or AddGhostCellsArray()
  methods to ensure that additional point- and/or cell-data arrays are added
  if your simulation duplicates values at shared processor boundaries.
  See :ref:`data_model` for details on what values should be in these
  arrays if your simulation needs to provide them.
* If arrays mentioned in the mesh metadata are required for the analysis,
  then the analysis will call your adaptor's AddArray() method with the
  mesh provided by the GetMesh() call above plus a description of the array
  that it needs.
  This method may be called multiple times (once for each array needed by
  an analysis).
  This method should translate the given array into a vtkAbstractArray
  instance and add it to the data object passed to AddArray().
  As with the mesh objects themselves, ownership is passed from your
  adaptor to the mesh object provided to the AddArray method; once
  the mesh is deleted, the array will automatically be deleted.
  Don't worry: this will not delete your simulation state unless you specifically
  instruct the VTK data array that it owns your simulation state's memory.
* Perform its analysis on the data object.
* Call ReleaseData() on your data adaptor to indicate that any
  memory your adaptor has allocated should be released.
  This method is not usually required since the data objects themselves
  use VTK's reference counting and will be deleted by the analysis adaptor
  just before its Execute() method completes.

.. _from_analysis:

Fetching data for an analysis
-----------------------------

When writing an analysis adaptor, you will need to fetch simulation
state from the data adaptor your Execute() method is provided.
The data adaptor provides methods named GetNumberOfMeshes() and
GetMeshMetadata() that allow you to query a data adaptor for all of
a simulation's state.
See :ref:`data_model` for details about the mesh metadata that these
methods return.
However, a typical pattern is for analyses to be configured with the
names of mesh and arrays to use.

However your adaptor determines which meshes it needs,
you should fetch the mesh from the data adaptor by calling
GetMesh() with the mesh's name.
This function also takes a boolean `structureOnly` argument that,
when true, indicates your analysis does not need explicit cell connectivity.
An example of this is the histogram analysis provided with SENSEI;
since it only iterates over the data arrays attached as point-, cell-, or
field-data and does not use cell connectivity, it calls GetMesh() with
`structureOnly` set to true.

The mesh returned by GetMesh() will not have any point-, cell-, or field-data
arrays added.
In order to obtain them, you must call the data adaptor's AddArray() method
once for each array you need.
This way, simulations that must copy memory to adapt to VTK's data structure
will only copy what is absolutely required.

Because many simulation processes hold point or cell information on boundaries
shared with other processes, you should be careful to call the data adaptor's
AddGhostNodesArray() and/or AddGhostCellsArray() as needed so that you don't
bias your analysis results by processing the same point or cell multiple times.

Once your analysis is complete, be sure to

* Call the Delete() method on each vtkDataObject returned by the data adaptor's GetMesh() method.
* Call the data adaptor's ReleaseData() method so it can recover any other memory it allocated
  for analysis purposes.
