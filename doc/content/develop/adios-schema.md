---
markdown:
  gfm: false
---
# SENSEI dataset schema and ADIOS

VTK has two ways of representing parallel distributed data. In the first each
MPI rank has a single object derived from
[vtkDataSet](https://www.vtk.org/doc/nightly/html/classvtkDataSet.html). We
will call this the legacy approach. In the second case each MPI rank has a
single object derived from
[vtkCompositeDataSet](https://www.vtk.org/doc/nightly/html/classvtkCompositeDataSet.html),
which contains any number of local and remote datasets. We will call this the
composite approach. We treat the legacy approach as a special case of the
composite approach. Use of a data object type enumeration at the top level
enables differentiation between the two cases.

Each VTK object that can be serialized has a corresponding serilaizer derived
from [senseiADIOS::Schema](#senseiadiosschema). Serializers are expected to serialize data and metadata
needed to represent the object, or pass the object to lower level serializers
to accomplish this task. As composite data objects are traversed each high
level serializer class is given the chance to serialize each leaf dataset. When
they are passed a dataset that they can't handle they will ignore it.

At the highest level we provide serialization of collections of VTK data objects,
each of which can be a composite data object containing any number of datasets or
nested composite objects. Unique id's are given to each object in the collection,
and also each dataset in the object. Top level data objects are given a unique id,
called a `doid`, while nested datasets make use of the so called flat index provided by
VTK, called a `dsid`. For example the first dataset in the second object is identified by
the path:
```
data_object_1/dataset_0
```

The remainder of this document details each serializer, and what it writes to
the ADIOS file/stream.

# senseiADIOS::Schema
This is the base class defining API to serialize/deserialize VTK collections of
VTK data objects using in ADIOS. The [senseiADIOS::Schema](#senseiadiosschema) class declares API to
accomplish steps involved in writing/reading data with ADIOS. A common theme
when dealing with parallel distributed VTK datasets is traversing the composite
objects and operating on leaf datasets. Thus the class provides default
implementation for the traversal of composite datasets leaving derived classes
to implement an override to process leaf datasets.

# senseiADIOS::DataObjectCollectionSchema
This class serializes/deserializes collections of
[vtkDataOject](https://www.vtk.org/doc/nightly/html/classvtkDataObject.html)s
and global metadata such as object names, time, time step, and schema version.
Each object in the collection is serialized by
[senseiADIOS::DataObjectSchema](#senseiadiosdataobjectschema).

### writes/reads
 path | description
 ---  | ---
`SENSEIDataObjectSchema` | schema revision, unsigned int
`time` | current simulation time, double
`time_step` | current simulation step, double
`number_of_data_objects` | number of objects serialized, integer
`data_object_<doid>/name` | the name of each object, string

# senseiADIOS::DataObjectSchema
This class serializes/deserializes metadata for
[vtkDataOject](https://www.vtk.org/doc/nightly/html/classvtkDataObject.html)
and passes the data object off to the
([senseiADIOS::DatasetSchema](#senseiadiosdatasetschema)) for serialization of leaf
datasets.

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/number_of_datasets` | number of leaves in composite dataset, integer
`data_object_<doid>/data_object_type` | VTK data object type enumeration, integer

# senseiADIOS::DatasetSchema
This class serializes/deserializes metadata for
[vtkDataSet](https://www.vtk.org/doc/nightly/html/classvtkDataSet.html) and
manages lower level specialized serialization objects
([senseiADIOS::CellsSchema](#senseiadioscellsschema),
[senseiADIOS::PointsSchema](#senseiadiospointsschema),
[senseiADIOS::DatasetAttributesSchema](#senseiadiosdatasetattributesschema),
[senseiADIOS::Extent3DSchema](#senseiadiosextent3dschema)) that serialize/deserialize VTK
datasets derived from vtkDataSet.

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/dataset_<dsid>/data_object_type` | VTK dataset type enumeration, integer

# senseiADIOS::Extent3DSchema
This class serializes/deserializes metadata needed to represent geometry of uniform
Cartesian meshes, in VTK the
[vtkImageData](https://www.vtk.org/doc/nightly/html/classvtkImageData.html)
datasets.

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/dataset_<dsid>/extent` | index space extents, 6 integers
`data_object_<doid>/dataset_<dsid>/origin` | coordinate system origin, 3 doubles
`data_object_<doid>/dataset_<dsid>/spacing` | grid spacing, 3 doubles

# senseiADIOS::PointsSchema
This class serializes/deserializes coordinates of unstructured meshes derived
from VTK's
[vtkPointSet](https://www.vtk.org/doc/nightly/html/classvtkPointSet.html)
datasets.

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/dataset_<dsid>/points/number_of_elements` | length of the array, unsigned long
`data_object_<doid>/dataset_<dsid>/points/type` | VTK data type enumeration, integer
`data_object_<doid>/dataset_<dsid>/points/data` | the array values

# senseiADIOS::CellsSchema
This class serializes/deserializes mesh topology for unstructured meshes of VTK's
[vtkUnstructuredGrid](https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html)
and [vtkPolyData](https://www.vtk.org/doc/nightly/html/classvtkPolyData.html)
datasets.

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/dataset_<dsid>/cells/number_of_cells` | number of cells, unsigned long
`data_object_<doid>/dataset_<dsid>/cells/cell_types` | array of VTK cell type enumeration
`data_object_<doid>/dataset_<dsid>/cells/number_of_elements` | length of the cell array
`data_object_<doid>/dataset_<dsid>/cells/data` | the cell array values

# senseiADIOS::DatasetAttributesSchema
This class serializes/deserializes
[vtkDataSetAttributes](https://www.vtk.org/doc/nightly/html/classvtkDataSetAttributes.html),
the containers for cell and point centered data arrays and the contained data
arrays. It is templated on attribute enumeration `att_t` which is used as a tag
in the schema. `att_str` is a string representation of `att_t`. For convenience
we define the following typedefs:
```C++
// specializations for common use cases
using PointDataSchema = DatasetAttributesSchema<vtkDataObject::POINT>;
using CellDataSchema = DatasetAttributesSchema<vtkDataObject::CELL>;

```

### writes/reads
 path | description
 ---  | ---
`data_object_<doid>/dataset_<dsid>/<att_str>/number_of_arrays` | number of arrays, integer
`data_object_<doid>/dataset_<dsid>/<att_str>/array_<i>/name` | name of the array, string
`data_object_<doid>/dataset_<dsid>/<att_str>/array_<i>/number_of_elements` | length of the array, unsigned long
`data_object_<doid>/dataset_<dsid>/<att_str>/array_<i>/number_of_components` | number of components, integer
`data_object_<doid>/dataset_<dsid>/<att_str>/array_<i>/element_type` | VTK data type enumeration, integer
`data_object_<doid>/dataset_<dsid>/<att_str>/array_<i>/data` | the array values

# Examples
## Aggregate data
This example shows the file structure of a collection comprised of a 2 block
multi-block uniform Cartesian mesh and a 2 block multi-block unstructured mesh
over 3 time steps. Each mesh contains 1 single precision cell data and 1 single
precision point data array. The code generating and writing the data,
`testADIOSWrite.py` is capable of writing BP files or streaming over FLEXPATH
and is part of the regression test suite distributed with the source code. Its
counterpart `testADIOSRead.py` can be used to deserialize the file/stream.

```bash
$mpiexec -np 2 python ../sensei/sensei/testing/testADIOSWrite.py test.bp MPI 3
STATUS[0] : initializing the VTKDataAdaptor step 0 time 0.0
STATUS[0] : executing ADIOSAnalysisAdaptor MPI step 0 time 0.0
WARNING: [0][/home/sensei/sc17/software/sensei/builds/sensei/sensei/ADIOSAnalysisAdaptor.cxx:82][v1.1.0]
WARNING: No subset specified. Writing all available data
STATUS[0] : finished writing 1 steps
```
The `bpls` tool that ships with ADIOS can be used to display the file structure
and dump arrays.
```bash
$bpls test.bp
  unsigned long long  time_step                                                        3*scalar
  double              time                                                             3*scalar
  integer             number_of_data_objects                                           3*scalar
  integer             data_object_0/name_len                                           3*scalar
  byte                data_object_0/name                                               3*{6}
  unsigned integer    data_object_0/number_of_datasets                                 3*scalar
  integer             data_object_0/data_object_type                                   3*scalar
  integer             data_object_0/dataset_1/data_object_type                         3*scalar
  integer             data_object_0/dataset_1/extent_len                               3*scalar
  integer             data_object_0/dataset_1/extent                                   3*{6}
  integer             data_object_0/dataset_1/origin_len                               3*scalar
  double              data_object_0/dataset_1/origin                                   3*{3}
  integer             data_object_0/dataset_1/spacing_len                              3*scalar
  double              data_object_0/dataset_1/spacing                                  3*{3}
  integer             data_object_0/dataset_1/point_data/number_of_arrays              3*scalar
  integer             data_object_0/dataset_1/point_data/array_0/name_len              3*scalar
  byte                data_object_0/dataset_1/point_data/array_0/name                  3*{11}
  long long           data_object_0/dataset_1/point_data/array_0/number_of_elements    3*scalar
  integer             data_object_0/dataset_1/point_data/array_0/number_of_components  3*scalar
  integer             data_object_0/dataset_1/point_data/array_0/element_type          3*scalar
  real                data_object_0/dataset_1/point_data/array_0/data                  3*{108}
  integer             data_object_0/dataset_1/cell_data/number_of_arrays               3*scalar
  integer             data_object_0/dataset_1/cell_data/array_0/name_len               3*scalar
  byte                data_object_0/dataset_1/cell_data/array_0/name                   3*{11}
  long long           data_object_0/dataset_1/cell_data/array_0/number_of_elements     3*scalar
  integer             data_object_0/dataset_1/cell_data/array_0/number_of_components   3*scalar
  integer             data_object_0/dataset_1/cell_data/array_0/element_type           3*scalar
  real                data_object_0/dataset_1/cell_data/array_0/data                   3*{34}
  integer             data_object_1/name_len                                           3*scalar
  byte                data_object_1/name                                               3*{13}
  unsigned integer    data_object_1/number_of_datasets                                 3*scalar
  integer             data_object_1/data_object_type                                   3*scalar
  integer             data_object_1/dataset_1/data_object_type                         3*scalar
  unsigned long long  data_object_1/dataset_1/cells/number_of_cells                    3*scalar
  unsigned byte       data_object_1/dataset_1/cells/cell_types                         3*{16}
  unsigned long long  data_object_1/dataset_1/cells/number_of_elements                 3*scalar
  long long           data_object_1/dataset_1/cells/data                               3*{32}
  unsigned long long  data_object_1/dataset_1/points/number_of_elements                3*scalar
  integer             data_object_1/dataset_1/points/elem_type                         3*scalar
  real                data_object_1/dataset_1/points/data                              3*{48}
  integer             data_object_1/dataset_1/point_data/number_of_arrays              3*scalar
  integer             data_object_1/dataset_1/point_data/array_0/name_len              3*scalar
  byte                data_object_1/dataset_1/point_data/array_0/name                  3*{11}
  long long           data_object_1/dataset_1/point_data/array_0/number_of_elements    3*scalar
  integer             data_object_1/dataset_1/point_data/array_0/number_of_components  3*scalar
  integer             data_object_1/dataset_1/point_data/array_0/element_type          3*scalar
  real                data_object_1/dataset_1/point_data/array_0/data                  3*{16}
  integer             data_object_1/dataset_1/cell_data/number_of_arrays               3*scalar
  integer             data_object_1/dataset_1/cell_data/array_0/name_len               3*scalar
  byte                data_object_1/dataset_1/cell_data/array_0/name                   3*{11}
  long long           data_object_1/dataset_1/cell_data/array_0/number_of_elements     3*scalar
  integer             data_object_1/dataset_1/cell_data/array_0/number_of_components   3*scalar
  integer             data_object_1/dataset_1/cell_data/array_0/element_type           3*scalar
  real                data_object_1/dataset_1/cell_data/array_0/data                   3*{16}
  integer             data_object_0/dataset_2/data_object_type                         3*scalar
  integer             data_object_0/dataset_2/extent_len                               3*scalar
  integer             data_object_0/dataset_2/extent                                   3*{6}
  integer             data_object_0/dataset_2/origin_len                               3*scalar
  double              data_object_0/dataset_2/origin                                   3*{3}
  integer             data_object_0/dataset_2/spacing_len                              3*scalar
  double              data_object_0/dataset_2/spacing                                  3*{3}
  integer             data_object_0/dataset_2/point_data/number_of_arrays              3*scalar
  integer             data_object_0/dataset_2/point_data/array_0/name_len              3*scalar
  byte                data_object_0/dataset_2/point_data/array_0/name                  3*{11}
  long long           data_object_0/dataset_2/point_data/array_0/number_of_elements    3*scalar
  integer             data_object_0/dataset_2/point_data/array_0/number_of_components  3*scalar
  integer             data_object_0/dataset_2/point_data/array_0/element_type          3*scalar
  real                data_object_0/dataset_2/point_data/array_0/data                  3*{108}
  integer             data_object_0/dataset_2/cell_data/number_of_arrays               3*scalar
  integer             data_object_0/dataset_2/cell_data/array_0/name_len               3*scalar
  byte                data_object_0/dataset_2/cell_data/array_0/name                   3*{11}
  long long           data_object_0/dataset_2/cell_data/array_0/number_of_elements     3*scalar
  integer             data_object_0/dataset_2/cell_data/array_0/number_of_components   3*scalar
  integer             data_object_0/dataset_2/cell_data/array_0/element_type           3*scalar
  real                data_object_0/dataset_2/cell_data/array_0/data                   3*{34}
  integer             data_object_1/dataset_2/data_object_type                         3*scalar
  unsigned long long  data_object_1/dataset_2/cells/number_of_cells                    3*scalar
  unsigned byte       data_object_1/dataset_2/cells/cell_types                         3*{16}
  unsigned long long  data_object_1/dataset_2/cells/number_of_elements                 3*scalar
  long long           data_object_1/dataset_2/cells/data                               3*{32}
  unsigned long long  data_object_1/dataset_2/points/number_of_elements                3*scalar
  integer             data_object_1/dataset_2/points/elem_type                         3*scalar
  real                data_object_1/dataset_2/points/data                              3*{48}
  integer             data_object_1/dataset_2/point_data/number_of_arrays              3*scalar
  integer             data_object_1/dataset_2/point_data/array_0/name_len              3*scalar
  byte                data_object_1/dataset_2/point_data/array_0/name                  3*{11}
  long long           data_object_1/dataset_2/point_data/array_0/number_of_elements    3*scalar
  integer             data_object_1/dataset_2/point_data/array_0/number_of_components  3*scalar
  integer             data_object_1/dataset_2/point_data/array_0/element_type          3*scalar
  byte                data_object_1/dataset_2/point_data/array_0/data                  3*{16}
  integer             data_object_1/dataset_2/cell_data/number_of_arrays               3*scalar
  integer             data_object_1/dataset_2/cell_data/array_0/name_len               3*scalar
  byte                data_object_1/dataset_2/cell_data/array_0/name                   3*{11}
  long long           data_object_1/dataset_2/cell_data/array_0/number_of_elements     3*scalar
  integer             data_object_1/dataset_2/cell_data/array_0/number_of_components   3*scalar
  integer             data_object_1/dataset_2/cell_data/array_0/element_type           3*scalar
  real                data_object_1/dataset_2/cell_data/array_0/data                   3*{16}
 ```
