#ifndef ADIOSVTK_h
#define ADIOSVTK_h

class vtkDataSet;
class vtkDataObject;
typedef struct _ADIOS_FILE ADIOS_FILE;

#include <adios_read.h>
#include <vtkDataObject.h>
#include <mpi.h>
#include <set>
#include <cstdint>

namespace senseiADIOS
{

struct InputStream;

/// Base class for representing VTK data in ADIOS.
// the 3 operations that need to be done to send data to ADIOS are:
//
// 1. declare variable types
// 2. compute a buffer size needed to write
// 3. write the data
// 4. read the data
//
// this class declares the API for these operations.
//
// The common theme that is encountered when writing VTK data to ADIOS
// is traversing the vtkDataObject and operating on leaf nodes. This
// class provides default methods that traverse the DataObject and call
// a DataSet overload provided by concrete implementations.
class Schema
{
public:
  virtual ~Schema() {}

  // sends ADIOS variable definitions
  virtual int DefineVariables(MPI_Comm comm, int64_t gh, vtkDataObject *dobj);
  virtual int DefineVariables(int64_t gh, unsigned int id, vtkDataSet *ds);

  // Gets number of bytes used in ADIOS representation
  virtual uint64_t GetSize(MPI_Comm comm, vtkDataObject *dobj);
  virtual uint64_t GetSize(vtkDataSet *ds);

  // Writes data to ADIOS
  virtual int Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj);
  virtual int Write(int64_t fh, unsigned int id, vtkDataSet *ds);

  // Read data from ADIOS
  virtual int Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj);
  virtual int Read(MPI_Comm comm, InputStream &iStream, unsigned int id, vtkDataSet *&ds);
};



/// ADIOS representation of vtkDataObject
// This class provides the user facing API managing the lower level
// objects internally. The write API defines variables needed for the
// ADIOS representation, computes their size for ADIOS buffers, and
// serializes complete VTK objects. The read API can deserialize complete
// representation and also includes methods that coorrelate to SENSEI
// data adaptor API which enables targeted reading of subsets of the
// data.
class DataObjectSchema : public Schema
{
public:
  DataObjectSchema();
  ~DataObjectSchema();

  int DefineVariables(MPI_Comm comm, int64_t gh, vtkDataObject* dobj) override;

  uint64_t GetSize(MPI_Comm comm, vtkDataObject *dobj) override;

  // Write time and time step metadata
  int WriteTimeStep(MPI_Comm comm, int64_t fh, unsigned long step, double time);

  int Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj) override;

  int Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj) override;

  // verifies that the file is ours
  int CanRead(MPI_Comm comm, InputStream &iStream);

  // creates the mesh matching what is on disk(or stream), indluding a domain
  // decomposition, but does not read data arrays. If structure_only is true
  // then points and cells are not read from disk.
  int ReadMesh(MPI_Comm comm, InputStream &iStream, bool structure_only,
    vtkDataObject *&dobj);

  // discover names of data arrays on disk(or stream)
  int ReadArrayNames(MPI_Comm comm, InputStream &iStream, vtkDataObject *dobj,
    int association, std::set<std::string> &array_names);

  // read a single array from disk(or stream), store it into the mesh
  int ReadArray(MPI_Comm comm, InputStream &iStream, vtkDataObject *dobj,
    int association, const std::string &name);

  // returns the current time and time step
  int ReadTimeStep(MPI_Comm comm, InputStream &iStream,
    unsigned long &time_step, double &time);

private:
  // create data object
  int InitializeDataObject(MPI_Comm comm, InputStream &iStream,
    vtkDataObject *&dobj);

  struct InternalsType;
  InternalsType *Internals;
};



/// ADIOS representation of vtkDataSet
// seriealizes/deserializes dataset level attributes and manages the lower
// level classes. See DataObjectSchema for user facing API.
class DatasetSchema : public Schema
{
public:
  DatasetSchema();
  ~DatasetSchema();

  int DefineVariables(int64_t gh, unsigned int id, vtkDataSet*ds) override;
  int DefineVariables(MPI_Comm comm, int64_t gh, vtkDataObject* dobj) override;

  uint64_t GetSize(MPI_Comm comm, vtkDataObject *dobj) override;
  uint64_t GetSize(vtkDataSet *ds) override;

  int Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj) override;
  int Write(int64_t fh, unsigned int id, vtkDataSet *ds) override;

  int Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj) override;

  int Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds) override;

  // creates the mesh matching what is on disk(or stream), indluding a domain
  // decomposition, but does not read data arrays. If structure_only is true
  // then points and cells are not read from disk.
  int ReadMesh(MPI_Comm comm, InputStream &iStream, bool structure_only,
    vtkDataObject *&dobj);

  int ReadMesh(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds, bool structure_only);

  // discover names of data arrays on disk(or stream)
  int ReadArrayNames(MPI_Comm comm, InputStream &iStream, vtkDataObject *dobj,
    int association, std::set<std::string> &array_names);

  // read a single array from disk(or stream), store it into the mesh
  int ReadArray(MPI_Comm comm, InputStream &iStream, vtkDataObject *dobj,
    int association, const std::string &name);

  // define the local domain decomposition by a start data object id
  // and length.
  void SetDecomp(unsigned int id, unsigned int n);
  void ClearDecomp();

private:
  struct InternalsType;
  InternalsType *Internals;
};



/// ADIOS representation of VTK's 3D extent(image data)
// serializes/deserializes properties unique to VTK's
// extent based datasets
class Extent3DSchema : public Schema
{
public:
  using Schema::DefineVariables;
  using Schema::Write;
  using Schema::Read;
  using Schema::GetSize;

  int DefineVariables(int64_t gh, unsigned int id, vtkDataSet*ds) override;

  uint64_t GetSize(vtkDataSet *ds) override;

  int Write(int64_t fh, unsigned int id, vtkDataSet *ds) override;

  int Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds) override;
};



/// ADIOS representation of vtkDataSetAttributes
// serializes/deserializes attribute data arrays
// templated on attribute type(POINT/CELL).
template<int att_t>
class DatasetAttributesSchema : public Schema
{
public:
  using Schema::DefineVariables;
  using Schema::Write;
  using Schema::Read;
  using Schema::GetSize;

  DatasetAttributesSchema();
  ~DatasetAttributesSchema();

  int DefineVariables(int64_t gh, unsigned int id, vtkDataSet* ds) override;

  uint64_t GetSize(vtkDataSet *ds) override;

  int Write(int64_t fh, unsigned int id, vtkDataSet *ds) override;

  int Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds) override;

  // discover names of data arrays
  int ReadArrayNames(MPI_Comm comm, InputStream &iStream,
    vtkDataObject *dobj, std::set<std::string> &array_names);

  int ReadArrayNames(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *ds, std::set<std::string> &array_names);

  // read a single array and store it into the mesh
  int ReadArray(MPI_Comm comm, InputStream &iStream,
    const std::string &array_name, vtkDataObject *dobj);

  int ReadArray(MPI_Comm comm, InputStream &iStream,
    const std::string &array_name, unsigned int id, vtkDataSet *ds);

private:
  struct InternalsType;
  InternalsType *Internals;
};

// specializations for common use cases
using PointDataSchema = DatasetAttributesSchema<vtkDataObject::POINT>;
using CellDataSchema = DatasetAttributesSchema<vtkDataObject::CELL>;



/// ADIOS representation of VTK's cells (includes cell types)
// serializes/deserializes cells
class CellsSchema : public Schema
{
public:
  using Schema::DefineVariables;
  using Schema::Write;
  using Schema::Read;
  using Schema::GetSize;

  int DefineVariables(int64_t gh, unsigned int id, vtkDataSet* ds) override;

  uint64_t GetSize(vtkDataSet *ds) override;

  int Write(int64_t fh, unsigned int id, vtkDataSet *ds) override;

  int Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds) override;

  // get length of arrays
  static unsigned long GetCellsLength(vtkDataSet *ds);

  // get size of arrays
  static uint64_t GetCellsSize(vtkDataSet *ds);

  // get the number of cells
  static unsigned long GetNumberOfCells(vtkDataSet *ds);
};



/// ADIOS representation of vtkPoints
// serializes/deserializes points
class PointsSchema : public Schema
{
public:
  using Schema::DefineVariables;
  using Schema::Write;
  using Schema::Read;
  using Schema::GetSize;

  int DefineVariables(int64_t gh, unsigned int id, vtkDataSet* ds) override;

  uint64_t GetSize(vtkDataSet *ds) override;

  int Write(int64_t fh, unsigned int id, vtkDataSet *ds) override;

  int Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
    vtkDataSet *&ds) override;

  // get length of arrays
  static unsigned long GetPointsLength(vtkDataSet *ds);

  // get size of arrays
  static uint64_t GetPointsSize(vtkDataSet *ds);

  // get the number of points
  static unsigned long GetNumberOfPoints(vtkDataSet *ds);
};



/// High level operations on an ADIOS file/stream
struct InputStream
{
  InputStream() : File(nullptr),
    ReadMethod(static_cast<ADIOS_READ_METHOD>(-1)) {}

  InputStream(ADIOS_FILE *file, ADIOS_READ_METHOD method)
    : File(file), ReadMethod(method) {}

  int Open(MPI_Comm comm, ADIOS_READ_METHOD method,
    const std::string &fileName);

  int AdvanceTimeStep();

  int Close();

  ADIOS_FILE *File;
  ADIOS_READ_METHOD ReadMethod;
};

}

#endif
