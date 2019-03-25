#ifndef ADIOSVTK_h
#define ADIOSVTK_h

class vtkDataSet;
class vtkDataObject;
typedef struct _ADIOS_FILE ADIOS_FILE;

#include "MeshMetadata.h"
#include <adios_read.h>
#include <vtkDataObject.h>
#include <vtkCompositeDataSet.h>
#include <mpi.h>
#include <set>
#include <cstdint>
#include <string>
#include <vector>

namespace senseiADIOS1
{

struct InputStream;

/// ADIOS representation of collections of vtkDataObject
// This class provides the user facing API managing the lower level
// objects internally. The write API defines variables needed for the
// ADIOS representation, computes their size for ADIOS buffers, and
// serializes complete VTK objects. The read API can deserialize complete
// representation and also includes methods that coorrelate to SENSEI
// data adaptor API which enables targeted reading of subsets of the
// data.
class DataObjectCollectionSchema
{
public:
  DataObjectCollectionSchema();
  ~DataObjectCollectionSchema();

  // declare variables for adios write
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::vector<sensei::MeshMetadataPtr> &metadata);

  // discover names of data objects on disk(or stream)
  int ReadMeshMetadata(MPI_Comm comm, InputStream &iStream);

  // get cached metadata for object i. Available after ReadMeshMetadata
  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md);

  // get the number of meshes available. Available after ReadMeshMetadata
  int GetNumberOfObjects(unsigned int &num);

  // write the object collection
  int Write(MPI_Comm comm, int64_t fh, unsigned long time_step, double time,
    const std::vector<sensei::MeshMetadataPtr> &metadata,
    const std::vector<vtkCompositeDataSet*> &objects);

  // return true if the file is one of ours and the version the file was
  // written with is compatible with this revision of the schema
  bool CanRead(InputStream &iStream);

  // creates the mesh matching what is on disk(or stream), including a domain
  // decomposition, but does not read data arrays. If structure_only is true
  // then points and cells are not read from disk.
  int ReadObject(MPI_Comm comm, InputStream &iStream, const std::string &name,
    vtkDataObject *&object, bool structure_only);

  // read a single array from disk(or stream), store it into the mesh
  int ReadArray(MPI_Comm comm, InputStream &iStream,
    const std::string &object_name, int association,
    const std::string &array_name, vtkDataObject *dobj);

  // returns the current time and time step
  int ReadTimeStep(MPI_Comm comm, InputStream &iStream,
    unsigned long &time_step, double &time);

private:
  // given a name get the id
  int GetObjectId(MPI_Comm comm,
    const std::string &object_name, unsigned int &doid);

  struct InternalsType;
  InternalsType *Internals;
};

/// High level operations on an ADIOS file/stream
struct InputStream
{
  InputStream() : File(nullptr),
    ReadMethod(static_cast<ADIOS_READ_METHOD>(-1)),
    FileName() {}

  int SetReadMethod(const std::string &method);

  int Open(MPI_Comm comm, ADIOS_READ_METHOD method,
    const std::string &fileName);

  int Open(MPI_Comm Comm);

  int AdvanceTimeStep();

  int Close();

  int Good() { return File != nullptr; }

  ADIOS_FILE *File;
  ADIOS_READ_METHOD ReadMethod;
  std::string FileName;
};

}

#endif
