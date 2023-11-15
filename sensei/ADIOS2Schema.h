#ifndef ADIOSVTK_h
#define ADIOSVTK_h

class svtkDataSet;
class svtkDataObject;

#include "MeshMetadata.h"
#include "SVTKUtils.h"

#include <adios2_c.h>
#include <adios2.h>
#include <svtkDataObject.h>
#include <svtkCompositeDataSet.h>
#include <mpi.h>
#include <set>
#include <cstdint>
#include <string>
#include <vector>
#include <limits.h>

namespace senseiADIOS2
{
const char *adios2_strerror(adios2_error err);

struct AdiosHandle
{
  AdiosHandle() : io(nullptr), engine(nullptr) {}
  adios2_io *io;
  adios2_engine *engine;
};

struct InputStream;

/// ADIOS representation of collections of svtkDataObject
// This class provides the user facing API managing the lower level
// objects internally. The write API defines variables needed for the
// ADIOS representation, computes their size for ADIOS buffers, and
// serializes complete SVTK objects. The read API can deserialize complete
// representation and also includes methods that coorrelate to SENSEI
// data adaptor API which enables targeted reading of subsets of the
// data.
class DataObjectCollectionSchema
{
public:
  DataObjectCollectionSchema();
  ~DataObjectCollectionSchema();

  // declare variables for adios write
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    const std::vector<sensei::MeshMetadataPtr> &metadata);

  // discover names of data objects on disk(or stream)
  int ReadMeshMetadata(MPI_Comm comm, InputStream &iStream);

  // get cached metadata for object i. Available after ReadMeshMetadata
  // this shows how the data is layed out on the sender side
  int GetSenderMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md);

  // set/get cached metadata for object i. this must be set before
  // reading any data. this controls how thye data is layed out on
  // the receiver side
  int GetReceiverMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md);
  int SetReceiverMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md);

  // get the number of meshes available. Available after ReadMeshMetadata
  int GetNumberOfObjects(unsigned int &num);

  // write the object collection
  int Write(MPI_Comm comm, AdiosHandle handles, unsigned long time_step, double time,
    const std::vector<sensei::MeshMetadataPtr> &metadata,
    const std::vector<svtkCompositeDataSetPtr> &objects);

  // return true if the file is one of ours and the version the file was
  // written with is compatible with this revision of the schema
  bool CanRead(InputStream &iStream);

  // creates the mesh matching what is on disk(or stream), including a domain
  // decomposition, but does not read data arrays. If structure_only is true
  // then points and cells are not read from disk.
  int ReadObject(MPI_Comm comm, InputStream &iStream, const std::string &name,
    svtkDataObject *&object, bool structure_only);

  // read a single array from disk(or stream), store it into the mesh
  int ReadArray(MPI_Comm comm, InputStream &iStream,
    const std::string &object_name, int association,
    const std::string &array_name, svtkDataObject *dobj);

  // returns the current time and time step
  int ReadTimeStep(MPI_Comm comm, InputStream &iStream,
    unsigned long &time_step, double &time);

private:
  // given a name get the id
  int GetObjectId(MPI_Comm comm,
    const std::string &object_name, unsigned int &doid);

  // generate an array on each block of the object filled with the BlockOwner
  int AddBlockOwnerArray(MPI_Comm comm, const std::string &name, int centering,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  struct InternalsType;
  InternalsType *Internals;
};



/// High level operations on an ADIOS file/stream
struct InputStream
{
  InputStream() : Handles(), Adios(nullptr),
    ReadEngine(""), FileName(""), FileSeries(0),
    StepsPerFile(0), FileIndex(0), StepIndex(0) {}

  // pass engine parameters to ADIOS2 in key value pairs
  void AddParameter(const std::string &key, const std::string &value);

  // set the ADIOS engine to use. Must be the same as on
  // the write side.
  void SetReadEngine(const std::string &engine)
  { this->ReadEngine = engine; }

  /// @brief Set the filename.
  /// Default value is "sensei.bp" which is suitable for use with streams or
  /// transport engines such as SST. When writing files to disk using the BP4
  /// engine one could SetStepsPerFile to prevent all steps being accumulated in
  /// a single file. In this case one should also use a printf like format
  /// specifier compatible with an int type in the file name. For example
  /// "sensei_%04d.bp".
  void SetFileName(const std::string &fileName);

  /// @brief Set the number of time steps to store in each file.  The default
  /// value is 0 which results in all the steps landing in a single file. If set
  /// to non-zero then multiple files per run are created each with this number
  /// of steps. An ordinal file index is incorporated in the file name. See
  /// notes in SetFileName for details on specifying the format specifier.
  void SetStepsPerFile(int stepsPerFile)
  { this->StepsPerFile = stepsPerFile; }

  int Initialize(MPI_Comm Comm);
  int Finalize();

  int Open();

  int BeginStep();

  int AdvanceTimeStep();

  int EndOfStream();

  int Close();

  int Good() { return Handles.engine != nullptr; }

  AdiosHandle Handles;
  adios2_adios *Adios;
  std::string ReadEngine;
  std::string FileName;
  int FileSeries;
  int StepsPerFile;
  int FileIndex;
  int StepIndex;
  std::vector<std::pair<std::string,std::string>> Parameters;
};

}

#endif
