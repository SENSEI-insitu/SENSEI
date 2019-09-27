#ifndef HDF5_VTK_h
#define HDF5_VTK_h

class vtkDataSet;
class vtkDataObject;
typedef struct _ADIOS_FILE ADIOS_FILE;

#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "hdf5.h"
//#include <adios_read.h>
#include <cstdint>
#include <mpi.h>
#include <set>
#include <string>
#include <vector>
#include <vtkCompositeDataSet.h>
#include <vtkDataObject.h>

namespace senseiHDF5
{

class HDF5GroupGuard
{
public:
  HDF5GroupGuard(hid_t &gid)
    : m_GroupID(gid)
  {
  }

  ~HDF5GroupGuard()
  {
    H5Gclose(m_GroupID);
    m_GroupID = -1;
  }

private:
  hid_t &m_GroupID;
};

class HDF5VarGuard
{
public:
  HDF5VarGuard(hid_t varID);

  ~HDF5VarGuard();

  void ReadAll(void *buf);

  void ReadSlice(void *buf,
                 int ndim,
                 const hsize_t *start,
                 const hsize_t *stride,
                 const hsize_t *count,
                 const hsize_t *block);

  hid_t m_VarID;
  hid_t m_VarType;
  hid_t m_VarSpace;
};

class HDF5SpaceGuard
{
public:
  HDF5SpaceGuard(hsize_t global, hsize_t s, hsize_t c)
  {
    m_ndim = 1;

    hsize_t total[1] = { global };
    m_FileSpaceID = H5Screate_simple(1, total, NULL);

    hsize_t offset[1] = { s };
    hsize_t count[1] = { c };

    H5Sselect_hyperslab(
      m_FileSpaceID, H5S_SELECT_SET, offset, NULL, count, NULL);

    m_MemSpaceID = H5Screate_simple(1, count, NULL);
  }

  ~HDF5SpaceGuard()
  {
    if(m_FileSpaceID >= 0)
      H5Sclose(m_FileSpaceID);
    if(m_MemSpaceID >= 0)
      H5Sclose(m_MemSpaceID);
  }

  // private:
  hid_t m_FileSpaceID;
  hid_t m_MemSpaceID;

  unsigned int m_ndim;
};

class BasicStream;
class ReadStream;
class WriteStream;

//
//
//
class StreamHandler
{
public:
  StreamHandler(bool m, const std::string &name, BasicStream *);
  virtual ~StreamHandler() {};

  // virtual bool OpenStream() = 0; // AdvanceStream is enough
  virtual bool CloseStream() = 0;
  virtual bool AdvanceStream() = 0;
  virtual bool IsValid() = 0;
  virtual bool Summary() = 0;

  hid_t m_TimeStepId;
  unsigned int m_TimeStepCounter = 0;

  bool m_InReadMode = true;

  std::string m_FileName;
  BasicStream *m_Client;
};

class DefaultStreamHandler : public StreamHandler
{
public:
  DefaultStreamHandler(const std::string &, ReadStream *);
  DefaultStreamHandler(const std::string &, WriteStream *);
  ~DefaultStreamHandler();

  bool OpenStream();
  bool CloseStream();
  bool AdvanceStream();

  bool IsValid();
  bool Summary();

private:
  hid_t m_HostFileId;
  unsigned int m_TimeStepTotal = 0;
};

class PerStepStreamHandler : public StreamHandler
{
public:
  PerStepStreamHandler(const std::string &filename, ReadStream *client);
  PerStepStreamHandler(const std::string &filename, WriteStream *client);
  ~PerStepStreamHandler();

  bool OpenStream();
  bool CloseStream();
  bool AdvanceStream();

  bool IsValid();
  bool Summary();

private:
  void GetStepFileName(std::string &stepName, int ts);

  // hid_t m_HostFileId;
  bool NoMoreStep();
  void GetCurrAvailStep();
  void UpdateAvailStep();

  int m_NumStepsWritten = -1; // -1 if not able to detect. otherwise >=1

  bool m_AllStepsWritten = false;
};

//
// IO stream
//
class BasicStream
{
public:
  BasicStream(MPI_Comm comm, bool);

  virtual ~BasicStream();

  virtual bool AdvanceTimeStep(unsigned long &time_step, double &time) = 0;
  virtual bool Init(const std::string &name) = 0;
  virtual void Close() = 0;

  void CloseTimeStep();
  void SetCollectiveTxf();
  MPI_Comm m_Comm;
  int m_Rank;
  int m_Size;

  bool m_StreamingOn = false;

  sensei::MeshMetadataMap m_AllMeshInfo; // sender
  sensei::MeshMetadataMap m_AllMeshInfoReceiver;

#ifdef NEVER
  hid_t m_TimeStepGroupId;
  unsigned int m_TimeStepCounter = 0;
  hid_t m_FileId;
#else
  StreamHandler *m_Streamer = nullptr;
#endif
  hid_t m_PropertyListId; // MPIO acceess

protected:
  hid_t m_CollectiveTxf = H5P_DEFAULT;
};

class WriteStream : public BasicStream
{
public:
  WriteStream(MPI_Comm comm, bool);
  ~WriteStream();
  bool Init(const std::string &name);

  bool AdvanceTimeStep(unsigned long &time_step, double &time);

  void Close() {}
  bool WriteMesh(sensei::MeshMetadataPtr &md, vtkCompositeDataSet *vtkPtr);

  bool WriteBinary(const std::string &name, sensei::BinaryStream &str);
  bool WriteMetadata(sensei::MeshMetadataPtr &md);
  bool WriteNativeAttr(const std::string &name,
                       void *val,
                       hid_t h5Type,
                       hid_t owner);

  hid_t CreateVar(const std::string &name,
                  const HDF5SpaceGuard &space,
                  hid_t h5Type);

  // bool WriteVar(const std::string& name, const HDF5SpaceGuard &space,
  // hid_t h5Type, void *data);

  bool WriteVar(hid_t &vid,
                const std::string &name,
                const HDF5SpaceGuard &space,
                hid_t h5Type,
                void *data);

private:
  unsigned int m_MeshCounter;
};

class ReadStream : public BasicStream
{
public:
  ReadStream(MPI_Comm comm, bool);
  ~ReadStream();

  bool AdvanceTimeStep(unsigned long &time_step, double &time);

  bool Init(const std::string &name);
  void Close();

  bool ReadMetadata(unsigned int &nMesh);

  int GetNumberOfMeshes() { return m_AllMeshInfo.Size(); }

  bool ReadSenderMeshMetaData(unsigned int i, sensei::MeshMetadataPtr &ptr);
  bool ReadReceiverMeshMetaData(unsigned int i, sensei::MeshMetadataPtr &ptr);

  bool ReadMesh(std::string name, vtkDataObject *&dobj, bool structure_only);

  bool ReadInArray(const std::string &meshName,
                   int association,
                   const std::string &array_name,
                   vtkDataObject *dobj);

  bool ReadNativeAttr(const std::string &name,
                      void *val,
                      hid_t h5Type,
                      hid_t hid);
  bool ReadBinary(const std::string &name, sensei::BinaryStream &str);
  bool ReadVar1D(const std::string &name, hsize_t s, hsize_t c, void *data);

private:
  unsigned int m_TimeStepTotal;
};

class ArrayFlow;

//
// a MESH is a vtkCompositeDataset
//
class MeshFlow
{
public:
  MeshFlow(vtkCompositeDataSet *, unsigned int meshID);
  ~MeshFlow();

  bool ReadBlockOwnerArray(ReadStream *input,
                           const std::string &array_name,
                           int association);

  bool ReadArray(ReadStream *input,
                 const std::string &array_name,
                 int association);
  bool ReadFrom(ReadStream *StreamPtr, bool structureOnly);
  bool Initialize(const sensei::MeshMetadataPtr &md, ReadStream *input);

  bool WriteTo(WriteStream *StreamPtr, const sensei::MeshMetadataPtr &md);

  vtkCompositeDataSet *m_VtkPtr;

private:
  bool ValidateMetaData(const sensei::MeshMetadataPtr &md);

  void Unload(ArrayFlow *arrayFlowPtr, 
	      const sensei::MeshMetadataPtr &md,
              WriteStream *output);
  void Load(ArrayFlow *arrayFlowPtr, 
	    const sensei::MeshMetadataPtr &md,
            ReadStream *reader);


  unsigned int m_MeshID;
};

class VTKObjectFlow
{
public:
  VTKObjectFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  virtual ~VTKObjectFlow();
  virtual bool load(unsigned int block_id,
                    vtkCompositeDataIterator *it,
                    ReadStream *input) = 0;
  virtual bool update(unsigned int block_id) = 0;
  virtual bool unload(unsigned int block_id,
                      vtkCompositeDataIterator *it,
                      WriteStream *output) = 0;

protected:
  const sensei::MeshMetadataPtr &m_Metadata;
  unsigned int m_MeshID;

  unsigned long long m_TotalCell = 0;
  unsigned long long m_TotalArraySize = 0;

  hid_t m_PointType = -1;

  std::string m_CellArrayVarName;
  std::string m_CellTypeVarName;
  std::string m_PointVarName;

  hid_t m_CellArrayVarID = -1;
  hid_t m_CellTypeVarID = -1;
  hid_t m_PointVarID = -1;
};

class WorkerCollection
{
public:
  WorkerCollection(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  ~WorkerCollection();

  bool load(unsigned int block_id,
            vtkCompositeDataIterator *it,
            ReadStream *input);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *input);
  bool update(unsigned int block_id);

protected:
  std::vector<VTKObjectFlow *> m_Workers;
};

class ArrayFlow : public VTKObjectFlow {
public:
  // regular array init
  ArrayFlow(const sensei::MeshMetadataPtr &md, 
	    unsigned int meshID,
            unsigned int arrayID);
  // ghost array init
  ArrayFlow(unsigned int meshID, 
	    int centering,
            const sensei::MeshMetadataPtr &md);
  ~ArrayFlow();

  bool load(unsigned int block_id, 
	    vtkCompositeDataIterator *it, 
	    ReadStream *);
  bool unload(unsigned int block_id, 
	      vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int block_id);

  int GetArrayType();
  const std::string &GetArrayName();

protected:
  unsigned long long getLocalElement(unsigned int block_id);

private:
  unsigned long long m_BlockOffset;
  std::string m_ArrayPath; // name in H5
  hid_t m_ArrayVarID;

  unsigned int m_ArrayID;
  bool m_IsGhostArray;
  int m_ArrayCenter;
  unsigned long long m_NumArrayComponent;
  unsigned long long m_ElementTotal = 0;
  ;
};


class PointFlow : public VTKObjectFlow
{
public:
  PointFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  ~PointFlow() {}
  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int block_id);

private:
  unsigned long long m_BlockOffset;
  unsigned long long m_GlobalTotal;
};

class PolydataCellFlow : public VTKObjectFlow
{
public:
  PolydataCellFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  ~PolydataCellFlow();

  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int block_id);

private:
  unsigned long long m_CellTypesBlockOffset = 0;
  unsigned long long m_CellArrayBlockOffset = 0;
};

class UniformCartesianFlow : public VTKObjectFlow
{
public:
  UniformCartesianFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  ~UniformCartesianFlow();
  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int) { return true; }

private:
  std::string m_OriginPath;
  std::string m_SpacingPath;

  hid_t m_SpacingVarID;
  hid_t m_OriginVarID;
};

class LogicallyCartesianFlow : public VTKObjectFlow
{
public:
  LogicallyCartesianFlow(const sensei::MeshMetadataPtr &md,
                         unsigned int meshID);
  ~LogicallyCartesianFlow();

  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int) { return true; }

private:
  std::string m_ExtentPath;
  hid_t m_ExtentID;
};

class StretchedCartesianFlow : public VTKObjectFlow
{
public:
  StretchedCartesianFlow(const sensei::MeshMetadataPtr &md,
                         unsigned int meshID);
  ~StretchedCartesianFlow();
  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int);

private:
  void GetLocal(int block_id, unsigned long long (&out)[3]);
  std::string m_XPath;
  std::string m_YPath;
  std::string m_ZPath;

  hid_t m_PosID[3] = { -1, -1, -1 };

  unsigned long long m_Total[3] = { 0, 0, 0 };
  unsigned long long m_BlockOffset[3] = { 0, 0, 0 };
};

class UnstructuredCellFlow : public VTKObjectFlow
{
public:
  UnstructuredCellFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID);
  ~UnstructuredCellFlow() {}

  bool load(unsigned int block_id, vtkCompositeDataIterator *it, ReadStream *);
  bool unload(unsigned int block_id,
              vtkCompositeDataIterator *it,
              WriteStream *output);
  bool update(unsigned int block_id);

private:
  unsigned long long m_CellTypesBlockOffset = 0;
  unsigned long long m_CellArrayBlockOffset = 0;
};

//
// blocks  of a mesh have the <same> type
//
/*
class BlockFlow: public VTKObjectFlow
{
 public:
  BlockFlow(sensei::MeshMetadataPtr& md, vtkDataObject*);

  void ToStream(StreamPtr);
  void FromStream(StreamPtr);

 private:
  vtkDataObject* _vtkPtr;
};
*/

//
//
//
} // namespace senseiHDF5

#endif
