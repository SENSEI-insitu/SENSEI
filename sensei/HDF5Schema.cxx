#include "HDF5Schema.h"
#include "Profiler.h"
#include "SVTKUtils.h"

#include <svtkCellArray.h>
#include <svtkCellData.h>
#include <svtkCellTypes.h>
#include <svtkCharArray.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkHierarchicalBoxDataSet.h>
#include <svtkHyperTreeGrid.h>
#include <svtkIdTypeArray.h>
#include <svtkImageData.h>
#include <svtkInformation.h>
#include <svtkIntArray.h>
#include <svtkLongArray.h>
#include <svtkLongLongArray.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkMultiPieceDataSet.h>
#include <svtkNonOverlappingAMR.h>
#include <svtkObjectFactory.h>
#include <svtkOverlappingAMR.h>
#include <svtkPointData.h>
#include <svtkPoints.h>
#include <svtkPolyData.h>
#include <svtkRectilinearGrid.h>
#include <svtkSmartPointer.h>
#include <svtkStructuredGrid.h>
#include <svtkStructuredPoints.h>
#include <svtkTable.h>
#include <svtkUniformGrid.h>
#include <svtkUniformGridAMR.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkUnsignedLongLongArray.h>
#include <svtkUnstructuredGrid.h>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "BlockPartitioner.h"

namespace senseiHDF5
{
static const std::string ATTRNAME_TIMESTEP = "timestep";
static const std::string ATTRNAME_TIME = "time";
static const std::string ATTRNAME_NUM_TIMESTEP = "num_timestep";
static const std::string ATTRNAME_NUM_MESH = "num_meshs";
static const std::string TAG_MESH = "mesh_";
static const std::string TAG_ARRAY = "array_";
static const std::string TAG_SVTK_GHOST =
  "svtkGhostType"; // reserved  to indicate ghost array for cell/points

static bool gCheckSenseiCall(int code)
{
  return (code == 0); // burlen return 0 for success
}

static void gGetTimeStepString(std::string &stepName, int ts)
{
  stepName = "/Step" + std::to_string(ts);
}

static void gGetNameStr(std::string &out,
                        unsigned int meshID,
                        const std::string &suffix)
{
  std::ostringstream ons;

  if(suffix.size() == 0)
    ons << TAG_MESH << meshID;
  else
    ons << TAG_MESH << meshID << "/" << suffix;

  out = ons.str();
}

static void gGetArrayNameStr(std::string &out,
                             unsigned int meshID,
                             unsigned int array_id)
{
  std::ostringstream ons;
  ons << TAG_MESH << meshID << "/" << TAG_ARRAY << array_id << "_data";
  out = ons.str();
}

hid_t gHDF5_IDType()
{
  if(sizeof(svtkIdType) == sizeof(int64_t))
    {
      return H5T_NATIVE_LONG; // 64 bits
    }
  else if(sizeof(svtkIdType) == sizeof(int32_t))
    {
      return H5T_NATIVE_INT; // 32 bits
    }
  else
    {
      SENSEI_ERROR("No conversion from svtkIdType to HDF5 NativeDATATYPES");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return -1;
}

hid_t gGetHDF5Type(svtkDataArray *da)
{
  if(dynamic_cast<svtkFloatArray *>(da))
    {
      return H5T_NATIVE_FLOAT;
    }
  else if(dynamic_cast<svtkDoubleArray *>(da))
    {
      return H5T_NATIVE_DOUBLE;
    }
  else if(dynamic_cast<svtkCharArray *>(da))
    {
      return H5T_NATIVE_CHAR;
    }
  else if(dynamic_cast<svtkIntArray *>(da))
    {
      return H5T_NATIVE_INT;
    }
  else if(dynamic_cast<svtkLongArray *>(da))
    {
      if(sizeof(long) == 4)
        return H5T_NATIVE_INT; // 32 bits
      return H5T_NATIVE_LONG;  // 64 bits
    }
  else if(dynamic_cast<svtkLongLongArray *>(da))
    {
      return H5T_NATIVE_LONG; // 64 bits
    }
  else if(dynamic_cast<svtkUnsignedCharArray *>(da))
    {
      return H5T_NATIVE_UCHAR;
    }
  else if(dynamic_cast<svtkUnsignedIntArray *>(da))
    {
      return H5T_NATIVE_UINT;
    }
  else if(dynamic_cast<svtkUnsignedLongArray *>(da))
    {
      if(sizeof(unsigned long) == 4)
        return H5T_NATIVE_UINT; // 32 bits
      return H5T_NATIVE_ULONG;  // 64 bits
    }
  else if(dynamic_cast<svtkUnsignedLongLongArray *>(da))
    {
      return H5T_NATIVE_ULONG; // 64 bits
    }
  else if(dynamic_cast<svtkIdTypeArray *>(da))
    {
      return gHDF5_IDType();
    }
  else
    {
      SENSEI_ERROR("the HDF5  type for data array \""
                   << da->GetClassName() << "\" is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return -1;
}

hid_t gSVTKToH5Type(int svtkt)
{
  switch(svtkt)
    {
    case SVTK_FLOAT:
      return H5T_NATIVE_FLOAT;
      break;
    case SVTK_DOUBLE:
      return H5T_NATIVE_DOUBLE;
      break;
    case SVTK_CHAR:
      return H5T_NATIVE_CHAR;
      break;
    case SVTK_UNSIGNED_CHAR:
      return H5T_NATIVE_UCHAR;
      break;
    case SVTK_INT:
      return H5T_NATIVE_INT;
      break;
    case SVTK_UNSIGNED_INT:
      return H5T_NATIVE_UINT;
      break;
    case SVTK_LONG:
      if(sizeof(long) == 4)
        return H5T_NATIVE_INT; // 32 bits
      return H5T_NATIVE_LONG;  // 64 bits
      break;
    case SVTK_UNSIGNED_LONG:
      if(sizeof(long) == 4)
        return H5T_NATIVE_UINT; // 32 bits
      return H5T_NATIVE_ULONG;  // 64 bits
      break;
    case SVTK_LONG_LONG:
      return H5T_NATIVE_LONG;
      break;
    case SVTK_UNSIGNED_LONG_LONG:
      return H5T_NATIVE_ULONG; // 64 bits
      break;
    case SVTK_ID_TYPE:
      return gHDF5_IDType();
      break;
    default:
    {
      SENSEI_ERROR("the HDF5 type for svtk type enumeration "
                   << svtkt << " is currently not implemented");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    }
  return -1;
}


//
//
//

//
//
//
HDF5VarGuard::HDF5VarGuard(hid_t varID)
  : m_VarID(varID)
{
  m_VarType = H5Dget_type(varID);
  m_VarSpace = H5Dget_space(varID);
}

HDF5VarGuard::~HDF5VarGuard()
{
  H5Dclose(m_VarID);
  H5Sclose(m_VarSpace);
}

void HDF5VarGuard::ReadAll(void *buf)
{
  H5Dread(m_VarID, m_VarType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
}

void HDF5VarGuard::ReadSlice(void *buf,
                             int ndim,
                             const hsize_t *start,
                             const hsize_t *stride,
                             const hsize_t *count,
                             const hsize_t *block)
{
  hid_t memDataSpace = H5Screate_simple(ndim, count, NULL);
  hssize_t bytes = H5Sget_simple_extent_npoints(memDataSpace);

  std::ostringstream  oss;   oss<<"H5BytesRead="<<bytes;
  std::string evtName = oss.str();
  sensei::TimeEvent<128> mark(evtName.c_str());
  H5Sselect_hyperslab(m_VarSpace, H5S_SELECT_SET, start, stride, count, block);

  H5Dread(m_VarID, m_VarType, memDataSpace, m_VarSpace, H5P_DEFAULT, buf);

  H5Sclose(memDataSpace);
}

//
//
//
StreamHandler::StreamHandler(bool readmode,
                             const std::string &hostFile,
                             BasicStream *client)
  : m_TimeStepId(-1)
  , m_InReadMode(readmode)
  , m_FileName(hostFile)
  , m_Client(client)
{
}

//
//
//
DefaultStreamHandler::DefaultStreamHandler(const std::string &hostFile,
    ReadStream *client)
  : StreamHandler(true, hostFile, client)
{
  m_HostFileId =
    H5Fopen(hostFile.c_str(), H5F_ACC_RDONLY, client->m_PropertyListId);

  if(m_HostFileId >= 0)
    client->ReadNativeAttr(senseiHDF5::ATTRNAME_NUM_TIMESTEP,
                           &(m_TimeStepTotal),
                           H5T_NATIVE_UINT,
                           m_HostFileId);
}

DefaultStreamHandler::DefaultStreamHandler(const std::string &hostFile,
    WriteStream *client)
  : StreamHandler(false, hostFile, client)
{
  m_HostFileId = H5Fcreate(
                   hostFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, client->m_PropertyListId);
}

DefaultStreamHandler::~DefaultStreamHandler()
{
  if(m_HostFileId >= 0)
    H5Fclose(m_HostFileId);
}

bool DefaultStreamHandler::IsValid()
{
  if(m_HostFileId < 0)
    return false;

  if(m_TimeStepTotal == 0)
    return false;

  return true;
}

bool DefaultStreamHandler::CloseStream()
{
  if(this->m_TimeStepId > -1)
    {
      H5Gclose(this->m_TimeStepId);
      this->m_TimeStepId = -1;
    }

  return true;
}

/*
bool DefaultStreamHandler::OpenStream()
{
  if (!m_InReadMode)  return false;

  std::string stepName;
  senseiHDF5::gGetTimeStepString(stepName, m_TimeStepCounter);
  m_TimeStepId = H5Gopen(m_HostFileId, stepName.c_str(), H5P_DEFAULT);

  if (m_TimeStepId < 0) return false;

  m_TimeStepCounter ++;

  return true;
}
*/

bool DefaultStreamHandler::AdvanceStream()
{
  std::string stepName;
  gGetTimeStepString(stepName, m_TimeStepCounter);

  if(m_InReadMode)
    {
      CloseStream();
      if(m_TimeStepCounter == m_TimeStepTotal)
        return false;
      m_TimeStepId = H5Gopen(m_HostFileId, stepName.c_str(), H5P_DEFAULT);
    }
  else
    {
      if(m_TimeStepCounter > 0)
        CloseStream();
      m_TimeStepId = H5Gcreate2(
                       m_HostFileId, stepName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }

  if(m_TimeStepId < 0)
    return false;

  m_TimeStepCounter++;

  return true;
}

bool DefaultStreamHandler::Summary()
{
  if(m_InReadMode)
    return true;

  WriteStream *writer = (WriteStream *)m_Client;
  if(!writer->WriteNativeAttr(senseiHDF5::ATTRNAME_NUM_TIMESTEP,
                              &(m_TimeStepCounter),
                              H5T_NATIVE_UINT,
                              m_HostFileId))
    return false;

  return true;
}

// use char instead of lines in the timestep records?
// if using char, do 11111 and use 0 to indicate end... so only need to check
// whether the last char is 0?

//
//
//
PerStepStreamHandler::PerStepStreamHandler(const std::string &hostFile,
    ReadStream *client)
  : StreamHandler(true, hostFile, client)
{
  unsigned int AllowedTimeOutSec = 300;

  // loop until timestep showed
  unsigned int counter = 0;
  while(counter < AllowedTimeOutSec)
    {
      GetCurrAvailStep();
      if(m_NumStepsWritten > 0)
        break;
      sleep(2);
      counter += 5;
    }
}

PerStepStreamHandler::PerStepStreamHandler(const std::string &hostFile,
    WriteStream *client)
  : StreamHandler(false, hostFile, client)
{
  if(m_Client->m_Rank == 0)
    std::remove(m_FileName.c_str());
  MPI_Barrier(m_Client->m_Comm);
}

PerStepStreamHandler::~PerStepStreamHandler() {}

bool PerStepStreamHandler::IsValid()
{
  return (m_NumStepsWritten > 0);
}

void PerStepStreamHandler::GetStepFileName(std::string &stepName, int ts)
{
  stepName = m_FileName + "_s" + std::to_string(ts);
}

bool PerStepStreamHandler::OpenStream()
{
  if(!m_InReadMode)
    return false;

  std::string currStepFileName;
  GetStepFileName(currStepFileName, m_TimeStepCounter);
  m_TimeStepId = H5Fopen(
                   currStepFileName.c_str(), H5F_ACC_RDONLY, m_Client->m_PropertyListId);

  if(m_TimeStepId < 0)
    return false;

  m_TimeStepCounter++;

  return true;
}

bool PerStepStreamHandler::Summary()
{
  if(m_InReadMode)
    {
      MPI_Barrier(m_Client->m_Comm);
      if(m_Client->m_Rank == 0)
        std::remove(m_FileName.c_str());

      return true;
    }

  if(m_Client->m_Rank > 0)
    return true;

  std::ofstream outfile;
  outfile.open(m_FileName, std::ios::out | std::ios::app);

  if(outfile.fail())
    throw std::ios_base::failure(std::strerror(errno));

  outfile << 0;

  outfile.close();

  return true;
}

void PerStepStreamHandler::GetCurrAvailStep()
{
  if(m_AllStepsWritten)
    return;

  int counter = -1;
  if(m_Client->m_Rank == 0)
    {
      std::ifstream curr(m_FileName);
      std::string line;
      if(curr.is_open())
        {
          // while ( getline (curr,line) )
          getline(curr, line);
          curr.close();

          counter = line.size();
          if('0' == line.back())
            {
              counter--;
              m_AllStepsWritten = true;
            }
        } // else {
      // counter = -1; // no such file
      //}
      m_NumStepsWritten = counter;

      counter *= 10;
      if(m_AllStepsWritten)
        counter += 1;
      MPI_Bcast(&counter, 1, MPI_INT, 0, m_Client->m_Comm);
    }
  else
    {
      MPI_Bcast(&counter, 1, MPI_INT, 0, m_Client->m_Comm);
      int c = counter % 10;
      if(1 == c)
        {
          counter = (counter - 1) / 10;
          m_AllStepsWritten = true;
        }
      else
        counter = counter / 10;

      m_NumStepsWritten = counter;
    }
}

void PerStepStreamHandler::UpdateAvailStep()
{
  // testing sleep(m_TimeStepCounter*10);
  if(m_Client->m_Rank > 0)
    {
      return;
    }

  std::ofstream outfile;

  outfile.open(m_FileName, std::ios::out | std::ios::app);

  if(outfile.fail())
    throw std::ios_base::failure(std::strerror(errno));

  outfile << 1;
  outfile.flush();
  outfile.close();
}

bool PerStepStreamHandler::NoMoreStep()
{
  if((m_NumStepsWritten - m_TimeStepCounter) > 0)
    return false;

  if(m_AllStepsWritten && (0 == (m_NumStepsWritten - m_TimeStepCounter)))
    return true;

  unsigned int counter = 0;
  unsigned int MAXWAIT = 900;

  while (true)
  {
    GetCurrAvailStep();
    if ((m_NumStepsWritten < 0) && (m_TimeStepCounter > 0))
      return true; // all finished

    if ((m_NumStepsWritten - m_TimeStepCounter) > 0)
      return false;

    if (m_AllStepsWritten && (0 == (m_NumStepsWritten - m_TimeStepCounter)))
      return true;

    sleep(1);

    counter++;
    if(counter > MAXWAIT)
      return true;
  }
  return true;
}

bool PerStepStreamHandler::CloseStream()
{
  if(m_TimeStepId < 0)
    return true; // nothing to do

  char name[100];
  H5Fget_name(m_TimeStepId, name, 100);

  H5Fclose(m_TimeStepId);
  m_TimeStepId = -1;
  if(!m_InReadMode)
    UpdateAvailStep();

  if((m_InReadMode) && (m_Client->m_Rank == 0))
    {
      // std::cout<<" ... [TODO]: will remove this file when read is done ...
      // "<<name<<std::endl;
      std::remove(name);
    }

  return true;
}

bool PerStepStreamHandler::AdvanceStream()
{
  std::string stepName;
  GetStepFileName(stepName, m_TimeStepCounter);

  if(m_InReadMode)
    {
      CloseStream();
      if(NoMoreStep())
        return false;
      m_TimeStepId =
        H5Fopen(stepName.c_str(), H5F_ACC_RDONLY, m_Client->m_PropertyListId);
    }
  else
    {
      if(m_TimeStepCounter > 0)
        CloseStream();
      m_TimeStepId = H5Fcreate(stepName.c_str(),
                               H5F_ACC_TRUNC,
                               H5P_DEFAULT,
                               m_Client->m_PropertyListId);
    }

  if(m_TimeStepId < 0)
    return false;

  m_TimeStepCounter++;

  return true;
}

//
//
//
BasicStream::BasicStream(MPI_Comm comm, bool streaming)
  : m_Comm(comm)
  , m_StreamingOn(streaming)
{
  MPI_Comm_rank(comm, &m_Rank);
  MPI_Comm_size(comm, &m_Size);

  m_Comm = comm;

  m_PropertyListId = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(m_PropertyListId, comm, MPI_INFO_NULL);

  m_CollectiveTxf = H5P_DEFAULT;
}

void BasicStream::SetCollectiveTxf()
{
  if(m_Size > 1)
    {
      m_CollectiveTxf = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(m_CollectiveTxf, H5FD_MPIO_COLLECTIVE);
    }
}

BasicStream::~BasicStream()
{
  H5Pclose(m_PropertyListId);
  if(H5P_DEFAULT != m_CollectiveTxf)
    H5Pclose(m_CollectiveTxf);

  CloseTimeStep();
  if(m_Streamer != nullptr)
    delete m_Streamer;
}

void BasicStream::CloseTimeStep()
{
  if(m_Streamer)
    m_Streamer->CloseStream();
}

//
//
//
ReadStream::ReadStream(MPI_Comm comm, bool streaming)
  : BasicStream(comm, streaming)
{
  m_TimeStepTotal = 0;
}

ReadStream::~ReadStream()
{
  m_Streamer->Summary();
}

bool ReadStream::AdvanceTimeStep(unsigned long &time_step, double &time)
{
  m_AllMeshInfo.Clear();
  m_AllMeshInfoReceiver.Clear();

  if(!m_Streamer->AdvanceStream())
    return false;

  if(!ReadNativeAttr(
        senseiHDF5::ATTRNAME_TIMESTEP, &time_step, H5T_NATIVE_ULONG, -1))
    return false;
  if(!ReadNativeAttr(senseiHDF5::ATTRNAME_TIME, &time, H5T_NATIVE_DOUBLE, -1))
    return false;

  return true;
}

bool ReadStream::ReadNativeAttr(const std::string &name,
                                void *val,
                                hid_t h5Type,
                                hid_t hid)
{
  if(hid == -1)
    hid = m_Streamer->m_TimeStepId;

  hid_t attr = H5Aopen(hid, name.c_str(), H5P_DEFAULT);
  if(attr < 0)
    {
      SENSEI_ERROR("Failed to open H5 attr: " << name);
      return false;
    }

  H5Aread(attr, h5Type, val);

  H5Aclose(attr);

  return true;
}

bool ReadStream::ReadVar1D(const std::string &name,
                           hsize_t s,
                           hsize_t c,
                           void *data)
{
  hid_t varId = H5Dopen(m_Streamer->m_TimeStepId, name.c_str(), H5P_DEFAULT);

  if(varId < 0)
    {
      SENSEI_ERROR("Failed to open H5 dataset: " << name);
      return false;
    }

  HDF5VarGuard g(varId);

  hsize_t start[1] = { s };
  hsize_t count[1] = { c };
  hsize_t stride[1] = { 1 };

  g.ReadSlice(data, 1, start, stride, count, NULL);

  return true;
}

bool ReadStream::ReadBinary(const std::string &name, sensei::BinaryStream &str)
{
  hid_t varID = H5Dopen(m_Streamer->m_TimeStepId, name.c_str(), H5P_DEFAULT);

  if(varID < 0)
    {
      SENSEI_ERROR("Failed to open H5 dataset: " << name);
      return false;
    }

  HDF5VarGuard g(varID);

  hsize_t nbytes = H5Sget_simple_extent_npoints(g.m_VarSpace);
  str.Resize(nbytes);
  str.SetReadPos(0);
  str.SetWritePos(nbytes);

  std::ostringstream  oss;   oss<<"H5BytesReadBinary="<<nbytes;
  std::string evtName = oss.str();
  sensei::TimeEvent<128> mark(evtName.c_str());
  g.ReadAll(str.GetData());

  return true;
}

bool ReadStream::Init(const std::string &filename)
{
  if(m_StreamingOn)
    m_Streamer = new PerStepStreamHandler(filename, this);
  else
    m_Streamer = new DefaultStreamHandler(filename, this);
  return m_Streamer->IsValid();
}

void ReadStream::Close() {}

bool ReadStream::ReadMetadata(unsigned int &nMesh)
{
  if(!ReadNativeAttr(
        senseiHDF5::ATTRNAME_NUM_MESH, &(nMesh), H5T_NATIVE_UINT, -1))
    return false;

  if(nMesh == 0)
    return false;

  for(unsigned int i = 0; i < nMesh; ++i)
    {
      std::string path;
      gGetNameStr(path, i, "meshdata");

      sensei::BinaryStream bs;
      if(!ReadBinary(path, bs))
        return false;

      sensei::MeshMetadataPtr md = sensei::MeshMetadata::New();
      md->FromStream(bs);

      // add internally generated arrays
      md->ArrayName.push_back("SenderBlockOwner");
      md->ArrayCentering.push_back(svtkDataObject::CELL);
      md->ArrayComponents.push_back(1);
      md->ArrayType.push_back(SVTK_INT);

      md->ArrayName.push_back("ReceiverBlockOwner");
      md->ArrayCentering.push_back(svtkDataObject::CELL);
      md->ArrayComponents.push_back(1);
      md->ArrayType.push_back(SVTK_INT);

      md->NumArrays += 2;

      m_AllMeshInfo.PushBack(md);
      m_AllMeshInfoReceiver.PushBack(md);

    }
  return true;
}

bool ReadStream::ReadSenderMeshMetaData(unsigned int i, sensei::MeshMetadataPtr &ptr)
{
  if(i >= m_AllMeshInfo.Size())
    return false;

  return (gCheckSenseiCall(m_AllMeshInfo.GetMeshMetadata(i, ptr)));
}

bool ReadStream::ReadReceiverMeshMetaData(unsigned int i, sensei::MeshMetadataPtr &ptr)
{
  if(i >= m_AllMeshInfoReceiver.Size())
    return false;

  return (gCheckSenseiCall(m_AllMeshInfoReceiver.GetMeshMetadata(i, ptr)));
}


bool ReadStream::ReadMesh(std::string name,
                          svtkDataObject *&dobj,
                          bool structure_only)
{
  // sensei::MeshMetadataPtr meshMetaPtr;
  // if (!ReadMeshMetadata(name, meshMetaPtr))
  //  return false;

  unsigned int meshId;
  if(m_AllMeshInfo.GetMeshId(name, meshId) < 0)
    return false;

  svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet *>(dobj);

  MeshFlow m(cd, meshId);
  if(!m.ReadFrom(this, structure_only))
    {
      SENSEI_ERROR("Failed to read object " << meshId << name << "\"");
      return false;
    }

  dobj = m.m_VtkPtr;

  return true;
}

bool ReadStream::ReadInArray(const std::string &meshName,
                             int association,
                             const std::string &array_name,
                             svtkDataObject *dobj)
{
  unsigned int meshId;
  if(m_AllMeshInfo.GetMeshId(meshName, meshId) < 0)
    return false;

  MeshFlow m(dynamic_cast<svtkCompositeDataSet *>(dobj), meshId);

  if(!m.ReadArray(this, array_name, association))
    {
      SENSEI_ERROR("Failed to read "
                   << sensei::SVTKUtils::GetAttributesName(association)
                   << " data array \"" << array_name << "\" from object \""
                   << meshName << "\"");
      return false;
    }
  return true;
}

//
//
//
MeshFlow::MeshFlow(svtkCompositeDataSet *cd, unsigned int meshID)
  : m_VtkPtr(cd)
  , m_MeshID(meshID)
{
}

MeshFlow::~MeshFlow() {}

bool MeshFlow::ValidateMetaData(const sensei::MeshMetadataPtr &md)
{
  unsigned int num_arrays = md->NumArrays;
  for(unsigned int i = 0; i < num_arrays; ++i)
    {
      int array_cen = md->ArrayCentering[i];
      if((array_cen != svtkDataObject::POINT) &&
          (array_cen != svtkDataObject::CELL))
        {
          SENSEI_ERROR("Invalid array centering at array " << i);
          return false;
        }
    }

  return true;
}

bool MeshFlow::ReadBlockOwnerArray(ReadStream *reader,
                                   const std::string &array_name,
                                   int association)
{
  sensei::MeshMetadataPtr md;
  if(array_name.compare("SenderBlockOwner") == 0)
    {
      reader->ReadSenderMeshMetaData(m_MeshID, md);
    }
  else if(array_name.compare("ReceiverBlockOwner") == 0)
    {
      reader->ReadReceiverMeshMetaData(m_MeshID, md);
    }
  else
    {
      return false;
    }

  svtkCompositeDataIterator *it = m_VtkPtr->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  unsigned int num_blocks = md->NumBlocks;

  // read each block
  for(unsigned int j = 0; j < num_blocks; ++j)
    {
      // get the block size
      unsigned long long num_elem_local = (association== svtkDataObject::POINT ?
                                           md->BlockNumPoints[j] : md->BlockNumCells[j]);

      // define the variable for a local block
      svtkDataSet *ds = dynamic_cast<svtkDataSet *>(it->GetCurrentDataObject());
      if(ds)
        {
          // create arrays filled with sender and receiver ranks
          svtkDataArray *bo = svtkIntArray::New();
          bo->SetNumberOfTuples(num_elem_local);
          bo->SetName(array_name.c_str());
          bo->FillComponent(0, md->BlockOwner[j]);

          svtkDataSetAttributes *dsa = association == svtkDataObject::POINT ?
                                      dynamic_cast<svtkDataSetAttributes *>(ds->GetPointData()) :
                                      dynamic_cast<svtkDataSetAttributes *>(ds->GetCellData());

          dsa->AddArray(bo);
          bo->Delete();
        }

      // next block
      it->GoToNextItem();
    }

  it->Delete();

  return true;
}

bool MeshFlow::ReadArray(ReadStream *reader,
                         const std::string &array_name,
                         int association)
{
  if(ReadBlockOwnerArray(reader, array_name, association))
    return true;

  sensei::MeshMetadataPtr md;
  reader->ReadReceiverMeshMetaData(m_MeshID, md);

  //unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  if (array_name == TAG_SVTK_GHOST) {
    ArrayFlow arrayFlow(m_MeshID, association, md);
    Load(&arrayFlow, md, reader);
    return true;
  }

  // read data arrays
  for (unsigned int i = 0; i < num_arrays; ++i) {
    // skip all but the requested array
    if ((association != md->ArrayCentering[i]) ||
        (array_name != md->ArrayName[i]))
      continue;

    ArrayFlow arrayFlow(md, m_MeshID, i);
    Load(&arrayFlow, md, reader);
  }

  return true;
}

void MeshFlow::Load(ArrayFlow *arrayFlowPtr, const sensei::MeshMetadataPtr &md,
                    ReadStream *reader) {
  unsigned int num_blocks = md->NumBlocks;

  svtkCompositeDataIterator *it = m_VtkPtr->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  for (unsigned int j = 0; j < num_blocks; ++j) {
    if (md->BlockOwner[j] == reader->m_Rank) {
      arrayFlowPtr->load(j, it, reader);
    }
    arrayFlowPtr->update(j);
    it->GoToNextItem();
  }

  it->Delete();
}


bool MeshFlow::Initialize(const sensei::MeshMetadataPtr &md, ReadStream *input)
{
  //
  m_VtkPtr = nullptr;
  svtkMultiBlockDataSet *mbds = svtkMultiBlockDataSet::New();
  int num_blocks = md->NumBlocks;
  mbds->SetNumberOfBlocks(num_blocks);

  int rank = input->m_Rank;
  /*
   */

  for(int i = 0; i < num_blocks; ++i)
    {
      if(rank == md->BlockOwner[i])
        {
          svtkDataObject *ds = sensei::SVTKUtils::NewDataObject(md->BlockType);
          mbds->SetBlock(md->BlockIds[i], ds);
          ds->Delete();
        }
    }

  m_VtkPtr = mbds;

  return true;
}

bool MeshFlow::ReadFrom(ReadStream *input, bool structure_only)
{
  sensei::MeshMetadataPtr md;
  input->ReadReceiverMeshMetaData(m_MeshID, md);

  unsigned int num_blocks = md->NumBlocks;

  // create the data object
  if(!this->Initialize(md, input))
    {
      SENSEI_ERROR("Failed to initialize data object");
      return -1;
    }

  if(structure_only)
    return true;

  {
    svtkCompositeDataIterator *it = m_VtkPtr->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    WorkerCollection workerPool(md, m_MeshID);
    for(unsigned int j = 0; j < num_blocks; ++j)
      {
        if(input->m_Rank == md->BlockOwner[j])
          {
            workerPool.load(j, it, input);
          }
        workerPool.update(j);
        it->GoToNextItem();
      }

    it->Delete();
  }

  return true;
}

bool MeshFlow::WriteTo(WriteStream *output, const sensei::MeshMetadataPtr &md)
{
  unsigned int num_blocks = md->NumBlocks;
  {
    svtkCompositeDataIterator *it = m_VtkPtr->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    WorkerCollection workerPool(md, m_MeshID);
    for(unsigned int j = 0; j < num_blocks; ++j)
      {
        if(output->m_Rank == md->BlockOwner[j])
          {
            workerPool.unload(j, it, output);
          }
        workerPool.update(j);
        it->GoToNextItem();
      }
    it->Delete();
  }

  {
    unsigned int num_arrays = md->NumArrays;
    for (unsigned int i = 0; i < num_arrays; ++i) {
      ArrayFlow arrayFlow(md, m_MeshID, i);
      Unload(&arrayFlow, md, output);
    }
  }

  {
    if (md->NumGhostCells) {
      ArrayFlow arrayFlow(m_MeshID, svtkDataObject::CELL, md);
      Unload(&arrayFlow, md, output);
    }

    if (md->NumGhostNodes) {
      ArrayFlow arrayFlow(m_MeshID, svtkDataObject::POINT, md);
      Unload(&arrayFlow, md, output);
    }
  }

  return true;
}

void MeshFlow::Unload(ArrayFlow *arrayFlowPtr,
                      const sensei::MeshMetadataPtr &md, 
		      WriteStream *output) 
{
  unsigned int num_blocks = md->NumBlocks;

  svtkCompositeDataIterator *it = m_VtkPtr->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  // ArrayFlow arrayFlow(md, m_MeshID, i);
  for (unsigned int j = 0; j < num_blocks; ++j) {
    if (output->m_Rank == md->BlockOwner[j]) {
      arrayFlowPtr->unload(j, it, output);
    }
    arrayFlowPtr->update(j);
    it->GoToNextItem();
  }

  it->Delete();
}

//
//
//
WorkerCollection::WorkerCollection(const sensei::MeshMetadataPtr &md,
                                   unsigned int meshID)
{
  if(sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md) ||
      sensei::SVTKUtils::Polydata(md))
    {
      m_Workers.push_back(new PointFlow(md, meshID));
    }
  if(sensei::SVTKUtils::Unstructured(md))
    {
      m_Workers.push_back(new UnstructuredCellFlow(md, meshID));
    }
  if(sensei::SVTKUtils::Polydata(md))
    {
      m_Workers.push_back(new PolydataCellFlow(md, meshID));
    }
  if(sensei::SVTKUtils::UniformCartesian(md))
    {
      m_Workers.push_back(new UniformCartesianFlow(md, meshID));
    }
  if(sensei::SVTKUtils::StretchedCartesian(md))
    {
      m_Workers.push_back(new StretchedCartesianFlow(md, meshID));
    }
  if(sensei::SVTKUtils::LogicallyCartesian(md))
    {
      m_Workers.push_back(new LogicallyCartesianFlow(md, meshID));
    }

  if(m_Workers.size() == 0)
    {
      std::cout << "..... Nothing to save to HDF5!! ..... " << std::endl;
    }
}

WorkerCollection::~WorkerCollection()
{
  for(size_t i = 0; i < m_Workers.size(); i++)
    {
      delete m_Workers[i];
    }
  m_Workers.clear();
}

bool WorkerCollection::load(unsigned int block_id,
                            svtkCompositeDataIterator *it,
                            ReadStream *reader)
{
  for(size_t i = 0; i < m_Workers.size(); i++)
    {
      if(!m_Workers[i]->load(block_id, it, reader))
        return false;
    }

  return true;
}

bool WorkerCollection::unload(unsigned int block_id,
                              svtkCompositeDataIterator *it,
                              WriteStream *writer)
{
  for(size_t i = 0; i < m_Workers.size(); i++)
    {
      if(!m_Workers[i]->unload(block_id, it, writer))
        return false;
    }
  return true;
}

bool WorkerCollection::update(unsigned int block_id)
{
  for(size_t i = 0; i < m_Workers.size(); i++)
    {
      if(!m_Workers[i]->update(block_id))
        return false;
    }
  return true;
}
//
//
//
ArrayFlow::ArrayFlow(const sensei::MeshMetadataPtr &md,
                     unsigned int meshID,
                     unsigned int arrayID)
  : SVTKObjectFlow(md, meshID)
  , m_BlockOffset(0)
  , m_ArrayID(arrayID)
{
  m_IsGhostArray = false;

  m_ArrayCenter = md->ArrayCentering[arrayID];
  m_NumArrayComponent = md->ArrayComponents[arrayID];

  gGetArrayNameStr(m_ArrayPath, m_MeshID, arrayID);
  m_ArrayVarID = -1;

  for(int j = 0; j < md->NumBlocks; ++j)
    {
      m_ElementTotal += getLocalElement(j);
    }

  m_ElementTotal *= m_NumArrayComponent;
}

ArrayFlow::ArrayFlow(unsigned int meshID, int GhostCentering,
                     const sensei::MeshMetadataPtr &md)
    : SVTKObjectFlow(md, meshID), m_BlockOffset(0) {
  m_ArrayCenter = GhostCentering;
  m_NumArrayComponent = 1; // md->ArrayComponents[arrayID];

  m_IsGhostArray = true;

  if (GhostCentering == svtkDataObject::CELL)
    gGetNameStr(m_ArrayPath, m_MeshID, "ghostcell");
  else if (GhostCentering == svtkDataObject::POINT)
    gGetNameStr(m_ArrayPath, m_MeshID, "ghostpoint");

  m_ArrayVarID = -1;

  for (int j = 0; j < md->NumBlocks; ++j) {
    m_ElementTotal += getLocalElement(j);
  }

  m_ElementTotal *= m_NumArrayComponent;
}

ArrayFlow::~ArrayFlow()
{
  if(-1 != m_ArrayVarID)
    H5Dclose(m_ArrayVarID);
}

int ArrayFlow::GetArrayType() {
  if (!m_IsGhostArray)
    return m_Metadata->ArrayType[m_ArrayID];
  return SVTK_UNSIGNED_CHAR; // ghost data type
}

const std::string &ArrayFlow::GetArrayName() {
  if (!m_IsGhostArray)
    return m_Metadata->ArrayName[m_ArrayID];
  return TAG_SVTK_GHOST;
}


bool ArrayFlow::load(unsigned int block_id,
                     svtkCompositeDataIterator *it,
                     ReadStream *reader)
{
  unsigned long long num_elem_local =
    m_NumArrayComponent * (m_ArrayCenter == svtkDataObject::POINT
                           ? m_Metadata->BlockNumPoints[block_id]
                           : m_Metadata->BlockNumCells[block_id]);

  uint64_t start = m_BlockOffset;
  uint64_t count = num_elem_local;
  ;

  svtkDataArray *array = svtkDataArray::CreateDataArray(GetArrayType());
  array->SetNumberOfComponents(m_NumArrayComponent);
  array->SetName(GetArrayName().c_str());
  array->SetNumberOfTuples(num_elem_local);

  if(!reader->ReadVar1D(m_ArrayPath, start, count, array->GetVoidPointer(0)))
    return false;

  // pass to svtk
  svtkDataSet *ds = dynamic_cast<svtkDataSet *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id << " rank"
                   << reader->m_Rank);
      return false;
    }

  svtkDataSetAttributes *dsa =
    (m_ArrayCenter == svtkDataObject::POINT)
    ? dynamic_cast<svtkDataSetAttributes *>(ds->GetPointData())
    : dynamic_cast<svtkDataSetAttributes *>(ds->GetCellData());

  dsa->AddArray(array);
  array->Delete();

  return true;
}

bool ArrayFlow::unload(unsigned int block_id,
                       svtkCompositeDataIterator *it,
                       WriteStream *output)
{
  svtkDataSet *ds = dynamic_cast<svtkDataSet *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      // dob->Print(std::cerr);
      m_Metadata->ToStream(std::cerr);
      it->Print(std::cerr);
      svtkDataObject *d = it->GetCurrentDataObject();
      d->Print(std::cerr);
      return false;
    }

  svtkDataSetAttributes *dsa =
    m_ArrayCenter == svtkDataObject::POINT
    ? dynamic_cast<svtkDataSetAttributes *>(ds->GetPointData())
    : dynamic_cast<svtkDataSetAttributes *>(ds->GetCellData());

  svtkDataArray *da = dsa->GetArray(GetArrayName().c_str()); 
  if(!da)
    {
      SENSEI_ERROR("Failed to get array \"" << GetArrayName()
                   << "\"");
      return false;
    }

  hid_t h5TypeCurrArray = gGetHDF5Type(da);
  unsigned long long num_elem_local =
    m_NumArrayComponent * getLocalElement(block_id);

  HDF5SpaceGuard arraySpace(m_ElementTotal, m_BlockOffset, num_elem_local);

  // if (-1 == m_ArrayVarID)
  // m_ArrayVarID = output->CreateVar(m_ArrayPath, arraySpace, h5TypeCurrArray);

  output->WriteVar(m_ArrayVarID,
                   m_ArrayPath,
                   arraySpace,
                   h5TypeCurrArray,
                   da->GetVoidPointer(0));

  return true;
}

unsigned long long ArrayFlow::getLocalElement(unsigned int block_id)
{
  return (m_ArrayCenter == svtkDataObject::POINT
          ? m_Metadata->BlockNumPoints[block_id]
          : m_Metadata->BlockNumCells[block_id]);
}

bool ArrayFlow::update(unsigned int block_id)
{
  unsigned long long num_elem_local =
    m_NumArrayComponent * getLocalElement(block_id);

  m_BlockOffset += num_elem_local;

  return true;
}

//
//
//
PolydataCellFlow::PolydataCellFlow(const sensei::MeshMetadataPtr &md,
                                   unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
{
  m_CellTypesBlockOffset = 0;
  m_CellArrayBlockOffset = 0;
}

PolydataCellFlow::~PolydataCellFlow() {}

bool PolydataCellFlow::load(unsigned int block_id,
                            svtkCompositeDataIterator *it,
                            ReadStream *reader)
{
  // /data_object_<id>/cell_types
  unsigned long long cell_array_size_local =
    m_Metadata->BlockCellArraySize[block_id];
  unsigned long long num_cells_local = m_Metadata->BlockNumCells[block_id];

  std::vector<svtkIdType> cell_array(cell_array_size_local);
  std::vector<unsigned char> cell_types(num_cells_local);

  uint64_t ct_start = m_CellTypesBlockOffset;
  uint64_t ct_count = num_cells_local;
  if(!reader->ReadVar1D(
        m_CellTypeVarName, ct_start, ct_count, cell_types.data()))
    return false;

  uint64_t ca_start = m_CellArrayBlockOffset;
  uint64_t ca_count = cell_array_size_local;
  ;
  if(!reader->ReadVar1D(
        m_CellArrayVarName, ca_start, ca_count, cell_array.data()))
    return false;

  unsigned char *p_types = cell_types.data();
  svtkIdType *p_cells = cell_array.data();

  // find first and last vert and number of verts
  unsigned long i = 0;
  unsigned long n_verts = 0;
  svtkIdType *vert_begin = p_cells;
  while((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
    {
      p_cells += p_cells[0] + 1;
      ++n_verts;
      ++i;
    }
  svtkIdType *vert_end = p_cells;

  // find first and last line and number of lines
  unsigned long n_lines = 0;
  svtkIdType *line_begin = p_cells;
  while((i < num_cells_local) && (p_types[i] == SVTK_LINE))
    {
      p_cells += p_cells[0] + 1;
      ++n_lines;
      ++i;
    }
  svtkIdType *line_end = p_cells;

  // find first and last poly and number of polys
  unsigned long n_polys = 0;
  svtkIdType *poly_begin = p_cells;
  while((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
    {
      p_cells += p_cells[0] + 1;
      ++n_polys;
      ++i;
    }
  svtkIdType *poly_end = p_cells;

  // find first and last strip and number of strips
  unsigned long n_strips = 0;
  svtkIdType *strip_begin = p_cells;
  while((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
    {
      p_cells += p_cells[0] + 1;
      ++n_strips;
      ++i;
    }
  svtkIdType *strip_end = p_cells;

  // pass into svtk
  svtkPolyData *pd = dynamic_cast<svtkPolyData *>(it->GetCurrentDataObject());
  if(!pd)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      return -1;
    }

  // pass verts
  unsigned long n_tups = vert_end - vert_begin;
  svtkIdTypeArray *verts = svtkIdTypeArray::New();
  verts->SetNumberOfTuples(n_tups);
  svtkIdType *p_verts = verts->GetPointer(0);

  for(unsigned long j = 0; j < n_tups; ++j)
    p_verts[j] = vert_begin[j];

  svtkCellArray *ca = svtkCellArray::New();
  ca->SetCells(n_verts, verts);
  verts->Delete();

  pd->SetVerts(ca);
  ca->Delete();

  // pass lines
  n_tups = line_end - line_begin;
  svtkIdTypeArray *lines = svtkIdTypeArray::New();
  lines->SetNumberOfTuples(n_tups);
  svtkIdType *p_lines = lines->GetPointer(0);

  for(unsigned long j = 0; j < n_tups; ++j)
    p_lines[j] = line_begin[j];

  ca = svtkCellArray::New();
  ca->SetCells(n_lines, lines);
  lines->Delete();

  pd->SetLines(ca);
  ca->Delete();

  // pass polys
  n_tups = poly_end - poly_begin;
  svtkIdTypeArray *polys = svtkIdTypeArray::New();
  polys->SetNumberOfTuples(n_tups);
  svtkIdType *p_polys = polys->GetPointer(0);

  for(unsigned long j = 0; j < n_tups; ++j)
    p_polys[j] = poly_begin[j];

  ca = svtkCellArray::New();
  ca->SetCells(n_polys, polys);
  polys->Delete();

  pd->SetPolys(ca);
  ca->Delete();

  // pass strips
  n_tups = strip_end - strip_begin;
  svtkIdTypeArray *strips = svtkIdTypeArray::New();
  strips->SetNumberOfTuples(n_tups);
  svtkIdType *p_strips = strips->GetPointer(0);

  for(unsigned long j = 0; j < n_tups; ++j)
    p_strips[j] = strip_begin[j];

  ca = svtkCellArray::New();
  ca->SetCells(n_strips, strips);
  strips->Delete();

  pd->SetStrips(ca);
  ca->Delete();

  pd->BuildCells();

  return true;
}

bool PolydataCellFlow::unload(unsigned int block_id,
                              svtkCompositeDataIterator *it,
                              WriteStream *output)
{
  hid_t h5TypeCellArray = senseiHDF5::gHDF5_IDType();
  hid_t h5TypeCellType = H5T_NATIVE_CHAR;

  unsigned long long num_cells_local = m_Metadata->BlockNumCells[block_id];
  unsigned long long cell_array_size_local =
    m_Metadata->BlockCellArraySize[block_id];

  svtkPolyData *pd = dynamic_cast<svtkPolyData *>(it->GetCurrentDataObject());

  if(!pd)
    {
      SENSEI_ERROR("Failed to get poly data from block " << block_id);
      return false;
    }

  std::vector<char> types;
  std::vector<svtkIdType> cells;

  svtkIdType nv = pd->GetNumberOfVerts();
  if(nv)
    {
      types.insert(types.end(), nv, SVTK_VERTEX);
      svtkIdType *pv = pd->GetVerts()->GetData()->GetPointer(0);
      cells.insert(
        cells.end(), pv, pv + pd->GetVerts()->GetData()->GetNumberOfTuples());
    }

  svtkIdType nl = pd->GetNumberOfLines();
  if(nl)
    {
      types.insert(types.end(), nl, SVTK_LINE);
      svtkIdType *pl = pd->GetLines()->GetData()->GetPointer(0);
      cells.insert(
        cells.end(), pl, pl + pd->GetLines()->GetData()->GetNumberOfTuples());
    }

  svtkIdType np = pd->GetNumberOfPolys();
  if(np)
    {
      types.insert(types.end(), np, SVTK_POLYGON);
      svtkIdType *pp = pd->GetPolys()->GetData()->GetPointer(0);
      cells.insert(
        cells.end(), pp, pp + pd->GetPolys()->GetData()->GetNumberOfTuples());
    }

  svtkIdType ns = pd->GetNumberOfStrips();
  if(ns)
    {
      types.insert(types.end(), ns, SVTK_TRIANGLE_STRIP);
      svtkIdType *ps = pd->GetStrips()->GetData()->GetPointer(0);
      cells.insert(
        cells.end(), ps, ps + pd->GetStrips()->GetData()->GetNumberOfTuples());
    }

  HDF5SpaceGuard cellArraySpace(
    m_TotalArraySize, m_CellArrayBlockOffset, cell_array_size_local);

  // if (-1 == m_CellArrayVarID)
  // m_CellArrayVarID = output->CreateVar(    m_CellArrayVarName,
  // cellArraySpace, h5TypeCellArray);
  output->WriteVar(m_CellArrayVarID,
                   m_CellArrayVarName,
                   cellArraySpace,
                   h5TypeCellArray,
                   cells.data());

  HDF5SpaceGuard cellTypeSpace(
    m_TotalCell, m_CellTypesBlockOffset, num_cells_local);

  // if (-1 == m_CellTypeVarID)
  // m_CellTypeVarID = output->CreateVar(m_CellTypeVarName, cellTypeSpace,
  // h5TypeCellType);
  output->WriteVar(m_CellTypeVarID,
                   m_CellTypeVarName,
                   cellTypeSpace,
                   h5TypeCellType,
                   types.data());

  return true;
}

bool PolydataCellFlow::update(unsigned int block_id)
{
  m_CellTypesBlockOffset += m_Metadata->BlockNumCells[block_id];
  ;
  m_CellArrayBlockOffset += m_Metadata->BlockCellArraySize[block_id];

  return true;
}

//
//
//
UnstructuredCellFlow::UnstructuredCellFlow(const sensei::MeshMetadataPtr &md,
    unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
{
  m_CellTypesBlockOffset = 0;
  m_CellArrayBlockOffset = 0;
}

bool UnstructuredCellFlow::load(unsigned int block_id,
                                svtkCompositeDataIterator *it,
                                ReadStream *reader)
{
  // /data_object_<id>/cell_types
  unsigned long long cell_array_size_local =
    m_Metadata->BlockCellArraySize[block_id];
  unsigned long long num_cells_local = m_Metadata->BlockNumCells[block_id];
  uint64_t ct_start = m_CellTypesBlockOffset;
  uint64_t ct_count = num_cells_local;

  svtkUnsignedCharArray *cell_types = svtkUnsignedCharArray::New();
  cell_types->SetNumberOfComponents(1);
  cell_types->SetNumberOfTuples(ct_count);
  cell_types->SetName("cell_types");

  if(!reader->ReadVar1D(
        m_CellTypeVarName, ct_start, ct_count, cell_types->GetVoidPointer(0)))
    return false;

  // /data_object_<id>/cell_array
  uint64_t ca_start = m_CellArrayBlockOffset;
  uint64_t ca_count = cell_array_size_local;
  ;

  svtkIdTypeArray *cell_array = svtkIdTypeArray::New();
  cell_array->SetNumberOfComponents(1);
  cell_array->SetNumberOfTuples(cell_array_size_local);
  cell_array->SetName("cell_array");

  if(!reader->ReadVar1D(
        m_CellArrayVarName, ca_start, ca_count, cell_array->GetVoidPointer(0)))
    return false;

  svtkUnstructuredGrid *ds =
    dynamic_cast<svtkUnstructuredGrid *>(it->GetCurrentDataObject());

  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      return -1;
    }
  // build locations
  svtkIdTypeArray *cell_locs = svtkIdTypeArray::New();
  cell_locs->SetNumberOfTuples(m_Metadata->BlockNumCells[block_id]);
  svtkIdType *p_locs = cell_locs->GetPointer(0);
  svtkIdType *p_cells = cell_array->GetPointer(0);
  p_locs[0] = 0;
  for(unsigned long i = 1; i < num_cells_local; ++i)
    p_locs[i] = p_locs[i - 1] + p_cells[p_locs[i - 1]] + 1;

  // pass types, cell_locs, and cells
  svtkCellArray *ca = svtkCellArray::New();
  ca->SetCells(num_cells_local, cell_array);
  cell_array->Delete();

  ds->SetCells(cell_types, cell_locs, ca);

  cell_locs->Delete();
  cell_array->Delete();
  cell_types->Delete();

  return true;
}

bool UnstructuredCellFlow::update(unsigned int block_id)
{
  m_CellTypesBlockOffset += m_Metadata->BlockNumCells[block_id];
  ;
  m_CellArrayBlockOffset += m_Metadata->BlockCellArraySize[block_id];

  return true;
}

bool UnstructuredCellFlow::unload(unsigned int block_id,
                                  svtkCompositeDataIterator *it,
                                  WriteStream *output)
{
  hid_t h5TypeCellArray = senseiHDF5::gHDF5_IDType();
  hid_t h5TypeCellType = H5T_NATIVE_CHAR;

  unsigned long long num_cells_local = m_Metadata->BlockNumCells[block_id];
  unsigned long long cell_array_size_local =
    m_Metadata->BlockCellArraySize[block_id];

  svtkUnstructuredGrid *ds =
    dynamic_cast<svtkUnstructuredGrid *>(it->GetCurrentDataObject());

  if(!ds)
    {
      SENSEI_ERROR("Failed to get unstructured block " << block_id);
      return false;
    }

  HDF5SpaceGuard cellArraySpace(
    m_TotalArraySize, m_CellArrayBlockOffset, cell_array_size_local);
  output->WriteVar(m_CellArrayVarID,
                   m_CellArrayVarName,
                   cellArraySpace,
                   h5TypeCellArray,
                   ds->GetCells()->GetData()->GetVoidPointer(0));

  HDF5SpaceGuard cellTypeSpace(
    m_TotalCell, m_CellTypesBlockOffset, num_cells_local);
  output->WriteVar(m_CellTypeVarID,
                   m_CellTypeVarName,
                   cellTypeSpace,
                   h5TypeCellType,
                   ds->GetCellTypesArray()->GetVoidPointer(0));

  return true;
}

//
//
//
SVTKObjectFlow::SVTKObjectFlow(const sensei::MeshMetadataPtr &md,
                             unsigned int meshID)
  : m_Metadata(md)
  , m_MeshID(meshID)
{
  if (md->NumBlocks != static_cast<long>(md->BlockCellArraySize.size()))
    {
    return;
    }

  gGetNameStr(m_CellTypeVarName, m_MeshID, "cell_types");
  gGetNameStr(m_CellArrayVarName, m_MeshID, "cell_array");
  gGetNameStr(m_PointVarName, m_MeshID, "points");

  unsigned int num_blocks = md->NumBlocks;
  m_TotalCell = 0;
  m_TotalArraySize = 0;

  for(unsigned int j = 0; j < num_blocks; ++j)
    {
    m_TotalCell += md->BlockNumCells[j];
    m_TotalArraySize += md->BlockCellArraySize[j];
    }

  m_PointType = senseiHDF5::gSVTKToH5Type(m_Metadata->CoordinateType);
}

SVTKObjectFlow::~SVTKObjectFlow()
{
  if(-1 != m_CellTypeVarID)
    H5Dclose(m_CellTypeVarID);

  if(-1 != m_CellArrayVarID)
    H5Dclose(m_CellArrayVarID);

  if(-1 != m_PointVarID)
    H5Dclose(m_PointVarID);
}

//
//
//
PointFlow::PointFlow(const sensei::MeshMetadataPtr &md, unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
  , m_BlockOffset(0)
  , m_GlobalTotal(0)
{
  unsigned int num_blocks = md->NumBlocks;
  // calc global size

  for(unsigned int j = 0; j < num_blocks; ++j)
    {
      m_GlobalTotal += md->BlockNumPoints[j];
    }
}

bool PointFlow::load(unsigned int block_id,
                     svtkCompositeDataIterator *it,
                     ReadStream *reader)
{
  uint64_t start = 3 * m_BlockOffset;
  uint64_t count = 3 * m_Metadata->BlockNumPoints[block_id];

  svtkDataArray *points =
    svtkDataArray::CreateDataArray(m_Metadata->CoordinateType);
  points->SetNumberOfComponents(3);
  points->SetNumberOfTuples(m_Metadata->BlockNumPoints[block_id]);
  points->SetName("points");

  // std::string path = ons + "points";
  // std::string path;

  if(!reader->ReadVar1D(
        m_PointVarName, start, count, points->GetVoidPointer(0)))
    return false;

  // pass into svtk
  svtkPoints *pts = svtkPoints::New();
  pts->SetData(points);
  points->Delete();

  svtkPointSet *ds = dynamic_cast<svtkPointSet *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      return -1;
    }

  ds->SetPoints(pts);
  pts->Delete();

  return true;
}

bool PointFlow::unload(unsigned int block_id,
                       svtkCompositeDataIterator *it,
                       WriteStream *output)
{
  uint64_t start = 3 * m_BlockOffset;
  uint64_t count = 3 * m_Metadata->BlockNumPoints[block_id];

  svtkPointSet *ds = dynamic_cast<svtkPointSet *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      return false;
    }

  HDF5SpaceGuard space(3 * m_GlobalTotal, start, count);

  // if (-1 == m_PointVarID)
  // m_PointVarID = output->CreateVar(m_PointVarName, space, m_PointType);

  output->WriteVar(m_PointVarID,
                   m_PointVarName,
                   space,
                   m_PointType,
                   ds->GetPoints()->GetData()->GetVoidPointer(0));

  return true;
}

bool PointFlow::update(unsigned int j)
{
  m_BlockOffset += m_Metadata->BlockNumPoints[j];
  return true;
}

//
//
//

UniformCartesianFlow::UniformCartesianFlow(const sensei::MeshMetadataPtr &md,
    unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
{
  gGetNameStr(m_OriginPath, m_MeshID, "origin");
  gGetNameStr(m_SpacingPath, m_MeshID, "spacing");

  m_OriginVarID = -1;
  m_SpacingVarID = -1;
}

UniformCartesianFlow::~UniformCartesianFlow()
{
  if(-1 != m_OriginVarID)
    H5Dclose(m_OriginVarID);

  if(-1 != m_SpacingVarID)
    H5Dclose(m_SpacingVarID);
}

bool UniformCartesianFlow::load(unsigned int block_id,
                                svtkCompositeDataIterator *it,
                                ReadStream *reader)
{
  uint64_t triplet_start = 3 * block_id;
  uint64_t triplet_count = 3;

  double x0[3] = { 0.0 };
  if(!reader->ReadVar1D(m_OriginPath, triplet_start, triplet_count, x0))
    return false;

  double dx[3] = { 0.0 };
  if(!reader->ReadVar1D(m_SpacingPath, triplet_start, triplet_count, dx))
    return false;

  // update the svtk object
  svtkImageData *ds = dynamic_cast<svtkImageData *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id << " not image data");
      return false;
    }

  ds->SetOrigin(x0);
  ds->SetSpacing(dx);

  return true;
}

bool UniformCartesianFlow::unload(unsigned int block_id,
                                  svtkCompositeDataIterator *it,
                                  WriteStream *output)
{
  unsigned int num_blocks = m_Metadata->NumBlocks;

  HDF5SpaceGuard space(3 * num_blocks, 3 * block_id, 3);

  svtkImageData *ds = dynamic_cast<svtkImageData *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id << " not image data");
      return false;
    }

  // if (0 == block_id) {
  // m_OriginVarID  = output->CreateVar(m_OriginPath, space, H5T_NATIVE_DOUBLE);
  // m_SpacingVarID = output->CreateVar(m_SpacingPath, space,
  // H5T_NATIVE_DOUBLE);
  //}

  output->WriteVar(
    m_OriginVarID, m_OriginPath, space, H5T_NATIVE_DOUBLE, ds->GetOrigin());
  output->WriteVar(
    m_SpacingVarID, m_SpacingPath, space, H5T_NATIVE_DOUBLE, ds->GetSpacing());

  return true;
}

//
//
//
LogicallyCartesianFlow::LogicallyCartesianFlow(
  const sensei::MeshMetadataPtr &md,
  unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
{
  gGetNameStr(m_ExtentPath, m_MeshID, "extent");
  m_ExtentID = -1;
}

LogicallyCartesianFlow::~LogicallyCartesianFlow()
{
  if(-1 < m_ExtentID)
    H5Dclose(m_ExtentID);
}

bool LogicallyCartesianFlow::unload(unsigned int block_id,
                                    svtkCompositeDataIterator *it,
                                    WriteStream *output)
{
  svtkDataObject *dobj = it->GetCurrentDataObject();
  if(!dobj)
    {
      SENSEI_ERROR("Failed to get logically cartesian data from block"
                   << block_id);
      return false;
    }

  unsigned int num_blocks = m_Metadata->NumBlocks;
  HDF5SpaceGuard space(6 * num_blocks, 6 * block_id, 6);

  // if (-1 == m_ExtentID)
  // m_ExtentID = output->CreateVar(m_ExtentPath, space, H5T_NATIVE_INT);

  switch(m_Metadata->BlockType)
    {
    case SVTK_RECTILINEAR_GRID:
      output->WriteVar(m_ExtentID,
                       m_ExtentPath,
                       space,
                       H5T_NATIVE_INT,
                       dynamic_cast<svtkRectilinearGrid *>(dobj)->GetExtent());
      break;
    case SVTK_IMAGE_DATA:
      output->WriteVar(m_ExtentID,
                       m_ExtentPath,
                       space,
                       H5T_NATIVE_INT,
                       dynamic_cast<svtkImageData *>(dobj)->GetExtent());
      break;
    case SVTK_STRUCTURED_GRID:
      output->WriteVar(m_ExtentID,
                       m_ExtentPath,
                       space,
                       H5T_NATIVE_INT,
                       dynamic_cast<svtkStructuredGrid *>(dobj)->GetExtent());
      break;
    }

  return true;
}

bool LogicallyCartesianFlow::load(unsigned int block_id,
                                  svtkCompositeDataIterator *it,
                                  ReadStream *reader)
{
  uint64_t hexlet_start = 6 * block_id;
  uint64_t hexlet_count = 6;

  int ext[6] = { 0 };
  if(!reader->ReadVar1D(m_ExtentPath, hexlet_start, hexlet_count, ext))
    return false;

  svtkDataObject *dobj = it->GetCurrentDataObject();
  if(!dobj)
    {
      SENSEI_ERROR("Failed to get block " << block_id);
      return false;
    }
  switch(m_Metadata->BlockType)
    {
    case SVTK_RECTILINEAR_GRID:
      dynamic_cast<svtkRectilinearGrid *>(dobj)->SetExtent(ext);
      break;
    case SVTK_IMAGE_DATA:
      dynamic_cast<svtkImageData *>(dobj)->SetExtent(ext);
      break;
    case SVTK_STRUCTURED_GRID:
      dynamic_cast<svtkStructuredGrid *>(dobj)->SetExtent(ext);
      break;
    }

  return true;
}

//
//
//
StretchedCartesianFlow::StretchedCartesianFlow(
  const sensei::MeshMetadataPtr &md,
  unsigned int meshID)
  : SVTKObjectFlow(md, meshID)
{
  gGetNameStr(m_XPath, m_MeshID, "x_coords");
  gGetNameStr(m_YPath, m_MeshID, "y_coords");
  gGetNameStr(m_ZPath, m_MeshID, "z_coords");

  unsigned long long temp[3];
  for(int j = 0; j < m_Metadata->NumBlocks; ++j)
    {
      GetLocal(j, temp);
      m_Total[0] += temp[0];
      m_Total[1] += temp[1];
      m_Total[2] += temp[2];
    }
}

StretchedCartesianFlow::~StretchedCartesianFlow()
{
  if(-1 != m_PosID[0])
    {
      H5Dclose(m_PosID[0]);
      H5Dclose(m_PosID[1]);
      H5Dclose(m_PosID[2]);
    }
}
void StretchedCartesianFlow::GetLocal(int block_id,
                                      unsigned long long (&out)[3])
{
  int *ext = m_Metadata->BlockExtents[block_id].data();
  out[0] = ext[1] - ext[0] + 2;
  out[1] = ext[3] - ext[2] + 2;
  out[2] = ext[5] - ext[4] + 2;
}

bool StretchedCartesianFlow::load(unsigned int block_id,
                                  svtkCompositeDataIterator *it,
                                  ReadStream *reader)
{
  unsigned long long local[3];
  GetLocal(block_id, local);

  svtkRectilinearGrid *ds =
    dynamic_cast<svtkRectilinearGrid *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get rectilinear data fromblock " << block_id);
      return false;
    }

  svtkDataArray *x_coords =
    svtkDataArray::CreateDataArray(m_Metadata->CoordinateType);
  x_coords->SetNumberOfComponents(1);
  x_coords->SetNumberOfTuples(local[0]);
  x_coords->SetName("x_coords");

  svtkDataArray *y_coords =
    svtkDataArray::CreateDataArray(m_Metadata->CoordinateType);
  y_coords->SetNumberOfComponents(1);
  y_coords->SetNumberOfTuples(local[1]);
  y_coords->SetName("y_coords");

  svtkDataArray *z_coords =
    svtkDataArray::CreateDataArray(m_Metadata->CoordinateType);
  z_coords->SetNumberOfComponents(1);
  z_coords->SetNumberOfTuples(local[2]);
  z_coords->SetName("z_coords");

  if(!reader->ReadVar1D(
        m_XPath, m_BlockOffset[0], local[0], x_coords->GetVoidPointer(0)))
    return false;
  if(!reader->ReadVar1D(
        m_YPath, m_BlockOffset[1], local[1], y_coords->GetVoidPointer(0)))
    return false;
  if(!reader->ReadVar1D(
        m_ZPath, m_BlockOffset[2], local[2], z_coords->GetVoidPointer(0)))
    return false;

  ds->SetXCoordinates(x_coords);
  ds->SetYCoordinates(y_coords);
  ds->SetZCoordinates(z_coords);

  x_coords->Delete();
  y_coords->Delete();
  z_coords->Delete();

  return true;
}

bool StretchedCartesianFlow::unload(unsigned int block_id,
                                    svtkCompositeDataIterator *it,
                                    WriteStream *output)
{
  unsigned long long local[3];
  GetLocal(block_id, local);

  svtkRectilinearGrid *ds =
    dynamic_cast<svtkRectilinearGrid *>(it->GetCurrentDataObject());
  if(!ds)
    {
      SENSEI_ERROR("Failed to get block " << block_id << " not unstructured");
      return false;
    }

  {
    HDF5SpaceGuard spaceX(m_Total[0], m_BlockOffset[0], local[0]);
    // if (-1 == m_PosID[0])
    // m_PosID[0] = output->CreateVar(m_XPath, spaceX, m_PointType);

    output->WriteVar(m_PosID[0],
                     m_XPath,
                     spaceX,
                     m_PointType,
                     ds->GetXCoordinates()->GetVoidPointer(0));
  }
  {
    HDF5SpaceGuard spaceY(m_Total[1], m_BlockOffset[1], local[1]);
    // if (-1 == m_PosID[1])
    // m_PosID[1] = output->CreateVar(m_YPath, spaceY, m_PointType);

    output->WriteVar(m_PosID[1],
                     m_YPath,
                     spaceY,
                     m_PointType,
                     ds->GetYCoordinates()->GetVoidPointer(0));
  }
  {
    HDF5SpaceGuard spaceZ(m_Total[2], m_BlockOffset[2], local[2]);
    // if (-1 == m_PosID[2])
    // m_PosID[2] = output->CreateVar(m_ZPath, spaceZ, m_PointType);
    output->WriteVar(m_PosID[2],
                     m_ZPath,
                     spaceZ,
                     m_PointType,
                     ds->GetZCoordinates()->GetVoidPointer(0));
  }

  return true;
}

bool StretchedCartesianFlow::update(unsigned int block_id)
{
  unsigned long long temp[3];
  GetLocal(block_id, temp);

  m_BlockOffset[0] += temp[0];
  m_BlockOffset[1] += temp[1];
  m_BlockOffset[2] += temp[2];

  return true;
}

//
//
//
WriteStream::WriteStream(MPI_Comm comm, bool streaming)
  : BasicStream(comm, streaming)
{
  m_MeshCounter = 0;
}

bool WriteStream::Init(const std::string &filename)
{
  if(m_StreamingOn)
    m_Streamer = new PerStepStreamHandler(filename, this);
  else
    m_Streamer = new DefaultStreamHandler(filename, this);

  return m_Streamer->IsValid();
}

bool WriteStream::AdvanceTimeStep(unsigned long &time_step, double &time)
{
  if(m_Streamer->m_TimeStepCounter > 0)
    WriteNativeAttr(
      senseiHDF5::ATTRNAME_NUM_MESH, &(m_MeshCounter), H5T_NATIVE_UINT, -1);

  m_MeshCounter = 0;
  m_Streamer->AdvanceStream();

  WriteNativeAttr(
    senseiHDF5::ATTRNAME_TIMESTEP, &time_step, H5T_NATIVE_ULONG, -1);
  WriteNativeAttr(senseiHDF5::ATTRNAME_TIME, &time, H5T_NATIVE_DOUBLE, -1);

  return true;
}

bool WriteStream::WriteNativeAttr(const std::string &name,
                                  void *val,
                                  hid_t h5Type,
                                  hid_t owner)
{
  if(owner == -1)
    owner = m_Streamer->m_TimeStepId;

  hid_t s = H5Screate(H5S_SCALAR);
  hid_t attr =
    H5Acreate(owner, name.c_str(), h5Type, s, H5P_DEFAULT, H5P_DEFAULT);

  if(attr < 0)
    return false;

  H5Awrite(attr, h5Type, val);

  H5Sclose(s);
  H5Aclose(attr);

  return true;
}

hid_t WriteStream::CreateVar(const std::string &name,
                             const HDF5SpaceGuard &space,
                             hid_t h5Type)
{
  hid_t varID = H5Dcreate(m_Streamer->m_TimeStepId,
                          name.c_str(),
                          h5Type,
                          space.m_FileSpaceID,
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);

  return varID;
}

bool WriteStream::WriteVar(hid_t &varID,
                           const std::string &name,
                           const HDF5SpaceGuard &space,
                           hid_t h5Type,
                           void *data)
{
  hsize_t bytes= H5Sget_simple_extent_npoints(space.m_MemSpaceID);
  std::ostringstream  oss;   oss<<"H5BytesWrote="<<bytes;
  //oss<<" WVrank="<<m_Rank<<"  name=["<<name<<"]"<<varID;
  //std::cout<< oss.str()<<std::endl;
  std::string evtName = oss.str();
  sensei::TimeEvent<128> mark(evtName.c_str());

  if(-1 == varID)
    varID = CreateVar(name, space, h5Type);

  H5Dwrite(varID,
           h5Type,
           space.m_MemSpaceID,
           space.m_FileSpaceID,
           m_CollectiveTxf,
           data);

  return true;
}

/*
bool WriteStream::WriteVar(const std::string& name,
                             const HDF5SpaceGuard& space,
                             hid_t h5Type,
                             void* data)
{
  hid_t varID = H5Dcreate(m_Streamer->m_TimeStepId,
                          name.c_str(),
                          h5Type,
                          space.m_FileSpaceID,
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);

  WriteVar(varID, space, h5Type, data);

  H5Dclose(varID);

  return true;
}
*/

bool WriteStream::WriteMetadata(sensei::MeshMetadataPtr &md)
{
  std::string path;
  gGetNameStr(path, m_MeshCounter, "meshdata");

  sensei::BinaryStream bs;
  md->ToStream(bs);

  WriteBinary(path, bs);
  return true;
}

// --------------------------------------------------------------------------
WriteStream::~WriteStream()
{
  if(m_Streamer->m_TimeStepCounter > 0)
    {
      WriteNativeAttr(
        senseiHDF5::ATTRNAME_NUM_MESH, &(m_MeshCounter), H5T_NATIVE_UINT, -1);
      CloseTimeStep();
    }
  m_Streamer->Summary();
}

// --------------------------------------------------------------------------
bool WriteStream::WriteBinary(const std::string &name,
                              sensei::BinaryStream &str)
{
  std::ostringstream  oss;   oss<<"H5BytesWroteBinary="<<str.Size();
  std::string evtName=oss.str();
  sensei::TimeEvent<128> mark(evtName.c_str());

  hid_t h5Type = H5T_NATIVE_CHAR;

  hsize_t strlen[1] = { str.Size() };
  hid_t fileSpace = H5Screate_simple(1, strlen, NULL);

  hid_t varID = H5Dcreate(m_Streamer->m_TimeStepId,
                          name.c_str(),
                          h5Type,
                          fileSpace,
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);

  H5Dwrite(varID, h5Type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, str.GetData());
  H5Dclose(varID);

  return true;
}

bool WriteStream::WriteMesh(sensei::MeshMetadataPtr &md,
                            svtkCompositeDataSet *svtkPtr)
{
  std::string meshName;
  gGetNameStr(meshName, m_MeshCounter, "");

  hid_t meshID = H5Gcreate2(m_Streamer->m_TimeStepId,
                            meshName.c_str(),
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

  if(meshID < 0)
    return false;

  HDF5GroupGuard g(meshID);

  WriteMetadata(md);

  MeshFlow m(svtkPtr, m_MeshCounter);
  m.WriteTo(this, md);

  m_MeshCounter++;
  return true;
}

} // namespace senseiHDF5
