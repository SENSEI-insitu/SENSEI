#include "HDF5DataAdaptor.h"

#include "Error.h"
#include "Profiler.h"

#include "BlockPartitioner.h"
#include "MeshMetadata.h"
#include "Partitioner.h"
#include "SVTKUtils.h"
#include <svtkCompositeDataIterator.h>
#include <svtkDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkInformation.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkObjectFactory.h>
#include <svtkSmartPointer.h>

#include <pugixml.hpp>
#include <sstream>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(HDF5DataAdaptor);

//----------------------------------------------------------------------------
HDF5DataAdaptor::HDF5DataAdaptor()
  : m_HDF5Reader(nullptr)
{
}

//----------------------------------------------------------------------------
HDF5DataAdaptor::~HDF5DataAdaptor()
{
  delete m_HDF5Reader;
}

//----------------------------------------------------------------------------
void HDF5DataAdaptor::SetStreamName(const std::string& name)
{
  m_StreamName = name;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::Initialize(pugi::xml_node& node)
{
  TimeEvent<128> mark("HDF5DataAdaptor::Initialize");

  this->InTransitDataAdaptor::Initialize(node);

  pugi::xml_attribute filename = node.attribute("filename");
  pugi::xml_attribute file_name = node.attribute("file_name");
  pugi::xml_attribute methodAttr = node.attribute("method");

  if (filename)
    SetStreamName(filename.value());

  if (file_name)
    SetStreamName(file_name.value());

  if (methodAttr)
    {
      std::string method = methodAttr.value();

      if (method.size() > 0)
        {
          bool doStreaming = ('s' == method[0]);
          bool doCollectiveTxf = ((method.size() > 1) && ('c' == method[1]));

          SetStreaming(doStreaming);
          SetCollective(doCollectiveTxf);
        }
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::OpenStream()
{
  TimeEvent<128> mark("HDF5DataAdaptor::OpenStream");

  if (m_StreamName.size() == 0)
    {
      SENSEI_ERROR("Failed to specify stream name:");
      return -1;
    }

  if (this->m_HDF5Reader == NULL)
    {
      this->m_HDF5Reader =
        new senseiHDF5::ReadStream(this->GetCommunicator(), m_Streaming);
    }

  if (!this->m_HDF5Reader->Init(m_StreamName))
    {
      SENSEI_ERROR("Failed to open \"" << m_StreamName << "\"");
      return -1;
    }

  // initialize the time step
  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::StreamGood()
{
  TimeEvent<128> mark("HDF5DataAdaptor::StreamGood");
  if (this->m_HDF5Reader == nullptr)
    return -1;

  if (this->m_HDF5Reader->m_Streamer == nullptr)
    return -1;

  if (this->m_HDF5Reader->m_Streamer->IsValid())
    return 0; //

  return -1;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::CloseStream()
{
  TimeEvent<128> mark("HDF5DataAdaptor::CloseStream");
  int m_Rank;
  MPI_Comm_rank(GetCommunicator(), &m_Rank);

  if (this->m_HDF5Reader != nullptr)
    {
      delete this->m_HDF5Reader;
      this->m_HDF5Reader = nullptr;
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::Finalize()
{
  TimeEvent<128> mark("HDF5DataAdaptor::Finalize");
  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AdvanceStream()
{
  TimeEvent<128> mark("HDF5DataAdaptor::Advance");

  return this->UpdateTimeStep();
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::UpdateTimeStep()
{
  TimeEvent<128> mark("HDF5DataAdaptor::UpdateTimeStep");

  // update data object time and time step
  unsigned long timeStep = 0;
  double time = 0.0;

  if (!this->m_HDF5Reader->AdvanceTimeStep(timeStep, time))
    {
      // SENSEI_ERROR("Failed to update time step");
      return -1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  // read metadata

  unsigned int nMeshes = 0;
  if (!this->m_HDF5Reader->ReadMetadata(nMeshes))
    {
      SENSEI_ERROR("Failed to read metadata at timestep: " << timeStep);
      return -1;
    }

  if (nMeshes == 0)
    {
      SENSEI_ERROR("No Mesh at this timestep found");
      return -1;
    }
  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetSenderMeshMetadata(unsigned int id,
                                           MeshMetadataPtr& metadata)
{
  TimeEvent<128> mark("HDF5DataAdaptor::GetSenderMeshMetadata");
  if (this->m_HDF5Reader->ReadSenderMeshMetaData(id, metadata))
    return 0;
  else
    {
      SENSEI_ERROR("Failed to get metadata for object " << id)
      return -1;
    }
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetNumberOfMeshes(unsigned int& numMeshes)
{
  TimeEvent<128> mark("HDF5DataAdaptor::GetNumberOfMeshes");
  if (this->m_HDF5Reader)
    {
      numMeshes = this->m_HDF5Reader->GetNumberOfMeshes();
      return 0;
    }

  return -1;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr& metadata)
{
  TimeEvent<128> mark("HDF5DataAdaptor::GetMeshMetadata");
  // check if some user analysis told us how the data should land by
  // passing in reciever metadata
  if (this->GetReceiverMeshMetadata(id, metadata))
    {
      // none set, we'll use the partitioner to figure it out
      // first take a look at what's available
      MeshMetadataPtr senderMd;
      if (this->GetSenderMeshMetadata(id, senderMd))
        {
          SENSEI_ERROR("Failed to get sender metadata")
          return -1;
        }

      // get the partitioner, default to the block based layout
      PartitionerPtr part = this->GetPartitioner();
      if (!part)
        {
          part = BlockPartitioner::New();
        }

      MeshMetadataPtr recverMd;
      if (part->GetPartition(this->GetCommunicator(), senderMd, recverMd))
        {
          SENSEI_ERROR(
            "Failed to determine a suitable layout to receive the data");
          this->CloseStream();
        }

      metadata = recverMd;
      //
      // use this meshmetadata to read objects
      //
      this->m_HDF5Reader->m_AllMeshInfoReceiver.SetMeshMetadata(id, recverMd);
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetMesh(const std::string& meshName,
                             bool structureOnly,
                             svtkDataObject*& mesh)
{
  TimeEvent<128> mark("HDF5DataAdaptor::GetMesh");

  mesh = nullptr;

  // other wise we need to read the mesh at the current time step
  if (!this->m_HDF5Reader->ReadMesh(meshName, mesh, structureOnly))
    {
      SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"");
      return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddGhostNodesArray(svtkDataObject* mesh,
                                        const std::string& meshName)
{
  return AddArray(mesh, meshName, svtkDataObject::POINT, "svtkGhostType");
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddGhostCellsArray(svtkDataObject* mesh,
                                        const std::string& meshName)
{
  return AddArray(mesh, meshName, svtkDataObject::CELL, "svtkGhostType");
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddArray(svtkDataObject* mesh,
                              const std::string& meshName,
                              int association,
                              const std::string& arrayName)
{
  std::ostringstream  oss;
  oss<<"HDF5DataAdaptor::AddArray mesh="<<meshName<<" array="<<arrayName;

  std::string evtName = oss.str();
  TimeEvent<128> mark(evtName.c_str());
  //TimeEvent<128> mark("HDF5DataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh)
    {
      SENSEI_ERROR("Invalid mesh object");
      return -1;
    }

  if (!this->m_HDF5Reader->ReadInArray(meshName, association, arrayName, mesh))
    {
      SENSEI_ERROR("Failed to read " << SVTKUtils::GetAttributesName(association)
                                     << " data array \"" << arrayName
                                     << "\" from mesh \"" << meshName << "\"");
      return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::ReleaseData()
{
  TimeEvent<128> mark("HDF5DataAdaptor::ReleaseData");
  return 0;
}

}
