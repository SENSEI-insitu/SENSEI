#include "ADIOS2DataAdaptor.h"
#include "MeshMetadata.h"
#include "Partitioner.h"
#include "BlockPartitioner.h"
#include "Error.h"
#include "Profiler.h"
#include "ADIOS2Schema.h"
#include "SVTKUtils.h"
#include "XMLUtils.h"

#include <svtkCompositeDataIterator.h>
#include <svtkDataSetAttributes.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkObjectFactory.h>
#include <svtkSmartPointer.h>
#include <svtkDataSet.h>

#include <pugixml.hpp>

#include <sstream>

namespace sensei
{
struct ADIOS2DataAdaptor::InternalsType
{
  InternalsType() : Stream() {}

  senseiADIOS2::InputStream Stream;
  senseiADIOS2::DataObjectCollectionSchema Schema;
};

//----------------------------------------------------------------------------
senseiNewMacro(ADIOS2DataAdaptor);

//----------------------------------------------------------------------------
ADIOS2DataAdaptor::ADIOS2DataAdaptor() : Internals(nullptr)
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
ADIOS2DataAdaptor::~ADIOS2DataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void ADIOS2DataAdaptor::SetFileName(const std::string &fileName)
{
  this->Internals->Stream.SetFileName(fileName);
}

//----------------------------------------------------------------------------
void ADIOS2DataAdaptor::SetReadEngine(const std::string &engine)
{
  this->Internals->Stream.SetReadEngine(engine);
}

//----------------------------------------------------------------------------
void ADIOS2DataAdaptor::AddParameter(const std::string &name,
  const std::string &value)
{
  this->Internals->Stream.AddParameter(name, value);
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::Initialize(pugi::xml_node &node)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::Initialize");

  // let the base class handle initialization of the partitioner etc
  if (this->InTransitDataAdaptor::Initialize(node))
    {
    SENSEI_ERROR("Failed to intialize the ADIOSDataAdaptor")
    return -1;
    }

  // required attributes
  if (XMLUtils::RequireAttribute(node, "engine") ||
    XMLUtils::RequireAttribute(node, "filename"))
    {
    SENSEI_ERROR("Failed to initialize ADIOS2DataAdaptor");
    return -1;
    }

  this->SetFileName(node.attribute("filename").value());
  this->SetReadEngine(node.attribute("engine").value());

  // optional attributes
  if (node.attribute("timeout"))
    this->AddParameter("OpenTimeoutSecs", node.attribute("timeout").value());

  pugi::xml_node params = node.child("engine_parameters");
  if (params)
    {
    std::vector<std::string> name;
    std::vector<std::string> value;
    XMLUtils::ParseNameValuePairs(params, name, value);
    size_t n = name.size();
    for (size_t i = 0; i < n; ++i)
        this->AddParameter(name[i], value[i]);
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::Finalize()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::Finalize");
  this->Internals->Stream.Finalize();
  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::OpenStream()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::OpenStream");

  // initialize adios
  if (this->Internals->Stream.Initialize(this->GetCommunicator()))
    return -1;

  // open the stream or file
  if (this->Internals->Stream.Open())
    return -1;

  // start the first timestep
  if (this->Internals->Stream.BeginStep())
    return -1;

  // pull metadata from the first the time step
  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::StreamGood()
{
  return this->Internals->Stream.Good();
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::CloseStream()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::CloseStream");

  this->Internals->Stream.Close();
  this->Internals->Stream.Finalize();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AdvanceStream()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::AdvanceStream");

  if (this->Internals->Stream.AdvanceTimeStep())
    return -1;

  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::UpdateTimeStep()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::UpdateTimeStep");

  // update data object time and time step
  unsigned long timeStep = 0;
  double time = 0.0;

  if (this->Internals->Schema.ReadTimeStep(this->GetCommunicator(),
    this->Internals->Stream, timeStep, time))
    {
    SENSEI_ERROR("Failed to update time step")
    return -1;
    }

  if (timeStep == std::numeric_limits<int64_t>::max())
    {
    SENSEI_STATUS("End of stream detected")
    return 1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  // read metadata
  if (this->Internals->Schema.ReadMeshMetadata(this->GetCommunicator(),
    this->Internals->Stream))
    {
    SENSEI_ERROR("Failed to read metadata")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetSenderMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::SenderMeshMetadata");
  if (this->Internals->Schema.GetSenderMeshMetadata(id, metadata))
    {
    SENSEI_ERROR("Failed to get metadata for object " << id)
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::GetNumberOfMeshes");
  numMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(numMeshes))
    return -1;
  return  0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::GetMeshMetadata");
  // check if an analysis told us how the data should land by
  // passing in reciever metadata
  if (this->GetReceiverMeshMetadata(id, metadata))
    {
    // layout was set by an analysis. did we do this already?
    if (this->Internals->Schema.GetReceiverMeshMetadata(id, metadata))
      {
      SENSEI_ERROR("Failed to get mesh metadata")
      this->CloseStream();
      return -1;
      }

    // we did this already, return cached layout
    if (metadata)
       return 0;

    // first time through. use the partitioner to figure it out.
    // get the sender layout.
    MeshMetadataPtr senderMd;
    if (this->GetSenderMeshMetadata(id, senderMd))
      {
      SENSEI_ERROR("Failed to get sender metadata")
      this->CloseStream();
      return -1;
      }

    // get the partitioner, default to the block partitioner
    PartitionerPtr part = this->GetPartitioner();
    if (!part)
      {
      SENSEI_WARNING("No partitoner specified, using BlockParititoner")
      part = BlockPartitioner::New();
      }

    MeshMetadataPtr receiverMd;
    if (part->GetPartition(this->GetCommunicator(), senderMd, receiverMd))
      {
      SENSEI_ERROR("Failed to determine a suitable layout to receive the data")
      this->CloseStream();
      }

    // cache and return the new layout
    this->Internals->Schema.SetReceiverMeshMetadata(id, receiverMd);
    metadata = receiverMd;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetMesh(const std::string &meshName,
   bool structureOnly, svtkDataObject *&mesh)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::GetMesh");

  mesh = nullptr;

  // other wise we need to read the mesh at the current time step
  if (this->Internals->Schema.ReadObject(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, structureOnly))
    {
    SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AddGhostNodesArray(svtkDataObject *mesh,
  const std::string &meshName)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::AddGhostNodesArray");
  return AddArray(mesh, meshName, svtkDataObject::POINT, "svtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AddGhostCellsArray(svtkDataObject *mesh,
  const std::string &meshName)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::AddGhostCellsArray");
  return AddArray(mesh, meshName, svtkDataObject::CELL, "svtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AddArray(svtkDataObject* mesh,
  const std::string &meshName, int association, const std::string& arrayName)
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh)
    {
    SENSEI_ERROR("Invalid mesh object")
    return -1;
    }

  if (this->Internals->Schema.ReadArray(this->GetCommunicator(),
    this->Internals->Stream, meshName, association, arrayName, mesh))
    {
    SENSEI_ERROR("Failed to read " << SVTKUtils::GetAttributesName(association)
      << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::ReleaseData()
{
  TimeEvent<128> mark("ADIOS2DataAdaptor::ReleaseData");
  return 0;
}

}
