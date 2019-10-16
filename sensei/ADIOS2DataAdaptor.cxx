#include "ADIOS2DataAdaptor.h"


#include "MeshMetadata.h"
#include "Partitioner.h"
#include "BlockPartitioner.h"
#include "Error.h"
#include "Timer.h"
#include "ADIOS2Schema.h"
#include "VTKUtils.h"
#include "XMLUtils.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

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
int ADIOS2DataAdaptor::SetFileName(const std::string &fileName)
{
  this->Internals->Stream.FileName = fileName;
  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::SetReadEngine(const std::string &engine)
{
  return this->Internals->Stream.SetReadEngine(engine);
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::Initialize(pugi::xml_node &node)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::Initialize");

  // let the base class handle initialization of the partitioner etc
  if (this->InTransitDataAdaptor::Initialize(node))
    {
    SENSEI_ERROR("Failed to intialize InTransitDataAdaptor")
    return -1;
    }

  if (node.attribute("file_name"))
    this->SetFileName(node.attribute("file_name").value());

  if (node.attribute("read_engine") &&
    this->SetReadEngine(node.attribute("read_engine").value()))
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::Finalize()
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::Finalize");
  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::OpenStream()
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::OpenStream");

  if (this->Internals->Stream.Open(this->GetCommunicator()))
    {
    SENSEI_ERROR("Failed to open stream")
    return -1;
    }

  // initialize the time step
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
  timer::MarkEvent mark("ADIOS2DataAdaptor::CloseStream");

  this->Internals->Stream.Close();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AdvanceStream()
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::AdvanceStream");

  if (this->Internals->Stream.AdvanceTimeStep())
    return -1;

  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::UpdateTimeStep()
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::UpdateTimeStep");

  // update data object time and time step
  unsigned long timeStep = 0;
  double time = 0.0;

  if (this->Internals->Schema.ReadTimeStep(this->GetCommunicator(),
    this->Internals->Stream, timeStep, time))
    {
    SENSEI_ERROR("Failed to update time step")
    return -1;
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

  // set up the name-object map
  unsigned int nMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(nMeshes))
    {
    SENSEI_ERROR("Failed to get the number of meshes")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetSenderMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::SenderMeshMetadata");
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
  timer::MarkEvent mark("ADIOS2DataAdaptor::GetNumberOfMeshes");
  numMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(numMeshes))
    return -1;
  return  0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::GetMeshMetadata");
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
   bool structureOnly, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::GetMesh");

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
int ADIOS2DataAdaptor::AddGhostNodesArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::AddGhostNodesArray");
  return AddArray(mesh, meshName, vtkDataObject::POINT, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AddGhostCellsArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::AddGhostCellsArray");
  return AddArray(mesh, meshName, vtkDataObject::CELL, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string& arrayName)
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::AddArray");

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
    SENSEI_ERROR("Failed to read " << VTKUtils::GetAttributesName(association)
      << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2DataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("ADIOS2DataAdaptor::ReleaseData");
  return 0;
}

//----------------------------------------------------------------------------
void ADIOS2DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->sensei::DataAdaptor::PrintSelf(os, indent);
}

}
