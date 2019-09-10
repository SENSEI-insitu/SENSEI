#include "libISDataAdaptor.h"
#include "is_sim.h"


#include "MeshMetadata.h"
#include "Partitioner.h"
#include "BlockPartitioner.h"
#include "Error.h"
#include "Timer.h"
#include "libISSchema.h"
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
struct libISDataAdaptor::InternalsType
{
  InternalsType() : Stream() {}

  senseilibIS::InputStream Stream;
  senseilibIS::DataObjectCollectionSchema Schema;
};

//----------------------------------------------------------------------------
senseiNewMacro(libISDataAdaptor);

//----------------------------------------------------------------------------
libISDataAdaptor::libISDataAdaptor() : Internals(nullptr)
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
libISDataAdaptor::~libISDataAdaptor()
{
  delete this->Internals;
}

/***
//----------------------------------------------------------------------------
int libISDataAdaptor::SetFileName(const std::string &fileName)
{
  this->Internals->Stream.FileName = fileName;
  return 0;
}
***/

/***
//----------------------------------------------------------------------------
int libISDataAdaptor::SetReadMethod(const std::string &method)
{
  return this->Internals->Stream.SetReadMethod(method);
}
***/


/***
//----------------------------------------------------------------------------
int libISDataAdaptor::SetReadMethod(ADIOS_READ_METHOD method)
{
  this->Internals->Stream.ReadMethod = method;
  return 0;
}
***/

//----------------------------------------------------------------------------
int libISDataAdaptor::Initialize(pugi::xml_node &node)
{
  timer::MarkEvent mark("libISDataAdaptor::Initialize");

  // let the base class handle initialization of the partitioner etc
  if (this->InTransitDataAdaptor::Initialize(node))
    {
    SENSEI_ERROR("Failed to intialize InTransitDataAdaptor")
    return -1;
    }

  //if (node.attribute("file_name"))
  //  this->SetFileName(node.attribute("file_name").value());

  //if (node.attribute("read_method") &&
  //  this->SetReadMethod(node.attribute("read_method").value()))
  //  return -1;

  return 0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::Finalize()
{
  timer::MarkEvent mark("libISDataAdaptor::Finalize");
  return 0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::OpenStream()
{
  timer::MarkEvent mark("libISDataAdaptor::OpenStream");

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
int libISDataAdaptor::StreamGood()
{
  return this->Internals->Stream.Good();
}

//----------------------------------------------------------------------------
int libISDataAdaptor::CloseStream()
{
  timer::MarkEvent mark("libISDataAdaptor::CloseStream");

  this->Internals->Stream.Close();

  return 0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::AdvanceStream()
{
  timer::MarkEvent mark("libISDataAdaptor::AdvanceStream");

  if (this->Internals->Stream.AdvanceTimeStep())
    return -1;

  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::UpdateTimeStep()
{
  timer::MarkEvent mark("libISDataAdaptor::UpdateTimeStep");

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
int libISDataAdaptor::GetSenderMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (this->Internals->Schema.GetSenderMeshMetadata(id, metadata))
    {
    SENSEI_ERROR("Failed to get metadata for object " << id)
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(numMeshes))
    return -1;
  return  0;
}

//----------------------------------------------------------------------------
int libISDataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
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
      SENSEI_WARNING("No partitioner specified, using BlockParititoner")
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
int libISDataAdaptor::GetMesh(const std::string &meshName,
   bool structureOnly, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("libISDataAdaptor::GetMesh");

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
int libISDataAdaptor::AddGhostNodesArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::POINT, "vtkGhostType");
}

//----------------------------------------------------------------------------
int libISDataAdaptor::AddGhostCellsArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::CELL, "vtkGhostType");
}

//----------------------------------------------------------------------------
int libISDataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string& arrayName)
{
  timer::MarkEvent mark("libISDataAdaptor::AddArray");

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
int libISDataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("libISDataAdaptor::ReleaseData");
  return 0;
}

//----------------------------------------------------------------------------
void libISDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->sensei::DataAdaptor::PrintSelf(os, indent);
}

}
