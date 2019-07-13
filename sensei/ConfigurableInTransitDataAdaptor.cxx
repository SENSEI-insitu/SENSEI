#include "ConfigurableInTransitDataAdaptor.h"
#include "InTransitDataAdaptor.h"
#include "XMLUtils.h"
#include "Error.h"
#ifdef ENABLE_ADIOS1
#include "ADIOS1DataAdaptor.h"
#endif
#ifdef ENABLE_HDF5
#include "HDF5DataAdaptor.h"
#endif

#include <pugixml.hpp>
#include <string>

#include <vtkObjectFactory.h>

namespace sensei
{

struct ConfigurableInTransitDataAdaptor::InternalsType
{
  InternalsType() : Adaptor(nullptr) {}

  ~InternalsType()
  {
    if (this->Adaptor)
      Adaptor->Delete();
  }

  InTransitDataAdaptor *Adaptor;
};

//----------------------------------------------------------------------------
senseiNewMacro(ConfigurableInTransitDataAdaptor);

// -------------------------------------------------------------------------------
ConfigurableInTransitDataAdaptor::ConfigurableInTransitDataAdaptor() :
  Internals(new InternalsType)
{
}

// -------------------------------------------------------------------------------
ConfigurableInTransitDataAdaptor::~ConfigurableInTransitDataAdaptor()
{
  delete this->Internals;
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::Initialize(const std::string &fileName)
{
  MPI_Comm comm = this->GetCommunicator();

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  pugi::xml_document doc;
  if (XMLUtils::Parse(comm, fileName, doc))
    {
    if (rank == 0)
      SENSEI_ERROR("failed to parse configuration")
    return -1;
    }

  pugi::xml_node root = doc.child("sensei");

  return this->Initialize(root);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::Initialize(pugi::xml_node &root)
{
  // get the transport element and its type attribute
  if (XMLUtils::RequireChild(root, "transport"))
    return -1;

  pugi::xml_node node = root.child("transport");

  if (XMLUtils::RequireAttribute(node, "type"))
    return -1;

  std::string type = node.attribute("type").value();

  // construct the requested adaptor type
  InTransitDataAdaptor *adaptor = nullptr;
  if (type == "adios1")
    {
#ifndef ENABLE_ADIOS1
    SENSEI_ERROR("ADIOS1 transport requested but is disabled in this build")
    return -1;
#else
    adaptor = ADIOS1DataAdaptor::New();
#endif
    }
  else if (type == "adios2")
    {
#ifndef ENABLE_ADIOS2
    SENSEI_ERROR("ADIOS2 transport requested but is disabled in this build")
    return -1;
#else
    SENSEI_ERROR("ADIOS2 not yet available")
    return -1;
#endif
    }
  else if (type == "hdf5")
    {
#ifndef ENABLE_HDF5
    SENSEI_ERROR("HDF5 transport requested but is disabled in this build")
    return -1;
#else
    adaptor = HDF5DataAdaptor::New();
#endif
    }
  else if (type == "libis")
    {
#ifndef ENABLE_LIBIS
    SENSEI_ERROR("libis transport requested but is disabled in this build")
    return -1;
#else
    SENSEI_ERROR("libis not yet available")
    return -1;
#endif
    }
  else
    {
    SENSEI_ERROR("No \"" << type << "\" in transit data adaptor")
    return -1;
    }

  // intialize the adaptor. the partitioner is typically iniitialized
  // by the default initialize in the InTransitDataAdaptor
  if (adaptor->Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize \"" << type << "\" data adaptor")
    return -1;
    }

  // everything is good, take ownership of the concrete instance
  if (this->Internals->Adaptor)
    this->Internals->Adaptor->Delete();

  this->Internals->Adaptor = adaptor;

  SENSEI_STATUS("Configured \"" << adaptor->GetClassName())

  return 0;
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetSenderMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetSenderMeshMetadata(id, metadata);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetReceiverMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetReceiverMeshMetadata(id, metadata);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::SetReceiverMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->SetReceiverMeshMetadata(id, metadata);
}

// -------------------------------------------------------------------------------
void ConfigurableInTransitDataAdaptor::SetPartitioner(const sensei::PartitionerPtr &partitioner)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    }

  return this->Internals->Adaptor->SetPartitioner(partitioner);
}

// -------------------------------------------------------------------------------
sensei::PartitionerPtr ConfigurableInTransitDataAdaptor::GetPartitioner()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return nullptr;
    }

  return this->Internals->Adaptor->GetPartitioner();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::OpenStream()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->OpenStream();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::CloseStream()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->CloseStream();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::AdvanceStream()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->AdvanceStream();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::StreamGood()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->StreamGood();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::Finalize()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->Finalize();
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetNumberOfMeshes(numMeshes);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetMeshMetadata(id, metadata);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetMesh(const std::string &meshName,
  bool structureOnly, vtkDataObject *&mesh)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetMesh(meshName, structureOnly, mesh);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::GetMesh(const std::string &meshName,
  bool structureOnly, vtkCompositeDataSet *&mesh)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->GetMesh(meshName, structureOnly, mesh);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::AddGhostNodesArray(vtkDataObject* mesh,
  const std::string &meshName)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->AddGhostNodesArray(mesh, meshName);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::AddGhostCellsArray(vtkDataObject* mesh,
  const std::string &meshName)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->AddGhostCellsArray(mesh, meshName);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string &arrayName)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->AddArray(mesh, meshName, association, arrayName);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::AddArrays(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::vector<std::string> &arrayName)
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->AddArrays(mesh, meshName, association, arrayName);
}

// -------------------------------------------------------------------------------
int ConfigurableInTransitDataAdaptor::ReleaseData()
{
  if (!this->Internals->Adaptor)
    {
    SENSEI_ERROR("No InTransitDataAdaptor instance")
    return -1;
    }

  return this->Internals->Adaptor->ReleaseData();
}

// -------------------------------------------------------------------------------
double ConfigurableInTransitDataAdaptor::GetDataTime()
{
  return this->Internals->Adaptor->GetDataTime();
}

// -------------------------------------------------------------------------------
void ConfigurableInTransitDataAdaptor::SetDataTime(double time)
{
  this->Internals->Adaptor->SetDataTime(time);
}

// -------------------------------------------------------------------------------
long ConfigurableInTransitDataAdaptor::GetDataTimeStep()
{
  return this->Internals->Adaptor->GetDataTimeStep();
}

// -------------------------------------------------------------------------------
void ConfigurableInTransitDataAdaptor::SetDataTimeStep(long index)
{
  this->Internals->Adaptor->SetDataTimeStep(index);
}

}
