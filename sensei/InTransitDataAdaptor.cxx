#include "InTransitDataAdaptor.h"
#include "MeshMetadata.h"
#include "Partitioner.h"
#include "ConfigurablePartitioner.h"
#include "BlockPartitioner.h"
#include "Error.h"
#include "Profiler.h"

#include <pugixml.hpp>

#include <vtkDataObject.h>
#include <vtkObjectFactory.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

namespace sensei
{

struct InTransitDataAdaptor::InternalsType
{
  InternalsType() : Part(BlockPartitioner::New()) {}
  ~InternalsType() {}

  PartitionerPtr Part;
  std::map<unsigned int, MeshMetadataPtr> ReceiverMetadata;
};

//----------------------------------------------------------------------------
InTransitDataAdaptor::InTransitDataAdaptor()
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
int InTransitDataAdaptor::Initialize(pugi::xml_node &node)
{
  TimeEvent<128> mark("InTransitDataAdaptor::Initialize");

  // look for the presense of an optional partitioner spec
  pugi::xml_node partNode = node.child("partitioner");
  if (partNode)
    {
    // create and initialize the partitioner
    PartitionerPtr tmp = ConfigurablePartitioner::New();
    if (tmp->Initialize(partNode))
      {
      SENSEI_ERROR("Failed to initialize the partitioner from XML")
      return -1;
      }
    this->Internals->Part = tmp;
    }

  return 0;
}

//----------------------------------------------------------------------------
InTransitDataAdaptor::~InTransitDataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void InTransitDataAdaptor::SetPartitioner(const sensei::PartitionerPtr &partitioner)
{
  this->Internals->Part = partitioner;
}

//----------------------------------------------------------------------------
sensei::PartitionerPtr InTransitDataAdaptor::GetPartitioner()
{
  return this->Internals->Part;
}

//----------------------------------------------------------------------------
int InTransitDataAdaptor::GetReceiverMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  TimeEvent<128> mark("InTransitDataAdaptor::GetReceiverMeshMetadata");
  std::map<unsigned int, MeshMetadataPtr>::iterator it =
    this->Internals->ReceiverMetadata.find(id);

  // don't report the error here, as caller may handle it
  if (it == this->Internals->ReceiverMetadata.end())
    return -1;

  metadata = it->second;
  return 0;
}

//----------------------------------------------------------------------------
int InTransitDataAdaptor::SetReceiverMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  this->Internals->ReceiverMetadata[id] = metadata;
  return 0;
}
}
