#include "InTransitDataAdaptor.h"
#include "MeshMetadata.h"
#include "Partitioner.h"
#include "Error.h"

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
  InternalsType() {}
  ~InternalsType() {}

  std::unique_ptr<sensei::Partitioner> Part;
  std::map<unsigned int, MeshMetadataPtr> ReceiverMetadata;
};

//----------------------------------------------------------------------------
InTransitDataAdaptor::InTransitDataAdaptor()
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
InTransitDataAdaptor::~InTransitDataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void InTransitDataAdaptor::SetPartitioner(sensei::Partitioner *partitioner)
{
  this->Internals->Part = std::unique_ptr<sensei::Partitioner>(partitioner);
}

//----------------------------------------------------------------------------
sensei::Partitioner *InTransitDataAdaptor::GetPartitioner()
{
  return this->Internals->Part.get();
}

//----------------------------------------------------------------------------
int InTransitDataAdaptor::GetReceiverMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
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
/*
//----------------------------------------------------------------------------
void InTransitDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
*/
}
