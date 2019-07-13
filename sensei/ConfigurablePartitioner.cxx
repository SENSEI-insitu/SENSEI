#include "ConfigurablePartitioner.h"
#include "Partitioner.h"
#include "BlockPartitioner.h"
#include "MappedPartitioner.h"
#include "PlanarPartitioner.h"
#include "PlanarSlicePartitioner.h"
#include "XMLUtils.h"

#include <pugixml.hpp>

#include <memory>

namespace sensei
{

struct ConfigurablePartitioner::InternalsType
{
  PartitionerPtr Part;
};

// ---------------------------------------------------------------------------
ConfigurablePartitioner::ConfigurablePartitioner()
{
  this->Internals = new InternalsType;
}

// ---------------------------------------------------------------------------
ConfigurablePartitioner::~ConfigurablePartitioner()
{
  delete this->Internals;
}

// ---------------------------------------------------------------------------
int ConfigurablePartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
    MeshMetadataPtr &out)
{
  if (!this->Internals->Part)
    {
    SENSEI_ERROR("Partitioner has not been initialized")
    return -1;
    }

  return this->Internals->Part->GetPartition(comm, in, out);
}

// ---------------------------------------------------------------------------
int ConfigurablePartitioner::Initialize(pugi::xml_node &partNode)
{
  // partNode must contain a valid <partitioner> element, any problems
  // indicate a critical error

  if (XMLUtils::RequireAttribute(partNode, "type"))
    {
    SENSEI_ERROR("Failed to construct a partitioner. "
      "Missing \"type\" attribute");
    return -1;
    }

  //get the type and construct an instance
  PartitionerPtr tmp;
  std::string partType = partNode.attribute("type").value();
  if (partType == "block")
    {
    tmp = BlockPartitioner::New();
    }
  else if (partType == "planar")
    {
    tmp = PlanarPartitioner::New();
    }
  else if (partType == "mapped")
    {
    tmp = MappedPartitioner::New();
    }
  else if (partType == "planar_slice")
    {
    tmp = PlanarSlicePartitioner::New();
    }
  else
    {
    SENSEI_ERROR("Failed to construct a partitioner. \""
      << partType << "\" is not a recognized partitioner.")
    return -1;
    }

  // let the instance initialize itself
  if (tmp->Initialize(partNode))
    {
    SENSEI_ERROR("Failed to initialize the \"" << partType << "\" partitioner")
    return -1;
    }

  // everything is good, update internal state
  this->Internals->Part = tmp;

  return 0;
}

}
