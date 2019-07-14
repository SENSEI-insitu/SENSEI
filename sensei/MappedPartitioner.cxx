#include "MappedPartitioner.h"
#include "XMLUtils.h"
#include "STLUtils.h"
#include "Timer.h"

#include <pugixml.hpp>
#include <sstream>

namespace sensei
{
using namespace STLUtils; // for operator<<

// --------------------------------------------------------------------------
MappedPartitioner::MappedPartitioner(const std::vector<int> &blkOwner,
  const std::vector<int> &blkIds) : BlockOwner(blkOwner), BlockIds(blkIds)
{
}

// --------------------------------------------------------------------------
void MappedPartitioner::SetBlockOwner(const std::vector<int> &blkOwner)
{
  this->BlockOwner = blkOwner;
}

// --------------------------------------------------------------------------
void MappedPartitioner::SetBlockIds(const std::vector<int> &blkIds)
{
  this->BlockIds = blkIds;
}

// --------------------------------------------------------------------------
int MappedPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  (void)comm;
  timer::MarkEvent mark("MappedPartitioner::GetPartition");

  mdOut = mdIn->NewCopy();

  mdOut->BlockOwner = this->BlockOwner;
  mdOut->BlockIds = this->BlockIds;

  return 0;
}

// --------------------------------------------------------------------------
int MappedPartitioner::Initialize(pugi::xml_node &node)
{
  // parse owner and id map from the XML
  if (XMLUtils::RequireChild(node, "block_owner") ||
    XMLUtils::RequireChild(node, "block_id"))
    return -1;

  if (XMLUtils::ParseNumeric(node.child("block_owner"), this->BlockOwner))
    {
    SENSEI_ERROR("Failed to parse BlockOwner array")
    return -1;
    }

  if (XMLUtils::ParseNumeric(node.child("block_id"), this->BlockIds))
    {
    SENSEI_ERROR("Failed to parse BlockIds array")
    return -1;
    }

  // report the config
  std::ostringstream oss;
  oss << "BlockIds=" << this->BlockIds << " BlockOwner=" << this->BlockOwner;
  SENSEI_STATUS("Configured MappedPartitioner " << oss.str())

  return 0;
}

}
