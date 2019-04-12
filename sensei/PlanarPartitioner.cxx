#include "PlanarPartitioner.h"
#include <pugixml.hpp>

namespace sensei
{

// --------------------------------------------------------------------------
int PlanarPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  mdOut = mdIn->NewCopy();

  int blkCnt = 0;
  int rankIdx = 0;

  int numRanks = 1;
  MPI_Comm_size(comm, &numRanks);

  while (blkCnt < mdOut->NumBlocks)
    {
    int owner = rankIdx % numRanks;

    for (unsigned int j = 0; (j < this->PlaneSize) && (blkCnt < mdOut->NumBlocks); ++j, ++blkCnt)
      mdOut->BlockOwner[blkCnt] = owner;

    ++rankIdx;
    }

  return 0;
}

// --------------------------------------------------------------------------
int PlanarPartitioner::Initialize(pugi::xml_node &node)
{
  this->PlaneSize = node.attribute("plane_size").as_uint(1);
  SENSEI_STATUS("Configured PlanarPartitioner plane_size=" << this->PlaneSize)
  return 0;
}

}
