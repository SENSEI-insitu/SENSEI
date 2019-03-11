#include "PlanarPartitioner.h"

#include <pugixml.hpp>

namespace sensei
{

// --------------------------------------------------------------------------
PlanarPartitioner::PlanarPartitioner(unsigned int planeSize) : PlaneSize(planeSize)
{
}

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
  pugi::xml_node sizeNode = node.child("plane_size");
  if (!sizeNode || !sizeNode.text())
    return -1;

  this->PlaneSize = sizeNode.text().as_uint();

  return 0;
}

}
