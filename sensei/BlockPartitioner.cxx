
#include "BlockPartitioner.h"


namespace sensei
{

int BlockPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  mdOut = mdIn->NewCopy();

  int nRanks = 1;
  MPI_Comm_size(comm, &nRanks);

  // compute the domain decomposition
  int nLocal = mdOut->NumBlocks / nRanks;
  int nLarge = mdOut->NumBlocks % nRanks;

  for (int j = 0; j < nRanks; ++j)
    {
    // do it for all ranks, so every rank can know who
    // owns what
    int rank = j;

    int id0 = nLocal*rank + (rank < nLarge ? rank : nLarge);
    int id1 = id0 + nLocal + (rank < nLarge ? 1 : 0);

    // allocate the local dataset
    for (int i = id0; i <= id1; ++i)
      mdOut->BlockOwner[i] = rank;
    }

  return 0;
}

}
