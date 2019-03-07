
#include "CyclicPartitioner.h"


namespace sensei
{

int CyclicPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  int numLocalRanks = 0;
  MPI_Comm_size(comm, &numLocalRanks);
  mdOut = mdIn->NewCopy();
	
  int rankIdx = 0;
  for (auto& blkOwner : mdOut->BlockOwner)
    {
		blkOwner = rankIdx++ % numLocalRanks;
    }

  return 0;
}


}
