
#include <CyclicPartitioner.h>


namespace sensei
{

CyclicPartitioner::CyclicPartitioner(int numLocalRanks) :
  Partitioner(numLocalRanks)
{

}


int CyclicPartitioner::GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local)
{
	// implement the "cyclic" partitioner decomposition

   	// in this example...
   	// remote->BlockOwner is {0, 1, ..., 8}
   	// remote->BlockIds is {0, 1, ... , 8}
   
   	// in this example...
   	// local->BlockOwner = {0, 1, 0, 1, 0, 1, 0, 1, 0};
   	// local->BlockIds = {0, 1, ... , 8}

   	local = remote->NewCopy();
	
	int rank_idx = 0;
	for (auto& blk_o : local->BlockOwner)
	{
		blk_o = rank_idx++ % this->_NumLocalRanks;
	}
}


}
