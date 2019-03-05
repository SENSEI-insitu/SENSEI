
#include <BlockPartitioner.h>
#include <cmath>


namespace sensei
{

BlockPartitioner::BlockPartitioner(int numLocalRanks) :
  Partitioner(numLocalRanks)
{

}


int BlockPartitioner::GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local)
{
	// in this example...
   	// remote->BlockOwner is {0, 1, ..., 8}
   	// remote->BlockIds is {0, 1, ... , 8}
   
   	// in this example...
   	// local->BlockOwner = {0, 0, 0, 0, 0, 1, 1, 1, 1};
   	// local->BlockIds = {0, 1, ... , 8}

	local = remote->NewCopy();
	
	int num_blks = local->BlockIds.size();
	float blk_per_rank = (float)num_blks / this->_NumLocalRanks;
	int blk_per_rank_floor = std::floor(blk_per_rank);
	int blk_per_rank_ceil = std::ceil(blk_per_rank);
	bool switch_flag = false;
	int blk_cnt = 0;
	int rank_idx = 0;

	while (blk_cnt < num_blks)
	{
		int blk_per_rank_cur = switch_flag ? blk_per_rank_floor : blk_per_rank_ceil;
		for (int j = 0; j < blk_per_rank_cur && blk_cnt < num_blks; ++j)
			local->BlockOwner[blk_cnt++] = rank_idx;
		rank_idx++;
		switch_flag = !switch_flag;
	}

}


}
