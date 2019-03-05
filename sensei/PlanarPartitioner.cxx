
#include <PlanarPartitioner.h>


namespace sensei
{

PlanarPartitioner::PlanarPartitioner(unsigned int numLocalRanks, unsigned int planeSize) :
  Partitioner(numLocalRanks), _PlaneSize(planeSize) 
{

}


int PlanarPartitioner::GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local)
{
	// details omitted

	local = remote->NewCopy();
	
	int num_blks = local->BlockIds.size();
	int blk_cnt = 0;
	int rank_idx = 0;

	while (blk_cnt < num_blks)
	{
		for (int j = 0; j < this->_PlaneSize && blk_cnt < num_blks; ++j)
			local->BlockOwner[blk_cnt++] = rank_idx;
		rank_idx++;
	}
}


}
