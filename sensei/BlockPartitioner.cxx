
#include <BlockPartitioner.h>


namespace sensei
{

int BlockPartitioner::GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local)
{
	// implement the "block" partitioner decomposition

   	// in this example...
   	// remote->BlockOwner is {0, 1, ..., 8}
   	// remote->BlockIds is {0, 1, ... , 8}
   
   	// in this example...
   	// local->BlockOwner = {0, 0, 0, 0, 0, 1, 1, 1, 1};
   	// local->BlockIds = {0, 1, ... , 8}

}


}
