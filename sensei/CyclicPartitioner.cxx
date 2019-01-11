
#include <CyclicPartitioner.h>


namespace sensei
{

int CyclicPartitioner::GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local)
{
	// implement the "cyclic" partitioner decomposition

   	// in this example...
   	// remote->BlockOwner is {0, 1, ..., 8}
   	// remote->BlockIds is {0, 1, ... , 8}
   
   	// in this example...
   	// local->BlockOwner = {0, 1, 0, 1, 0, 1, 0, 1, 0};
   	// local->BlockIds = {0, 1, ... , 8}
}


}
