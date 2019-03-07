#ifndef sensei_MappedPartitioner_h
#define sensei_MappedPartitioner_h

#include "Partitioner.h"

namespace sensei
{
/// @class MappedPartitioner
/// @brief MappedPartitioner is class that represents the mapped partitioning mode for in-transit operation.
///
/// The mapped partitioning mode will allocate blocks in-order as listed in a nested 'block_owner' 
/// and 'block_id' elements. Each entry in the block element has a corresponding entry in the proc 
/// element naming the mpi rank where the block lands.
class MappedPartitioner : public sensei::Partitioner 
{
public:
  MappedPartitioner(const std::vector<int>& blkOwner = {}, const std::vector<int>& blkIds = {});
  ~MappedPartitioner() {}

  MappedPartitioner(const MappedPartitioner&) = delete;
  void operator=(const MappedPartitioner&) = delete;

  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in, MeshMetadataPtr &out);

  int Initialize(pugi::xml_node &node);

protected:
  std::vector<int> BlockOwner;
  std::vector<int> BlockIds;
};

}

#endif
