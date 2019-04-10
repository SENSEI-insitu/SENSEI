#ifndef sensei_BlockPartitioner_h
#define sensei_BlockPartitioner_h

#include "Partitioner.h"

namespace sensei
{

class BlockPartitioner;
using BlockPartitionerPtr = std::shared_ptr<BlockPartitioner>;

/// @class BlockPartitioner
/// The block partitioning mode will distribute blocks to a rank such that
/// consecutive blocks share a rank.
class BlockPartitioner : public sensei::Partitioner
{
public:
  static BlockPartitionerPtr New()
  { return BlockPartitionerPtr(new BlockPartitioner); }

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  // distributes blocks to a rank such that consecutive blocks share a rank.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
    MeshMetadataPtr &out) override;

protected:
  BlockPartitioner() = default;
  BlockPartitioner(const BlockPartitioner &) = default;
};

}

#endif
