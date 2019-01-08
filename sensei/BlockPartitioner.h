#ifndef sensei_BlockPartitioner_h
#define sensei_BlockPartitioner_h

#include "Partitioner.h"

namespace sensei
{
/// @class BlockPartitioner
/// @brief BlockPartitioner is class that represents the block partitioning mode for in-transit operation.
///
/// The block partitioning mode will distribute blocks to a rank such that consecutive blocks share a rank.
///  
class BlockPartitioner : public sensei::Partitioner
{
public:
  BlockPartitioner();
  ~BlockPartitioner();

  BlockPartitioner(const BlockPartitioner&) = delete;
  void operator=(const BlockPartitioner&) = delete;

  int GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);
};

}
#endif
