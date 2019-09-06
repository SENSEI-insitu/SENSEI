#ifndef sensei_PencilPartitioner_h
#define sensei_PencilPartitioner_h

#include "Partitioner.h"
#include <vector>

namespace sensei
{

class PencilPartitioner;
using PencilPartitionerPtr = std::shared_ptr<PencilPartitioner>;

/// @class PencilPartitioner
/// @brief represents the mapped partitioning mode for in-transit operation.
///
/// The mapped partitioner enables one to explicitly control a block based
/// paritioning of the data. The poarallel vectors 'BlockIds' and 'BlockOwner'
/// contain the block id of each data block and the rank which should own it.
/// These may be set programatically or through XML.
class PencilPartitioner : public sensei::Partitioner
{
public:
  static PencilPartitionerPtr New()
  { return PencilPartitionerPtr(new PencilPartitioner); }

  const char *GetClassName() override { return "PencilPartitioner"; }

  // construct initialzed from vectors of owner and block ids.
  PencilPartitioner(const std::vector<int> &blkOwner,
    const std::vector<int> &blkIds);

  // Initialize the partitioner from the 'block_owner' and 'block_id' XML
  // elements nested below the current node.
  int Initialize(pugi::xml_node &node) override;
  int Initialize(int dir);

  // Set the block onwer list
  void SetBlockOwner(const std::vector<int> &blkOwner);

  // Set the block id list
  void SetBlockIds(const std::vector<int> &blkIds);

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  // distributes blocks to a rank such that consecutive blocks share a rank.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
     MeshMetadataPtr &out) override;

protected:
  PencilPartitioner() = default;
  PencilPartitioner(const PencilPartitioner &) = default;

private:
  std::vector<int> BlockOwner;
  std::vector<int> BlockIds;
  int dir = 0;
};

}

#endif
