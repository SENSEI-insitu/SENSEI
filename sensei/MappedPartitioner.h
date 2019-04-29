#ifndef sensei_MappedPartitioner_h
#define sensei_MappedPartitioner_h

#include "Partitioner.h"
#include <vector>

namespace sensei
{

class MappedPartitioner;
using MappedPartitionerPtr = std::shared_ptr<MappedPartitioner>;

/// @class MappedPartitioner
/// @brief represents the mapped partitioning mode for in-transit operation.
///
/// The mapped partitioner enables one to explicitly control a block based
/// paritioning of the data. The poarallel vectors 'BlockIds' and 'BlockOwner'
/// contain the block id of each data block and the rank which should own it.
/// These may be set programatically or through XML.
class MappedPartitioner : public sensei::Partitioner
{
public:
  static MappedPartitionerPtr New()
  { return MappedPartitionerPtr(new MappedPartitioner); }

  const char *GetClassName() override { return "MappedPartitioner"; }

  // construct initialzed from vectors of owner and block ids.
  MappedPartitioner(const std::vector<int> &blkOwner,
    const std::vector<int> &blkIds);

  // Initialize the partitioner from the 'block_owner' and 'block_id' XML
  // elements nested below the current node.
  int Initialize(pugi::xml_node &node) override;

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
  MappedPartitioner() = default;
  MappedPartitioner(const MappedPartitioner &) = default;

private:
  std::vector<int> BlockOwner;
  std::vector<int> BlockIds;
};

}

#endif
