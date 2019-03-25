#ifndef sensei_CyclicPartitioner_h
#define sensei_CyclicPartitioner_h

#include "Partitioner.h"

namespace sensei
{

/// @class CyclicPartitioner
/// The cyclic distribution method will distribute blocks to a rank
/// such that consecutive blocks are distributed over consecutive
/// ranks (in a round-robin fashion).
class CyclicPartitioner : public sensei::Partitioner
{
public:
  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  // blocks are distributed in a round-robin fashion.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
     MeshMetadataPtr &out) override;
};

}

#endif
