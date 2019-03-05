#ifndef sensei_CyclicPartitioner_h
#define sensei_CyclicPartitioner_h

#include "Partitioner.h"

namespace sensei
{
/// @class CyclicPartitioner
/// @brief CyclicPartitioner is class that represents the cyclic partitioning mode for in-transit operation.
///
/// The cyclic partitioning mode will distribute blocks to a rank such that consecutive blocks are distributed 
/// over consecutive ranks (in a round-robin fashion).
class CyclicPartitioner : public sensei::Partitioner
{
public:
  CyclicPartitioner(int numLocalRanks);
  ~CyclicPartitioner();

  CyclicPartitioner(const CyclicPartitioner&) = delete;
  void operator=(const CyclicPartitioner&) = delete;

  int GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);
  
};

}
#endif
