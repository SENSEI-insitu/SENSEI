#ifndef sensei_Partitioner_h
#define sensei_Partitioner_h

namespace sensei
{
/// @class Partitioner
/// @brief Partitioner is an abstract base class that represents the way data is partitioned for in-transit operation mode.
///
/// Partitioner defines the interface to get the partitioning mode. Each specific partitioning mode is
/// represented by a class that inherits from Partitioner class. However, the core modes in SENSEI 3 are:
///  
///  block  The block distribution method will distribute blocks to a rank
///         such that consecutive blocks share a rank.
///
///  cyclic The cyclic distribution method will distribute blocks to a rank
///         such that consecutive blocks are distributed over consecutive
///         ranks (in a round-robin fashion).
///
///  planar The  blocks are distributed in blocks of a specified size.
///         The size is specified in the 'plane_size' attribute. Note
///         block is a special case of planar with a plane_size of 1
///
///  mapped The mapped method of distribution will allocate blocks
///         in-order as listed in a nested 'block_owner' and 'block_id'
///         elements.  each entry in the block element has a
///         corresponding entry in the proc element naming the mpi rank
///         where the block lands
class Partitioner 
{
public:
  virtual int GetPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local) = 0;

protected:
  Partitioner();
  virtual ~Partitioner();

  Partitioner(const Partitioner&) = delete;
  void operator=(const Partitioner&) = delete;
};

}
#endif
