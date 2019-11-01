#ifndef sensei_PlanarPartitioner_h
#define sensei_PlanarPartitioner_h

#include "Partitioner.h"

namespace sensei
{

class PlanarPartitioner;
using PlanarPartitionerPtr = std::shared_ptr<sensei::PlanarPartitioner>;

/// @class PlanarPartitioner
/// The cyclic distribution method will distribute blocks to a rank
/// such that consecutive blocks are distributed over consecutive
/// ranks in a round-robin fashion in chunks of length specified by
/// the PlaneSize. The class looks for PlaneSize in the XML attribute
/// `cycle_size`.
class PlanarPartitioner : public sensei::Partitioner
{
public:
  static sensei::PlanarPartitionerPtr New()
  { return PlanarPartitionerPtr(new PlanarPartitioner); }

  const char *GetClassName() override { return "PlanarPartitioner"; }

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  // blocks are distributed in a round-robin fashion in chunks of length
  // specified by the PlaneSize.
  int GetPartition(MPI_Comm comm, const sensei::MeshMetadataPtr &in,
     sensei::MeshMetadataPtr &out) override;

  // Set/get the PlaneSize. This controls how many blocks are assigned to
  // each rank in each iteration of the partitioning process.
  void SetPlaneSize(unsigned int size) { this->PlaneSize = size; }
  unsigned int GetPlaneSize(){ return this->PlaneSize; }

  // Initialize from XML
  int Initialize(pugi::xml_node &node) override;

protected:
  PlanarPartitioner() : PlaneSize(1) {}
  PlanarPartitioner(const PlanarPartitioner &) = default;

  unsigned int PlaneSize;
};

}

#endif
