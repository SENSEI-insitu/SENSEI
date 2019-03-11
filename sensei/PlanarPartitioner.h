#ifndef sensei_PlanarPartitioner_h
#define sensei_PlanarPartitioner_h

#include "Partitioner.h"

namespace sensei
{
/// @class PlanarPartitioner
/// @brief Represents the planar partitioning mode for in-transit operation.
///
/// In planar partitioning mode the blocks are distributed in blocks of a
/// specified size. The size is specified in the 'plane_size' attribute. Note
/// cyclic partitioner is a special case of the planar with a plane_size of 1.
class PlanarPartitioner : public sensei::Partitioner
{
public:
  PlanarPartitioner() : PlaneSize(1) {}
  ~PlanarPartitioner() {}

  PlanarPartitioner(unsigned int planeSize);

  // Set the plane size
  void SetPlaneSize(unsigned int planeSize)
  { this->PlaneSize = planeSize; }

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  // distributes blocks to a rank such that are assigned round ribbin in chuncks
  // of plane size blocks.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
    MeshMetadataPtr &out) override;

  // Initialize from the XML attribute 'plane_size' of node.
  int Initialize(pugi::xml_node &node) override;

protected:
  unsigned int PlaneSize;
};

}

#endif
