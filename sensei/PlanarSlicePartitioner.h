#ifndef sensei_PlanarSlicePartitioner_h
#define sensei_PlanarSlicePartitioner_h

#include "Partitioner.h"
#include <array>

namespace sensei
{

class PlanarSlicePartitioner;
using PlanarSlicePartitionerPtr = std::shared_ptr<sensei::PlanarSlicePartitioner>;

/// @class PlanarSlicePartitioner
/// The slice paritioner determins which blocks intersect the plane
/// defined by a given point and normal. This blocks are partitioned
/// in consecutive blocks to ranks such that each rank gets approximately
/// the same number. Ranks will differ by at most 1 block.
class PlanarSlicePartitioner : public sensei::Partitioner
{
public:
  static sensei::PlanarSlicePartitionerPtr New()
  { return PlanarSlicePartitionerPtr(new PlanarSlicePartitioner); }

  const char *GetClassName() override { return "PlanarSlicePartitioner"; }

  // set the point defining the slice plane
  void SetPoint(const std::array<double,3> &p) { this->Point = p; }
  void GetPoint(std::array<double,3> &p) { p = this->Point; }

  // set the normal defining the slice plane
  void SetNormal(const std::array<double,3> &n) { this->Normal = n; }
  void GetNormal(std::array<double,3> &n) { n = this->Normal; }

  // Initialize from XML
  int Initialize(pugi::xml_node &node) override;

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  int GetPartition(MPI_Comm comm, const sensei::MeshMetadataPtr &in,
    sensei::MeshMetadataPtr &out) override;

protected:
  PlanarSlicePartitioner() : Point{0.,0.,0.}, Normal{1.,0.,0.} {}
  PlanarSlicePartitioner(const PlanarSlicePartitioner &) = default;

  std::array<double,3> Point;
  std::array<double,3> Normal;
};

}

#endif
