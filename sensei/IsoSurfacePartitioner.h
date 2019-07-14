#ifndef sensei_IsoSurfacePartitioner_h
#define sensei_IsoSurfacePartitioner_h

#include "Partitioner.h"

#include <set>
#include <vector>
#include <string>

namespace sensei
{

class IsoSurfacePartitioner;
using IsoSurfacePartitionerPtr = std::shared_ptr<IsoSurfacePartitioner>;

/// @class IsoSurfacePartitioner
/// The IsoSurfacePartitioner selects only blocks that are needed to
/// compute the desired set of iso surfaces. These blocks are partitioned
/// in consecutive spans to ranks such that each rank gets approximately
/// the same number. The number of blocks per rank will differ by at most 1.
class IsoSurfacePartitioner : public sensei::Partitioner
{
public:
  static IsoSurfacePartitionerPtr New()
  { return IsoSurfacePartitionerPtr(new IsoSurfacePartitioner); }

  const char *GetClassName() override { return "IsoSurfacePartitioner"; }

  // add iso values
  void SetIsoValues(const std::string &meshName,
    const std::string &arrayName, int arrayCentering,
    const std::vector<double> &vals);

  int GetIsoValues(std::string &meshName, std::string &arrayName,
    int &arrayCentering, std::vector<double> &vals) const;

  // Initialize from XML
  int Initialize(pugi::xml_node &node) override;

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
    MeshMetadataPtr &out) override;

protected:
  IsoSurfacePartitioner() = default;
  IsoSurfacePartitioner(const IsoSurfacePartitioner &) = default;

  std::string MeshName;
  std::string ArrayName;
  int ArrayCentering;
  std::vector<double> IsoValues;
};

}

#endif
