#include "PlanarSlicePartitioner.h"
#include "XMLUtils.h"
#include "STLUtils.h"
#include "VTKUtils.h"
#include "Timer.h"

#include <cstdlib>
#include <sstream>
#include <limits>

#include <pugixml.hpp>

namespace sensei
{
using namespace STLUtils; // for operator<<

// --------------------------------------------------------------------------
int PlanarSlicePartitioner::Initialize(pugi::xml_node &node)
{
  if (XMLUtils::RequireChild(node, "point") ||
    XMLUtils::RequireChild(node, "normal"))
    return -1;

  if (XMLUtils::ParseNumeric(node.child("point"), this->Point) ||
    XMLUtils::ParseNumeric(node.child("normal"), this->Normal))
    return -1;

  // report configuration
  std::ostringstream oss;
  oss << "point=" << this->Point << " normal=" << this->Normal;
  SENSEI_STATUS("Configured PlanareSlicePartitioner " << oss.str())

  return 0;
}

// --------------------------------------------------------------------------
int PlanarSlicePartitioner::GetPartition(MPI_Comm comm,
  const MeshMetadataPtr &mdIn, MeshMetadataPtr &mdOut)
{
  timer::MarkEvent("PlanarSlicePartitioner::GetPartition");

  // require block bounds
  if (mdIn->BlockBounds.size() != static_cast<unsigned int>(mdIn->NumBlocks))
    {
    SENSEI_ERROR("Block bounds are required")
    return -1;
    }

  // build the list of active blocks
  std::vector<int> activeBlocks;
  for (int i = 0; i < mdIn->NumBlocks; ++i)
    {
    // compute the distance from  each corner of the block bounding box
    // to the plane. if the block intersects the plane, at least one
    // corner will have a different sign.
    const std::array<double,6> &bounds = mdIn->BlockBounds[i];

    // triplets defining corner points
    int pt_ids[] = {0,2,4, 0,3,4, 1,3,4, 1,2,4,
      0,2,5, 0,3,5, 1,3,5, 1,2,5};

    double min_d = std::numeric_limits<double>::max();
    double max_d = std::numeric_limits<double>::lowest();

    for (int q = 0; q < 8; ++q)
      {
      double d = 0.0;
      for (int j = 0; j < 3; ++j)
        d += this->Normal[j] * (bounds[pt_ids[q*3 + j]] - this->Point[j]);

      min_d = std::min(min_d, d);
      max_d = std::max(max_d, d);

      if ((min_d <= 0.0) && (max_d > 0.0))
        {
        activeBlocks.push_back(i);
        break;
        }
      }
    }

  // partition the remaining blocks to ranks equally
  int nRanks = 1;
  MPI_Comm_size(comm, &nRanks);

  int numActiveBlocks = activeBlocks.size();

  std::vector<int> activeBlockOwner(numActiveBlocks);

  int nLocal = numActiveBlocks / nRanks;
  int nLarge = numActiveBlocks % nRanks;

  for (int j = 0; j < nRanks; ++j)
    {
    // do it for all ranks, so every rank can know who
    // owns what
    int rank = j;

    int id0 = nLocal*rank + (rank < nLarge ? rank : nLarge);
    int id1 = id0 + nLocal + (rank < nLarge ? 1 : 0);

    // allocate the local dataset
    for (int i = id0; i < id1; ++i)
      activeBlockOwner[i] = rank;
    }

  // update the metadata with the new decomp
  mdOut = mdIn->NewCopy();

  // start out with all block assigned to no rank
  for (int i = 0; i < mdOut->NumBlocks; ++i)
    mdOut->BlockOwner[i] = -1;

  // assign the active blocks to the correct rank
  for (int i = 0; i < numActiveBlocks; ++i)
    mdOut->BlockOwner[activeBlocks[i]] = activeBlockOwner[i];

  // report the decomp
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  static unsigned long it = 0;
  int verbose = this->GetVerbose();
  if ((rank == 0) && (verbose && (!mdIn->StaticMesh || (it == 0))))
    {
    // write a file for visualizaing the receiver domain decomp
    std::ostringstream oss;
    if (verbose > 1)
      {
      oss << it;
      VTKUtils::WriteDomainDecomp(comm, mdOut,
        "planar_slice_partitioner_receiver_decomp_" + oss.str() + ".vtk");
      oss.str();
      }

    // calculate number of cells moved
    long long numCellsLeft = 0;
    long long numCellsMoved = 0;
    for (int i = 0; i < mdOut->NumBlocks; ++i)
      {
      if (mdOut->BlockOwner[i] >= 0)
        numCellsMoved += mdOut->BlockNumCells[i];
      else
        numCellsLeft += mdOut->BlockNumCells[i];
      }

    // report number of blocks moved
    oss << "PlanarSlicePartitioner: NumBlocks=" << mdIn->NumBlocks
      << " NumActiveBlocks=" << activeBlocks.size() << " numCellsMoved=" << numCellsMoved
      << " numCellsLeft=" << numCellsLeft << " movedFraction="
      << double(numCellsMoved)/double(numCellsMoved + numCellsLeft);

    // report sender and receiver decomp
    if (verbose > 2)
      {
      oss << std::endl << "BlockIds=" << mdIn->BlockIds << std::endl
        << "sender BlockOwner=" << mdIn->BlockOwner << std::endl
        << "receiver BlockOwner=" << mdOut->BlockOwner;
      }

    SENSEI_STATUS(<< oss.str())

    it += 1;
    }

  return 0;
}

}
