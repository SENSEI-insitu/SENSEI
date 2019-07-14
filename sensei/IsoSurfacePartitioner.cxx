#include "IsoSurfacePartitioner.h"
#include "XMLUtils.h"
#include "STLUtils.h"
#include "VTKUtils.h"

#include <array>
#include <cstdlib>
#include <sstream>

#include <pugixml.hpp>

namespace sensei
{

using namespace STLUtils;

// --------------------------------------------------------------------------
int IsoSurfacePartitioner::Initialize(pugi::xml_node &node)
{
  // TODO
  (void)node;
  SENSEI_ERROR("Not yet implemented")
  return 0;
}

// --------------------------------------------------------------------------
void IsoSurfacePartitioner::SetIsoValues(const std::string &meshName,
  const std::string &arrayName, int arrayCentering,
  const std::vector<double> &isoVals)
{
  this->MeshName = meshName;
  this->ArrayName = arrayName;
  this->ArrayCentering = arrayCentering;
  this->IsoValues = isoVals;
}

// --------------------------------------------------------------------------
int IsoSurfacePartitioner::GetIsoValues(std::string &meshName,
  std::string &arrayName, int &arrayCentering, std::vector<double> &isoVals) const
{
  if (this->IsoValues.empty())
    return -1;

  meshName = this->MeshName;
  arrayName = this->ArrayName;
  arrayCentering = this->ArrayCentering;
  isoVals = this->IsoValues;

  return 0;
}

// --------------------------------------------------------------------------
int IsoSurfacePartitioner::GetPartition(MPI_Comm comm,
  const MeshMetadataPtr &mdIn, MeshMetadataPtr &mdOut)
{
  // find the set of arrays and values for this mesh
  if (this->MeshName != mdIn->MeshName)
    {
    SENSEI_ERROR("No iso values set for mesh \"" << mdIn->MeshName << "\"")
    return -1;
    }

  // locate the active blocks
  std::set<int> activeBlocks;
  for (int i = 0; i < mdIn->NumArrays; ++i)
    {
    // see if this array is being used, if not skip it
    const std::string &array = mdIn->ArrayName[i];
    if (this->ArrayName != array)
      continue;

    // walk blocks and see if any of them are needed
    const std::vector<double> &vals = this->IsoValues;
    int nvals = vals.size();
    for (int j = 0; j < mdIn->NumBlocks; ++j)
      {
      std::array<double,2> &rng = mdIn->BlockArrayRange[j][i];
      for (int k = 0; k < nvals; ++k)
        {
        // if the value is in the range then this block is needed
        double val = vals[k];
        if ((val >= rng[0]) && (val <= rng[1]))
          activeBlocks.insert(j);
        }
      }
    }

  // partition the needed blocks to ranks equally
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

  // copy the input metadata and fix up the domain decomp
  mdOut = mdIn->NewCopy();

  // start out with all block assigned to no rank
  for (int i = 0; i < mdOut->NumBlocks; ++i)
    mdOut->BlockOwner[i] = -1;

  // assign the active blocks to the correct rank
  std::set<int>::iterator bit = activeBlocks.begin();
  for (int i = 0; i < numActiveBlocks; ++i, ++bit)
    mdOut->BlockOwner[*bit] = activeBlockOwner[i];

  // report the decomp
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int verbose = this->GetVerbose();
  static unsigned long it = 0;
  if ((rank == 0) && (verbose > 0))
    {
    std::ostringstream oss;

    // write a file for visualizaing the receiver domain decomp
    if (verbose > 1)
      {
      oss << it;
      VTKUtils::WriteDomainDecomp(comm, mdOut,
        "iso_surface_partitioner_receiver_decomp_" + oss.str() + ".vtk");
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
    oss << "IsoSurfacePartitioner: NumBlocks=" << mdIn->NumBlocks
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
