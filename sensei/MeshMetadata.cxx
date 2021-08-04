#include "MeshMetadata.h"
#include "Profiler.h"
#include "MPIUtils.h"
#include "STLUtils.h"
#include "Error.h"

#include <utility>
#include <algorithm>

namespace sensei
{
// for various operator<< overloads
using namespace STLUtils;


// --------------------------------------------------------------------------
int MeshMetadataFlags::ToStream(sensei::BinaryStream &str) const
{
  str.Pack(this->Flags);
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadataFlags::FromStream(sensei::BinaryStream &str)
{
  str.Unpack(this->Flags);
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadataFlags::ToStream(ostream &str) const
{
  str << "0x" << std::hex << this->Flags << " ";

  int nSet = 0;

  if (this->Flags & DECOMP)
    {
    str << (nSet ? "|" : "") << "DECOMP";
    nSet += 1;
    }

  if (this->Flags & SIZE)
    {
    str << (nSet ? "|" : "") << "SIZE";
    nSet += 1;
    }

  if (this->Flags & EXTENTS)
    {
    str << (nSet ? "|" : "") << "EXTENTS";
    nSet += 0;
    }

  if (this->Flags & BOUNDS)
    {
    str << (nSet ? "|" : "") << "BOUNDS";
    nSet += 1;
    }

  if (this->Flags & RANGE)
    {
    str << (nSet ? "|" : "") << "RANGE";
    nSet += 1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::ToStream(sensei::BinaryStream &str) const
{
  str.Pack(this->GlobalView);
  str.Pack(this->MeshName);
  str.Pack(this->MeshType);
  str.Pack(this->BlockType);
  str.Pack(this->NumBlocks);
  str.Pack(this->NumBlocksLocal);
  str.Pack(this->Extent);
  str.Pack(this->Bounds);
  str.Pack(this->CoordinateType);
  str.Pack(this->NumPoints);
  str.Pack(this->NumCells);
  str.Pack(this->CellArraySize);
  str.Pack(this->CellArrayType);
  str.Pack(this->NumArrays);
  str.Pack(this->NumGhostCells);
  str.Pack(this->NumGhostNodes);
  str.Pack(this->NumLevels);
  str.Pack(this->StaticMesh);
  str.Pack(this->ArrayName);
  str.Pack(this->ArrayCentering);
  str.Pack(this->ArrayComponents);
  str.Pack(this->ArrayType);
  str.Pack(this->ArrayRange);
  str.Pack(this->BlockOwner);
  str.Pack(this->BlockIds);
  str.Pack(this->BlockNumPoints);
  str.Pack(this->BlockNumCells);
  str.Pack(this->BlockCellArraySize);
  str.Pack(this->BlockExtents);
  str.Pack(this->BlockBounds);
  str.Pack(this->BlockArrayRange);
  str.Pack(this->RefRatio);
  str.Pack(this->BlocksPerLevel);
  str.Pack(this->BlockLevel);
  str.Pack(this->PeriodicBoundary);
  this->Flags.ToStream(str);

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::FromStream(sensei::BinaryStream &str)
{
  str.Unpack(this->GlobalView);
  str.Unpack(this->MeshName);
  str.Unpack(this->MeshType);
  str.Unpack(this->BlockType);
  str.Unpack(this->NumBlocks);
  str.Unpack(this->NumBlocksLocal);
  str.Unpack(this->Extent);
  str.Unpack(this->Bounds);
  str.Unpack(this->CoordinateType);
  str.Unpack(this->NumPoints);
  str.Unpack(this->NumCells);
  str.Unpack(this->CellArraySize);
  str.Unpack(this->CellArrayType);
  str.Unpack(this->NumArrays);
  str.Unpack(this->NumGhostCells);
  str.Unpack(this->NumGhostNodes);
  str.Unpack(this->NumLevels);
  str.Unpack(this->StaticMesh);
  str.Unpack(this->ArrayName);
  str.Unpack(this->ArrayCentering);
  str.Unpack(this->ArrayComponents);
  str.Unpack(this->ArrayType);
  str.Unpack(this->ArrayRange);
  str.Unpack(this->BlockOwner);
  str.Unpack(this->BlockIds);
  str.Unpack(this->BlockNumPoints);
  str.Unpack(this->BlockNumCells);
  str.Unpack(this->BlockCellArraySize);
  str.Unpack(this->BlockExtents);
  str.Unpack(this->BlockBounds);
  str.Unpack(this->BlockArrayRange);
  str.Unpack(this->RefRatio);
  str.Unpack(this->BlocksPerLevel);
  str.Unpack(this->BlockLevel);
  str.Unpack(this->PeriodicBoundary);
  this->Flags.FromStream(str);

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::ToStream(ostream &str) const
{
  str << "{";
  str << "GlobalView = " << this->GlobalView << std::endl;
  str << "MeshName = " << this->MeshName << std::endl;
  str << "MeshType = " << this->MeshType << std::endl;
  str << "BlockType = " << this->BlockType << std::endl;
  str << "NumBlocks = " << this->NumBlocks << std::endl;
  str << "Bounds = " << this->Bounds << std::endl;
  str << "Extent = " << this->Extent << std::endl;
  str << "NumPoints = " << this->NumPoints << std::endl;
  str << "NumCells = " << this->NumCells << std::endl;
  str << "CellArraySize = " << this->CellArraySize << std::endl;
  str << "CellArrayType = " << this->CellArrayType << std::endl;
  str << "NumArrays = " << this->NumArrays << std::endl;
  str << "NumGhostCells = " << this->NumGhostCells << std::endl;
  str << "NumGhostNodes = " << this->NumGhostNodes << std::endl;
  str << "NumLevels = " << this->NumLevels << std::endl;
  str << "StaticMesh = " << this->StaticMesh << std::endl;
  str << "ArrayName = " << this->ArrayName << std::endl;
  str << "ArrayCentering = " << this->ArrayCentering << std::endl;
  str << "ArrayComponents = " << this->ArrayComponents << std::endl;
  str << "ArrayType = " << this->ArrayType << std::endl;
  str << "ArrayRange = " << this->ArrayRange << std::endl;
  str << "BlockOwner = " << this->BlockOwner << std::endl;
  str << "BlockIds = " << this->BlockIds << std::endl;
  str << "BlockNumPoints = " << this->BlockNumPoints << std::endl;
  str << "BlockNumCells = " << this->BlockNumCells << std::endl;
  str << "BlockCellArraySize = " << this->BlockCellArraySize << std::endl;
  str << "BlockExtents = " << this->BlockExtents << std::endl;
  str << "BlockBounds = " << this->BlockBounds << std::endl;
  str << "BlockArrayRange = " << this->BlockArrayRange << std::endl;
  str << "RefRatio = " << this->RefRatio << std::endl;
  str << "BlocksPerLevel = " << this->BlocksPerLevel << std::endl;
  str << "BlockLevel = " << this->BlockLevel << std::endl;
  str << "PeriodicBoundary = " << this->PeriodicBoundary << std::endl;
  str << "Flags = "; this->Flags.ToStream(str); str << std::endl;
  str << "}";
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::Validate(MPI_Comm comm, const MeshMetadataFlags &requiredFlags)
{
  bool err = false;

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

  // an empty dataset is a valid scenario
  // if the dataset is empty there may not be any metadata. for instance
  // in the VTKDataAdaptor metadata is determined by examining the available
  // data.
  bool localBlocks = ((this->NumBlocks > 0) ||
    ((this->NumBlocksLocal.size() > 0) && ((this->GlobalView ?
    this->NumBlocksLocal[rank] : this->NumBlocksLocal[0]) > 0)));

  if (!localBlocks)
    return 0;

  // figure out what the valid size is in this case
  bool global = this->GlobalView || (this->MeshType == VTK_OVERLAPPING_AMR);
  bool haveLocal = this->NumBlocksLocal.size() > 0;
  bool haveAllLocal = this->NumBlocksLocal.size() == unsigned(nRanks);
  unsigned long validSize = (global || !haveLocal ? this->NumBlocks :
     (haveAllLocal ? this->NumBlocksLocal[rank] : this->NumBlocksLocal[0]));

  // check block decomp
  if (this->Flags.BlockDecompSet() || requiredFlags.BlockDecompSet())
    {
    if (this->BlockOwner.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockOwner has " << this->BlockOwner.size()
        << " elements but should have " << validSize)
      err = true;
      }
    if (this->BlockIds.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockIds has " << this->BlockOwner.size()
        << " elements but should have " << validSize)
      err = true;
      }
    }

  // check block size
  if (this->Flags.BlockSizeSet() || requiredFlags.BlockSizeSet())
    {
    if (this->BlockNumCells.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockNumCells has " << this->BlockNumCells.size()
        << " elements but should have " << validSize)
      err = true;
      }
    if (this->BlockNumPoints.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockNumPoints has " << this->BlockNumPoints.size()
        << " elements but should have " << validSize)
      err = true;
      }
    if (((this->BlockType == VTK_UNSTRUCTURED_GRID) || (this->BlockType == VTK_POLY_DATA)) &&
      (this->BlockCellArraySize.size() != validSize))
      {
      SENSEI_ERROR("Metadata inconsistency. BlockCellArraySize has " << this->BlockCellArraySize.size()
        << " elements but should have " << validSize)
      err = true;
      }
    }

  // check block extent
  if ((this->Flags.BlockExtentsSet() || requiredFlags.BlockExtentsSet()) &&
    ((this->MeshType == VTK_OVERLAPPING_AMR) || (this->BlockType == VTK_IMAGE_DATA) ||
    (this->BlockType == VTK_RECTILINEAR_GRID) || (this->BlockType == VTK_STRUCTURED_GRID)))
    {
    if (this->BlockExtents.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockExtents has " << this->BlockExtents.size()
        << " elements but should have " << validSize)
      err = true;
      }
    if ((this->MeshType == VTK_OVERLAPPING_AMR) && (this->BlockLevel.size() != validSize))
      {
      SENSEI_ERROR("Metadata inconsistency. BlockLevel has " << this->BlockLevel.size()
        << " elements but should have " << validSize)
      err = true;
      }
    }

  // check block bounds
  if (this->Flags.BlockBoundsSet() || requiredFlags.BlockBoundsSet())
    {
    if (this->BlockBounds.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockBounds has " << this->BlockBounds.size()
        << " elements but should have " << validSize)
      err = true;
      }
    }

  // check arrays
  unsigned long numArrays = this->NumArrays;
  if (this->ArrayName.size() != numArrays)
    {
    SENSEI_ERROR("Metadata inconsistency. ArrayName has " << this->ArrayName.size()
      << " elements but should have " << numArrays)
    err = true;
    }
  if (this->ArrayCentering.size() != numArrays)
    {
    SENSEI_ERROR("Metadata inconsistency. ArrayCentering has " << this->ArrayCentering.size()
      << " elements but should have " << numArrays)
    err = true;
    }
  if (this->ArrayComponents.size() != numArrays)
    {
    SENSEI_ERROR("Metadata inconsistency. ArrayComponents has " << this->ArrayComponents.size()
      << " elements but should have " << numArrays)
    err = true;
    }
  if (this->ArrayType.size() != numArrays)
    {
    SENSEI_ERROR("Metadata inconsistency. ArrayType has " << this->ArrayType.size()
      << " elements but should have " << numArrays)
    err = true;
    }

  // check block array ranges
  if (this->Flags.BlockArrayRangeSet() || requiredFlags.BlockArrayRangeSet())
    {
    if (this->BlockArrayRange.size() != validSize)
      {
      SENSEI_ERROR("Metadata inconsistency. BlockArrayRange has "
        << this->BlockArrayRange.size() << " elements but should have " << validSize)
      err = true;
      }
    else
      {
      for (unsigned long i = 0; i < validSize; ++i)
        {
        if (this->BlockArrayRange[i].size() != numArrays)
          {
          SENSEI_ERROR("Metadata inconsistency. BlockArrayRange at block " << i
            << " has " << this->BlockArrayRange[i].size()
            << " elements but should have " << numArrays)
          err = true;
          }
        }
      }
    if (this->GlobalView && (this->ArrayRange.size() != numArrays))
      {
      SENSEI_ERROR("Metadata inconsistency. ArrayRange has " << this->ArrayRange.size()
        << " elements but should have " << numArrays)
      err = true;
      }
    }

  // check amr specific
  if (this->MeshType == VTK_OVERLAPPING_AMR)
    {
    if (this->Flags.BlockDecompSet())
      {
      if (this->RefRatio.size() != unsigned(this->NumLevels))
        {
        SENSEI_ERROR("Metadata inconsistency. RefRatio has " << this->RefRatio.size()
          << " elements but should have " << this->NumLevels)
        err = true;
        }

      if (this->BlocksPerLevel.size() != unsigned(this->NumLevels))
        {
        SENSEI_ERROR("Metadata inconsistency. BlocksPerLevel has " << this->BlocksPerLevel.size()
          << " elements but should have " << this->NumLevels)
        err = true;
        }

      if (this->BlockLevel.size() != validSize)
        {
        SENSEI_ERROR("Metadata inconsistency. BlockLevel has " << this->BlockLevel.size()
          << " elements but should have " << validSize)
        err = true;
        }
      }
    }


  return err ? -1 : 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::GlobalizeView(MPI_Comm comm)
{
  TimeEvent<128> mark("MeshMetadata::GlobalizeView");
  if (!this->GlobalView)
    {
    MPIUtils::GlobalViewV(comm, this->BlockOwner);
    MPIUtils::GlobalViewV(comm, this->BlockIds);
    MPIUtils::GlobalViewV(comm, this->NumBlocksLocal);
    MPIUtils::GlobalViewV(comm, this->BlockNumPoints);
    MPIUtils::GlobalViewV(comm, this->BlockNumCells);
    MPIUtils::GlobalViewV(comm, this->BlockCellArraySize);
    MPIUtils::GlobalViewV(comm, this->BlockExtents);
    MPIUtils::GlobalViewV(comm, this->BlockBounds);
    MPIUtils::GlobalViewV(comm, this->BlockArrayRange);
    MPIUtils::GlobalViewV(comm, this->BlockLevel);

    MPIUtils::GlobalCounts(comm, this->BlocksPerLevel);

    STLUtils::ReduceRange(this->BlockBounds, this->Bounds);
    STLUtils::ReduceRange(this->BlockExtents, this->Extent);
    STLUtils::ReduceRange(this->BlockArrayRange, this->ArrayRange);

    this->NumBlocks = STLUtils::Sum(this->NumBlocksLocal);
    this->NumPoints = STLUtils::Sum(this->BlockNumPoints);
    this->NumCells = STLUtils::Sum(this->BlockNumCells);
    this->CellArraySize = STLUtils::Sum(this->BlockCellArraySize);

    this->GlobalView = true;
    }

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::ClearBlockInfo()
{
  // clear out all block metadata
  this->NumBlocks = 0;

  this->NumPoints = 0;
  this->NumCells = 0;
  this->CellArraySize = 0;

  STLUtils::InitializeRange(this->Bounds);
  STLUtils::InitializeRange(this->Extent);

  this->BlockIds.clear();
  this->BlockOwner.clear();
  this->NumBlocksLocal.clear();

  this->BlockBounds.clear();
  this->BlockExtents.clear();

  this->BlockNumPoints.clear();
  this->BlockNumCells.clear();
  this->BlockCellArraySize.clear();

  this->BlockArrayRange.clear();

  this->ArrayRange.resize(this->NumArrays);
  STLUtils::InitializeRange(this->ArrayRange);

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::CopyBlockInfo(const MeshMetadataPtr &other, int i)
{
  this->NumBlocks += 1;

  if (other->BlockNumPoints.size())
    {
    this->NumPoints += other->BlockNumPoints[i];
    this->BlockNumPoints.push_back(other->BlockNumPoints[i]);
    }

  if (other->BlockNumCells.size())
    {
    this->NumCells += other->BlockNumCells[i];
    this->BlockNumCells.push_back(other->BlockNumCells[i]);
    }

  if (other->BlockCellArraySize.size())
    {
    this->CellArraySize += other->BlockCellArraySize[i];
    this->BlockCellArraySize.push_back(other->BlockCellArraySize[i]);
    }

  if (other->BlockOwner.size())
    this->BlockOwner.push_back(other->BlockOwner[i]);

  if (other->BlockIds.size())
    this->BlockIds.push_back(other->BlockIds[i]);

  if (other->BlockBounds.size())
    {
    const std::array<double,6> &obi = other->BlockBounds[i];
    this->BlockBounds.push_back(obi);
    STLUtils::ReduceRange(obi, this->Bounds);
    }

  if (other->BlockExtents.size())
    {
    const std::array<int,6> &obi = other->BlockExtents[i];
    this->BlockExtents.push_back(obi);
    STLUtils::ReduceRange(obi, this->Extent);
    }

  if (other->BlockArrayRange.size())
    {
    const std::vector<std::array<double,2>> &obari = other->BlockArrayRange[i];
    this->BlockArrayRange.push_back(obari);
    STLUtils::ReduceRange(obari, this->ArrayRange);
    }

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::ClearArrayInfo()
{
  this->NumArrays = 0;
  this->ArrayName.clear();
  this->ArrayCentering.clear();
  this->ArrayComponents.clear();
  this->ArrayType.clear();
  this->ArrayRange.clear();
  this->BlockArrayRange.clear();
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::CopyArrayInfo(const sensei::MeshMetadataPtr &other,
  const std::string &arrayName)
{
  // find the id of this array
  int aid = -1;
  for (int i = 0; i < other->NumArrays; ++i)
    {
    if (other->ArrayName[i] == arrayName)
      {
      aid = i;
      break;
      }
    }

  if (aid == -1)
    {
    SENSEI_ERROR("Failed to copy array \"" << arrayName
      << "\" it was not found in the source object")
    return -1;
    }

  // do a bounds check on the required members
  if (((int)other->ArrayCentering.size() < aid)
    || ((int)other->ArrayComponents.size() < aid)
    || ((int)other->ArrayType.size() < aid))
    {
    SENSEI_ERROR("Failed to copy metadata for array " << aid
      << ", invalid metadata")
    return -1;
    }

  // copy the required metadata
  this->ArrayName.push_back(other->ArrayName[aid]);
  this->ArrayCentering.push_back(other->ArrayCentering[aid]);
  this->ArrayComponents.push_back(other->ArrayComponents[aid]);
  this->ArrayType.push_back(other->ArrayType[aid]);

  // array ranges are optional, copy if present
  // per-block array ranges are ordered rirst by block then by array.
  // loop over blocks copy the ith entry from each block.
  int nBlocks = other->BlockArrayRange.size();
  if (nBlocks)
    {
    // copy global array range
    if ((int)other->ArrayRange.size() < aid)
      {
      SENSEI_ERROR("Invlaid ArrayRange metadata for array \""
        << arrayName << "\"")
      return -1;
      }
    this->ArrayRange.push_back(other->ArrayRange[aid]);

    // resize the first time through
    if ((int)this->BlockArrayRange.size() != nBlocks)
      this->BlockArrayRange.resize(nBlocks);

    for (int i = 0; i < nBlocks; ++i)
      {
      if ((int)other->BlockArrayRange[i].size() < aid)
        {
        SENSEI_ERROR("Invalid BlockARrayRange for array \""
          << arrayName << "\" at block " << i)
        return -1;
        }

      this->BlockArrayRange[i].push_back(other->BlockArrayRange[i][aid]);
      }
    }

  ++this->NumArrays;

  return 0;
}

}
