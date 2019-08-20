#include "MeshMetadata.h"
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
    nSet += 1;
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
  std::vector<int> rr; //(this->RefRatio.data(), this->RefRatio.data()+this->RefRatio.size());
  std::vector<int> pbc; //(this->PeriodicBoundary.begin(), this->PeriodicBoundary.data());

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
  str << "RefRatio = " << rr << std::endl;
  str << "BlocksPerLevel = " << this->BlocksPerLevel << std::endl;
  str << "BlockLevel = " << this->BlockLevel << std::endl;
  str << "PeriodicBoundary = " << pbc << std::endl;
  str << "Flags = "; this->Flags.ToStream(str); str << std::endl;
  str << "}";
  return 0;
}

#define CheckGlobalSize


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
  int localBlocks = ((this->NumBlocks > 0) &&
    ((this->NumBlocksLocal.size() > 0) && ((this->GlobalView ?
    this->NumBlocksLocal[rank] : this->NumBlocksLocal[0]) > 0)));

  if (localBlocks && this->Flags.BlockDecompSet() &&
    requiredFlags.BlockDecompSet() && (this->BlockOwner.empty() || this->BlockIds.empty()))
    {
    SENSEI_ERROR("Metadata is missing block deocomp arrays, BlockDecomp and/or BlockIds")
    err = true;
    }

  if (localBlocks && this->Flags.BlockSizeSet() && requiredFlags.BlockSizeSet() &&
    (this->BlockNumCells.empty() || this->BlockNumPoints.empty() ||
    (((this->BlockType == VTK_UNSTRUCTURED_GRID) ||
    (this->BlockType == VTK_POLY_DATA)) && this->BlockCellArraySize.empty())))
    {
    SENSEI_ERROR("Metadata is missing block sizes")
    err = true;
    }

  if (localBlocks && this->Flags.BlockExtentsSet() &&
    !((this->BlockType == VTK_UNSTRUCTURED_GRID) || (this->BlockType == VTK_POLY_DATA)))
    {
    if (this->MeshType == VTK_OVERLAPPING_AMR)
      {
      if (this->NumBlocks != int(this->BlockExtents.size()/6))
        {
        SENSEI_ERROR("BlockExtents are always a global view in AMR data")
        err = true;
        }
      if (this->Extent.size() != 6)
        {
        SENSEI_ERROR("Extent is required for AMR data")
        err = true;
        }
      }
    else if (((this->BlockType == VTK_IMAGE_DATA)
      || (this->BlockType == VTK_RECTILINEAR_GRID) || (this->BlockType == VTK_STRUCTURED_GRID)))
      {
      if (this->BlockExtents.empty())
        {
        if (requiredFlags.BlockExtentsSet())
          {
          SENSEI_ERROR("Metadata is missing block extents")
          err = true;
          }
        }
      }
    }

  if (localBlocks && this->Flags.BlockBoundsSet() && ((this->BlockType == VTK_IMAGE_DATA)
    || (this->BlockType == VTK_RECTILINEAR_GRID) || (this->BlockType == VTK_STRUCTURED_GRID)))
    {
    if (this->BlockBounds.empty())
      {
      if (requiredFlags.BlockBoundsSet())
        {
        SENSEI_ERROR("Metadata is missing block bounds")
        err = true;
        }
      }
    else if (this->Bounds.empty())
      {
      if (requiredFlags.BlockBoundsSet())
        {
        SENSEI_ERROR("Metadata is missing bounds")
        err = true;
        }
      }
    }

  if (localBlocks && this->Flags.BlockArrayRangeSet() &&
    (this->ArrayRange.empty() || this->BlockArrayRange.empty()))
    {
    SENSEI_ERROR("Metadata is missing array range and/or block array range")
    err = true;
    }

  return err ? -1 : 0;
}

// --------------------------------------------------------------------------
int MeshMetadata::GlobalizeView(MPI_Comm comm)
{
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

}
