#include "MeshMetadata.h"
#include "MPIUtils.h"
#include "Error.h"

#include <utility>
#include <algorithm>

namespace sensei
{
// helper for printing metadata
template<typename T, std::size_t N>
ostream &operator<<(ostream &os, const std::array<T,N> &vec)
{
  os << "{";
  if (N)
    {
    os << vec[0];
    for (size_t i = 1; i < N; ++i)
      os << ", " << vec[i];
    }
  os << "}";
  return os;
}

template<typename T>
ostream &operator<<(ostream &os, const std::vector<T> &vec)
{
  os << "{";
  size_t n = vec.size();
  if (n)
    {
    os << vec[0];
    for (size_t i = 1; i < n; ++i)
      os << ", " << vec[i];
    }
  os << "}";
  return os;
}

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

  if (this->Flags & NEIGHBORS)
    {
    str << (nSet ? "|" : "") << "NEIGHBORS";
    nSet += 1;
    }

  if (this->Flags & PARENTS)
    {
    str << (nSet ? "|" : "") << "PARENTS";
    nSet += 1;
    }

  if (this->Flags & CHILDREN)
    {
    str << (nSet ? "|" : "") << "CHILDREN";
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
  str.Pack(this->NumArrays);
  str.Pack(this->NumGhostCells);
  str.Pack(this->NumGhostNodes);
  str.Pack(this->NumLevels);
  str.Pack(this->StaticMesh);
  str.Pack(this->ArrayName);
  str.Pack(this->ArrayCentering);
  str.Pack(this->ArrayComponents);
  str.Pack(this->ArrayType);
  str.Pack(this->BlockOwner);
  str.Pack(this->BlockIds);
  str.Pack(this->BlockNumPoints);
  str.Pack(this->BlockNumCells);
  str.Pack(this->BlockCellArraySize);
  str.Pack(this->BlockExtents);
  str.Pack(this->BlockBounds);
  str.Pack(this->RefRatio);
  str.Pack(this->BlocksPerLevel);
  str.Pack(this->BlockLevel);
  str.Pack(this->BoxArray);
  str.Pack(this->PeriodicBoundary);
  str.Pack(this->BlockNeighbors);
  str.Pack(this->BlockParents);
  str.Pack(this->BlockChildren);
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
  str.Unpack(this->NumArrays);
  str.Unpack(this->NumGhostCells);
  str.Unpack(this->NumGhostNodes);
  str.Unpack(this->NumLevels);
  str.Unpack(this->StaticMesh);
  str.Unpack(this->ArrayName);
  str.Unpack(this->ArrayCentering);
  str.Unpack(this->ArrayComponents);
  str.Unpack(this->ArrayType);
  str.Unpack(this->BlockOwner);
  str.Unpack(this->BlockIds);
  str.Unpack(this->BlockNumPoints);
  str.Unpack(this->BlockNumCells);
  str.Unpack(this->BlockCellArraySize);
  str.Unpack(this->BlockExtents);
  str.Unpack(this->BlockBounds);
  str.Unpack(this->RefRatio);
  str.Unpack(this->BlocksPerLevel);
  str.Unpack(this->BlockLevel);
  str.Unpack(this->BoxArray);
  str.Unpack(this->PeriodicBoundary);
  str.Unpack(this->BlockNeighbors);
  str.Unpack(this->BlockParents);
  str.Unpack(this->BlockChildren);
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
  str << "NumArrays = " << this->NumArrays << std::endl;
  str << "NumGhostCells = " << this->NumGhostCells << std::endl;
  str << "NumGhostNodes = " << this->NumGhostNodes << std::endl;
  str << "NumLevels = " << this->NumLevels << std::endl;
  str << "StaticMesh = " << this->StaticMesh << std::endl;
  str << "ArrayName = " << this->ArrayName << std::endl;
  str << "ArrayCentering = " << this->ArrayCentering << std::endl;
  str << "ArrayComponents = " << this->ArrayComponents << std::endl;
  str << "ArrayType = " << this->ArrayType << std::endl;
  str << "BlockOwner = " << this->BlockOwner << std::endl;
  str << "BlockIds = " << this->BlockIds << std::endl;
  str << "BlockNumPoints = " << this->BlockNumPoints << std::endl;
  str << "BlockNumCells = " << this->BlockNumCells << std::endl;
  str << "BlockCellArraySize = " << this->BlockCellArraySize << std::endl;
  str << "BlockExtents = " << this->BlockExtents << std::endl;
  str << "BlockBounds = " << this->BlockBounds << std::endl;
  str << "RefRatio = " << rr << std::endl;
  str << "BlocksPerLevel = " << this->BlocksPerLevel << std::endl;
  str << "BlockLevel = " << this->BlockLevel << std::endl;
  str << "BoxArray = " << this->BoxArray << std::endl;
  str << "PeriodicBoundary = " << pbc << std::endl;
  str << "BlockNeighbors = " << this->BlockNeighbors << std::endl;
  str << "BlockParents = " << this->BlockParents << std::endl;
  str << "BlockChildren = " << this->BlockChildren << std::endl;
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

  if (this->Flags.BlockNeighborsSet() &&
    requiredFlags.BlockNeighborsSet() && this->BlockNeighbors.empty())
    {
    SENSEI_ERROR("Metadata is missing block neighbors")
    err = true;
    }

  if (this->Flags.BlockParentsSet() && requiredFlags.BlockParentsSet())
    {
    if (this->MeshType != VTK_OVERLAPPING_AMR)
      {
      SENSEI_ERROR("Block parents requested for a non-AMR dataset type")
      err = true;
      }

    if (this->BlockParents.empty())
      {
      SENSEI_ERROR("Metadata is missing block parents")
      err = true;
      }
    }

  if (this->Flags.BlockChildrenSet() && requiredFlags.BlockChildrenSet())
    {
    if ((this->MeshType == VTK_OVERLAPPING_AMR) && (this->BlockChildren.empty()))
      {
      SENSEI_ERROR("Metadata is missing block children array")
      err = true;
      }
    }

  if (this->Flags.BlockDecompSet() &&
    requiredFlags.BlockDecompSet() && (this->BlockOwner.empty() || this->BlockIds.empty()))
    {
    SENSEI_ERROR("Metadata is missing block deocomp arrays, BlockDecomp and/or BlockIds")
    err = true;
    }

  if (this->Flags.BlockSizeSet() && requiredFlags.BlockSizeSet() &&
    (this->BlockNumCells.empty() || this->BlockNumPoints.empty() ||
    (((this->BlockType == VTK_UNSTRUCTURED_GRID) ||
    (this->BlockType == VTK_POLY_DATA)) && this->BlockCellArraySize.empty())))
    {
    SENSEI_ERROR("Metadata is missing block sizes")
    err = true;
    }

  if (this->Flags.BlockExtentsSet() &&
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
      else if (this->Extent.empty())
        {
        // the global extent was not set, but we have the local extents
        // compute the global extent as a convenience.
        MPIUtils::GlobalBounds<int>(comm, this->BlockExtents, this->Extent);
        }
      }
    }

  if (this->Flags.BlockBoundsSet() && ((this->BlockType == VTK_IMAGE_DATA)
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
    else if ((this->MeshType == VTK_MULTIBLOCK_DATA_SET) && this->Bounds.empty())
      {
      // the global bounds were not set, but we have the local bounds
      // compute the global bounds as a convenience.
      MPIUtils::GlobalBounds<double>(comm, this->BlockBounds, this->Bounds);
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

  return err ? -1 : 0;
}

}
