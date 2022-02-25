/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkExtractGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkExtractStructuredGridHelper.h"

// SVTK includes
#include "svtkBoundingBox.h"
#include "svtkCellData.h"
#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkStructuredData.h"
#include "svtkStructuredExtent.h"

// C/C++ includes
#include <algorithm>
#include <cassert>
#include <vector>

// Some useful extent macros
#define EMIN(ext, dim) (ext[2 * dim])
#define EMAX(ext, dim) (ext[2 * dim + 1])
#define IMIN(ext) (ext[0])
#define IMAX(ext) (ext[1])
#define JMIN(ext) (ext[2])
#define JMAX(ext) (ext[3])
#define KMIN(ext) (ext[4])
#define KMAX(ext) (ext[5])

#define I(ijk) (ijk[0])
#define J(ijk) (ijk[1])
#define K(ijk) (ijk[2])

namespace svtk
{
namespace detail
{

// Index mapping works as:
// inputExtent = Mapping[dim][outputExtent - this->OutputWholeExtent[2*dim]]
struct svtkIndexMap
{
  std::vector<int> Mapping[3];
};

} // End namespace detail
} // End namespace svtk

svtkStandardNewMacro(svtkExtractStructuredGridHelper);

//-----------------------------------------------------------------------------
svtkExtractStructuredGridHelper::svtkExtractStructuredGridHelper()
{
  this->IndexMap = new svtk::detail::svtkIndexMap;
  this->Invalidate();
}

//-----------------------------------------------------------------------------
svtkExtractStructuredGridHelper::~svtkExtractStructuredGridHelper()
{
  delete this->IndexMap;
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::Invalidate()
{
  this->VOI[0] = 0;
  this->VOI[1] = -1;
  this->VOI[2] = 0;
  this->VOI[3] = -1;
  this->VOI[4] = 0;
  this->VOI[5] = -1;
  this->InputWholeExtent[0] = 0;
  this->InputWholeExtent[1] = -1;
  this->InputWholeExtent[2] = 0;
  this->InputWholeExtent[3] = -1;
  this->InputWholeExtent[4] = 0;
  this->InputWholeExtent[5] = -1;
  this->SampleRate[0] = 0;
  this->SampleRate[1] = 0;
  this->SampleRate[2] = 0;
  this->IncludeBoundary = true;

  this->OutputWholeExtent[0] = 0;
  this->OutputWholeExtent[1] = -1;
  this->OutputWholeExtent[2] = 0;
  this->OutputWholeExtent[3] = -1;
  this->OutputWholeExtent[4] = 0;
  this->OutputWholeExtent[5] = -1;

  for (int i = 0; i < 3; ++i)
  {
    this->IndexMap->Mapping[i].clear();
  }
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::Initialize(
  int inVoi[6], int wholeExtent[6], int sampleRate[3], bool includeBoundary)
{
  assert("pre: nullptr index map" && (this->IndexMap != nullptr));

  // Copy the VOI because we'll clamp it later:
  int voi[6];
  std::copy(inVoi, inVoi + 6, voi);

  // Have the parameters actually changed?
  if (std::equal(voi, voi + 6, this->VOI) &&
    std::equal(wholeExtent, wholeExtent + 6, this->InputWholeExtent) &&
    std::equal(sampleRate, sampleRate + 3, this->SampleRate) &&
    includeBoundary == this->IncludeBoundary)
  {
    // Nope.
    return;
  }

  // Is the VOI valid?
  if (voi[1] < voi[0] || voi[3] < voi[2] || voi[5] < voi[4])
  {
    this->Invalidate();
    svtkWarningMacro("Invalid volume of interest: ["
      << " [ " << voi[0] << ", " << voi[1] << " ], "
      << " [ " << voi[2] << ", " << voi[3] << " ], "
      << " [ " << voi[4] << ", " << voi[5] << " ] ]");
    return;
  }

  // Save the input parameters so we'll know when the map is out of date
  std::copy(voi, voi + 6, this->VOI);
  std::copy(wholeExtent, wholeExtent + 6, this->InputWholeExtent);
  std::copy(sampleRate, sampleRate + 3, this->SampleRate);
  this->IncludeBoundary = includeBoundary;

  svtkBoundingBox wExtB(
    wholeExtent[0], wholeExtent[1], wholeExtent[2], wholeExtent[3], wholeExtent[4], wholeExtent[5]);
  svtkBoundingBox voiB(voi[0], voi[1], voi[2], voi[3], voi[4], voi[5]);

  if (!wExtB.Intersects(voiB))
  {
    this->Invalidate();
    svtkDebugMacro(<< "Extent [" << wholeExtent[0] << ", " << wholeExtent[1] << ", "
                  << wholeExtent[2] << ", " << wholeExtent[3] << ", " << wholeExtent[4] << ", "
                  << wholeExtent[5] << "] does not contain VOI [" << voi[0] << ", " << voi[1]
                  << ", " << voi[2] << ", " << voi[3] << ", " << voi[4] << ", " << voi[5] << "].");
    return;
  }

  // Clamp VOI to Whole Extent
  svtkStructuredExtent::Clamp(voi, wholeExtent);

  // Create mapping between output extent and input extent.
  // Compute the output whole extent in the process.
  for (int dim = 0; dim < 3; ++dim)
  {
    // +2: +1 to include start/end points, +1 in case we need to append an
    // extra point for includeBoundary edge cases.
    this->IndexMap->Mapping[dim].resize(voi[2 * dim + 1] - voi[2 * dim] + 2);

    int outIdx = 0;
    // the start inIdx should account for the extent index offset
    int inIdx = voi[2 * dim] - wholeExtent[2 * dim];
    int idxSize = voi[2 * dim + 1] - wholeExtent[2 * dim];
    while (inIdx <= idxSize)
    {
      this->IndexMap->Mapping[dim][outIdx++] = inIdx;
      inIdx += sampleRate[dim];
    } // END for all points in this dimension, strided by the sample rate

    if (includeBoundary && this->IndexMap->Mapping[dim][outIdx - 1] != idxSize)
    {
      this->IndexMap->Mapping[dim][outIdx++] = idxSize;
    }
    this->IndexMap->Mapping[dim].resize(outIdx);

    // Preserve the extent range when sample rate is 1, otherwise extents start
    // at 0 if downsampling.
    int offset = this->SampleRate[dim] == 1 ? voi[2 * dim] : 0;

    // Update output whole extent
    this->OutputWholeExtent[2 * dim] = offset;
    this->OutputWholeExtent[2 * dim + 1] =
      offset + static_cast<int>(this->IndexMap->Mapping[dim].size() - 1);
  } // END for all dimensions
}

//-----------------------------------------------------------------------------
bool svtkExtractStructuredGridHelper::IsValid() const
{
  return this->OutputWholeExtent[0] <= this->OutputWholeExtent[1] &&
    this->OutputWholeExtent[2] <= this->OutputWholeExtent[3] &&
    this->OutputWholeExtent[4] <= this->OutputWholeExtent[5];
}

//-----------------------------------------------------------------------------
int svtkExtractStructuredGridHelper::GetMappedIndex(int dim, int outIdx)
{
  // Sanity Checks
  assert("pre: dimension dim is out-of-bounds!" && dim >= 0 && dim < 3);
  assert("pre: point index out-of-bounds!" && outIdx >= 0 && outIdx < this->GetSize(dim));
  return this->IndexMap->Mapping[dim][outIdx];
}

//-----------------------------------------------------------------------------
int svtkExtractStructuredGridHelper::GetMappedIndexFromExtentValue(int dim, int outExtVal)
{
  // Sanity Checks
  assert("pre: dimension dim is out-of-bounds!" && dim >= 0 && dim < 3);
  assert("pre: extent value out-of-bounds!" && outExtVal >= this->OutputWholeExtent[2 * dim] &&
    outExtVal <= this->OutputWholeExtent[2 * dim + 1]);
  int outIdx = outExtVal - this->OutputWholeExtent[2 * dim];
  return this->IndexMap->Mapping[dim][outIdx];
}

//-----------------------------------------------------------------------------
int svtkExtractStructuredGridHelper::GetMappedExtentValue(int dim, int outExtVal)
{
  // Sanity Checks
  assert("pre: dimension dim is out-of-bounds!" && dim >= 0 && dim < 3);
  assert("pre: extent value out-of-bounds!" && outExtVal >= this->OutputWholeExtent[2 * dim] &&
    outExtVal <= this->OutputWholeExtent[2 * dim + 1]);
  int outIdx = outExtVal - this->OutputWholeExtent[2 * dim];
  return this->IndexMap->Mapping[dim][outIdx] + this->InputWholeExtent[2 * dim];
}

//-----------------------------------------------------------------------------
int svtkExtractStructuredGridHelper::GetMappedExtentValueFromIndex(int dim, int outIdx)
{
  // Sanity Checks
  assert("pre: dimension dim is out-of-bounds!" && dim >= 0 && dim < 3);
  assert("pre: point index out-of-bounds!" && outIdx >= 0 && outIdx < this->GetSize(dim));
  return this->IndexMap->Mapping[dim][outIdx] + this->InputWholeExtent[2 * dim];
}

//-----------------------------------------------------------------------------
int svtkExtractStructuredGridHelper::GetSize(const int dim)
{
  assert("pre: dimension dim is out-of-bounds!" && (dim >= 0) && (dim < 3));
  return (static_cast<int>(this->IndexMap->Mapping[dim].size()));
}

//-----------------------------------------------------------------------------
namespace
{
int roundToInt(double r)
{
  return r > 0.0 ? r + 0.5 : r - 0.5;
}
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::ComputeBeginAndEnd(
  int inExt[6], int voi[6], int begin[3], int end[3])
{
  svtkBoundingBox inExtB(inExt[0], inExt[1], inExt[2], inExt[3], inExt[4], inExt[5]);
  svtkBoundingBox uExtB(voi[0], voi[1], voi[2], voi[3], voi[4], voi[5]);
  std::fill(begin, begin + 3, 0);
  std::fill(end, end + 3, -1);

  int uExt[6];
  if (uExtB.IntersectBox(inExtB))
  {

    for (int i = 0; i < 6; ++i)
    {
      uExt[i] = static_cast<int>(roundToInt(uExtB.GetBound(i)));
    }

    // Find the first and last indices in the map that are
    // within data extents. These are the extents of the
    // output data.
    for (int dim = 0; dim < 3; ++dim)
    {
      for (int idx = 0; idx < this->GetSize(dim); ++idx)
      {
        int extVal = this->GetMappedExtentValueFromIndex(dim, idx);
        if (extVal >= uExt[2 * dim] && extVal <= uExt[2 * dim + 1])
        {
          begin[dim] = idx;
          break;
        }
      } // END for all indices with

      for (int idx = this->GetSize(dim) - 1; idx >= 0; --idx)
      {
        int extVal = this->GetMappedExtentValueFromIndex(dim, idx);
        if (extVal <= uExt[2 * dim + 1] && extVal >= uExt[2 * dim])
        {
          end[dim] = idx;
          break;
        }
      } // END for all indices

    } // END for all dimensions

  } // END if box intersects
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::CopyPointsAndPointData(int inExt[6], int outExt[6],
  svtkPointData* pd, svtkPoints* inpnts, svtkPointData* outPD, svtkPoints* outpnts)
{
  assert("pre: nullptr input point-data!" && (pd != nullptr));
  assert("pre: nullptr output point-data!" && (outPD != nullptr));

  // short-circuit
  if ((pd->GetNumberOfArrays() == 0) && (inpnts == nullptr))
  {
    // nothing to copy
    return;
  }

  // Get the size of the input and output
  svtkIdType inSize = svtkStructuredData::GetNumberOfPoints(inExt);
  svtkIdType outSize = svtkStructuredData::GetNumberOfPoints(outExt);
  (void)inSize; // Prevent warnings, this is only used in debug builds.

  // Check if we can use some optimizations:
  bool canCopyRange = I(this->SampleRate) == 1;
  bool useMapping =
    !(I(this->SampleRate) == 1 && J(this->SampleRate) == 1 && K(this->SampleRate) == 1);

  if (inpnts != nullptr)
  {
    assert("pre: output points data-structure is nullptr!" && (outpnts != nullptr));
    outpnts->SetDataType(inpnts->GetDataType());
    outpnts->SetNumberOfPoints(outSize);
  }
  outPD->CopyAllocate(pd, outSize, outSize);

  // Lists for batching copy operations:
  svtkNew<svtkIdList> srcIds;
  svtkNew<svtkIdList> dstIds;
  if (!canCopyRange)
  {
    svtkIdType bufferSize = IMAX(outExt) - IMIN(outExt) + 1;
    srcIds->Allocate(bufferSize);
    dstIds->Allocate(bufferSize);
  }

  int ijk[3];
  int src_ijk[3];
  for (K(ijk) = KMIN(outExt); K(ijk) <= KMAX(outExt); ++K(ijk))
  {
    K(src_ijk) = useMapping ? this->GetMappedExtentValue(2, K(ijk)) : K(ijk);

    for (J(ijk) = JMIN(outExt); J(ijk) <= JMAX(outExt); ++J(ijk))
    {
      J(src_ijk) = useMapping ? this->GetMappedExtentValue(1, J(ijk)) : J(ijk);

      if (canCopyRange)
      {
        // Find the first point id:
        I(ijk) = IMIN(outExt);
        I(src_ijk) = I(ijk);

        svtkIdType srcStart = svtkStructuredData::ComputePointIdForExtent(inExt, src_ijk);
        svtkIdType dstStart = svtkStructuredData::ComputePointIdForExtent(outExt, ijk);
        svtkIdType num = IMAX(outExt) - IMIN(outExt) + 1;

        // Sanity checks
        assert("pre: srcStart out of bounds" && (srcStart >= 0) && (srcStart < inSize));
        assert("pre: dstStart out of bounds" && (dstStart >= 0) && (dstStart < outSize));

        if (inpnts != nullptr)
        {
          outpnts->InsertPoints(dstStart, num, srcStart, inpnts);
        }
        outPD->CopyData(pd, dstStart, num, srcStart);
      }
      else // canCopyRange
      {
        for (I(ijk) = IMIN(outExt); I(ijk) <= IMAX(outExt); ++I(ijk))
        {
          I(src_ijk) = useMapping ? this->GetMappedExtentValue(0, I(ijk)) : I(ijk);

          svtkIdType srcIdx = svtkStructuredData::ComputePointIdForExtent(inExt, src_ijk);
          svtkIdType targetIdx = svtkStructuredData::ComputePointIdForExtent(outExt, ijk);

          // Sanity checks
          assert("pre: srcIdx out of bounds" && (srcIdx >= 0) && (srcIdx < inSize));
          assert("pre: targetIdx out of bounds" && (targetIdx >= 0) && (targetIdx < outSize));

          srcIds->InsertNextId(srcIdx);
          dstIds->InsertNextId(targetIdx);

        } // END for all i

        if (inpnts != nullptr)
        {
          outpnts->InsertPoints(dstIds, srcIds, inpnts);
        } // END if
        outPD->CopyData(pd, srcIds, dstIds);
        srcIds->Reset();
        dstIds->Reset();

      } // END else canCopyRange

    } // END for all j

  } // END for all k
}

//-----------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::CopyCellData(
  int inExt[6], int outExt[6], svtkCellData* cd, svtkCellData* outCD)
{
  assert("pre: nullptr input cell-data!" && (cd != nullptr));
  assert("pre: nullptr output cell-data!" && (outCD != nullptr));

  // short-circuit
  if (cd->GetNumberOfArrays() == 0)
  {
    // nothing to copy
    return;
  }

  // Get the size of the output & allocate output
  svtkIdType inSize = svtkStructuredData::GetNumberOfCells(inExt);
  svtkIdType outSize = svtkStructuredData::GetNumberOfCells(outExt);
  (void)inSize; // Prevent warnings, this is only used in debug builds.
  outCD->CopyAllocate(cd, outSize, outSize);

  // Check if we can use some optimizations:
  bool canCopyRange = I(this->SampleRate) == 1;
  bool useMapping =
    !(I(this->SampleRate) == 1 && J(this->SampleRate) == 1 && K(this->SampleRate) == 1);

  int inpCellExt[6];
  svtkStructuredData::GetCellExtentFromPointExtent(inExt, inpCellExt);

  int outCellExt[6];
  svtkStructuredData::GetCellExtentFromPointExtent(outExt, outCellExt);

  // clamp outCellExt using inpCellExt. This is needed for the case where outExt
  // is the outer face of the dataset along any of the dimensions.
  for (int dim = 0; dim < 3; ++dim)
  {
    EMIN(outCellExt, dim) = std::min(EMAX(inpCellExt, dim), EMIN(outCellExt, dim));
    EMAX(outCellExt, dim) = std::min(EMAX(inpCellExt, dim), EMAX(outCellExt, dim));
  }

  // Lists for batching copy operations:
  svtkNew<svtkIdList> srcIds;
  svtkNew<svtkIdList> dstIds;
  if (!canCopyRange)
  {
    svtkIdType bufferSize = IMAX(outCellExt) - IMIN(outCellExt) + 1;
    srcIds->Allocate(bufferSize);
    dstIds->Allocate(bufferSize);
  }

  int ijk[3];
  int src_ijk[3];
  for (K(ijk) = KMIN(outCellExt); K(ijk) <= KMAX(outCellExt); ++K(ijk))
  {
    K(src_ijk) = useMapping ? this->GetMappedExtentValue(2, K(ijk)) : K(ijk);
    if (K(src_ijk) == KMAX(this->InputWholeExtent) &&
      KMIN(this->InputWholeExtent) != KMAX(this->InputWholeExtent))
    {
      --K(src_ijk);
    }

    for (J(ijk) = JMIN(outCellExt); J(ijk) <= JMAX(outCellExt); ++J(ijk))
    {
      J(src_ijk) = useMapping ? this->GetMappedExtentValue(1, J(ijk)) : J(ijk);
      if (J(src_ijk) == JMAX(this->InputWholeExtent) &&
        JMIN(this->InputWholeExtent) != JMAX(this->InputWholeExtent))
      {
        --J(src_ijk);
      }

      if (canCopyRange)
      {
        // Find the first cell id:
        I(ijk) = IMIN(outCellExt);
        I(src_ijk) = I(ijk);

        // NOTE: since we are operating on cell extents, ComputePointID below
        // really returns the cell ID
        svtkIdType srcStart = svtkStructuredData::ComputePointIdForExtent(inpCellExt, src_ijk);
        svtkIdType dstStart = svtkStructuredData::ComputePointIdForExtent(outCellExt, ijk);
        svtkIdType num = IMAX(outCellExt) - IMIN(outCellExt) + 1;

        // Sanity checks
        assert("pre: srcStart out of bounds" && (srcStart >= 0) && (srcStart < inSize));
        assert("pre: dstStart out of bounds" && (dstStart >= 0) && (dstStart < outSize));

        outCD->CopyData(cd, dstStart, num, srcStart);
      }
      else // canCopyRange
      {
        for (I(ijk) = IMIN(outCellExt); I(ijk) <= IMAX(outCellExt); ++I(ijk))
        {
          I(src_ijk) = useMapping ? this->GetMappedExtentValue(0, I(ijk)) : I(ijk);
          if (I(src_ijk) == IMAX(this->InputWholeExtent) &&
            IMIN(this->InputWholeExtent) != IMAX(this->InputWholeExtent))
          {
            --I(src_ijk);
          }

          // NOTE: since we are operating on cell extents, ComputePointID below
          // really returns the cell ID
          svtkIdType srcIdx = svtkStructuredData::ComputePointIdForExtent(inpCellExt, src_ijk);

          svtkIdType targetIdx = svtkStructuredData::ComputePointIdForExtent(outCellExt, ijk);

          srcIds->InsertNextId(srcIdx);
          dstIds->InsertNextId(targetIdx);
        } // END for all i

        outCD->CopyData(cd, srcIds, dstIds);
        srcIds->Reset();
        dstIds->Reset();

      } // END else canCopyRange
    }   // END for all j
  }     // END for all k
}

//------------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::GetPartitionedVOI(const int globalVOI[6],
  const int partitionedExtent[6], const int sampleRate[3], bool includeBoundary,
  int partitionedVOI[6])
{
  // 1D Example:
  //   InputWholeExtent = [0, 20]
  //   GlobalVOI = [3, 17]
  //   SampleRate = 2
  //   OutputWholeExtent = [0, 7]
  //   Processes = 2
  //
  // Process 0:
  //   PartitionedInputExtent = [0, 10]
  //   ClampedVOI = [3, 10]
  //   PartitionedVOI = [3, 9] (due to sampling)
  //
  // Process 1:
  //   PartitionedInputExtent = [10, 20]
  //   ClampedVOI = [10, 17]
  //   PartitionedVOI = [11, 17] (offset due to sampling)
  //
  // This method calculates the PartitionedVOI.

  // Start with filter's VOI (Ex: [3, 17] | [3, 17] )
  std::copy(globalVOI, globalVOI + 6, partitionedVOI);

  // Clamp to paritioned data (Ex: [3, 10] | [10, 17] )
  svtkStructuredExtent::Clamp(partitionedVOI, partitionedExtent);

  // Adjust for spacing: (Ex: [3, 9] | [11, 17] )
  for (int dim = 0; dim < 3; ++dim)
  {
    // Minimia:
    // Ex: 0 | 7
    int delta = EMIN(partitionedVOI, dim) - EMIN(globalVOI, dim);
    // Ex: 0 | 1
    delta %= sampleRate[dim];
    if (delta != 0)
    {
      delta = sampleRate[dim] - delta;
    }
    // Ex: 3 | 11
    EMIN(partitionedVOI, dim) += delta;

    if (includeBoundary && EMAX(partitionedVOI, dim) == EMAX(globalVOI, dim))
    {
      continue;
    }

    // Maxima:
    // Ex: 7 | 6
    delta = EMAX(partitionedVOI, dim) - EMIN(partitionedVOI, dim);
    // Ex: 1 | 0
    delta %= sampleRate[dim];
    EMAX(partitionedVOI, dim) -= delta;
  }
}

//------------------------------------------------------------------------------
void svtkExtractStructuredGridHelper::GetPartitionedOutputExtent(const int globalVOI[6],
  const int partitionedVOI[6], const int outputWholeExtent[6], const int sampleRate[3],
  bool includeBoundary, int partitionedOutputExtent[6])
{
  // 1D Example:
  //   InputWholeExtent = [0, 20]
  //   GlobalVOI = [3, 17]
  //   SampleRate = 2
  //   OutputWholeExtent = [0, 7]
  //   Processes = 2
  //
  // Process 0:
  //   PartitionedInputExtent = [0, 10]
  //   PartitionedVOI = [3, 9] (due to sampling)
  //   SerialOutputExtent = [0, 3]
  //   PartitionedOutputExtent = [0, 3]
  //
  // Process 1:
  //   PartitionedInputExtent = [10, 20]
  //   PartitionedVOI = [11, 17] (offset due to sampling)
  //   SerialOutputExtent = [0, 3]
  //   PartitionedOutputExtent = [4, 7]
  //
  // This method computes the PartitionedOutputExtent. The gap [3, 4] will be
  // cleaned up by the parallel filter using svtkStructuredImplicitConnectivity.
  for (int dim = 0; dim < 3; ++dim)
  {
    if (sampleRate[dim] == 1)
    {
      // If we're not downsampling, just return the partitioned VOI:
      EMIN(partitionedOutputExtent, dim) = EMIN(partitionedVOI, dim);
      EMAX(partitionedOutputExtent, dim) = EMAX(partitionedVOI, dim);
    }
    else
    {
      // If we downsample, the global output VOI will be offset to start at 0,
      // so we'll adjust the minimum
      // Ex: 0 | 4
      EMIN(partitionedOutputExtent, dim) =
        (EMIN(partitionedVOI, dim) - EMIN(globalVOI, dim)) / sampleRate[dim];

      if (includeBoundary && EMAX(partitionedVOI, dim) == EMAX(globalVOI, dim))
      {
        int length = EMAX(partitionedVOI, dim) - EMIN(globalVOI, dim);
        EMAX(partitionedOutputExtent, dim) = length / sampleRate[dim];
        EMAX(partitionedOutputExtent, dim) += ((length % sampleRate[dim]) == 0) ? 0 : 1;
      }
      else
      {
        // Ex: 3 | 7
        EMAX(partitionedOutputExtent, dim) =
          (EMAX(partitionedVOI, dim) - EMIN(globalVOI, dim)) / sampleRate[dim];
      }

      // Account for any offsets in the OutputWholeExtent:
      EMIN(partitionedOutputExtent, dim) += EMIN(outputWholeExtent, dim);
      EMAX(partitionedOutputExtent, dim) += EMIN(outputWholeExtent, dim);
    }
  }
}
