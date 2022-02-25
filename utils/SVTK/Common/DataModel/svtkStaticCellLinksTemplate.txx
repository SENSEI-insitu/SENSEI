/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStaticCellLinksTemplate.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkStaticCellLinksTemplate.h"

#ifndef svtkStaticCellLinksTemplate_txx
#define svtkStaticCellLinksTemplate_txx

#include "svtkCellArray.h"
#include "svtkDataArrayRange.h"
#include "svtkDataSet.h"
#include "svtkExplicitStructuredGrid.h"
#include "svtkPolyData.h"
#include "svtkSMPTools.h"
#include "svtkUnstructuredGrid.h"
#include <array>
#include <atomic>

#include <type_traits>

//----------------------------------------------------------------------------
// Note: this class is a faster, threaded version of svtkCellLinks. It uses
// svtkSMPTools and std::atomic.

//----------------------------------------------------------------------------
// Default constructor. BuildLinks() does most of the work.
template <typename TIds>
svtkStaticCellLinksTemplate<TIds>::svtkStaticCellLinksTemplate()
  : LinksSize(0)
  , NumPts(0)
  , NumCells(0)
  , Links(nullptr)
  , Offsets(nullptr)
{
  if (std::is_same<unsigned short, TIds>::value)
  {
    this->Type = svtkAbstractCellLinks::STATIC_CELL_LINKS_USHORT;
  }

  else if (std::is_same<unsigned int, TIds>::value)
  {
    this->Type = svtkAbstractCellLinks::STATIC_CELL_LINKS_UINT;
  }

  else if (std::is_same<svtkIdType, TIds>::value)
  {
    this->Type = svtkAbstractCellLinks::STATIC_CELL_LINKS_IDTYPE;
  }

  else
  {
    this->Type = svtkAbstractCellLinks::STATIC_CELL_LINKS_SPECIALIZED;
  }

  this->SequentialProcessing = false;
}

//----------------------------------------------------------------------------
template <typename TIds>
svtkStaticCellLinksTemplate<TIds>::~svtkStaticCellLinksTemplate()
{
  this->Initialize();
}

//----------------------------------------------------------------------------
// Clean up any previously allocated memory
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::Initialize()
{
  if (this->Links)
  {
    delete[] this->Links;
    this->Links = nullptr;
  }
  if (this->Offsets)
  {
    delete[] this->Offsets;
    this->Offsets = nullptr;
  }
}

//----------------------------------------------------------------------------
// Build the link list array for any dataset type. Specialized methods are
// used for dataset types that use svtkCellArrays to represent cells.
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::BuildLinks(svtkDataSet* ds)
{
  // Use a fast path if polydata or unstructured grid
  if (ds->GetDataObjectType() == SVTK_POLY_DATA)
  {
    return this->BuildLinks(static_cast<svtkPolyData*>(ds));
  }

  else if (ds->GetDataObjectType() == SVTK_UNSTRUCTURED_GRID)
  {
    return this->BuildLinks(static_cast<svtkUnstructuredGrid*>(ds));
  }

  else if (ds->GetDataObjectType() == SVTK_EXPLICIT_STRUCTURED_GRID)
  {
    return this->BuildLinks(static_cast<svtkExplicitStructuredGrid*>(ds));
  }

  // Any other type of dataset. Generally this is not called as datasets have
  // their own, more efficient ways of getting similar information.
  // Make sure that we clear out previous allocation.
  this->NumCells = ds->GetNumberOfCells();
  this->NumPts = ds->GetNumberOfPoints();

  svtkIdType npts, ptId;
  svtkIdType cellId, j;
  svtkIdList* cellPts = svtkIdList::New();

  // Traverse data to determine number of uses of each point. Also count the
  // number of links to allocate.
  this->Offsets = new TIds[this->NumPts + 1];
  std::fill_n(this->Offsets, this->NumPts, 0);

  for (this->LinksSize = 0, cellId = 0; cellId < this->NumCells; cellId++)
  {
    ds->GetCellPoints(cellId, cellPts);
    npts = cellPts->GetNumberOfIds();
    for (j = 0; j < npts; j++)
    {
      this->Offsets[cellPts->GetId(j)]++;
      this->LinksSize++;
    }
  }

  // Allocate space for links. Perform prefix sum.
  this->Links = new TIds[this->LinksSize + 1];
  this->Links[this->LinksSize] = this->NumPts;

  for (ptId = 0; ptId < this->NumPts; ++ptId)
  {
    npts = this->Offsets[ptId + 1];
    this->Offsets[ptId + 1] = this->Offsets[ptId] + npts;
  }

  // Now build the links. The summation from the prefix sum indicates where
  // the cells are to be inserted. Each time a cell is inserted, the offset
  // is decremented. In the end, the offset array is also constructed as it
  // points to the beginning of each cell run.
  for (cellId = 0; cellId < this->NumCells; ++cellId)
  {
    ds->GetCellPoints(cellId, cellPts);
    npts = cellPts->GetNumberOfIds();
    for (j = 0; j < npts; ++j)
    {
      ptId = cellPts->GetId(j);
      this->Offsets[ptId]--;
      this->Links[this->Offsets[ptId]] = cellId;
    }
  }
  this->Offsets[this->NumPts] = this->LinksSize;

  cellPts->Delete();
}

namespace svtkSCLT_detail
{

struct CountPoints
{
  template <typename CellStateT, typename TIds>
  void operator()(CellStateT& state,
    TIds* linkOffsets, // May be std::atomic<...>
    const svtkIdType beginCellId, const svtkIdType endCellId, const svtkIdType idOffset = 0)
  {
    using ValueType = typename CellStateT::ValueType;
    const svtkIdType connBeginId = state.GetBeginOffset(beginCellId);
    const svtkIdType connEndId = state.GetEndOffset(endCellId - 1);
    auto connRange = svtk::DataArrayValueRange<1>(state.GetConnectivity(), connBeginId, connEndId);

    // Count number of point uses
    for (const ValueType ptId : connRange)
    {
      ++linkOffsets[static_cast<size_t>(idOffset + ptId)];
    }
  }
};

// Serial version:
struct BuildLinks
{
  template <typename CellStateT, typename TIds>
  void operator()(CellStateT& state, TIds* linkOffsets, TIds* links, const svtkIdType idOffset = 0)
  {
    using ValueType = typename CellStateT::ValueType;

    const svtkIdType numCells = state.GetNumberOfCells();

    // Now build the links. The summation from the prefix sum indicates where
    // the cells are to be inserted. Each time a cell is inserted, the offset
    // is decremented. In the end, the offset array is also constructed as it
    // points to the beginning of each cell run.
    for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
    {
      const auto cell = state.GetCellRange(cellId);
      for (const ValueType cellPtId : cell)
      {
        const size_t ptId = static_cast<size_t>(cellPtId);
        --linkOffsets[ptId];
        links[linkOffsets[ptId]] = static_cast<TIds>(idOffset + cellId);
      }
    }
  }
};

// Parallel version:
struct BuildLinksThreaded
{
  template <typename CellStateT, typename TIds>
  void operator()(CellStateT& state, const TIds* offsets, std::atomic<TIds>* counts, TIds* links,
    const svtkIdType beginCellId, const svtkIdType endCellId, const TIds idOffset = 0)
  {
    using ValueType = typename CellStateT::ValueType;

    // Now build the links. The summation from the prefix sum indicates where
    // the cells are to be inserted. Each time a cell is inserted, the offset
    // is decremented. In the end, the offset array is also constructed as it
    // points to the beginning of each cell run.
    for (svtkIdType cellId = beginCellId; cellId < endCellId; ++cellId)
    {
      const auto cell = state.GetCellRange(cellId);
      for (const ValueType cellPtId : cell)
      {
        const size_t ptId = static_cast<size_t>(cellPtId);
        // memory_order_relaxed is safe here, since we're not using the atomics
        // for synchroniziation.
        const TIds offset =
          offsets[ptId] + counts[ptId].fetch_sub(1, std::memory_order_relaxed) - 1;
        links[offset] = idOffset + cellId;
      }
    }
  }
};

} // end namespace svtkSCLT_detail

//----------------------------------------------------------------------------
// Build the link list array for unstructured grids. Note this is a serial
// implementation: while there is another method (threaded) that is usually
// much faster, in certain pathological situations the serial version can be
// faster.
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::SerialBuildLinks(
  const svtkIdType numPts, const svtkIdType numCells, svtkCellArray* cellArray)
{
  // Basic information about the grid
  this->NumPts = numPts;
  this->NumCells = numCells;

  this->LinksSize = cellArray->GetConnectivityArray()->GetNumberOfValues();

  // Extra one allocated to simplify later pointer manipulation
  this->Links = new TIds[this->LinksSize + 1];
  this->Links[this->LinksSize] = this->NumPts;
  this->Offsets = new TIds[numPts + 1];
  std::fill_n(this->Offsets, this->NumPts + 1, 0);

  // Count how many cells each point appears in:
  cellArray->Visit(svtkSCLT_detail::CountPoints{}, this->Offsets, 0, numCells);

  // Perform prefix sum (inclusive scan)
  for (svtkIdType ptId = 0; ptId < this->NumPts; ++ptId)
  {
    const svtkIdType npts = this->Offsets[ptId + 1];
    this->Offsets[ptId + 1] = this->Offsets[ptId] + npts;
  }

  // Construct the links table and finalize the offsets:
  cellArray->Visit(svtkSCLT_detail::BuildLinks{}, this->Offsets, this->Links);

  this->Offsets[numPts] = this->LinksSize;
}

//----------------------------------------------------------------------------
// Threaded implementation of BuildLinks() using svtkSMPTools and std::atomic.

namespace
{ // anonymous

template <typename TIds>
struct CountUses
{
  svtkCellArray* CellArray;
  std::atomic<TIds>* Counts;

  CountUses(svtkCellArray* cellArray, std::atomic<TIds>* counts)
    : CellArray(cellArray)
    , Counts(counts)
  {
  }

  void operator()(svtkIdType cellId, svtkIdType endCellId)
  {
    this->CellArray->Visit(svtkSCLT_detail::CountPoints{}, this->Counts, cellId, endCellId);
  }
};

template <typename TIds>
struct InsertLinks
{
  svtkCellArray* CellArray;
  std::atomic<TIds>* Counts;
  const TIds* Offsets;
  TIds* Links;

  InsertLinks(svtkCellArray* cellArray, std::atomic<TIds>* counts, const TIds* offsets, TIds* links)
    : CellArray(cellArray)
    , Counts(counts)
    , Offsets(offsets)
    , Links(links)
  {
  }

  void operator()(svtkIdType cellId, svtkIdType endCellId)
  {
    this->CellArray->Visit(svtkSCLT_detail::BuildLinksThreaded{}, this->Offsets, this->Counts,
      this->Links, cellId, endCellId);
  }
};

} // anonymous

//----------------------------------------------------------------------------
// Build the link list array for unstructured grids. Note this is a threaded
// implementation: it uses SMPTools and atomics to prevent race situations.
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::ThreadedBuildLinks(
  const svtkIdType numPts, const svtkIdType numCells, svtkCellArray* cellArray)
{
  // Basic information about the grid
  this->NumPts = numPts;
  this->NumCells = numCells;

  // Trick follows: the size of the Links array is equal to
  // the size of the cell array, minus the number of cells.
  this->LinksSize = cellArray->GetNumberOfConnectivityIds();

  // Extra one allocated to simplify later pointer manipulation
  this->Links = new TIds[this->LinksSize + 1];
  this->Links[this->LinksSize] = this->NumPts;

  // Create an array of atomics with initial count=0. This will keep
  // track of point uses. Count them in parallel.
  std::atomic<TIds>* counts = new std::atomic<TIds>[numPts] {};
  CountUses<TIds> count(cellArray, counts);
  svtkSMPTools::For(0, numCells, count);

  // Perform prefix sum to determine offsets
  svtkIdType ptId, npts;
  this->Offsets = new TIds[numPts + 1];
  this->Offsets[0] = 0;
  for (ptId = 1; ptId < numPts; ++ptId)
  {
    npts = counts[ptId - 1];
    this->Offsets[ptId] = this->Offsets[ptId - 1] + npts;
  }
  this->Offsets[numPts] = this->LinksSize;

  // Now insert cell ids into cell links.
  InsertLinks<TIds> insertLinks(cellArray, counts, this->Offsets, this->Links);
  svtkSMPTools::For(0, numCells, insertLinks);

  // Clean up
  delete[] counts;
}

//----------------------------------------------------------------------------
// Build the link list array for unstructured grids
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::BuildLinks(svtkUnstructuredGrid* ugrid)
{
  // Basic information about the grid
  svtkIdType numPts = ugrid->GetNumberOfPoints();
  svtkIdType numCells = ugrid->GetNumberOfCells();

  // We're going to get into the guts of the class
  svtkCellArray* cellArray = ugrid->GetCells();

  // Use serial or threaded implementations
  if (!this->SequentialProcessing)
  {
    this->ThreadedBuildLinks(numPts, numCells, cellArray);
  }
  else
  {
    this->SerialBuildLinks(numPts, numCells, cellArray);
  }
}

//----------------------------------------------------------------------------
// Build the link list array for unstructured grids
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::BuildLinks(svtkExplicitStructuredGrid* esgrid)
{
  // Basic information about the grid
  svtkIdType numPts = esgrid->GetNumberOfPoints();
  svtkIdType numCells = esgrid->GetNumberOfCells();

  // We're going to get into the guts of the class
  svtkCellArray* cellArray = esgrid->GetCells();

  // Use serial implementation. TODO: add threaded implementation
  this->SerialBuildLinks(numPts, numCells, cellArray);
}

//----------------------------------------------------------------------------
// Build the link list array for poly data. This is more complex because there
// are potentially four different cell arrays to contend with.
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::BuildLinks(svtkPolyData* pd)
{
  // Basic information about the grid
  this->NumCells = pd->GetNumberOfCells();
  this->NumPts = pd->GetNumberOfPoints();

  svtkCellArray* cellArrays[4];
  svtkIdType numCells[4];
  svtkIdType sizes[4];
  int i, j;

  cellArrays[0] = pd->GetVerts();
  cellArrays[1] = pd->GetLines();
  cellArrays[2] = pd->GetPolys();
  cellArrays[3] = pd->GetStrips();

  for (i = 0; i < 4; ++i)
  {
    if (cellArrays[i] != nullptr)
    {
      numCells[i] = cellArrays[i]->GetNumberOfCells();
      sizes[i] = cellArrays[i]->GetConnectivityArray()->GetNumberOfValues();
    }
    else
    {
      numCells[i] = 0;
      sizes[i] = 0;
    }
  } // for the four polydata arrays

  // Allocate
  this->LinksSize = sizes[0] + sizes[1] + sizes[2] + sizes[3];
  this->Links = new TIds[this->LinksSize + 1];
  this->Links[this->LinksSize] = this->NumPts;
  this->Offsets = new TIds[this->NumPts + 1];
  this->Offsets[this->NumPts] = this->LinksSize;
  std::fill_n(this->Offsets, this->NumPts + 1, 0);

  // Now create the links.
  svtkIdType npts, CellId, ptId;

  // Visit the four arrays
  for (CellId = 0, j = 0; j < 4; ++j)
  {
    // Count number of point uses
    cellArrays[j]->Visit(svtkSCLT_detail::CountPoints{}, this->Offsets, 0, numCells[j], CellId);
    CellId += numCells[j];
  } // for each of the four polydata cell arrays

  // Perform prefix sum (inclusive scan)
  for (ptId = 0; ptId < this->NumPts; ++ptId)
  {
    npts = this->Offsets[ptId + 1];
    this->Offsets[ptId + 1] = this->Offsets[ptId] + npts;
  }

  // Now build the links. The summation from the prefix sum indicates where
  // the cells are to be inserted. Each time a cell is inserted, the offset
  // is decremented. In the end, the offset array is also constructed as it
  // points to the beginning of each cell run.
  for (CellId = 0, j = 0; j < 4; ++j)
  {
    cellArrays[j]->Visit(svtkSCLT_detail::BuildLinks{}, this->Offsets, this->Links, CellId);
    CellId += numCells[j];
  } // for each of the four polydata arrays
  this->Offsets[this->NumPts] = this->LinksSize;
}

//----------------------------------------------------------------------------
// Satisfy svtkAbstractCellLinks API
template <typename TIds>
unsigned long svtkStaticCellLinksTemplate<TIds>::GetActualMemorySize()
{
  unsigned long total = 0;
  if (Links != nullptr)
  {
    total = static_cast<unsigned long>((this->LinksSize + 1) * sizeof(TIds));
    total += static_cast<unsigned long>((this->NumPts + 1) * sizeof(TIds));
  }
  return total;
}

//----------------------------------------------------------------------------
// Satisfy svtkAbstractCellLinks API
template <typename TIds>
void svtkStaticCellLinksTemplate<TIds>::DeepCopy(svtkAbstractCellLinks* src)
{
  svtkStaticCellLinksTemplate<TIds>* links = dynamic_cast<svtkStaticCellLinksTemplate<TIds>*>(src);

  if (links)
  {
    this->LinksSize = links->LinksSize;
    this->NumPts = links->NumPts;
    this->NumCells = links->NumCells;

    if (this->Links != nullptr)
    {
      delete[] this->Links;
    }
    this->Links = new TIds[this->LinksSize + 1];
    std::copy(links->Links, links->Links + (this->LinksSize + 1), this->Links);

    if (this->Offsets != nullptr)
    {
      delete[] this->Offsets;
    }
    this->Offsets = new TIds[this->NumPts + 1];
    std::copy(links->Offsets, links->Offsets + (this->NumPts + 1), this->Offsets);
  }
}

#endif
