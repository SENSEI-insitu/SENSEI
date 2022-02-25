/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPolyData.h"

#include "svtkBoundingBox.h"
#include "svtkCellArray.h"
#include "svtkCellArrayIterator.h"
#include "svtkCellData.h"
#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkLine.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPointLocator.h"
#include "svtkPolyLine.h"
#include "svtkPolyVertex.h"
#include "svtkPolygon.h"
#include "svtkQuad.h"
#include "svtkSMPTools.h"
#include "svtkSmartPointer.h"
#include "svtkTriangle.h"
#include "svtkTriangleStrip.h"
#include "svtkUnsignedCharArray.h"
#include "svtkVertex.h"

#include <stdexcept>

// svtkPolyDataInternals.h methods:
namespace svtkPolyData_detail
{

svtkStandardNewMacro(CellMap);
CellMap::CellMap() = default;
CellMap::~CellMap() = default;

} // end namespace svtkPolyData_detail

svtkStandardNewMacro(svtkPolyData);

//----------------------------------------------------------------------------
// Initialize static member.  This member is used to simplify traversal
// of verts, lines, polygons, and triangle strips lists.  It basically
// "marks" empty lists so that the traversal method "GetNextCell"
// works properly.

struct svtkPolyDataDummyContainter
{
  svtkSmartPointer<svtkCellArray> Dummy;

  svtkPolyDataDummyContainter() { this->Dummy.TakeReference(svtkCellArray::New()); }
};

svtkPolyDataDummyContainter svtkPolyData::DummyContainer;

//----------------------------------------------------------------------------
unsigned char svtkPolyData::GetCell(svtkIdType cellId, svtkIdType const*& cell)
{
  svtkIdType npts;
  const svtkIdType* pts;
  const auto type = this->GetCellPoints(cellId, npts, pts);

  if (type == SVTK_EMPTY_CELL)
  { // Cell is deleted
    cell = nullptr;
  }
  else
  {
    this->LegacyBuffer->SetNumberOfIds(npts + 1);
    this->LegacyBuffer->SetId(0, npts);
    for (svtkIdType i = 0; i < npts; ++i)
    {
      this->LegacyBuffer->SetId(i, pts[i]);
    }

    cell = this->LegacyBuffer->GetPointer(0);
  }

  return type;
}

//----------------------------------------------------------------------------
svtkCell* svtkPolyData::GetCell(int svtkNotUsed(i), int svtkNotUsed(j), int svtkNotUsed(k))
{
  svtkErrorMacro("ijk indices are only valid with structured data!");
  return nullptr;
}

//----------------------------------------------------------------------------
svtkPolyData::svtkPolyData()
{
  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_PIECES_EXTENT);
  this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);
}

//----------------------------------------------------------------------------
svtkPolyData::~svtkPolyData() = default;

//----------------------------------------------------------------------------
int svtkPolyData::GetPiece()
{
  return this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
}

//----------------------------------------------------------------------------
int svtkPolyData::GetNumberOfPieces()
{
  return this->Information->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());
}

//----------------------------------------------------------------------------
int svtkPolyData::GetGhostLevel()
{
  return this->Information->Get(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS());
}

//----------------------------------------------------------------------------
// Copy the geometric and topological structure of an input poly data object.
void svtkPolyData::CopyStructure(svtkDataSet* ds)
{
  svtkPolyData* pd = svtkPolyData::SafeDownCast(ds);
  if (!pd)
  {
    svtkErrorMacro("Input dataset is not a polydata!");
    return;
  }

  svtkPointSet::CopyStructure(ds);

  this->Verts = pd->Verts;
  this->Lines = pd->Lines;
  this->Polys = pd->Polys;
  this->Strips = pd->Strips;

  this->Cells = nullptr;
  this->Links = nullptr;
}

//----------------------------------------------------------------------------
svtkCell* svtkPolyData::GetCell(svtkIdType cellId)
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);

  svtkIdType numPts;
  const svtkIdType* pts;
  svtkCell* cell = nullptr;
  switch (tag.GetCellType())
  {
    case SVTK_VERTEX:
      if (!this->Vertex)
      {
        this->Vertex = svtkSmartPointer<svtkVertex>::New();
      }
      cell = this->Vertex;
      this->Verts->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 1);
      break;

    case SVTK_POLY_VERTEX:
      if (!this->PolyVertex)
      {
        this->PolyVertex = svtkSmartPointer<svtkPolyVertex>::New();
      }
      cell = this->PolyVertex;
      this->Verts->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts);
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_LINE:
      if (!this->Line)
      {
        this->Line = svtkSmartPointer<svtkLine>::New();
      }
      cell = this->Line;
      this->Lines->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 2);
      break;

    case SVTK_POLY_LINE:
      if (!this->PolyLine)
      {
        this->PolyLine = svtkSmartPointer<svtkPolyLine>::New();
      }
      cell = this->PolyLine;
      this->Lines->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts);
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_TRIANGLE:
      if (!this->Triangle)
      {
        this->Triangle = svtkSmartPointer<svtkTriangle>::New();
      }
      cell = this->Triangle;
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 3);
      break;

    case SVTK_QUAD:
      if (!this->Quad)
      {
        this->Quad = svtkSmartPointer<svtkQuad>::New();
      }
      cell = this->Quad;
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 4);
      break;

    case SVTK_POLYGON:
      if (!this->Polygon)
      {
        this->Polygon = svtkSmartPointer<svtkPolygon>::New();
      }
      cell = this->Polygon;
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts);
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_TRIANGLE_STRIP:
      if (!this->TriangleStrip)
      {
        this->TriangleStrip = svtkSmartPointer<svtkTriangleStrip>::New();
      }
      cell = this->TriangleStrip;
      this->Strips->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts);
      cell->Points->SetNumberOfPoints(numPts);
      break;

    default:
      if (!this->EmptyCell)
      {
        this->EmptyCell = svtkSmartPointer<svtkEmptyCell>::New();
      }
      cell = this->EmptyCell;
      return cell;
  }

  for (svtkIdType i = 0; i < numPts; ++i)
  {
    cell->PointIds->SetId(i, pts[i]);
    cell->Points->SetPoint(i, this->Points->GetPoint(pts[i]));
  }

  return cell;
}

//----------------------------------------------------------------------------
void svtkPolyData::GetCell(svtkIdType cellId, svtkGenericCell* cell)
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);

  svtkIdType numPts;
  const svtkIdType* pts;
  switch (tag.GetCellType())
  {
    case SVTK_VERTEX:
      cell->SetCellTypeToVertex();
      this->Verts->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 1);
      break;

    case SVTK_POLY_VERTEX:
      cell->SetCellTypeToPolyVertex();
      this->Verts->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts); // reset number of points
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_LINE:
      cell->SetCellTypeToLine();
      this->Lines->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 2);
      break;

    case SVTK_POLY_LINE:
      cell->SetCellTypeToPolyLine();
      this->Lines->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts); // reset number of points
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_TRIANGLE:
      cell->SetCellTypeToTriangle();
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 3);
      break;

    case SVTK_QUAD:
      cell->SetCellTypeToQuad();
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      assert(numPts == 4);
      break;

    case SVTK_POLYGON:
      cell->SetCellTypeToPolygon();
      this->Polys->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts); // reset number of points
      cell->Points->SetNumberOfPoints(numPts);
      break;

    case SVTK_TRIANGLE_STRIP:
      cell->SetCellTypeToTriangleStrip();
      this->Strips->GetCellAtId(tag.GetCellId(), numPts, pts);
      cell->PointIds->SetNumberOfIds(numPts); // reset number of points
      cell->Points->SetNumberOfPoints(numPts);
      break;

    default:
      cell->SetCellTypeToEmptyCell();
      numPts = 0;
      return;
  }

  double x[3];
  for (svtkIdType i = 0; i < numPts; ++i)
  {
    cell->PointIds->SetId(i, pts[i]);
    this->Points->GetPoint(pts[i], x);
    cell->Points->SetPoint(i, x);
  }
}

//----------------------------------------------------------------------------
void svtkPolyData::CopyCells(svtkPolyData* pd, svtkIdList* idList, svtkIncrementalPointLocator* locator)
{
  svtkIdType cellId, ptId, newId, newCellId, locatorPtId;
  svtkIdType numPts, numCellPts, i;
  svtkPoints* newPoints;
  svtkIdList* pointMap = svtkIdList::New(); // maps old pt ids into new
  svtkIdList *cellPts, *newCellPts = svtkIdList::New();
  svtkGenericCell* cell = svtkGenericCell::New();
  double x[3];
  svtkPointData* outPD = this->GetPointData();
  svtkCellData* outCD = this->GetCellData();

  numPts = pd->GetNumberOfPoints();

  if (this->GetPoints() == nullptr)
  {
    this->Points = svtkPoints::New();
  }

  newPoints = this->GetPoints();

  pointMap->SetNumberOfIds(numPts);
  for (i = 0; i < numPts; i++)
  {
    pointMap->SetId(i, -1);
  }

  // Filter the cells
  for (cellId = 0; cellId < idList->GetNumberOfIds(); cellId++)
  {
    pd->GetCell(idList->GetId(cellId), cell);
    cellPts = cell->GetPointIds();
    numCellPts = cell->GetNumberOfPoints();

    for (i = 0; i < numCellPts; i++)
    {
      ptId = cellPts->GetId(i);
      if ((newId = pointMap->GetId(ptId)) < 0)
      {
        pd->GetPoint(ptId, x);
        if (locator != nullptr)
        {
          if ((locatorPtId = locator->IsInsertedPoint(x)) == -1)
          {
            newId = newPoints->InsertNextPoint(x);
            locator->InsertNextPoint(x);
            pointMap->SetId(ptId, newId);
            outPD->CopyData(pd->GetPointData(), ptId, newId);
          }
          else
          {
            newId = locatorPtId;
          }
        }
        else
        {
          newId = newPoints->InsertNextPoint(x);
          pointMap->SetId(ptId, newId);
          outPD->CopyData(pd->GetPointData(), ptId, newId);
        }
      }
      newCellPts->InsertId(i, newId);
    }
    newCellId = this->InsertNextCell(cell->GetCellType(), newCellPts);
    outCD->CopyData(pd->GetCellData(), idList->GetId(cellId), newCellId);
    newCellPts->Reset();
  } // for all cells
  newCellPts->Delete();
  pointMap->Delete();
  cell->Delete();
}

//----------------------------------------------------------------------------
// Fast implementation of GetCellBounds().  Bounds are calculated without
// constructing a cell. This method is expected to be thread-safe.
void svtkPolyData::GetCellBounds(svtkIdType cellId, double bounds[6])
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);
  if (tag.IsDeleted())
  {
    std::fill_n(bounds, 6, 0.);
    return;
  }

  svtkIdType numPts;
  const svtkIdType* pts;
  svtkCellArray* cells = this->GetCellArrayInternal(tag);
  svtkSmartPointer<svtkCellArrayIterator> iter;
  if (cells->IsStorageShareable())
  {
    // much faster and thread-safe if storage is shareable
    cells->GetCellAtId(tag.GetCellId(), numPts, pts);
  }
  else
  {
    // guaranteed thread safe
    iter = svtk::TakeSmartPointer(cells->NewIterator());
    iter->GetCellAtId(tag.GetCellId(), numPts, pts);
  }

  // carefully compute the bounds
  double x[3];
  if (numPts)
  {
    this->Points->GetPoint(pts[0], x);
    bounds[0] = x[0];
    bounds[2] = x[1];
    bounds[4] = x[2];
    bounds[1] = x[0];
    bounds[3] = x[1];
    bounds[5] = x[2];
    for (svtkIdType i = 1; i < numPts; ++i)
    {
      this->Points->GetPoint(pts[i], x);
      bounds[0] = std::min(x[0], bounds[0]);
      bounds[1] = std::max(x[0], bounds[1]);
      bounds[2] = std::min(x[1], bounds[2]);
      bounds[3] = std::max(x[1], bounds[3]);
      bounds[4] = std::min(x[2], bounds[4]);
      bounds[5] = std::max(x[2], bounds[5]);
    }
  }
  else
  {
    svtkMath::UninitializeBounds(bounds);
  }
}

//----------------------------------------------------------------------------
// This method only considers points that are used by one or more cells. Thus
// unused points make no contribution to the bounding box computation. This
// is more costly to compute than using just the points, but for rendering
// and historical reasons, produces preferred results.
void svtkPolyData::ComputeBounds()
{
  if (this->GetMeshMTime() > this->ComputeTime)
  {
    // If there are no cells, but there are points, compute the bounds from the
    // parent class svtkPointSet (which just examines points).
    svtkIdType numPts = this->GetNumberOfPoints();
    svtkIdType numCells = this->GetNumberOfCells();
    if (numCells <= 0 && numPts > 0)
    {
      svtkPointSet::ComputeBounds();
      return;
    }

    // We are going to compute the bounds
    this->ComputeTime.Modified();

    // Make sure this svtkPolyData has points.
    if (this->Points == nullptr || numPts <= 0)
    {
      svtkMath::UninitializeBounds(this->Bounds);
      return;
    }

    // With cells available, loop over the cells of the polydata.
    // Mark points that are used by one or more cells. Unmarked
    // points do not contribute.
    unsigned char* ptUses = new unsigned char[numPts];
    std::fill_n(ptUses, numPts, 0); // initially unvisited

    svtkCellArray* cellA[4];
    cellA[0] = this->GetVerts();
    cellA[1] = this->GetLines();
    cellA[2] = this->GetPolys();
    cellA[3] = this->GetStrips();

    // Process each cell array separately. Note that threading is only used
    // if the model is big enough (since there is a cost to spinning up the
    // thread pool).
    for (auto ca = 0; ca < 4; ca++)
    {
      if ((numCells = cellA[ca]->GetNumberOfCells()) > 250000)
      {
        // Lambda to threaded compute bounds
        svtkSMPTools::For(0, numCells, [&](svtkIdType cellId, svtkIdType endCellId) {
          svtkIdType npts, ptIdx;
          const svtkIdType* pts;
          auto iter = svtk::TakeSmartPointer(cellA[ca]->NewIterator());
          for (; cellId < endCellId; ++cellId)
          {
            iter->GetCellAtId(cellId, npts, pts); // thread-safe
            for (ptIdx = 0; ptIdx < npts; ++ptIdx)
            {
              ptUses[pts[ptIdx]] = 1;
            }
          }
        }); // end lambda
      }
      else if (numCells > 0) // serial
      {
        svtkIdType npts, ptIdx;
        const svtkIdType* pts;
        for (auto cellId = 0; cellId < numCells; ++cellId)
        {
          cellA[ca]->GetCellAtId(cellId, npts, pts);
          for (ptIdx = 0; ptIdx < npts; ++ptIdx)
          {
            ptUses[pts[ptIdx]] = 1;
          }
        }
      }
    } // for all cell arrays

    // Perform the bounding box computation
    svtkBoundingBox::ComputeBounds(this->Points, ptUses, this->Bounds);
    delete[] ptUses;
  }
}

//----------------------------------------------------------------------------
// Set the cell array defining vertices.
void svtkPolyData::SetVerts(svtkCellArray* v)
{
  if (v == this->DummyContainer.Dummy)
  {
    v = nullptr;
  }

  if (v != this->Verts)
  {
    this->Verts = v;

    // Reset the cell table:
    this->Cells = nullptr;

    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Get the cell array defining vertices. If there are no vertices, an
// empty array will be returned (convenience to simplify traversal).
svtkCellArray* svtkPolyData::GetVerts()
{
  if (!this->Verts)
  {
    return this->DummyContainer.Dummy;
  }
  else
  {
    return this->Verts;
  }
}

//----------------------------------------------------------------------------
// Set the cell array defining lines.
void svtkPolyData::SetLines(svtkCellArray* l)
{
  if (l == this->DummyContainer.Dummy)
  {
    l = nullptr;
  }

  if (l != this->Lines)
  {
    this->Lines = l;

    // Reset the cell table:
    this->Cells = nullptr;

    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Get the cell array defining lines. If there are no lines, an
// empty array will be returned (convenience to simplify traversal).
svtkCellArray* svtkPolyData::GetLines()
{
  if (!this->Lines)
  {
    return this->DummyContainer.Dummy;
  }
  else
  {
    return this->Lines;
  }
}

//----------------------------------------------------------------------------
// Set the cell array defining polygons.
void svtkPolyData::SetPolys(svtkCellArray* p)
{
  if (p == this->DummyContainer.Dummy)
  {
    p = nullptr;
  }

  if (p != this->Polys)
  {
    this->Polys = p;

    // Reset the cell table:
    this->Cells = nullptr;

    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Get the cell array defining polygons. If there are no polygons, an
// empty array will be returned (convenience to simplify traversal).
svtkCellArray* svtkPolyData::GetPolys()
{
  if (!this->Polys)
  {
    return this->DummyContainer.Dummy;
  }
  else
  {
    return this->Polys;
  }
}

//----------------------------------------------------------------------------
// Set the cell array defining triangle strips.
void svtkPolyData::SetStrips(svtkCellArray* s)
{
  if (s == this->DummyContainer.Dummy)
  {
    s = nullptr;
  }

  if (s != this->Strips)
  {
    this->Strips = s;

    // Reset the cell table:
    this->Cells = nullptr;

    this->Modified();
  }
}

//----------------------------------------------------------------------------
// Get the cell array defining triangle strips. If there are no
// triangle strips, an empty array will be returned (convenience to
// simplify traversal).
svtkCellArray* svtkPolyData::GetStrips()
{
  if (!this->Strips)
  {
    return this->DummyContainer.Dummy;
  }
  else
  {
    return this->Strips;
  }
}

//----------------------------------------------------------------------------
void svtkPolyData::Cleanup()
{
  this->Vertex = nullptr;
  this->PolyVertex = nullptr;
  this->Line = nullptr;
  this->PolyLine = nullptr;
  this->Triangle = nullptr;
  this->Quad = nullptr;
  this->Polygon = nullptr;
  this->TriangleStrip = nullptr;
  this->EmptyCell = nullptr;

  this->Verts = nullptr;
  this->Lines = nullptr;
  this->Polys = nullptr;
  this->Strips = nullptr;

  this->Cells = nullptr;
  this->Links = nullptr;
}

//----------------------------------------------------------------------------
// Restore object to initial state. Release memory back to system.
void svtkPolyData::Initialize()
{
  svtkPointSet::Initialize();

  this->Cleanup();

  if (this->Information)
  {
    this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
    this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 0);
    this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);
  }
}

//----------------------------------------------------------------------------
int svtkPolyData::GetMaxCellSize()
{
  int maxCellSize = 0;

  if (this->Verts)
  {
    maxCellSize = std::max(maxCellSize, this->Verts->GetMaxCellSize());
  }

  if (this->Lines)
  {
    maxCellSize = std::max(maxCellSize, this->Lines->GetMaxCellSize());
  }

  if (this->Polys)
  {
    maxCellSize = std::max(maxCellSize, this->Polys->GetMaxCellSize());
  }

  if (this->Strips)
  {
    maxCellSize = std::max(maxCellSize, this->Strips->GetMaxCellSize());
  }

  return maxCellSize;
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateEstimate(svtkIdType numCells, svtkIdType maxCellSize)
{
  return this->AllocateExact(numCells, numCells * maxCellSize);
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateEstimate(svtkIdType numVerts, svtkIdType maxVertSize, svtkIdType numLines,
  svtkIdType maxLineSize, svtkIdType numPolys, svtkIdType maxPolySize, svtkIdType numStrips,
  svtkIdType maxStripSize)
{
  return this->AllocateExact(numVerts, maxVertSize * numVerts, numLines, maxLineSize * numLines,
    numPolys, maxPolySize * numPolys, numStrips, maxStripSize * numStrips);
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateExact(svtkIdType numCells, svtkIdType connectivitySize)
{
  return this->AllocateExact(numCells, connectivitySize, numCells, connectivitySize, numCells,
    connectivitySize, numCells, connectivitySize);
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateExact(svtkIdType numVerts, svtkIdType vertConnSize, svtkIdType numLines,
  svtkIdType lineConnSize, svtkIdType numPolys, svtkIdType polyConnSize, svtkIdType numStrips,
  svtkIdType stripConnSize)
{
  auto initCellArray = [](svtkSmartPointer<svtkCellArray>& cellArray, svtkIdType numCells,
                         svtkIdType connSize) -> bool {
    cellArray = nullptr;
    if (numCells == 0 && connSize == 0)
    {
      return true;
    }
    cellArray = svtkSmartPointer<svtkCellArray>::New();
    return cellArray->AllocateExact(numCells, connSize);
  };

  // Reset the cell table:
  this->Cells = nullptr;

  return (initCellArray(this->Verts, numVerts, vertConnSize) &&
    initCellArray(this->Lines, numLines, lineConnSize) &&
    initCellArray(this->Polys, numPolys, polyConnSize) &&
    initCellArray(this->Strips, numStrips, stripConnSize));
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateCopy(svtkPolyData* pd)
{
  return this->AllocateProportional(pd, 1.);
}

//----------------------------------------------------------------------------
bool svtkPolyData::AllocateProportional(svtkPolyData* pd, double ratio)
{

  auto* verts = pd->GetVerts();
  auto* lines = pd->GetLines();
  auto* polys = pd->GetPolys();
  auto* strips = pd->GetStrips();

  return this->AllocateExact(static_cast<svtkIdType>(verts->GetNumberOfCells() * ratio),
    static_cast<svtkIdType>(verts->GetNumberOfConnectivityIds() * ratio),
    static_cast<svtkIdType>(lines->GetNumberOfCells() * ratio),
    static_cast<svtkIdType>(lines->GetNumberOfConnectivityIds() * ratio),
    static_cast<svtkIdType>(polys->GetNumberOfCells() * ratio),
    static_cast<svtkIdType>(polys->GetNumberOfConnectivityIds() * ratio),
    static_cast<svtkIdType>(strips->GetNumberOfCells() * ratio),
    static_cast<svtkIdType>(strips->GetNumberOfConnectivityIds() * ratio));
}

//----------------------------------------------------------------------------
void svtkPolyData::DeleteCells()
{
  // if we have Links, we need to delete them (they are no longer valid)
  this->Links = nullptr;
  this->Cells = nullptr;
}

namespace
{

struct BuildCellsImpl
{
  // Typer functor must take a svtkIdType cell size and convert it into a
  // SVTKCellType. The functor must ensure that the input size and returned cell
  // type are valid for the target cell array or throw a std::runtime_error.
  template <typename CellStateT, typename SizeToTypeFunctor>
  void operator()(CellStateT& state, svtkPolyData_detail::CellMap* map, SizeToTypeFunctor&& typer)
  {
    const svtkIdType numCells = state.GetNumberOfCells();
    if (numCells == 0)
    {
      return;
    }

    if (!map->ValidateCellId(numCells - 1))
    {
      throw std::runtime_error("Cell map storage capacity exceeded.");
    }

    for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
    {
      map->InsertNextCell(cellId, typer(state.GetCellSize(cellId)));
    }
  }
};

} // end anon namespace

//----------------------------------------------------------------------------
// Create data structure that allows random access of cells.
void svtkPolyData::BuildCells()
{
  svtkCellArray* verts = this->GetVerts();
  svtkCellArray* lines = this->GetLines();
  svtkCellArray* polys = this->GetPolys();
  svtkCellArray* strips = this->GetStrips();

  // here are the number of cells we have
  const svtkIdType nVerts = verts->GetNumberOfCells();
  const svtkIdType nLines = lines->GetNumberOfCells();
  const svtkIdType nPolys = polys->GetNumberOfCells();
  const svtkIdType nStrips = strips->GetNumberOfCells();

  // pre-allocate the space we need
  const svtkIdType nCells = nVerts + nLines + nPolys + nStrips;

  this->Cells = svtkSmartPointer<CellMap>::New();
  this->Cells->SetCapacity(nCells);

  try
  {
    if (nVerts > 0)
    {
      verts->Visit(BuildCellsImpl{}, this->Cells, [](svtkIdType size) -> SVTKCellType {
        if (size < 1)
        {
          throw std::runtime_error("Invalid cell size for verts.");
        }
        return size == 1 ? SVTK_VERTEX : SVTK_POLY_VERTEX;
      });
    }

    if (nLines > 0)
    {
      lines->Visit(BuildCellsImpl{}, this->Cells, [](svtkIdType size) -> SVTKCellType {
        if (size < 2)
        {
          throw std::runtime_error("Invalid cell size for lines.");
        }
        return size == 2 ? SVTK_LINE : SVTK_POLY_LINE;
      });
    }

    if (nPolys > 0)
    {
      polys->Visit(BuildCellsImpl{}, this->Cells, [](svtkIdType size) -> SVTKCellType {
        if (size < 3)
        {
          throw std::runtime_error("Invalid cell size for polys.");
        }

        switch (size)
        {
          case 3:
            return SVTK_TRIANGLE;
          case 4:
            return SVTK_QUAD;
          default:
            return SVTK_POLYGON;
        }
      });
    }

    if (nStrips > 0)
    {
      strips->Visit(BuildCellsImpl{}, this->Cells, [](svtkIdType size) -> SVTKCellType {
        if (size < 3)
        {
          throw std::runtime_error("Invalid cell size for polys.");
        }
        return SVTK_TRIANGLE_STRIP;
      });
    }
  }
  catch (std::runtime_error& e)
  {
    this->Cells = nullptr;
    svtkErrorMacro("Error while constructing cell map: " << e.what());
  }
}
//----------------------------------------------------------------------------
void svtkPolyData::DeleteLinks()
{
  this->Links = nullptr;
}

//----------------------------------------------------------------------------
// Create upward links from points to cells that use each point. Enables
// topologically complex queries.
void svtkPolyData::BuildLinks(int initialSize)
{
  if (this->Cells == nullptr)
  {
    this->BuildCells();
  }

  this->Links = svtkSmartPointer<svtkCellLinks>::New();
  if (initialSize > 0)
  {
    this->Links->Allocate(initialSize);
  }
  else
  {
    this->Links->Allocate(this->GetNumberOfPoints());
  }

  this->Links->BuildLinks(this);
}

//----------------------------------------------------------------------------
// Copy a cells point ids into list provided. (Less efficient.)
void svtkPolyData::GetCellPoints(svtkIdType cellId, svtkIdList* ptIds)
{
  if (this->Cells == nullptr)
  {
    this->BuildCells();
  }

  svtkIdType npts;
  const svtkIdType* pts;
  this->GetCellPoints(cellId, npts, pts);

  ptIds->SetNumberOfIds(npts);
  for (svtkIdType i = 0; i < npts; ++i)
  {
    ptIds->SetId(i, pts[i]);
  }
}

//----------------------------------------------------------------------------
void svtkPolyData::GetPointCells(svtkIdType ptId, svtkIdList* cellIds)
{
  svtkIdType* cells;
  svtkIdType numCells;
  svtkIdType i;

  if (!this->Links)
  {
    this->BuildLinks();
  }
  cellIds->Reset();

  numCells = this->Links->GetNcells(ptId);
  cells = this->Links->GetCells(ptId);

  for (i = 0; i < numCells; i++)
  {
    cellIds->InsertId(i, cells[i]);
  }
}

//----------------------------------------------------------------------------
// Insert a cell of type SVTK_VERTEX, SVTK_POLY_VERTEX, SVTK_LINE, SVTK_POLY_LINE,
// SVTK_TRIANGLE, SVTK_QUAD, SVTK_POLYGON, or SVTK_TRIANGLE_STRIP.  Make sure that
// the PolyData::Allocate() function has been called first or that vertex,
// line, polygon, and triangle strip arrays have been supplied.
// Note: will also insert SVTK_PIXEL, but converts it to SVTK_QUAD.
svtkIdType svtkPolyData::InsertNextCell(int type, int npts, const svtkIdType ptsIn[])
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const svtkIdType* pts = ptsIn;
  svtkIdType pixPts[4];

  // Docs say we need to handle SVTK_PIXEL:
  if (type == SVTK_PIXEL)
  {
    // need to rearrange vertices
    pixPts[0] = pts[0];
    pixPts[1] = pts[1];
    pixPts[2] = pts[3];
    pixPts[3] = pts[2];

    type = SVTK_QUAD;
    pts = pixPts;
  }

  // Make sure the type is supported by the dataset (and thus safe to use with
  // the TaggedCellId):
  if (!CellMap::ValidateCellType(SVTKCellType(type)))
  {
    svtkErrorMacro("Invalid cell type: " << type);
    return -1;
  }

  // Insert next cell into the lookup map:
  TaggedCellId& tag = this->Cells->InsertNextCell(SVTKCellType(type));
  svtkCellArray* cells = this->GetCellArrayInternal(tag);

  // Validate and update the internal cell id:
  const svtkIdType internalCellId = cells->InsertNextCell(npts, pts);
  if (internalCellId < 0)
  {
    svtkErrorMacro("Internal error: Invalid cell id (" << internalCellId << ").");
    return -1;
  }
  if (!CellMap::ValidateCellId(internalCellId))
  {
    svtkErrorMacro("Internal cell array storage exceeded.");
    return -1;
  }
  tag.SetCellId(internalCellId);

  // Return the dataset cell id:
  return this->Cells->GetNumberOfCells() - 1;
}

//----------------------------------------------------------------------------
// Insert a cell of type SVTK_VERTEX, SVTK_POLY_VERTEX, SVTK_LINE, SVTK_POLY_LINE,
// SVTK_TRIANGLE, SVTK_QUAD, SVTK_POLYGON, or SVTK_TRIANGLE_STRIP.  Make sure that
// the PolyData::Allocate() function has been called first or that vertex,
// line, polygon, and triangle strip arrays have been supplied.
// Note: will also insert SVTK_PIXEL, but converts it to SVTK_QUAD.
svtkIdType svtkPolyData::InsertNextCell(int type, svtkIdList* pts)
{
  return this->InsertNextCell(type, static_cast<int>(pts->GetNumberOfIds()), pts->GetPointer(0));
}

//----------------------------------------------------------------------------
// Recover extra allocated memory when creating data whose initial size
// is unknown. Examples include using the InsertNextCell() method, or
// when using the CellArray::EstimateSize() method to create vertices,
// lines, polygons, or triangle strips.
void svtkPolyData::Squeeze()
{
  if (this->Verts != nullptr)
  {
    this->Verts->Squeeze();
  }
  if (this->Lines != nullptr)
  {
    this->Lines->Squeeze();
  }
  if (this->Polys != nullptr)
  {
    this->Polys->Squeeze();
  }
  if (this->Strips != nullptr)
  {
    this->Strips->Squeeze();
  }
  if (this->Cells != nullptr)
  {
    this->Cells->Squeeze();
  }

  svtkPointSet::Squeeze();
}

//----------------------------------------------------------------------------
// Begin inserting data all over again. Memory is not freed but otherwise
// objects are returned to their initial state.
void svtkPolyData::Reset()
{
  if (this->Verts != nullptr)
  {
    this->Verts->Reset();
  }
  if (this->Lines != nullptr)
  {
    this->Lines->Reset();
  }
  if (this->Polys != nullptr)
  {
    this->Polys->Reset();
  }
  if (this->Strips != nullptr)
  {
    this->Strips->Reset();
  }

  if (this->GetPoints() != nullptr)
  {
    this->GetPoints()->Reset();
  }

  // discard Links and Cells
  this->DeleteLinks();
  this->DeleteCells();
}

//----------------------------------------------------------------------------
// Reverse the order of point ids defining the cell.
void svtkPolyData::ReverseCell(svtkIdType cellId)
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);
  svtkCellArray* cells = this->GetCellArrayInternal(tag);
  cells->ReverseCellAtId(tag.GetCellId());
}

//----------------------------------------------------------------------------
// Add a point to the cell data structure (after cell pointers have been
// built). This method allocates memory for the links to the cells.  (To
// use this method, make sure points are available and BuildLinks() has been invoked.)
svtkIdType svtkPolyData::InsertNextLinkedPoint(int numLinks)
{
  return this->Links->InsertNextPoint(numLinks);
}

//----------------------------------------------------------------------------
// Add a point to the cell data structure (after cell pointers have been
// built). This method adds the point and then allocates memory for the
// links to the cells.  (To use this method, make sure points are available
// and BuildLinks() has been invoked.)
svtkIdType svtkPolyData::InsertNextLinkedPoint(double x[3], int numLinks)
{
  this->Links->InsertNextPoint(numLinks);
  return this->Points->InsertNextPoint(x);
}

//----------------------------------------------------------------------------
// Add a new cell to the cell data structure (after cell pointers have been
// built). This method adds the cell and then updates the links from the points
// to the cells. (Memory is allocated as necessary.)
svtkIdType svtkPolyData::InsertNextLinkedCell(int type, int npts, const svtkIdType pts[])
{
  svtkIdType i, id;

  id = this->InsertNextCell(type, npts, pts);

  for (i = 0; i < npts; i++)
  {
    this->Links->ResizeCellList(pts[i], 1);
    this->Links->AddCellReference(id, pts[i]);
  }

  return id;
}

//----------------------------------------------------------------------------
// Remove a reference to a cell in a particular point's link list. You may also
// consider using RemoveCellReference() to remove the references from all the
// cell's points to the cell. This operator does not reallocate memory; use the
// operator ResizeCellList() to do this if necessary.
void svtkPolyData::RemoveReferenceToCell(svtkIdType ptId, svtkIdType cellId)
{
  this->Links->RemoveCellReference(cellId, ptId);
}

//----------------------------------------------------------------------------
// Add a reference to a cell in a particular point's link list. (You may also
// consider using AddCellReference() to add the references from all the
// cell's points to the cell.) This operator does not realloc memory; use the
// operator ResizeCellList() to do this if necessary.
void svtkPolyData::AddReferenceToCell(svtkIdType ptId, svtkIdType cellId)
{
  this->Links->AddCellReference(cellId, ptId);
}

//----------------------------------------------------------------------------
void svtkPolyData::ReplaceCell(svtkIdType cellId, svtkIdList* ids)
{
  this->ReplaceCell(cellId, static_cast<int>(ids->GetNumberOfIds()), ids->GetPointer(0));
}

//----------------------------------------------------------------------------
// Replace the points defining cell "cellId" with a new set of points. This
// operator is (typically) used when links from points to cells have not been
// built (i.e., BuildLinks() has not been executed). Use the operator
// ReplaceLinkedCell() to replace a cell when cell structure has been built.
void svtkPolyData::ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[])
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);
  svtkCellArray* cells = this->GetCellArrayInternal(tag);
  cells->ReplaceCellAtId(tag.GetCellId(), npts, pts);
}

//----------------------------------------------------------------------------
// Replace one cell with another in cell structure. This operator updates the
// connectivity list and the point's link list. It does not delete references
// to the old cell in the point's link list. Use the operator
// RemoveCellReference() to delete all references from points to (old) cell.
// You may also want to consider using the operator ResizeCellList() if the
// link list is changing size.
void svtkPolyData::ReplaceLinkedCell(svtkIdType cellId, int npts, const svtkIdType pts[])
{
  this->ReplaceCell(cellId, npts, pts);
  for (int i = 0; i < npts; i++)
  {
    this->Links->InsertNextCellReference(pts[i], cellId);
  }
}

//----------------------------------------------------------------------------
// Get the neighbors at an edge. More efficient than the general
// GetCellNeighbors(). Assumes links have been built (with BuildLinks()),
// and looks specifically for edge neighbors.
void svtkPolyData::GetCellEdgeNeighbors(
  svtkIdType cellId, svtkIdType p1, svtkIdType p2, svtkIdList* cellIds)
{
  cellIds->Reset();

  const svtkCellLinks::Link& link1(this->Links->GetLink(p1));
  const svtkCellLinks::Link& link2(this->Links->GetLink(p2));

  const svtkIdType* cells1 = link1.cells;
  const svtkIdType* cells1End = cells1 + link1.ncells;

  const svtkIdType* cells2 = link2.cells;
  const svtkIdType* cells2End = cells2 + link2.ncells;

  while (cells1 != cells1End)
  {
    if (*cells1 != cellId)
    {
      const svtkIdType* cells2Cur(cells2);
      while (cells2Cur != cells2End)
      {
        if (*cells1 == *cells2Cur)
        {
          cellIds->InsertNextId(*cells1);
          break;
        }
        ++cells2Cur;
      }
    }
    ++cells1;
  }
}

//----------------------------------------------------------------------------
void svtkPolyData::GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds)
{
  svtkIdType i, j, numPts, cellNum;
  int allFound, oneFound;

  if (!this->Links)
  {
    this->BuildLinks();
  }

  cellIds->Reset();

  // load list with candidate cells, remove current cell
  svtkIdType ptId = ptIds->GetId(0);
  int numPrime = this->Links->GetNcells(ptId);
  svtkIdType* primeCells = this->Links->GetCells(ptId);
  numPts = ptIds->GetNumberOfIds();

  // for each potential cell
  for (cellNum = 0; cellNum < numPrime; cellNum++)
  {
    // ignore the original cell
    if (primeCells[cellNum] != cellId)
    {
      // are all the remaining points in the cell ?
      for (allFound = 1, i = 1; i < numPts && allFound; i++)
      {
        ptId = ptIds->GetId(i);
        int numCurrent = this->Links->GetNcells(ptId);
        svtkIdType* currentCells = this->Links->GetCells(ptId);
        oneFound = 0;
        for (j = 0; j < numCurrent; j++)
        {
          if (primeCells[cellNum] == currentCells[j])
          {
            oneFound = 1;
            break;
          }
        }
        if (!oneFound)
        {
          allFound = 0;
        }
      }
      if (allFound)
      {
        cellIds->InsertNextId(primeCells[cellNum]);
      }
    }
  }
}

//----------------------------------------------------------------------------
int svtkPolyData::IsEdge(svtkIdType p1, svtkIdType p2)
{
  svtkIdType ncells;
  svtkIdType cellType;
  svtkIdType npts;
  svtkIdType i, j;
  svtkIdType* cells;
  const svtkIdType* pts;

  svtkIdType nbPoints = this->GetNumberOfPoints();
  if (p1 >= nbPoints || p2 >= nbPoints)
  {
    return 0;
  }

  this->GetPointCells(p1, ncells, cells);
  for (i = 0; i < ncells; i++)
  {
    cellType = this->GetCellType(cells[i]);
    switch (cellType)
    {
      case SVTK_EMPTY_CELL:
      case SVTK_VERTEX:
      case SVTK_POLY_VERTEX:
      case SVTK_LINE:
      case SVTK_POLY_LINE:
        break;
      case SVTK_TRIANGLE:
        if (this->IsPointUsedByCell(p2, cells[i]))
        {
          return 1;
        }
        break;
      case SVTK_QUAD:
        this->GetCellPoints(cells[i], npts, pts);
        for (j = 0; j < npts - 1; j++)
        {
          if (((pts[j] == p1) && (pts[j + 1] == p2)) || ((pts[j] == p2) && (pts[j + 1] == p1)))
          {
            return 1;
          }
        }
        if (((pts[0] == p1) && (pts[npts - 1] == p2)) || ((pts[0] == p2) && (pts[npts - 1] == p1)))
        {
          return 1;
        }
        break;
      case SVTK_TRIANGLE_STRIP:
        this->GetCellPoints(cells[i], npts, pts);
        for (j = 0; j < npts - 2; j++)
        {
          if ((((pts[j] == p1) && (pts[j + 1] == p2)) || ((pts[j] == p2) && (pts[j + 1] == p1))) ||
            (((pts[j] == p1) && (pts[j + 2] == p2)) || ((pts[j] == p2) && (pts[j + 2] == p1))))
          {
            return 1;
          }
        }
        if (((pts[npts - 2] == p1) && (pts[npts - 1] == p2)) ||
          ((pts[npts - 2] == p2) && (pts[npts - 1] == p1)))
        {
          return 1;
        }
        break;
      default:
        this->GetCellPoints(cells[i], npts, pts);
        for (j = 0; j < npts; j++)
        {
          if (p1 == pts[j])
          {
            if ((pts[(j - 1 + npts) % npts] == p2) || (pts[(j + 1) % npts] == p2))
            {
              return 1;
            }
          }
        }
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
unsigned long svtkPolyData::GetActualMemorySize()
{
  unsigned long size = this->svtkPointSet::GetActualMemorySize();
  if (this->Verts)
  {
    size += this->Verts->GetActualMemorySize();
  }
  if (this->Lines)
  {
    size += this->Lines->GetActualMemorySize();
  }
  if (this->Polys)
  {
    size += this->Polys->GetActualMemorySize();
  }
  if (this->Strips)
  {
    size += this->Strips->GetActualMemorySize();
  }
  if (this->Cells)
  {
    size += this->Cells->GetActualMemorySize();
  }
  if (this->Links)
  {
    size += this->Links->GetActualMemorySize();
  }
  return size;
}

//----------------------------------------------------------------------------
void svtkPolyData::ShallowCopy(svtkDataObject* dataObject)
{
  svtkPolyData* polyData = svtkPolyData::SafeDownCast(dataObject);
  if (this == polyData)
    return;

  if (polyData != nullptr)
  {
    this->SetVerts(polyData->GetVerts());
    this->SetLines(polyData->GetLines());
    this->SetPolys(polyData->GetPolys());
    this->SetStrips(polyData->GetStrips());

    // I do not know if this is correct but.
    // Me either! But it's been 20 years so I think it'll be ok.
    this->Cells = polyData->Cells;
    this->Links = polyData->Links;
  }

  // Do superclass
  this->svtkPointSet::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkPolyData::DeepCopy(svtkDataObject* dataObject)
{
  // Do superclass
  // We have to do this BEFORE we call BuildLinks, else there are no points
  // to build the links on (the parent DeepCopy copies the points)
  this->svtkPointSet::DeepCopy(dataObject);

  svtkPolyData* polyData = svtkPolyData::SafeDownCast(dataObject);

  if (polyData != nullptr)
  {
    this->Verts = svtkSmartPointer<svtkCellArray>::New();
    this->Verts->DeepCopy(polyData->GetVerts());

    this->Lines = svtkSmartPointer<svtkCellArray>::New();
    this->Lines->DeepCopy(polyData->GetLines());

    this->Polys = svtkSmartPointer<svtkCellArray>::New();
    this->Polys->DeepCopy(polyData->GetPolys());

    this->Strips = svtkSmartPointer<svtkCellArray>::New();
    this->Strips->DeepCopy(polyData->GetStrips());

    // only instantiate this if the input dataset has one
    if (polyData->Cells)
    {
      this->Cells = svtkSmartPointer<CellMap>::New();
      this->Cells->DeepCopy(polyData->Cells);
    }
    else
    {
      this->Cells = nullptr;
    }

    if (this->Links)
    {
      this->Links = nullptr;
    }
    if (polyData->Links)
    {
      this->BuildLinks();
    }
  }
}

//----------------------------------------------------------------------------
void svtkPolyData::RemoveGhostCells()
{
  // Get a pointer to the cell ghost level array.
  svtkUnsignedCharArray* temp = this->GetCellGhostArray();
  if (temp == nullptr)
  {
    svtkDebugMacro("Could not find cell ghost array.");
    return;
  }
  if (temp->GetNumberOfComponents() != 1 || temp->GetNumberOfTuples() < this->GetNumberOfCells())
  {
    svtkErrorMacro("Poorly formed ghost array.");
    return;
  }
  unsigned char* cellGhosts = temp->GetPointer(0);

  svtkIdType numCells = this->GetNumberOfCells();

  svtkIntArray* types = svtkIntArray::New();
  types->SetNumberOfValues(numCells);

  for (svtkIdType i = 0; i < numCells; i++)
  {
    types->SetValue(i, this->GetCellType(i));
  }

  this->DeleteCells();

  // we have to make new copies of Verts, Lines, Polys
  // and Strips since they may be shared with other polydata
  svtkSmartPointer<svtkCellArray> verts;
  if (this->Verts)
  {
    verts = this->Verts;
    verts->InitTraversal();
    this->Verts = svtkSmartPointer<svtkCellArray>::New();
  }

  svtkSmartPointer<svtkCellArray> lines;
  if (this->Lines)
  {
    lines = this->Lines;
    lines->InitTraversal();
    this->Lines = svtkSmartPointer<svtkCellArray>::New();
  }

  svtkSmartPointer<svtkCellArray> polys;
  if (this->Polys)
  {
    polys = this->Polys;
    polys->InitTraversal();
    this->Polys = svtkSmartPointer<svtkCellArray>::New();
  }

  svtkSmartPointer<svtkCellArray> strips;
  if (this->Strips)
  {
    strips = this->Strips;
    strips->InitTraversal();
    this->Strips = svtkSmartPointer<svtkCellArray>::New();
  }

  svtkCellData* newCellData = svtkCellData::New();
  // ensure that all attributes are copied over, including global ids.
  newCellData->CopyAllOn(svtkDataSetAttributes::COPYTUPLE);
  newCellData->CopyAllocate(this->CellData, numCells);

  const svtkIdType* pts;
  svtkIdType n;

  svtkIdType cellId;

  for (svtkIdType i = 0; i < numCells; i++)
  {
    int type = types->GetValue(i);

    if (type == SVTK_VERTEX || type == SVTK_POLY_VERTEX)
    {
      verts->GetNextCell(n, pts);

      if (!(cellGhosts[i] & svtkDataSetAttributes::DUPLICATECELL))
      {
        cellId = this->InsertNextCell(type, n, pts);
        newCellData->CopyData(this->CellData, i, cellId);
      }
    }
    else if (type == SVTK_LINE || type == SVTK_POLY_LINE)
    {
      lines->GetNextCell(n, pts);

      if (!(cellGhosts[i] & svtkDataSetAttributes::DUPLICATECELL))
      {
        cellId = this->InsertNextCell(type, n, pts);
        newCellData->CopyData(this->CellData, i, cellId);
      }
    }
    else if (type == SVTK_POLYGON || type == SVTK_TRIANGLE || type == SVTK_QUAD)
    {
      polys->GetNextCell(n, pts);

      if (!(cellGhosts[i] & svtkDataSetAttributes::DUPLICATECELL))
      {
        cellId = this->InsertNextCell(type, n, pts);
        newCellData->CopyData(this->CellData, i, cellId);
      }
    }
    else if (type == SVTK_TRIANGLE_STRIP)
    {
      strips->GetNextCell(n, pts);

      if (!(cellGhosts[i] & svtkDataSetAttributes::DUPLICATECELL))
      {
        cellId = this->InsertNextCell(type, n, pts);
        newCellData->CopyData(this->CellData, i, cellId);
      }
    }
  }

  newCellData->Squeeze();

  this->CellData->ShallowCopy(newCellData);
  newCellData->Delete();

  types->Delete();

  // If there are no more ghost levels, then remove all arrays.
  this->CellData->RemoveArray(svtkDataSetAttributes::GhostArrayName());

  this->Squeeze();
}

//----------------------------------------------------------------------------
void svtkPolyData::RemoveDeletedCells()
{
  if (!this->Cells)
  {
    return;
  }

  svtkNew<svtkPolyData> oldData;
  oldData->ShallowCopy(this);
  this->DeleteCells();

  if (this->Verts)
  {
    this->Verts = svtkSmartPointer<svtkCellArray>::New();
  }
  if (this->Lines)
  {
    this->Lines = svtkSmartPointer<svtkCellArray>::New();
  }
  if (this->Polys)
  {
    this->Polys = svtkSmartPointer<svtkCellArray>::New();
  }
  if (this->Strips)
  {
    this->Strips = svtkSmartPointer<svtkCellArray>::New();
  }

  this->CellData->CopyAllocate(oldData->GetCellData());

  const svtkIdType numCells = oldData->GetNumberOfCells();
  svtkCell* cell;
  svtkIdType cellId;
  svtkIdList* pointIds;
  int type;
  for (svtkIdType i = 0; i < numCells; i++)
  {
    type = oldData->GetCellType(i);

    if (type != SVTK_EMPTY_CELL)
    {
      cell = oldData->GetCell(i);
      pointIds = cell->GetPointIds();
      cellId = this->InsertNextCell(type, pointIds);
      this->CellData->CopyData(oldData->GetCellData(), i, cellId);
    }
  }

  this->CellData->Squeeze();
}

//----------------------------------------------------------------------------
svtkPolyData* svtkPolyData::GetData(svtkInformation* info)
{
  return info ? svtkPolyData::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkPolyData* svtkPolyData::GetData(svtkInformationVector* v, int i)
{
  return svtkPolyData::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkPolyData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number Of Vertices: " << this->GetNumberOfVerts() << "\n";
  os << indent << "Number Of Lines: " << this->GetNumberOfLines() << "\n";
  os << indent << "Number Of Polygons: " << this->GetNumberOfPolys() << "\n";
  os << indent << "Number Of Triangle Strips: " << this->GetNumberOfStrips() << "\n";

  os << indent << "Number Of Pieces: " << this->GetNumberOfPieces() << endl;
  os << indent << "Piece: " << this->GetPiece() << endl;
  os << indent << "Ghost Level: " << this->GetGhostLevel() << endl;
}

//----------------------------------------------------------------------------
int svtkPolyData::GetScalarFieldCriticalIndex(svtkIdType pointId, svtkDataArray* scalarField)
{
  /*
   * implements scalar field critical point classification for manifold
   * 2D meshes.
   */

  /*
   * returned value:
   *   -4: no such field
   *   -3: attribute check failed
   *   -2: non 2-manifold star
   *   -1: regular point
   *   0: minimum
   *   1: saddle
   *   2: maximum
   */

  bool is_min = true, is_max = true;
  svtkIdList *starTriangleList = svtkIdList::New(), *lowerLinkPointList = svtkIdList::New(),
            *upperLinkPointList = svtkIdList::New(), *pointList = nullptr;
  double pointFieldValue = scalarField->GetComponent(pointId, 0);

  if (this->GetNumberOfPoints() != scalarField->GetSize())
  {
    return svtkPolyData::ERR_INCORRECT_FIELD;
  }

  /* make sure the connectivity is built */
  if (!this->Links)
  {
    this->BuildLinks();
  }

  /* build the lower and upper links */
  this->GetPointCells(pointId, starTriangleList);
  int starNb = starTriangleList->GetNumberOfIds();
  for (int i = 0; i < starNb; i++)
  {
    svtkCell* c = this->GetCell(starTriangleList->GetId(i));
    pointList = c->GetPointIds();
    int pointNb = pointList->GetNumberOfIds();
    if (pointNb != 3)
    {
      starTriangleList->Delete();
      lowerLinkPointList->Delete();
      upperLinkPointList->Delete();
      return svtkPolyData::ERR_NON_MANIFOLD_STAR;
    }

    for (int j = 0; j < pointNb; j++)
    {
      svtkIdType currentPointId = pointList->GetId(j);

      /* quick check for extrema */
      double neighborFieldValue = scalarField->GetComponent(currentPointId, 0);
      if ((currentPointId != pointId) && (neighborFieldValue == pointFieldValue))
      {
        /* simulation of simplicity (Edelsbrunner et al. ACM ToG 1990) */
        if (currentPointId > pointId)
        {
          is_max = false;
          upperLinkPointList->InsertUniqueId(currentPointId);
        }
        if (currentPointId < pointId)
        {
          is_min = false;
          lowerLinkPointList->InsertUniqueId(currentPointId);
        }
      }
      else
      {
        if (neighborFieldValue > pointFieldValue)
        {
          is_max = false;
          upperLinkPointList->InsertUniqueId(currentPointId);
        }
        if (neighborFieldValue < pointFieldValue)
        {
          is_min = false;
          lowerLinkPointList->InsertUniqueId(currentPointId);
        }
      }
    }
  }

  if ((is_max) || (is_min))
  {
    starTriangleList->Delete();
    lowerLinkPointList->Delete();
    upperLinkPointList->Delete();
    if (is_max)
      return svtkPolyData::MAXIMUM;
    if (is_min)
      return svtkPolyData::MINIMUM;
  }

  /*
   * is the vertex really regular?
   * (lower and upper links are BOTH simply connected)
   */
  int visitedPointNb = 0, stackBottom = 0, lowerLinkPointNb = lowerLinkPointList->GetNumberOfIds(),
      upperLinkPointNb = upperLinkPointList->GetNumberOfIds();

  /* first, check lower link's simply connectedness */
  svtkIdList* stack = svtkIdList::New();
  stack->InsertUniqueId(lowerLinkPointList->GetId(0));
  do
  {
    svtkIdType currentPointId = stack->GetId(stackBottom);
    svtkIdType nextPointId = -1;

    stackBottom++;
    svtkIdList* triangleList = svtkIdList::New();
    this->GetPointCells(currentPointId, triangleList);
    int triangleNb = triangleList->GetNumberOfIds();

    for (int i = 0; i < triangleNb; i++)
    {
      svtkCell* c = this->GetCell(triangleList->GetId(i));
      pointList = c->GetPointIds();
      int pointNb = pointList->GetNumberOfIds();

      if (pointList->IsId(pointId) >= 0)
      {
        // those two triangles are in the star of pointId
        int j = 0;
        do
        {
          nextPointId = pointList->GetId(j);
          j++;
        } while (((nextPointId == pointId) || (nextPointId == currentPointId)) && (j < pointNb));
      }

      if (lowerLinkPointList->IsId(nextPointId) >= 0)
      {
        stack->InsertUniqueId(nextPointId);
      }
    }

    triangleList->Delete();
    visitedPointNb++;
  } while (stackBottom < stack->GetNumberOfIds());

  if (visitedPointNb != lowerLinkPointNb)
  {
    // the lower link is not simply connected, then it's a saddle
    stack->Delete();
    starTriangleList->Delete();
    lowerLinkPointList->Delete();
    upperLinkPointList->Delete();
    return svtkPolyData::SADDLE;
  }

  /*
   * then, check upper link's simply connectedness.
   * BOTH need to be checked if the 2-manifold has boundary components.
   */
  stackBottom = 0;
  visitedPointNb = 0;
  stack->Delete();
  stack = svtkIdList::New();
  stack->InsertUniqueId(upperLinkPointList->GetId(0));
  do
  {
    svtkIdType currentPointId = stack->GetId(stackBottom);
    svtkIdType nextPointId = -1;
    stackBottom++;
    svtkIdList* triangleList = svtkIdList::New();
    this->GetPointCells(currentPointId, triangleList);
    int triangleNb = triangleList->GetNumberOfIds();

    for (int i = 0; i < triangleNb; i++)
    {
      svtkCell* c = this->GetCell(triangleList->GetId(i));
      pointList = c->GetPointIds();
      int pointNb = pointList->GetNumberOfIds();

      if (pointList->IsId(pointId) >= 0)
      {
        // those two triangles are in the star of pointId
        int j = 0;
        do
        {
          nextPointId = pointList->GetId(j);
          j++;
        } while (((nextPointId == pointId) || (nextPointId == currentPointId)) && (j < pointNb));
      }

      if (upperLinkPointList->IsId(nextPointId) >= 0)
      {
        stack->InsertUniqueId(nextPointId);
      }
    }

    triangleList->Delete();
    visitedPointNb++;
  } while (stackBottom < stack->GetNumberOfIds());

  if (visitedPointNb != upperLinkPointNb)
  {
    // the upper link is not simply connected, then it's a saddle
    stack->Delete();
    starTriangleList->Delete();
    lowerLinkPointList->Delete();
    upperLinkPointList->Delete();
    return svtkPolyData::SADDLE;
  }

  /* else it's necessarily a regular point (only 4 cases in 2D)*/
  stack->Delete();
  starTriangleList->Delete();
  lowerLinkPointList->Delete();
  upperLinkPointList->Delete();
  return svtkPolyData::REGULAR_POINT;
}

//----------------------------------------------------------------------------
int svtkPolyData::GetScalarFieldCriticalIndex(svtkIdType pointId, const char* fieldName)
{
  /*
   * returned value:
   *   -4: no such field
   *   -3: attribute check failed
   *   -2: non 2-manifold star
   *   -1: regular point
   *   0: minimum
   *   1: saddle
   *   2: maximum
   */

  int fieldId = 0;

  svtkPointData* pointData = this->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(fieldName, fieldId);

  if (!scalarField)
    return svtkPolyData::ERR_NO_SUCH_FIELD;

  return this->GetScalarFieldCriticalIndex(pointId, scalarField);
}

//----------------------------------------------------------------------------
int svtkPolyData::GetScalarFieldCriticalIndex(svtkIdType pointId, int fieldId)
{
  /*
   * returned value:
   *   -4: no such field
   *   -3: attribute check failed
   *   -2: non 2-manifold star
   *   -1: regular point
   *   0: minimum
   *   1: saddle
   *   2: maximum
   */

  svtkPointData* pointData = this->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(fieldId);

  if (!scalarField)
    return svtkPolyData::ERR_NO_SUCH_FIELD;

  return this->GetScalarFieldCriticalIndex(pointId, scalarField);
}

//----------------------------------------------------------------------------
svtkMTimeType svtkPolyData::GetMeshMTime()
{
  svtkMTimeType time = this->Points ? this->Points->GetMTime() : 0;
  if (this->Verts)
  {
    time = svtkMath::Max(this->Verts->GetMTime(), time);
  }
  if (this->Lines)
  {
    time = svtkMath::Max(this->Lines->GetMTime(), time);
  }
  if (this->Polys)
  {
    time = svtkMath::Max(this->Polys->GetMTime(), time);
  }
  if (this->Strips)
  {
    time = svtkMath::Max(this->Strips->GetMTime(), time);
  }
  return time;
}

//----------------------------------------------------------------------------
svtkMTimeType svtkPolyData::GetMTime()
{
  svtkMTimeType time = this->Superclass::GetMTime();
  if (this->Verts)
  {
    time = svtkMath::Max(this->Verts->GetMTime(), time);
  }
  if (this->Lines)
  {
    time = svtkMath::Max(this->Lines->GetMTime(), time);
  }
  if (this->Polys)
  {
    time = svtkMath::Max(this->Polys->GetMTime(), time);
  }
  if (this->Strips)
  {
    time = svtkMath::Max(this->Strips->GetMTime(), time);
  }
  return time;
}
