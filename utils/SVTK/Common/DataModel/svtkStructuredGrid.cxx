/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStructuredGrid.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkStructuredGrid.h"

#include "svtkCellData.h"
#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkHexahedron.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkQuad.h"
#include "svtkUnsignedCharArray.h"
#include "svtkVertex.h"

svtkStandardNewMacro(svtkStructuredGrid);

unsigned char svtkStructuredGrid::MASKED_CELL_VALUE =
  svtkDataSetAttributes::HIDDENCELL | svtkDataSetAttributes::REFINEDCELL;

#define svtkAdjustBoundsMacro(A, B)                                                                 \
  A[0] = (B[0] < A[0] ? B[0] : A[0]);                                                              \
  A[1] = (B[0] > A[1] ? B[0] : A[1]);                                                              \
  A[2] = (B[1] < A[2] ? B[1] : A[2]);                                                              \
  A[3] = (B[1] > A[3] ? B[1] : A[3]);                                                              \
  A[4] = (B[2] < A[4] ? B[2] : A[4]);                                                              \
  A[5] = (B[2] > A[5] ? B[2] : A[5])

svtkStructuredGrid::svtkStructuredGrid()
{
  this->Vertex = svtkVertex::New();
  this->Line = svtkLine::New();
  this->Quad = svtkQuad::New();
  this->Hexahedron = svtkHexahedron::New();
  this->EmptyCell = svtkEmptyCell::New();

  this->Dimensions[0] = 0;
  this->Dimensions[1] = 0;
  this->Dimensions[2] = 0;
  this->DataDescription = SVTK_EMPTY;

  int extent[6] = { 0, -1, 0, -1, 0, -1 };
  memcpy(this->Extent, extent, 6 * sizeof(int));

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_3D_EXTENT);
  this->Information->Set(svtkDataObject::DATA_EXTENT(), this->Extent, 6);
}

//----------------------------------------------------------------------------
svtkStructuredGrid::~svtkStructuredGrid()
{
  this->Vertex->Delete();
  this->Line->Delete();
  this->Quad->Delete();
  this->Hexahedron->Delete();
  this->EmptyCell->Delete();
}

//----------------------------------------------------------------------------
// Copy the geometric and topological structure of an input structured grid.
void svtkStructuredGrid::CopyStructure(svtkDataSet* ds)
{
  svtkStructuredGrid* sg = static_cast<svtkStructuredGrid*>(ds);
  svtkPointSet::CopyStructure(ds);
  int i;

  for (i = 0; i < 3; i++)
  {
    this->Dimensions[i] = sg->Dimensions[i];
  }
  this->SetExtent(sg->GetExtent());

  this->DataDescription = sg->DataDescription;

  if (ds->HasAnyBlankPoints())
  {
    // there is blanking
    this->GetPointData()->AddArray(ds->GetPointGhostArray());
    this->PointGhostArray = nullptr;
  }
  if (ds->HasAnyBlankCells())
  {
    // there is blanking
    this->GetCellData()->AddArray(ds->GetCellGhostArray());
    this->CellGhostArray = nullptr;
  }
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::Initialize()
{
  this->Superclass::Initialize();

  if (this->Information)
  {
    this->SetDimensions(0, 0, 0);
  }
}

//----------------------------------------------------------------------------
int svtkStructuredGrid::GetCellType(svtkIdType cellId)
{
  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return SVTK_EMPTY_CELL;
  }

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return SVTK_EMPTY_CELL;

    case SVTK_SINGLE_POINT:
      return SVTK_VERTEX;

    case SVTK_X_LINE:
    case SVTK_Y_LINE:
    case SVTK_Z_LINE:
      return SVTK_LINE;

    case SVTK_XY_PLANE:
    case SVTK_YZ_PLANE:
    case SVTK_XZ_PLANE:
      return SVTK_QUAD;

    case SVTK_XYZ_GRID:
      return SVTK_HEXAHEDRON;

    default:
      svtkErrorMacro(<< "Bad data description!");
      return SVTK_EMPTY_CELL;
  }
}

//----------------------------------------------------------------------------
svtkCell* svtkStructuredGrid::GetCell(svtkIdType cellId)
{
  svtkCell* cell = nullptr;
  svtkIdType idx;
  int i, j, k;
  int d01, offset1, offset2;

  // Make sure data is defined
  if (!this->Points)
  {
    svtkErrorMacro(<< "No data");
    return nullptr;
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return this->EmptyCell;
  }

  // Update dimensions
  this->GetDimensions();

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return this->EmptyCell;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell = this->Vertex;
      cell->PointIds->SetId(0, 0);
      break;

    case SVTK_X_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Y_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Z_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_XY_PLANE:
      cell = this->Quad;
      i = cellId % (this->Dimensions[0] - 1);
      j = cellId / (this->Dimensions[0] - 1);
      idx = i + j * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_YZ_PLANE:
      cell = this->Quad;
      j = cellId % (this->Dimensions[1] - 1);
      k = cellId / (this->Dimensions[1] - 1);
      idx = j + k * this->Dimensions[1];
      offset1 = 1;
      offset2 = this->Dimensions[1];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XZ_PLANE:
      cell = this->Quad;
      i = cellId % (this->Dimensions[0] - 1);
      k = cellId / (this->Dimensions[0] - 1);
      idx = i + k * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XYZ_GRID:
      cell = this->Hexahedron;
      d01 = this->Dimensions[0] * this->Dimensions[1];
      i = cellId % (this->Dimensions[0] - 1);
      j = (cellId / (this->Dimensions[0] - 1)) % (this->Dimensions[1] - 1);
      k = cellId / ((this->Dimensions[0] - 1) * (this->Dimensions[1] - 1));
      idx = i + j * this->Dimensions[0] + k * d01;
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      idx += d01;
      cell->PointIds->SetId(4, idx);
      cell->PointIds->SetId(5, idx + offset1);
      cell->PointIds->SetId(6, idx + offset1 + offset2);
      cell->PointIds->SetId(7, idx + offset2);
      break;

    default:
      svtkErrorMacro(<< "Invalid DataDescription.");
      return nullptr;
  }

  // Extract point coordinates and point ids. NOTE: the ordering of the svtkQuad
  // and svtkHexahedron cells are tricky.
  int NumberOfIds = cell->PointIds->GetNumberOfIds();
  for (i = 0; i < NumberOfIds; i++)
  {
    idx = cell->PointIds->GetId(i);
    cell->Points->SetPoint(i, this->Points->GetPoint(idx));
  }
  return cell;
}

//----------------------------------------------------------------------------
svtkCell* svtkStructuredGrid::GetCell(int i, int j, int k)
{
  svtkIdType cellId = i + (j + (k * (this->Dimensions[1] - 1))) * (this->Dimensions[0] - 1);
  svtkCell* cell = nullptr;
  svtkIdType idx;
  int d01, offset1, offset2;

  // Make sure data is defined
  if (!this->Points)
  {
    svtkErrorMacro(<< "No data");
    return nullptr;
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return this->EmptyCell;
  }

  // Update dimensions
  this->GetDimensions();

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return this->EmptyCell;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell = this->Vertex;
      cell->PointIds->SetId(0, 0);
      break;

    case SVTK_X_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Y_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Z_LINE:
      cell = this->Line;
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_XY_PLANE:
      cell = this->Quad;
      idx = i + j * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_YZ_PLANE:
      cell = this->Quad;
      idx = j + k * this->Dimensions[1];
      offset1 = 1;
      offset2 = this->Dimensions[1];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XZ_PLANE:
      cell = this->Quad;
      idx = i + k * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XYZ_GRID:
      cell = this->Hexahedron;
      d01 = this->Dimensions[0] * this->Dimensions[1];
      idx = i + j * this->Dimensions[0] + k * d01;
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      idx += d01;
      cell->PointIds->SetId(4, idx);
      cell->PointIds->SetId(5, idx + offset1);
      cell->PointIds->SetId(6, idx + offset1 + offset2);
      cell->PointIds->SetId(7, idx + offset2);
      break;

    default:
      svtkErrorMacro(<< "Invalid DataDescription.");
      return nullptr;
  }

  // Extract point coordinates and point ids. NOTE: the ordering of the svtkQuad
  // and svtkHexahedron cells are tricky.
  int NumberOfIds = cell->PointIds->GetNumberOfIds();
  for (i = 0; i < NumberOfIds; i++)
  {
    idx = cell->PointIds->GetId(i);
    cell->Points->SetPoint(i, this->Points->GetPoint(idx));
  }
  return cell;
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::GetCell(svtkIdType cellId, svtkGenericCell* cell)
{
  svtkIdType idx;
  int i, j, k;
  int d01, offset1, offset2;
  double x[3];

  // Make sure data is defined
  if (!this->Points)
  {
    svtkErrorMacro(<< "No data");
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    cell->SetCellTypeToEmptyCell();
    return;
  }

  // Update dimensions
  this->GetDimensions();

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      cell->SetCellTypeToEmptyCell();
      return;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell->SetCellTypeToVertex();
      cell->PointIds->SetId(0, 0);
      break;

    case SVTK_X_LINE:
      cell->SetCellTypeToLine();
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Y_LINE:
      cell->SetCellTypeToLine();
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_Z_LINE:
      cell->SetCellTypeToLine();
      cell->PointIds->SetId(0, cellId);
      cell->PointIds->SetId(1, cellId + 1);
      break;

    case SVTK_XY_PLANE:
      cell->SetCellTypeToQuad();
      i = cellId % (this->Dimensions[0] - 1);
      j = cellId / (this->Dimensions[0] - 1);
      idx = i + j * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_YZ_PLANE:
      cell->SetCellTypeToQuad();
      j = cellId % (this->Dimensions[1] - 1);
      k = cellId / (this->Dimensions[1] - 1);
      idx = j + k * this->Dimensions[1];
      offset1 = 1;
      offset2 = this->Dimensions[1];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XZ_PLANE:
      cell->SetCellTypeToQuad();
      i = cellId % (this->Dimensions[0] - 1);
      k = cellId / (this->Dimensions[0] - 1);
      idx = i + k * this->Dimensions[0];
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      break;

    case SVTK_XYZ_GRID:
      cell->SetCellTypeToHexahedron();
      d01 = this->Dimensions[0] * this->Dimensions[1];
      i = cellId % (this->Dimensions[0] - 1);
      j = (cellId / (this->Dimensions[0] - 1)) % (this->Dimensions[1] - 1);
      k = cellId / ((this->Dimensions[0] - 1) * (this->Dimensions[1] - 1));
      idx = i + j * this->Dimensions[0] + k * d01;
      offset1 = 1;
      offset2 = this->Dimensions[0];

      cell->PointIds->SetId(0, idx);
      cell->PointIds->SetId(1, idx + offset1);
      cell->PointIds->SetId(2, idx + offset1 + offset2);
      cell->PointIds->SetId(3, idx + offset2);
      idx += d01;
      cell->PointIds->SetId(4, idx);
      cell->PointIds->SetId(5, idx + offset1);
      cell->PointIds->SetId(6, idx + offset1 + offset2);
      cell->PointIds->SetId(7, idx + offset2);
      break;
  }

  // Extract point coordinates and point ids. NOTE: the ordering of the svtkQuad
  // and svtkHexahedron cells are tricky.
  int NumberOfIds = cell->PointIds->GetNumberOfIds();
  for (i = 0; i < NumberOfIds; i++)
  {
    idx = cell->PointIds->GetId(i);
    this->Points->GetPoint(idx, x);
    cell->Points->SetPoint(i, x);
  }
}

//----------------------------------------------------------------------------
// Fast implementation of GetCellBounds().  Bounds are calculated without
// constructing a cell.
void svtkStructuredGrid::GetCellBounds(svtkIdType cellId, double bounds[6])
{
  svtkIdType idx = 0;
  int i, j, k;
  svtkIdType d01;
  int offset1 = 0;
  int offset2 = 0;
  double x[3];

  // Make sure data is defined
  if (!this->Points)
  {
    svtkErrorMacro(<< "No data");
    return;
  }

  svtkMath::UninitializeBounds(bounds);

  // Update dimensions
  this->GetDimensions();

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return;
    case SVTK_SINGLE_POINT: // cellId can only be = 0
      this->Points->GetPoint(0, x);
      bounds[0] = bounds[1] = x[0];
      bounds[2] = bounds[3] = x[1];
      bounds[4] = bounds[5] = x[2];
      break;

    case SVTK_X_LINE:
    case SVTK_Y_LINE:
    case SVTK_Z_LINE:
      this->Points->GetPoint(cellId, x);
      bounds[0] = bounds[1] = x[0];
      bounds[2] = bounds[3] = x[1];
      bounds[4] = bounds[5] = x[2];

      this->Points->GetPoint(cellId + 1, x);
      svtkAdjustBoundsMacro(bounds, x);
      break;

    case SVTK_XY_PLANE:
    case SVTK_YZ_PLANE:
    case SVTK_XZ_PLANE:
      if (this->DataDescription == SVTK_XY_PLANE)
      {
        i = cellId % (this->Dimensions[0] - 1);
        j = cellId / (this->Dimensions[0] - 1);
        idx = i + j * this->Dimensions[0];
        offset1 = 1;
        offset2 = this->Dimensions[0];
      }
      else if (this->DataDescription == SVTK_YZ_PLANE)
      {
        j = cellId % (this->Dimensions[1] - 1);
        k = cellId / (this->Dimensions[1] - 1);
        idx = j + k * this->Dimensions[1];
        offset1 = 1;
        offset2 = this->Dimensions[1];
      }
      else if (this->DataDescription == SVTK_XZ_PLANE)
      {
        i = cellId % (this->Dimensions[0] - 1);
        k = cellId / (this->Dimensions[0] - 1);
        idx = i + k * this->Dimensions[0];
        offset1 = 1;
        offset2 = this->Dimensions[0];
      }

      this->Points->GetPoint(idx, x);
      bounds[0] = bounds[1] = x[0];
      bounds[2] = bounds[3] = x[1];
      bounds[4] = bounds[5] = x[2];

      this->Points->GetPoint(idx + offset1, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset1 + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      break;

    case SVTK_XYZ_GRID:
      d01 = this->Dimensions[0] * this->Dimensions[1];
      i = cellId % (this->Dimensions[0] - 1);
      j = (cellId / (this->Dimensions[0] - 1)) % (this->Dimensions[1] - 1);
      k = cellId / ((this->Dimensions[0] - 1) * (this->Dimensions[1] - 1));
      idx = i + j * this->Dimensions[0] + k * d01;
      offset1 = 1;
      offset2 = this->Dimensions[0];

      this->Points->GetPoint(idx, x);
      bounds[0] = bounds[1] = x[0];
      bounds[2] = bounds[3] = x[1];
      bounds[4] = bounds[5] = x[2];

      this->Points->GetPoint(idx + offset1, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset1 + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      idx += d01;

      this->Points->GetPoint(idx, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset1, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset1 + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      this->Points->GetPoint(idx + offset2, x);
      svtkAdjustBoundsMacro(bounds, x);

      break;
  }
}

//----------------------------------------------------------------------------
// Turn off a particular data point.
void svtkStructuredGrid::BlankPoint(svtkIdType ptId)
{
  svtkUnsignedCharArray* ghosts = this->GetPointGhostArray();
  if (!ghosts)
  {
    ghosts = this->AllocatePointGhostArray();
  }
  ghosts->SetValue(ptId, ghosts->GetValue(ptId) | svtkDataSetAttributes::HIDDENPOINT);
  assert(!this->IsPointVisible(ptId));
}

//----------------------------------------------------------------------------
// Turn on a particular data point.
void svtkStructuredGrid::UnBlankPoint(svtkIdType ptId)
{
  svtkUnsignedCharArray* ghosts = this->GetPointGhostArray();
  if (ghosts)
  {
    ghosts->SetValue(ptId, ghosts->GetValue(ptId) & ~svtkDataSetAttributes::HIDDENPOINT);
  }
  assert(this->IsPointVisible(ptId));
}

//----------------------------------------------------------------------------
// Turn off a particular data cell.
void svtkStructuredGrid::BlankCell(svtkIdType cellId)
{
  svtkUnsignedCharArray* ghosts = this->GetCellGhostArray();
  if (!ghosts)
  {
    ghosts = this->AllocateCellGhostArray();
  }
  ghosts->SetValue(cellId, ghosts->GetValue(cellId) | svtkDataSetAttributes::HIDDENCELL);
  assert(!this->IsCellVisible(cellId));
}

//----------------------------------------------------------------------------
// Turn on a particular data cell.
void svtkStructuredGrid::UnBlankCell(svtkIdType cellId)
{
  svtkUnsignedCharArray* ghosts = this->GetCellGhostArray();
  if (ghosts)
  {
    ghosts->SetValue(cellId, ghosts->GetValue(cellId) & ~svtkDataSetAttributes::HIDDENCELL);
  }
}

//----------------------------------------------------------------------------
unsigned char svtkStructuredGrid::IsPointVisible(svtkIdType pointId)
{
  svtkUnsignedCharArray* ghosts = this->GetPointGhostArray();
  if (ghosts && (ghosts->GetValue(pointId) & svtkDataSetAttributes::HIDDENPOINT))
  {
    return 0;
  }
  return 1;
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::GetCellDims(int cellDims[3])
{
  for (int i = 0; i < 3; ++i)
  {
    cellDims[i] = ((this->Dimensions[i] - 1) < 1) ? 1 : this->Dimensions[i] - 1;
  }
}

//----------------------------------------------------------------------------
// Return non-zero if the specified cell is visible (i.e., not blanked)
unsigned char svtkStructuredGrid::IsCellVisible(svtkIdType cellId)
{
  svtkUnsignedCharArray* ghosts = this->GetCellGhostArray();
  if (ghosts && (ghosts->GetValue(cellId) & MASKED_CELL_VALUE))
  {
    return 0;
  }

  if (!this->GetPointGhostArray())
  {
    return (this->DataDescription == SVTK_EMPTY) ? 0 : 1;
  }

  // Update dimensions
  this->GetDimensions();

  int numIds = 0;
  svtkIdType ptIds[8];
  int iMin, iMax, jMin, jMax, kMin, kMax;
  svtkIdType d01 = this->Dimensions[0] * this->Dimensions[1];
  iMin = iMax = jMin = jMax = kMin = kMax = 0;

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return 0;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      numIds = 1;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      break;

    case SVTK_X_LINE:
      iMin = cellId;
      iMax = cellId + 1;
      numIds = 2;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMax + jMin * this->Dimensions[0] + kMin * d01;
      break;

    case SVTK_Y_LINE:
      jMin = cellId;
      jMax = cellId + 1;
      numIds = 2;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMin + jMax * this->Dimensions[0] + kMin * d01;
      break;

    case SVTK_Z_LINE:
      kMin = cellId;
      kMax = cellId + 1;
      numIds = 2;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMin + jMin * this->Dimensions[0] + kMax * d01;
      break;

    case SVTK_XY_PLANE:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      jMin = cellId / (this->Dimensions[0] - 1);
      jMax = jMin + 1;
      numIds = 4;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMax + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[2] = iMax + jMax * this->Dimensions[0] + kMin * d01;
      ptIds[3] = iMin + jMax * this->Dimensions[0] + kMin * d01;
      break;

    case SVTK_YZ_PLANE:
      jMin = cellId % (this->Dimensions[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / (this->Dimensions[1] - 1);
      kMax = kMin + 1;
      numIds = 4;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMin + jMax * this->Dimensions[0] + kMin * d01;
      ptIds[2] = iMin + jMax * this->Dimensions[0] + kMax * d01;
      ptIds[3] = iMin + jMin * this->Dimensions[0] + kMax * d01;
      break;

    case SVTK_XZ_PLANE:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      kMin = cellId / (this->Dimensions[0] - 1);
      kMax = kMin + 1;
      numIds = 4;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMax + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[2] = iMax + jMin * this->Dimensions[0] + kMax * d01;
      ptIds[3] = iMin + jMin * this->Dimensions[0] + kMax * d01;
      break;

    case SVTK_XYZ_GRID:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      jMin = (cellId / (this->Dimensions[0] - 1)) % (this->Dimensions[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / ((this->Dimensions[0] - 1) * (this->Dimensions[1] - 1));
      kMax = kMin + 1;
      numIds = 8;
      ptIds[0] = iMin + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[1] = iMax + jMin * this->Dimensions[0] + kMin * d01;
      ptIds[2] = iMax + jMax * this->Dimensions[0] + kMin * d01;
      ptIds[3] = iMin + jMax * this->Dimensions[0] + kMin * d01;
      ptIds[4] = iMin + jMin * this->Dimensions[0] + kMax * d01;
      ptIds[5] = iMax + jMin * this->Dimensions[0] + kMax * d01;
      ptIds[6] = iMax + jMax * this->Dimensions[0] + kMax * d01;
      ptIds[7] = iMin + jMax * this->Dimensions[0] + kMax * d01;
      break;
  }

  for (int i = 0; i < numIds; i++)
  {
    if (!this->IsPointVisible(ptIds[i]))
    {
      return 0;
    }
  }

  return 1;
}

//----------------------------------------------------------------------------
// Set dimensions of structured grid dataset.
void svtkStructuredGrid::SetDimensions(int i, int j, int k)
{
  this->SetExtent(0, i - 1, 0, j - 1, 0, k - 1);
}

//----------------------------------------------------------------------------
// Set dimensions of structured grid dataset.
void svtkStructuredGrid::SetDimensions(const int dim[3])
{
  this->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);
}

//----------------------------------------------------------------------------
// Get the points defining a cell. (See svtkDataSet for more info.)
void svtkStructuredGrid::GetCellPoints(svtkIdType cellId, svtkIdList* ptIds)
{
  // Update dimensions
  this->GetDimensions();

  int iMin, iMax, jMin, jMax, kMin, kMax;
  svtkIdType d01 = this->Dimensions[0] * this->Dimensions[1];

  ptIds->Reset();
  iMin = iMax = jMin = jMax = kMin = kMax = 0;

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      ptIds->SetNumberOfIds(1);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      break;

    case SVTK_X_LINE:
      iMin = cellId;
      iMax = cellId + 1;
      ptIds->SetNumberOfIds(2);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMax + jMin * this->Dimensions[0] + kMin * d01);
      break;

    case SVTK_Y_LINE:
      jMin = cellId;
      jMax = cellId + 1;
      ptIds->SetNumberOfIds(2);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMin + jMax * this->Dimensions[0] + kMin * d01);
      break;

    case SVTK_Z_LINE:
      kMin = cellId;
      kMax = cellId + 1;
      ptIds->SetNumberOfIds(2);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMin + jMin * this->Dimensions[0] + kMax * d01);
      break;

    case SVTK_XY_PLANE:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      jMin = cellId / (this->Dimensions[0] - 1);
      jMax = jMin + 1;
      ptIds->SetNumberOfIds(4);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMax + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(2, iMax + jMax * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(3, iMin + jMax * this->Dimensions[0] + kMin * d01);
      break;

    case SVTK_YZ_PLANE:
      jMin = cellId % (this->Dimensions[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / (this->Dimensions[1] - 1);
      kMax = kMin + 1;
      ptIds->SetNumberOfIds(4);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMin + jMax * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(2, iMin + jMax * this->Dimensions[0] + kMax * d01);
      ptIds->SetId(3, iMin + jMin * this->Dimensions[0] + kMax * d01);
      break;

    case SVTK_XZ_PLANE:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      kMin = cellId / (this->Dimensions[0] - 1);
      kMax = kMin + 1;
      ptIds->SetNumberOfIds(4);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMax + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(2, iMax + jMin * this->Dimensions[0] + kMax * d01);
      ptIds->SetId(3, iMin + jMin * this->Dimensions[0] + kMax * d01);
      break;

    case SVTK_XYZ_GRID:
      iMin = cellId % (this->Dimensions[0] - 1);
      iMax = iMin + 1;
      jMin = (cellId / (this->Dimensions[0] - 1)) % (this->Dimensions[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / ((this->Dimensions[0] - 1) * (this->Dimensions[1] - 1));
      kMax = kMin + 1;
      ptIds->SetNumberOfIds(8);
      ptIds->SetId(0, iMin + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(1, iMax + jMin * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(2, iMax + jMax * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(3, iMin + jMax * this->Dimensions[0] + kMin * d01);
      ptIds->SetId(4, iMin + jMin * this->Dimensions[0] + kMax * d01);
      ptIds->SetId(5, iMax + jMin * this->Dimensions[0] + kMax * d01);
      ptIds->SetId(6, iMax + jMax * this->Dimensions[0] + kMax * d01);
      ptIds->SetId(7, iMin + jMax * this->Dimensions[0] + kMax * d01);
      break;
  }
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::SetExtent(int extent[6])
{
  int description;

  description = svtkStructuredData::SetExtent(extent, this->Extent);

  if (description < 0) // improperly specified
  {
    svtkErrorMacro(<< "Bad Extent, retaining previous values");
  }

  if (description == SVTK_UNCHANGED)
  {
    return;
  }

  this->DataDescription = description;

  this->Modified();
  this->Dimensions[0] = extent[1] - extent[0] + 1;
  this->Dimensions[1] = extent[3] - extent[2] + 1;
  this->Dimensions[2] = extent[5] - extent[4] + 1;
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::SetExtent(int xMin, int xMax, int yMin, int yMax, int zMin, int zMax)
{
  int extent[6];

  extent[0] = xMin;
  extent[1] = xMax;
  extent[2] = yMin;
  extent[3] = yMax;
  extent[4] = zMin;
  extent[5] = zMax;

  this->SetExtent(extent);
}

int* svtkStructuredGrid::GetDimensions()
{
  this->GetDimensions(this->Dimensions);
  return this->Dimensions;
}

void svtkStructuredGrid::GetDimensions(int dim[3])
{
  const int* extent = this->Extent;
  dim[0] = extent[1] - extent[0] + 1;
  dim[1] = extent[3] - extent[2] + 1;
  dim[2] = extent[5] - extent[4] + 1;
}

class CellVisibility
{
public:
  CellVisibility(svtkStructuredGrid* input)
    : Input(input)
  {
  }
  bool operator()(const svtkIdType id) { return !Input->IsCellVisible(id); }

private:
  svtkStructuredGrid* Input;
};

//----------------------------------------------------------------------------
void svtkStructuredGrid::GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds)
{
  int numPtIds = ptIds->GetNumberOfIds();

  // Use special methods for speed
  switch (numPtIds)
  {
    case 0:
      cellIds->Reset();
      return;

    case 1:
    case 2:
    case 4: // vertex, edge, face neighbors
      svtkStructuredData::GetCellNeighbors(cellId, ptIds, cellIds, this->GetDimensions());
      break;

    default:
      this->svtkDataSet::GetCellNeighbors(cellId, ptIds, cellIds);
  }

  // If blanking, remove blanked cells.
  if (this->GetPointGhostArray() || this->GetCellGhostArray())
  {
    svtkIdType* pCellIds = cellIds->GetPointer(0);
    svtkIdType* end =
      std::remove_if(pCellIds, pCellIds + cellIds->GetNumberOfIds(), CellVisibility(this));
    cellIds->Resize(std::distance(pCellIds, end));
  }
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::GetCellNeighbors(
  svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds, int* seedLoc)
{
  int numPtIds = ptIds->GetNumberOfIds();

  // Use special methods for speed
  switch (numPtIds)
  {
    case 0:
      cellIds->Reset();
      return;

    case 1:
    case 2:
    case 4: // vertex, edge, face neighbors
      svtkStructuredData::GetCellNeighbors(cellId, ptIds, cellIds, this->GetDimensions(), seedLoc);
      break;

    default:
      this->svtkDataSet::GetCellNeighbors(cellId, ptIds, cellIds);
  }

  // If blanking, remove blanked cells.
  if (this->GetPointGhostArray() || this->GetCellGhostArray())
  {
    svtkIdType* pCellIds = cellIds->GetPointer(0);
    svtkIdType* end =
      std::remove_if(pCellIds, pCellIds + cellIds->GetNumberOfIds(), CellVisibility(this));
    cellIds->Resize(std::distance(pCellIds, end));
  }
}

//----------------------------------------------------------------------------
unsigned long svtkStructuredGrid::GetActualMemorySize()
{
  return this->svtkPointSet::GetActualMemorySize();
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::ShallowCopy(svtkDataObject* dataObject)
{
  svtkStructuredGrid* grid = svtkStructuredGrid::SafeDownCast(dataObject);
  if (grid != nullptr)
  {
    this->InternalStructuredGridCopy(grid);
  }
  this->svtkPointSet::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::DeepCopy(svtkDataObject* dataObject)
{
  svtkStructuredGrid* grid = svtkStructuredGrid::SafeDownCast(dataObject);
  if (grid != nullptr)
  {
    this->InternalStructuredGridCopy(grid);
  }
  this->svtkPointSet::DeepCopy(dataObject);
}

//----------------------------------------------------------------------------
// This copies all the local variables (but not objects).
void svtkStructuredGrid::InternalStructuredGridCopy(svtkStructuredGrid* src)
{
  int idx;

  this->DataDescription = src->DataDescription;

  // Update dimensions
  this->GetDimensions();

  for (idx = 0; idx < 3; ++idx)
  {
    this->Dimensions[idx] = src->Dimensions[idx];
  }
  memcpy(this->Extent, src->GetExtent(), 6 * sizeof(int));
}

//----------------------------------------------------------------------------
// Override this method because of blanking
void svtkStructuredGrid::ComputeScalarRange()
{
  if (this->GetMTime() > this->ScalarRangeComputeTime)
  {
    svtkDataArray* ptScalars = this->PointData->GetScalars();
    svtkDataArray* cellScalars = this->CellData->GetScalars();
    double ptRange[2];
    double cellRange[2];
    double s;

    ptRange[0] = SVTK_DOUBLE_MAX;
    ptRange[1] = SVTK_DOUBLE_MIN;
    if (ptScalars)
    {
      svtkIdType num = this->GetNumberOfPoints();
      for (svtkIdType id = 0; id < num; ++id)
      {
        if (this->IsPointVisible(id))
        {
          s = ptScalars->GetComponent(id, 0);
          if (s < ptRange[0])
          {
            ptRange[0] = s;
          }
          if (s > ptRange[1])
          {
            ptRange[1] = s;
          }
        }
      }
    }

    cellRange[0] = ptRange[0];
    cellRange[1] = ptRange[1];
    if (cellScalars)
    {
      svtkIdType num = this->GetNumberOfCells();
      for (svtkIdType id = 0; id < num; ++id)
      {
        if (this->IsCellVisible(id))
        {
          s = cellScalars->GetComponent(id, 0);
          if (s < cellRange[0])
          {
            cellRange[0] = s;
          }
          if (s > cellRange[1])
          {
            cellRange[1] = s;
          }
        }
      }
    }

    this->ScalarRange[0] = (cellRange[0] >= SVTK_DOUBLE_MAX ? 0.0 : cellRange[0]);
    this->ScalarRange[1] = (cellRange[1] <= SVTK_DOUBLE_MIN ? 1.0 : cellRange[1]);

    this->ScalarRangeComputeTime.Modified();
  }
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::Crop(const int* updateExtent)
{
  // Do nothing for empty datasets:
  for (int dim = 0; dim < 3; ++dim)
  {
    if (this->Extent[2 * dim] > this->Extent[2 * dim + 1])
    {
      svtkDebugMacro(<< "Refusing to crop empty dataset.");
      return;
    }
  }

  int i, j, k;
  int uExt[6];
  const int* extent = this->Extent;

  // If the update extent is larger than the extent,
  // we cannot do anything about it here.
  for (i = 0; i < 3; ++i)
  {
    uExt[i * 2] = updateExtent[i * 2];
    if (uExt[i * 2] < extent[i * 2])
    {
      uExt[i * 2] = extent[i * 2];
    }
    uExt[i * 2 + 1] = updateExtent[i * 2 + 1];
    if (uExt[i * 2 + 1] > extent[i * 2 + 1])
    {
      uExt[i * 2 + 1] = extent[i * 2 + 1];
    }
  }

  // If extents already match, then we need to do nothing.
  if (extent[0] == uExt[0] && extent[1] == uExt[1] && extent[2] == uExt[2] &&
    extent[3] == uExt[3] && extent[4] == uExt[4] && extent[5] == uExt[5])
  {
    return;
  }
  else
  {
    svtkStructuredGrid* newGrid;
    svtkPointData *inPD, *outPD;
    svtkCellData *inCD, *outCD;
    int outSize, jOffset, kOffset;
    svtkIdType idx, newId;
    svtkPoints *newPts, *inPts;
    int inInc1, inInc2;

    // Get the points.  Protect against empty data objects.
    inPts = this->GetPoints();
    if (inPts == nullptr)
    {
      return;
    }

    svtkDebugMacro(<< "Cropping Grid");

    newGrid = svtkStructuredGrid::New();
    inPD = this->GetPointData();
    inCD = this->GetCellData();
    outPD = newGrid->GetPointData();
    outCD = newGrid->GetCellData();

    // Allocate necessary objects
    //
    newGrid->SetExtent(uExt);
    outSize = (uExt[1] - uExt[0] + 1) * (uExt[3] - uExt[2] + 1) * (uExt[5] - uExt[4] + 1);
    newPts = inPts->NewInstance();
    newPts->SetDataType(inPts->GetDataType());
    newPts->SetNumberOfPoints(outSize);
    outPD->CopyAllocate(inPD, outSize, outSize);
    outCD->CopyAllocate(inCD, outSize, outSize);

    // Traverse this data and copy point attributes to output
    newId = 0;
    inInc1 = (extent[1] - extent[0] + 1);
    inInc2 = inInc1 * (extent[3] - extent[2] + 1);
    for (k = uExt[4]; k <= uExt[5]; ++k)
    {
      kOffset = (k - extent[4]) * inInc2;
      for (j = uExt[2]; j <= uExt[3]; ++j)
      {
        jOffset = (j - extent[2]) * inInc1;
        for (i = uExt[0]; i <= uExt[1]; ++i)
        {
          idx = (i - extent[0]) + jOffset + kOffset;
          newPts->SetPoint(newId, inPts->GetPoint(idx));
          outPD->CopyData(inPD, idx, newId++);
        }
      }
    }

    // Traverse input data and copy cell attributes to output
    newId = 0;
    inInc1 = (extent[1] - extent[0]);
    inInc2 = inInc1 * (extent[3] - extent[2]);
    for (k = uExt[4]; k < uExt[5]; ++k)
    {
      kOffset = (k - extent[4]) * inInc2;
      for (j = uExt[2]; j < uExt[3]; ++j)
      {
        jOffset = (j - extent[2]) * inInc1;
        for (i = uExt[0]; i < uExt[1]; ++i)
        {
          idx = (i - extent[0]) + jOffset + kOffset;
          outCD->CopyData(inCD, idx, newId++);
        }
      }
    }

    this->SetExtent(uExt);
    this->SetPoints(newPts);
    newPts->Delete();
    inPD->ShallowCopy(outPD);
    inCD->ShallowCopy(outCD);
    newGrid->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  int dim[3];
  this->GetDimensions(dim);
  os << indent << "Dimensions: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ")\n";

  const int* extent = this->Extent;
  os << indent << "Extent: " << extent[0] << ", " << extent[1] << ", " << extent[2] << ", "
     << extent[3] << ", " << extent[4] << ", " << extent[5] << endl;

  os << ")\n";
}

//----------------------------------------------------------------------------
svtkStructuredGrid* svtkStructuredGrid::GetData(svtkInformation* info)
{
  return info ? svtkStructuredGrid::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkStructuredGrid* svtkStructuredGrid::GetData(svtkInformationVector* v, int i)
{
  return svtkStructuredGrid::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkStructuredGrid::GetPoint(int i, int j, int k, double p[3], bool adjustForExtent)
{
  int extent[6];
  this->GetExtent(extent);

  if (i < extent[0] || i > extent[1] || j < extent[2] || j > extent[3] || k < extent[4] ||
    k > extent[5])
  {
    svtkErrorMacro("ERROR: IJK coordinates are outside of grid extent!");
    return; // out of bounds!
  }

  int pos[3];
  pos[0] = i;
  pos[1] = j;
  pos[2] = k;

  svtkIdType id;

  if (adjustForExtent)
  {
    id = svtkStructuredData::ComputePointIdForExtent(extent, pos);
  }
  else
  {
    int dim[3];
    this->GetDimensions(dim);
    id = svtkStructuredData::ComputePointId(dim, pos);
  }

  this->GetPoint(id, p);
}

//----------------------------------------------------------------------------
bool svtkStructuredGrid::HasAnyBlankPoints()
{
  return IsAnyBitSet(this->GetPointGhostArray(), svtkDataSetAttributes::HIDDENPOINT);
}

//----------------------------------------------------------------------------
bool svtkStructuredGrid::HasAnyBlankCells()
{
  int cellBlanking = IsAnyBitSet(this->GetCellGhostArray(), svtkDataSetAttributes::HIDDENCELL);
  return cellBlanking || this->HasAnyBlankPoints();
}
