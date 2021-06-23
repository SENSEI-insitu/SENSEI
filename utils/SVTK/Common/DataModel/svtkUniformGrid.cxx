/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformGrid.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkUniformGrid.h"

#include "svtkAMRBox.h"
#include "svtkCellData.h"
#include "svtkDataArray.h"
#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkImageData.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkLine.h"
#include "svtkObjectFactory.h"
#include "svtkPixel.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkStructuredData.h"
#include "svtkUnsignedCharArray.h"
#include "svtkVertex.h"
#include "svtkVoxel.h"

svtkStandardNewMacro(svtkUniformGrid);

unsigned char svtkUniformGrid::MASKED_CELL_VALUE =
  svtkDataSetAttributes::HIDDENCELL | svtkDataSetAttributes::REFINEDCELL;

//----------------------------------------------------------------------------
svtkUniformGrid::svtkUniformGrid()
{
  this->EmptyCell = nullptr;
}

//----------------------------------------------------------------------------
svtkUniformGrid::~svtkUniformGrid()
{
  if (this->EmptyCell)
  {
    this->EmptyCell->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkUniformGrid::Initialize()
{
  this->Superclass::Initialize();
}

//-----------------------------------------------------------------------------
int svtkUniformGrid::Initialize(const svtkAMRBox* def, double* origin, double* spacing)
{
  if (def->Empty())
  {
    svtkWarningMacro("Can't construct a data set from an empty box.");
    return 0;
  }
  if (def->ComputeDimension() == 2)
  {
    // NOTE: Define it 3D, with the third dim 0. eg. (X,X,0)(X,X,0)
    svtkWarningMacro("Can't construct a 3D data set from a 2D box.");
    return 0;
  }

  this->Initialize();
  int nPoints[3];
  def->GetNumberOfNodes(nPoints);

  this->SetDimensions(nPoints);
  this->SetSpacing(spacing);
  this->SetOrigin(origin);

  return 1;
}

//-----------------------------------------------------------------------------
int svtkUniformGrid::Initialize(
  const svtkAMRBox* def, double* origin, double* spacing, int nGhostsI, int nGhostsJ, int nGhostsK)
{
  if (!this->Initialize(def, origin, spacing))
  {
    return 0;
  }

  // Generate ghost cell array, with no ghosts marked.
  int nCells[3];
  def->GetNumberOfCells(nCells);
  svtkUnsignedCharArray* ghosts = svtkUnsignedCharArray::New();
  this->GetCellData()->AddArray(ghosts);
  ghosts->Delete();
  ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
  ghosts->SetNumberOfComponents(1);
  ghosts->SetNumberOfTuples(nCells[0] * nCells[1] * nCells[2]);
  ghosts->FillComponent(0, 0);
  // If there are ghost cells mark them.
  if (nGhostsI || nGhostsJ || nGhostsK)
  {
    unsigned char* pG = ghosts->GetPointer(0);
    const int* lo = def->GetLoCorner();
    const int* hi = def->GetHiCorner();
    // Identify & fill ghost regions
    if (nGhostsI)
    {
      svtkAMRBox left(lo[0], lo[1], lo[2], lo[0] + nGhostsI - 1, hi[1], hi[2]);
      FillRegion(pG, *def, left, static_cast<unsigned char>(1));
      svtkAMRBox right(hi[0] - nGhostsI + 1, lo[1], lo[2], hi[0], hi[1], hi[2]);
      FillRegion(pG, *def, right, static_cast<unsigned char>(1));
    }
    if (nGhostsJ)
    {
      svtkAMRBox front(lo[0], lo[1], lo[2], hi[0], lo[1] + nGhostsJ - 1, hi[2]);
      FillRegion(pG, *def, front, static_cast<unsigned char>(1));
      svtkAMRBox back(lo[0], hi[1] - nGhostsJ + 1, lo[2], hi[0], hi[1], hi[2]);
      FillRegion(pG, *def, back, static_cast<unsigned char>(1));
    }
    if (nGhostsK)
    {
      svtkAMRBox bottom(lo[0], lo[1], lo[2], hi[0], hi[1], lo[2] + nGhostsK - 1);
      FillRegion(pG, *def, bottom, static_cast<unsigned char>(1));
      svtkAMRBox top(lo[0], lo[1], hi[2] - nGhostsK + 1, hi[0], hi[1], hi[2]);
      FillRegion(pG, *def, top, static_cast<unsigned char>(1));
    }
  }
  return 1;
}

//-----------------------------------------------------------------------------
int svtkUniformGrid::Initialize(
  const svtkAMRBox* def, double* origin, double* spacing, const int nGhosts[3])
{
  return this->Initialize(def, origin, spacing, nGhosts[0], nGhosts[1], nGhosts[2]);
}

//-----------------------------------------------------------------------------
int svtkUniformGrid::Initialize(const svtkAMRBox* def, double* origin, double* spacing, int nGhosts)
{
  return this->Initialize(def, origin, spacing, nGhosts, nGhosts, nGhosts);
}

//----------------------------------------------------------------------------
int svtkUniformGrid::GetGridDescription()
{
  return (this->GetDataDescription());
}

//----------------------------------------------------------------------------
svtkEmptyCell* svtkUniformGrid::GetEmptyCell()
{
  if (!this->EmptyCell)
  {
    this->EmptyCell = svtkEmptyCell::New();
  }
  return this->EmptyCell;
}

//----------------------------------------------------------------------------
// Copy the geometric and topological structure of an input structured points
// object.
void svtkUniformGrid::CopyStructure(svtkDataSet* ds)
{
  this->Initialize();

  this->Superclass::CopyStructure(ds);

  if (ds->HasAnyBlankPoints())
  {
    // there is blanking
    this->GetPointData()->AddArray(ds->GetPointGhostArray());
    this->PointGhostArray = nullptr;
  }
  if (ds->HasAnyBlankCells())
  {
    // we assume there is blanking
    this->GetCellData()->AddArray(ds->GetCellGhostArray());
    this->CellGhostArray = nullptr;
  }
}

//----------------------------------------------------------------------------
svtkCell* svtkUniformGrid::GetCell(svtkIdType cellId)
{
  svtkCell* cell = nullptr;
  int loc[3];
  svtkIdType idx, npts;
  int iMin, iMax, jMin, jMax, kMin, kMax;
  double x[3];
  double* origin = this->GetOrigin();
  double* spacing = this->GetSpacing();
  int extent[6];
  this->GetExtent(extent);

  int dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
  int d01 = dims[0] * dims[1];

  iMin = iMax = jMin = jMax = kMin = kMax = 0;

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a cell from an empty image.");
    return this->GetEmptyCell();
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return this->GetEmptyCell();
  }

  switch (this->GetDataDescription())
  {
    case SVTK_EMPTY:
      return this->GetEmptyCell();

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell = this->Vertex;
      break;

    case SVTK_X_LINE:
      iMin = cellId;
      iMax = cellId + 1;
      cell = this->Line;
      break;

    case SVTK_Y_LINE:
      jMin = cellId;
      jMax = cellId + 1;
      cell = this->Line;
      break;

    case SVTK_Z_LINE:
      kMin = cellId;
      kMax = cellId + 1;
      cell = this->Line;
      break;

    case SVTK_XY_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = cellId / (dims[0] - 1);
      jMax = jMin + 1;
      cell = this->Pixel;
      break;

    case SVTK_YZ_PLANE:
      jMin = cellId % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / (dims[1] - 1);
      kMax = kMin + 1;
      cell = this->Pixel;
      break;

    case SVTK_XZ_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      kMin = cellId / (dims[0] - 1);
      kMax = kMin + 1;
      cell = this->Pixel;
      break;

    case SVTK_XYZ_GRID:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = (cellId / (dims[0] - 1)) % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / ((dims[0] - 1) * (dims[1] - 1));
      kMax = kMin + 1;
      cell = this->Voxel;
      break;

    default:
      svtkErrorMacro(<< "Invalid DataDescription.");
      return nullptr;
  }

  // Extract point coordinates and point ids
  // Ids are relative to extent min.
  npts = 0;
  for (loc[2] = kMin; loc[2] <= kMax; loc[2]++)
  {
    x[2] = origin[2] + (loc[2] + extent[4]) * spacing[2];
    for (loc[1] = jMin; loc[1] <= jMax; loc[1]++)
    {
      x[1] = origin[1] + (loc[1] + extent[2]) * spacing[1];
      for (loc[0] = iMin; loc[0] <= iMax; loc[0]++)
      {
        x[0] = origin[0] + (loc[0] + extent[0]) * spacing[0];

        idx = loc[0] + loc[1] * dims[0] + loc[2] * d01;
        cell->PointIds->SetId(npts, idx);
        cell->Points->SetPoint(npts++, x);
      }
    }
  }

  return cell;
}

//----------------------------------------------------------------------------
svtkCell* svtkUniformGrid::GetCell(int iMin, int jMin, int kMin)
{
  svtkIdType cellId = iMin + (jMin + (kMin * (this->Dimensions[1] - 1))) * (this->Dimensions[0] - 1);
  svtkCell* cell = nullptr;
  int loc[3];
  svtkIdType idx, npts;
  int iMax = 0, jMax = 0, kMax = 0;
  double x[3];
  double* origin = this->GetOrigin();
  double* spacing = this->GetSpacing();
  int extent[6];
  this->GetExtent(extent);

  int dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
  int d01 = dims[0] * dims[1];

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a cell from an empty image.");
    return this->GetEmptyCell();
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return this->GetEmptyCell();
  }

  switch (this->GetDataDescription())
  {
    case SVTK_EMPTY:
      return this->GetEmptyCell();

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell = this->Vertex;
      break;

    case SVTK_X_LINE:
      iMax = iMin + 1;
      jMax = jMin = 0;
      kMax = kMin = 0;
      cell = this->Line;
      break;

    case SVTK_Y_LINE:
      iMax = iMin = 0;
      jMax = jMin + 1;
      kMax = kMin = 0;
      cell = this->Line;
      break;

    case SVTK_Z_LINE:
      iMin = iMax = 0;
      jMin = jMax = 0;
      kMax = kMin + 1;
      cell = this->Line;
      break;

    case SVTK_XY_PLANE:
      iMax = iMin + 1;
      jMax = jMin + 1;
      kMin = kMax = 0;
      cell = this->Pixel;
      break;

    case SVTK_YZ_PLANE:
      iMin = iMax = 0;
      jMax = jMin + 1;
      kMax = kMin + 1;
      cell = this->Pixel;
      break;

    case SVTK_XZ_PLANE:
      iMax = iMin + 1;
      jMin = jMax = 0;
      kMax = kMin + 1;
      cell = this->Pixel;
      break;

    case SVTK_XYZ_GRID:
      iMax = iMin + 1;
      jMax = jMin + 1;
      kMax = kMin + 1;
      cell = this->Voxel;
      break;

    default:
      svtkErrorMacro(<< "Invalid DataDescription.");
      return nullptr;
  }

  // Extract point coordinates and point ids
  // Ids are relative to extent min.
  npts = 0;
  for (loc[2] = kMin; loc[2] <= kMax; loc[2]++)
  {
    x[2] = origin[2] + (loc[2] + extent[4]) * spacing[2];
    for (loc[1] = jMin; loc[1] <= jMax; loc[1]++)
    {
      x[1] = origin[1] + (loc[1] + extent[2]) * spacing[1];
      for (loc[0] = iMin; loc[0] <= iMax; loc[0]++)
      {
        x[0] = origin[0] + (loc[0] + extent[0]) * spacing[0];

        idx = loc[0] + loc[1] * dims[0] + loc[2] * d01;
        cell->PointIds->SetId(npts, idx);
        cell->Points->SetPoint(npts++, x);
      }
    }
  }

  return cell;
}

//----------------------------------------------------------------------------
void svtkUniformGrid::GetCell(svtkIdType cellId, svtkGenericCell* cell)
{
  svtkIdType npts, idx;
  int loc[3];
  int iMin, iMax, jMin, jMax, kMin, kMax;
  double* origin = this->GetOrigin();
  double* spacing = this->GetSpacing();
  double x[3];
  int extent[6];
  this->GetExtent(extent);

  int dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
  int d01 = dims[0] * dims[1];

  iMin = iMax = jMin = jMax = kMin = kMax = 0;

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a cell from an empty image.");
    cell->SetCellTypeToEmptyCell();
    return;
  }

  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    cell->SetCellTypeToEmptyCell();
    return;
  }

  switch (this->GetDataDescription())
  {
    case SVTK_EMPTY:
      cell->SetCellTypeToEmptyCell();
      return;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      cell->SetCellTypeToVertex();
      break;

    case SVTK_X_LINE:
      iMin = cellId;
      iMax = cellId + 1;
      cell->SetCellTypeToLine();
      break;

    case SVTK_Y_LINE:
      jMin = cellId;
      jMax = cellId + 1;
      cell->SetCellTypeToLine();
      break;

    case SVTK_Z_LINE:
      kMin = cellId;
      kMax = cellId + 1;
      cell->SetCellTypeToLine();
      break;

    case SVTK_XY_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = cellId / (dims[0] - 1);
      jMax = jMin + 1;
      cell->SetCellTypeToPixel();
      break;

    case SVTK_YZ_PLANE:
      jMin = cellId % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / (dims[1] - 1);
      kMax = kMin + 1;
      cell->SetCellTypeToPixel();
      break;

    case SVTK_XZ_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      kMin = cellId / (dims[0] - 1);
      kMax = kMin + 1;
      cell->SetCellTypeToPixel();
      break;

    case SVTK_XYZ_GRID:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = (cellId / (dims[0] - 1)) % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / ((dims[0] - 1) * (dims[1] - 1));
      kMax = kMin + 1;
      cell->SetCellTypeToVoxel();
      break;
  }

  // Extract point coordinates and point ids
  for (npts = 0, loc[2] = kMin; loc[2] <= kMax; loc[2]++)
  {
    x[2] = origin[2] + (loc[2] + extent[4]) * spacing[2];
    for (loc[1] = jMin; loc[1] <= jMax; loc[1]++)
    {
      x[1] = origin[1] + (loc[1] + extent[2]) * spacing[1];
      for (loc[0] = iMin; loc[0] <= iMax; loc[0]++)
      {
        x[0] = origin[0] + (loc[0] + extent[0]) * spacing[0];

        idx = loc[0] + loc[1] * dims[0] + loc[2] * d01;
        cell->PointIds->SetId(npts, idx);
        cell->Points->SetPoint(npts++, x);
      }
    }
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkUniformGrid::FindCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkGenericCell* svtkNotUsed(gencell), svtkIdType svtkNotUsed(cellId), double svtkNotUsed(tol2),
  int& subId, double pcoords[3], double* weights)
{
  return this->FindCell(x, static_cast<svtkCell*>(nullptr), 0, 0.0, subId, pcoords, weights);
}

//----------------------------------------------------------------------------
svtkIdType svtkUniformGrid::FindCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkIdType svtkNotUsed(cellId), double svtkNotUsed(tol2), int& subId, double pcoords[3],
  double* weights)
{
  int loc[3];
  int* dims = this->GetDimensions();

  if (this->ComputeStructuredCoordinates(x, loc, pcoords) == 0)
  {
    return -1;
  }

  this->Voxel->InterpolationFunctions(pcoords, weights);

  //
  //  From this location get the cell id
  //
  subId = 0;
  int extent[6];
  this->GetExtent(extent);

  svtkIdType cellId = (loc[2] - extent[4]) * (dims[0] - 1) * (dims[1] - 1) +
    (loc[1] - extent[2]) * (dims[0] - 1) + loc[0] - extent[0];

  if ((this->GetPointGhostArray() || this->GetCellGhostArray()) && !this->IsCellVisible(cellId))
  {
    return -1;
  }
  return cellId;
}

//----------------------------------------------------------------------------
svtkCell* svtkUniformGrid::FindAndGetCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkIdType svtkNotUsed(cellId), double svtkNotUsed(tol2), int& subId, double pcoords[3],
  double* weights)
{
  int i, j, k, loc[3];
  svtkIdType npts, idx;
  double xOut[3];
  int iMax = 0;
  int jMax = 0;
  int kMax = 0;
  svtkCell* cell = nullptr;
  double* origin = this->GetOrigin();
  double* spacing = this->GetSpacing();
  int extent[6];
  this->GetExtent(extent);

  int dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
  svtkIdType d01 = dims[0] * dims[1];

  if (this->ComputeStructuredCoordinates(x, loc, pcoords) == 0)
  {
    return nullptr;
  }

  svtkIdType cellId = loc[2] * (dims[0] - 1) * (dims[1] - 1) + loc[1] * (dims[0] - 1) + loc[0];

  if (!this->IsCellVisible(cellId))
  {
    return nullptr;
  }

  //
  // Get the parametric coordinates and weights for interpolation
  //
  switch (this->GetDataDescription())
  {
    case SVTK_EMPTY:
      return nullptr;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      iMax = loc[0];
      jMax = loc[1];
      kMax = loc[2];
      cell = this->Vertex;
      break;

    case SVTK_X_LINE:
      iMax = loc[0] + 1;
      jMax = loc[1];
      kMax = loc[2];
      cell = this->Line;
      break;

    case SVTK_Y_LINE:
      iMax = loc[0];
      jMax = loc[1] + 1;
      kMax = loc[2];
      cell = this->Line;
      break;

    case SVTK_Z_LINE:
      iMax = loc[0];
      jMax = loc[1];
      kMax = loc[2] + 1;
      cell = this->Line;
      break;

    case SVTK_XY_PLANE:
      iMax = loc[0] + 1;
      jMax = loc[1] + 1;
      kMax = loc[2];
      cell = this->Pixel;
      break;

    case SVTK_YZ_PLANE:
      iMax = loc[0];
      jMax = loc[1] + 1;
      kMax = loc[2] + 1;
      cell = this->Pixel;
      break;

    case SVTK_XZ_PLANE:
      iMax = loc[0] + 1;
      jMax = loc[1];
      kMax = loc[2] + 1;
      cell = this->Pixel;
      break;

    case SVTK_XYZ_GRID:
      iMax = loc[0] + 1;
      jMax = loc[1] + 1;
      kMax = loc[2] + 1;
      cell = this->Voxel;
      break;

    default:
      svtkErrorMacro(<< "Invalid DataDescription.");
      return nullptr;
  }
  cell->InterpolateFunctions(pcoords, weights);

  npts = 0;
  for (k = loc[2]; k <= kMax; k++)
  {
    xOut[2] = origin[2] + k * spacing[2];
    for (j = loc[1]; j <= jMax; j++)
    {
      xOut[1] = origin[1] + j * spacing[1];
      // make idx relative to the extent not the whole extent
      idx = loc[0] - extent[0] + (j - extent[2]) * dims[0] + (k - extent[4]) * d01;
      for (i = loc[0]; i <= iMax; i++, idx++)
      {
        xOut[0] = origin[0] + i * spacing[0];

        cell->PointIds->SetId(npts, idx);
        cell->Points->SetPoint(npts++, xOut);
      }
    }
  }
  subId = 0;

  return cell;
}

//----------------------------------------------------------------------------
int svtkUniformGrid::GetCellType(svtkIdType cellId)
{
  // see whether the cell is blanked
  if (!this->IsCellVisible(cellId))
  {
    return SVTK_EMPTY_CELL;
  }

  switch (this->GetDataDescription())
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
      return SVTK_PIXEL;

    case SVTK_XYZ_GRID:
      return SVTK_VOXEL;

    default:
      svtkErrorMacro(<< "Bad data description!");
      return SVTK_EMPTY_CELL;
  }
}

//----------------------------------------------------------------------------
void svtkUniformGrid::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
svtkImageData* svtkUniformGrid::NewImageDataCopy()
{
  svtkImageData* copy = svtkImageData::New();

  copy->ShallowCopy(this);

  double origin[3];
  double spacing[3];
  this->GetOrigin(origin);
  this->GetSpacing(spacing);
  // First set the extent of the copy to empty so that
  // the next call computes the DataDescription for us
  copy->SetExtent(0, -1, 0, -1, 0, -1);
  copy->SetExtent(this->GetExtent());
  copy->SetOrigin(origin);
  copy->SetSpacing(spacing);

  return copy;
}

//----------------------------------------------------------------------------
// Override this method because of blanking
void svtkUniformGrid::ComputeScalarRange()
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
// Turn off a particular data point.
void svtkUniformGrid::BlankPoint(svtkIdType ptId)
{
  svtkUnsignedCharArray* ghosts = this->GetPointGhostArray();
  if (!ghosts)
  {
    this->AllocatePointGhostArray();
    ghosts = this->GetPointGhostArray();
  }
  ghosts->SetValue(ptId, ghosts->GetValue(ptId) | svtkDataSetAttributes::HIDDENPOINT);
  assert(!this->IsPointVisible(ptId));
}

//----------------------------------------------------------------------------
void svtkUniformGrid::BlankPoint(const int i, const int j, const int k)
{
  int ijk[3];
  ijk[0] = i;
  ijk[1] = j;
  ijk[2] = k;
  int idx = svtkStructuredData::ComputePointId(this->Dimensions, ijk);
  this->BlankPoint(idx);
}

//----------------------------------------------------------------------------
// Turn on a particular data point.
void svtkUniformGrid::UnBlankPoint(svtkIdType ptId)
{
  svtkUnsignedCharArray* ghosts = this->GetPointGhostArray();
  if (!ghosts)
  {
    return;
  }
  ghosts->SetValue(ptId, ghosts->GetValue(ptId) & ~svtkDataSetAttributes::HIDDENPOINT);
}

//----------------------------------------------------------------------------
void svtkUniformGrid::UnBlankPoint(const int i, const int j, const int k)
{
  int ijk[3];
  ijk[0] = i;
  ijk[1] = j;
  ijk[2] = k;
  int idx = svtkStructuredData::ComputePointId(this->Dimensions, ijk);
  this->UnBlankPoint(idx);
}

//----------------------------------------------------------------------------
// Turn off a particular data cell.
void svtkUniformGrid::BlankCell(svtkIdType cellId)
{
  svtkUnsignedCharArray* ghost = this->GetCellGhostArray();
  if (!ghost)
  {
    this->AllocateCellGhostArray();
    ghost = this->GetCellGhostArray();
  }
  ghost->SetValue(cellId, ghost->GetValue(cellId) | svtkDataSetAttributes::HIDDENCELL);
  assert(!this->IsCellVisible(cellId));
}

//----------------------------------------------------------------------------
void svtkUniformGrid::BlankCell(const int i, const int j, const int k)
{
  int ijk[3];
  ijk[0] = i;
  ijk[1] = j;
  ijk[2] = k;
  int idx = svtkStructuredData::ComputeCellId(this->Dimensions, ijk);
  assert("cell id in range:" && ((idx >= 0) && (idx < this->GetNumberOfCells())));
  this->BlankCell(idx);
}

//----------------------------------------------------------------------------
// Turn on a particular data cell.
void svtkUniformGrid::UnBlankCell(svtkIdType cellId)
{
  svtkUnsignedCharArray* ghosts = this->GetCellGhostArray();
  if (!ghosts)
  {
    return;
  }
  ghosts->SetValue(cellId, ghosts->GetValue(cellId) & ~svtkDataSetAttributes::HIDDENCELL);
  assert(this->IsCellVisible(cellId));
}

//----------------------------------------------------------------------------
void svtkUniformGrid::UnBlankCell(const int i, const int j, const int k)
{
  int ijk[3];
  ijk[0] = i;
  ijk[1] = j;
  ijk[2] = k;
  int idx = svtkStructuredData::ComputeCellId(this->Dimensions, ijk);
  assert("cell id in range:" && ((idx >= 0) && (idx < this->GetNumberOfCells())));
  this->UnBlankCell(idx);
}

//----------------------------------------------------------------------------
unsigned char svtkUniformGrid::IsPointVisible(svtkIdType pointId)
{
  if (this->GetPointGhostArray() &&
    (this->GetPointGhostArray()->GetValue(pointId) & svtkDataSetAttributes::HIDDENPOINT))
  {
    return 0;
  }
  return 1;
}

//----------------------------------------------------------------------------
// Return non-zero if the specified cell is visible (i.e., not blanked)
unsigned char svtkUniformGrid::IsCellVisible(svtkIdType cellId)
{

  if (this->GetCellGhostArray() &&
    (this->GetCellGhostArray()->GetValue(cellId) & MASKED_CELL_VALUE))
  {
    return 0;
  }
  if (!this->GetPointGhostArray())
  {
    return (this->GetDataDescription() == SVTK_EMPTY) ? 0 : 1;
  }

  int iMin, iMax, jMin, jMax, kMin, kMax;
  int* dims = this->GetDimensions();

  iMin = iMax = jMin = jMax = kMin = kMax = 0;

  switch (this->GetDataDescription())
  {
    case SVTK_EMPTY:
      return 0;

    case SVTK_SINGLE_POINT: // cellId can only be = 0
      break;

    case SVTK_X_LINE:
      iMin = cellId;
      iMax = cellId + 1;
      break;

    case SVTK_Y_LINE:
      jMin = cellId;
      jMax = cellId + 1;
      break;

    case SVTK_Z_LINE:
      kMin = cellId;
      kMax = cellId + 1;
      break;

    case SVTK_XY_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = cellId / (dims[0] - 1);
      jMax = jMin + 1;
      break;

    case SVTK_YZ_PLANE:
      jMin = cellId % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / (dims[1] - 1);
      kMax = kMin + 1;
      break;

    case SVTK_XZ_PLANE:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      kMin = cellId / (dims[0] - 1);
      kMax = kMin + 1;
      break;

    case SVTK_XYZ_GRID:
      iMin = cellId % (dims[0] - 1);
      iMax = iMin + 1;
      jMin = (cellId / (dims[0] - 1)) % (dims[1] - 1);
      jMax = jMin + 1;
      kMin = cellId / ((dims[0] - 1) * (dims[1] - 1));
      kMax = kMin + 1;
      break;
  }

  // Extract point ids
  // Ids are relative to extent min.
  svtkIdType idx[8];
  svtkIdType npts = 0;
  int loc[3];
  int d01 = dims[0] * dims[1];
  for (loc[2] = kMin; loc[2] <= kMax; loc[2]++)
  {
    for (loc[1] = jMin; loc[1] <= jMax; loc[1]++)
    {
      for (loc[0] = iMin; loc[0] <= iMax; loc[0]++)
      {
        idx[npts] = loc[0] + loc[1] * dims[0] + loc[2] * d01;
        npts++;
      }
    }
  }

  for (int i = 0; i < npts; i++)
  {
    if (!this->IsPointVisible(idx[i]))
    {
      return 0;
    }
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkUniformGrid::GetCellDims(int cellDims[3])
{
  int nodeDims[3];
  this->GetDimensions(nodeDims);
  for (int i = 0; i < 3; ++i)
  {
    cellDims[i] = ((nodeDims[i] - 1) < 1) ? 1 : nodeDims[i] - 1;
  }
}

//----------------------------------------------------------------------------
svtkUniformGrid* svtkUniformGrid::GetData(svtkInformation* info)
{
  return info ? svtkUniformGrid::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkUniformGrid* svtkUniformGrid::GetData(svtkInformationVector* v, int i)
{
  return svtkUniformGrid::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
bool svtkUniformGrid::HasAnyBlankPoints()
{
  return IsAnyBitSet(this->GetPointGhostArray(), svtkDataSetAttributes::HIDDENPOINT);
}

//----------------------------------------------------------------------------
bool svtkUniformGrid::HasAnyBlankCells()
{
  int cellBlanking = IsAnyBitSet(this->GetCellGhostArray(), svtkDataSetAttributes::HIDDENCELL);
  return cellBlanking || this->HasAnyBlankPoints();
}
