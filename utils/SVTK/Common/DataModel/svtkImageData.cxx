/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImageData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkImageData.h"

#include "svtkCellData.h"
#include "svtkDataArray.h"
#include "svtkGenericCell.h"
#include "svtkInformation.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationVector.h"
#include "svtkLargeInteger.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkMatrix3x3.h"
#include "svtkMatrix4x4.h"
#include "svtkObjectFactory.h"
#include "svtkPixel.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkVertex.h"
#include "svtkVoxel.h"

svtkStandardNewMacro(svtkImageData);

//----------------------------------------------------------------------------
svtkImageData::svtkImageData()
{
  int idx;

  this->Vertex = nullptr;
  this->Line = nullptr;
  this->Pixel = nullptr;
  this->Voxel = nullptr;

  this->DataDescription = SVTK_EMPTY;

  for (idx = 0; idx < 3; ++idx)
  {
    this->Dimensions[idx] = 0;
    this->Increments[idx] = 0;
    this->Origin[idx] = 0.0;
    this->Spacing[idx] = 1.0;
    this->Point[idx] = 0.0;
  }

  this->DirectionMatrix = svtkMatrix3x3::New();
  this->IndexToPhysicalMatrix = svtkMatrix4x4::New();
  this->PhysicalToIndexMatrix = svtkMatrix4x4::New();
  this->DirectionMatrix->Identity();
  this->ComputeTransforms();

  int extent[6] = { 0, -1, 0, -1, 0, -1 };
  memcpy(this->Extent, extent, 6 * sizeof(int));

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_3D_EXTENT);
  this->Information->Set(svtkDataObject::DATA_EXTENT(), this->Extent, 6);
}

//----------------------------------------------------------------------------
svtkImageData::~svtkImageData()
{
  if (this->Vertex)
  {
    this->Vertex->Delete();
  }
  if (this->Line)
  {
    this->Line->Delete();
  }
  if (this->Pixel)
  {
    this->Pixel->Delete();
  }
  if (this->Voxel)
  {
    this->Voxel->Delete();
  }
  if (this->DirectionMatrix)
  {
    this->DirectionMatrix->Delete();
  }
  if (this->IndexToPhysicalMatrix)
  {
    this->IndexToPhysicalMatrix->Delete();
  }
  if (this->PhysicalToIndexMatrix)
  {
    this->PhysicalToIndexMatrix->Delete();
  }
}

//----------------------------------------------------------------------------
// Copy the geometric and topological structure of an input structured points
// object.
void svtkImageData::CopyStructure(svtkDataSet* ds)
{
  svtkImageData* sPts = static_cast<svtkImageData*>(ds);
  this->Initialize();

  int i;
  for (i = 0; i < 3; i++)
  {
    this->Dimensions[i] = sPts->Dimensions[i];
    this->Spacing[i] = sPts->Spacing[i];
    this->Origin[i] = sPts->Origin[i];
  }
  this->DirectionMatrix->DeepCopy(sPts->GetDirectionMatrix());
  this->ComputeTransforms();
  this->SetExtent(sPts->GetExtent());
}

//----------------------------------------------------------------------------
void svtkImageData::Initialize()
{
  this->Superclass::Initialize();
  if (this->Information)
  {
    this->SetDimensions(0, 0, 0);
  }
}

//----------------------------------------------------------------------------
void svtkImageData::CopyInformationFromPipeline(svtkInformation* information)
{
  // Let the superclass copy whatever it wants.
  this->Superclass::CopyInformationFromPipeline(information);

  // Copy origin and spacing from pipeline information to the internal
  // copies.
  if (information->Has(SPACING()))
  {
    this->SetSpacing(information->Get(SPACING()));
  }
  if (information->Has(ORIGIN()))
  {
    this->SetOrigin(information->Get(ORIGIN()));
  }
  if (information->Has(DIRECTION()))
  {
    this->SetDirectionMatrix(information->Get(DIRECTION()));
  }
}

//----------------------------------------------------------------------------
void svtkImageData::CopyInformationToPipeline(svtkInformation* info)
{
  // Let the superclass copy information to the pipeline.
  this->Superclass::CopyInformationToPipeline(info);

  // Copy the spacing, origin, direction, and scalar info
  info->Set(svtkDataObject::SPACING(), this->Spacing, 3);
  info->Set(svtkDataObject::ORIGIN(), this->Origin, 3);
  info->Set(svtkDataObject::DIRECTION(), this->DirectionMatrix->GetData(), 9);
  svtkDataObject::SetPointDataActiveScalarInfo(
    info, this->GetScalarType(), this->GetNumberOfScalarComponents());
}

//----------------------------------------------------------------------------
// Graphics filters reallocate every execute.  Image filters try to reuse
// the scalars.
void svtkImageData::PrepareForNewData()
{
  // free everything but the scalars
  svtkDataArray* scalars = this->GetPointData()->GetScalars();
  if (scalars)
  {
    scalars->Register(this);
  }
  this->Initialize();
  if (scalars)
  {
    this->GetPointData()->SetScalars(scalars);
    scalars->UnRegister(this);
  }
}

//----------------------------------------------------------------------------
template <class T>
unsigned long svtkImageDataGetTypeSize(T*)
{
  return sizeof(T);
}

//----------------------------------------------------------------------------
svtkCell* svtkImageData::GetCellTemplateForDataDescription()
{
  svtkCell* cell = nullptr;
  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      break;

    case SVTK_SINGLE_POINT:
      cell = this->Vertex;
      break;

    case SVTK_X_LINE:
    case SVTK_Y_LINE:
    case SVTK_Z_LINE:
      cell = this->Line;
      break;

    case SVTK_XY_PLANE:
    case SVTK_YZ_PLANE:
    case SVTK_XZ_PLANE:
      cell = this->Pixel;
      break;

    case SVTK_XYZ_GRID:
      cell = this->Voxel;
      break;

    default:
      svtkErrorMacro("Invalid DataDescription.");
      break;
  }
  return cell;
}

//----------------------------------------------------------------------------
bool svtkImageData::GetCellTemplateForDataDescription(svtkGenericCell* cell)
{
  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      cell->SetCellTypeToEmptyCell();
      break;

    case SVTK_SINGLE_POINT:
      cell->SetCellTypeToVertex();
      break;

    case SVTK_X_LINE:
    case SVTK_Y_LINE:
    case SVTK_Z_LINE:
      cell->SetCellTypeToLine();
      break;

    case SVTK_XY_PLANE:
    case SVTK_YZ_PLANE:
    case SVTK_XZ_PLANE:
      cell->SetCellTypeToPixel();
      break;

    case SVTK_XYZ_GRID:
      cell->SetCellTypeToVoxel();
      break;

    default:
      svtkErrorMacro("Invalid DataDescription.");
      return false;
  }
  return true;
}

//----------------------------------------------------------------------------
bool svtkImageData::GetIJKMinForCellId(svtkIdType cellId, int ijkMin[3])
{
  svtkIdType dims[3];
  this->GetDimensions(dims);

  ijkMin[0] = ijkMin[1] = ijkMin[2] = 0;

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a cell from an empty image.");
    return false;
  }

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return false;

    case SVTK_SINGLE_POINT:
      // cellId can only be = 0
      break;

    case SVTK_X_LINE:
      ijkMin[0] = cellId;
      break;

    case SVTK_Y_LINE:
      ijkMin[1] = cellId;
      break;

    case SVTK_Z_LINE:
      ijkMin[2] = cellId;
      break;

    case SVTK_XY_PLANE:
      ijkMin[0] = cellId % (dims[0] - 1);
      ijkMin[1] = cellId / (dims[0] - 1);
      break;

    case SVTK_YZ_PLANE:
      ijkMin[1] = cellId % (dims[1] - 1);
      ijkMin[2] = cellId / (dims[1] - 1);
      break;

    case SVTK_XZ_PLANE:
      ijkMin[0] = cellId % (dims[0] - 1);
      ijkMin[2] = cellId / (dims[0] - 1);
      break;

    case SVTK_XYZ_GRID:
      ijkMin[0] = cellId % (dims[0] - 1);
      ijkMin[1] = (cellId / (dims[0] - 1)) % (dims[1] - 1);
      ijkMin[2] = cellId / ((dims[0] - 1) * (dims[1] - 1));
      break;

    default:
      svtkErrorMacro("Invalid DataDescription.");
      return false;
  }
  return true;
}

//----------------------------------------------------------------------------
bool svtkImageData::GetIJKMaxForIJKMin(int ijkMin[3], int ijkMax[3])
{
  svtkIdType dims[3];
  this->GetDimensions(dims);

  ijkMax[0] = ijkMax[1] = ijkMax[2] = 0;

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a cell from an empty image.");
    return false;
  }

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return false;

    case SVTK_SINGLE_POINT:
      // cellId can only be = 0
      break;

    case SVTK_X_LINE:
      ijkMax[0] = ijkMin[0] + 1;
      break;

    case SVTK_Y_LINE:
      ijkMax[1] = ijkMin[1] + 1;
      break;

    case SVTK_Z_LINE:
      ijkMax[2] = ijkMin[2] + 1;
      break;

    case SVTK_XY_PLANE:
      ijkMax[0] = ijkMin[0] + 1;
      ijkMax[1] = ijkMin[1] + 1;
      break;

    case SVTK_YZ_PLANE:
      ijkMax[1] = ijkMin[1] + 1;
      ijkMax[2] = ijkMin[2] + 1;
      break;

    case SVTK_XZ_PLANE:
      ijkMax[0] = ijkMin[0] + 1;
      ijkMax[2] = ijkMin[2] + 1;
      break;

    case SVTK_XYZ_GRID:
      ijkMax[0] = ijkMin[0] + 1;
      ijkMax[1] = ijkMin[1] + 1;
      ijkMax[2] = ijkMin[2] + 1;
      break;

    default:
      svtkErrorMacro("Invalid DataDescription.");
      return false;
  }
  return true;
}

//----------------------------------------------------------------------------
void svtkImageData::AddPointsToCellTemplate(svtkCell* cell, int ijkMin[3], int ijkMax[3])
{
  int loc[3], i, j, k;
  svtkIdType idx, npts;
  double xyz[3];
  const int* extent = this->Extent;

  svtkIdType dims[3];
  this->GetDimensions(dims);
  svtkIdType d01 = dims[0] * dims[1];

  // Extract point coordinates and point ids
  // Ids are relative to extent min.
  npts = 0;
  for (loc[2] = ijkMin[2]; loc[2] <= ijkMax[2]; loc[2]++)
  {
    k = loc[2] + extent[4];
    for (loc[1] = ijkMin[1]; loc[1] <= ijkMax[1]; loc[1]++)
    {
      j = loc[1] + extent[2];
      for (loc[0] = ijkMin[0]; loc[0] <= ijkMax[0]; loc[0]++)
      {
        i = loc[0] + extent[0];
        this->TransformIndexToPhysicalPoint(i, j, k, xyz);

        idx = loc[0] + loc[1] * dims[0] + loc[2] * d01;
        cell->PointIds->SetId(npts, idx);
        cell->Points->SetPoint(npts++, xyz);
      }
    }
  }
}

//----------------------------------------------------------------------------
svtkCell* svtkImageData::GetCell(svtkIdType cellId)
{
  int ijkMin[3];
  if (!this->GetIJKMinForCellId(cellId, ijkMin))
  {
    return nullptr;
  }

  // Need to use vtImageData:: to avoid calling child classes implementation
  return this->svtkImageData::GetCell(ijkMin[0], ijkMin[1], ijkMin[2]);
}

//----------------------------------------------------------------------------
svtkCell* svtkImageData::GetCell(int iMin, int jMin, int kMin)
{
  svtkCell* cell = this->GetCellTemplateForDataDescription();
  if (cell == nullptr)
  {
    return nullptr;
  }

  int ijkMin[3] = { iMin, jMin, kMin };
  int ijkMax[3];
  if (!this->GetIJKMaxForIJKMin(ijkMin, ijkMax))
  {
    return nullptr;
  }

  this->AddPointsToCellTemplate(cell, ijkMin, ijkMax);
  return cell;
}

//----------------------------------------------------------------------------
void svtkImageData::GetCell(svtkIdType cellId, svtkGenericCell* cell)
{
  if (!this->GetCellTemplateForDataDescription(cell))
  {
    cell->SetCellTypeToEmptyCell();
    return;
  }

  int ijkMin[3];
  if (!this->GetIJKMinForCellId(cellId, ijkMin))
  {
    cell->SetCellTypeToEmptyCell();
    return;
  }

  int ijkMax[3];
  if (!this->GetIJKMaxForIJKMin(ijkMin, ijkMax))
  {
    cell->SetCellTypeToEmptyCell();
    return;
  }

  this->AddPointsToCellTemplate(cell, ijkMin, ijkMax);
}

//----------------------------------------------------------------------------
// Fast implementation of GetCellBounds().  Bounds are calculated without
// constructing a cell.
void svtkImageData::GetCellBounds(svtkIdType cellId, double bounds[6])
{
  int ijkMin[3];
  if (!this->GetIJKMinForCellId(cellId, ijkMin))
  {
    bounds[0] = bounds[1] = bounds[2] = bounds[3] = bounds[4] = bounds[5] = 0.0;
    return;
  }

  int ijkMax[3];
  if (!this->GetIJKMaxForIJKMin(ijkMin, ijkMax))
  {
    bounds[0] = bounds[1] = bounds[2] = bounds[3] = bounds[4] = bounds[5] = 0.0;
    return;
  }

  int loc[3], i, j, k;
  double xyz[3];
  const int* extent = this->Extent;

  // Compute the bounds
  if (ijkMax[2] >= ijkMin[2] && ijkMax[1] >= ijkMin[1] && ijkMax[0] >= ijkMin[0])
  {
    bounds[0] = bounds[2] = bounds[4] = SVTK_DOUBLE_MAX;
    bounds[1] = bounds[3] = bounds[5] = SVTK_DOUBLE_MIN;

    for (loc[2] = ijkMin[2]; loc[2] <= ijkMax[2]; loc[2]++)
    {
      k = loc[2] + extent[4];
      for (loc[1] = ijkMin[1]; loc[1] <= ijkMax[1]; loc[1]++)
      {
        j = loc[1] + extent[2];
        for (loc[0] = ijkMin[0]; loc[0] <= ijkMax[0]; loc[0]++)
        {
          i = loc[0] + extent[0];
          this->TransformIndexToPhysicalPoint(i, j, k, xyz);

          bounds[0] = (xyz[0] < bounds[0] ? xyz[0] : bounds[0]);
          bounds[1] = (xyz[0] > bounds[1] ? xyz[0] : bounds[1]);
          bounds[2] = (xyz[1] < bounds[2] ? xyz[1] : bounds[2]);
          bounds[3] = (xyz[1] > bounds[3] ? xyz[1] : bounds[3]);
          bounds[4] = (xyz[2] < bounds[4] ? xyz[2] : bounds[4]);
          bounds[5] = (xyz[2] > bounds[5] ? xyz[2] : bounds[5]);
        }
      }
    }
  }
  else
  {
    svtkMath::UninitializeBounds(bounds);
  }
}

//----------------------------------------------------------------------------
void svtkImageData::GetPoint(svtkIdType ptId, double x[3])
{
  const int* extent = this->Extent;

  svtkIdType dims[3];
  this->GetDimensions(dims);

  x[0] = x[1] = x[2] = 0.0;
  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0)
  {
    svtkErrorMacro("Requesting a point from an empty image.");
    return;
  }

  // "loc" holds the point x,y,z indices
  int loc[3];
  loc[0] = loc[1] = loc[2] = 0;

  switch (this->DataDescription)
  {
    case SVTK_EMPTY:
      return;

    case SVTK_SINGLE_POINT:
      break;

    case SVTK_X_LINE:
      loc[0] = ptId;
      break;

    case SVTK_Y_LINE:
      loc[1] = ptId;
      break;

    case SVTK_Z_LINE:
      loc[2] = ptId;
      break;

    case SVTK_XY_PLANE:
      loc[0] = ptId % dims[0];
      loc[1] = ptId / dims[0];
      break;

    case SVTK_YZ_PLANE:
      loc[1] = ptId % dims[1];
      loc[2] = ptId / dims[1];
      break;

    case SVTK_XZ_PLANE:
      loc[0] = ptId % dims[0];
      loc[2] = ptId / dims[0];
      break;

    case SVTK_XYZ_GRID:
      loc[0] = ptId % dims[0];
      loc[1] = (ptId / dims[0]) % dims[1];
      loc[2] = ptId / (dims[0] * dims[1]);
      break;
  }

  int i, j, k;
  i = loc[0] + extent[0];
  j = loc[1] + extent[2];
  k = loc[2] + extent[4];
  this->TransformIndexToPhysicalPoint(i, j, k, x);
}

//----------------------------------------------------------------------------
svtkIdType svtkImageData::FindPoint(double x[3])
{
  //
  //  Ensure valid spacing
  //
  const double* spacing = this->Spacing;
  svtkIdType dims[3];
  this->GetDimensions(dims);
  std::string ijkLabels[3] = { "I", "J", "K" };
  for (int i = 0; i < 3; i++)
  {
    if (spacing[i] == 0.0 && dims[i] > 1)
    {
      svtkWarningMacro("Spacing along the " << ijkLabels[i] << " axis is 0.");
      return -1;
    }
  }

  //
  //  Compute the ijk location
  //
  const int* extent = this->Extent;
  int loc[3];
  double ijk[3];
  this->TransformPhysicalPointToContinuousIndex(x, ijk);
  loc[0] = svtkMath::Floor(ijk[0] + 0.5);
  loc[1] = svtkMath::Floor(ijk[1] + 0.5);
  loc[2] = svtkMath::Floor(ijk[2] + 0.5);
  if (loc[0] < extent[0] || loc[0] > extent[1] || loc[1] < extent[2] || loc[1] > extent[3] ||
    loc[2] < extent[4] || loc[2] > extent[5])
  {
    return -1;
  }
  // since point id is relative to the first point actually stored
  loc[0] -= extent[0];
  loc[1] -= extent[2];
  loc[2] -= extent[4];

  //
  //  From this location get the point id
  //
  return loc[2] * dims[0] * dims[1] + loc[1] * dims[0] + loc[0];
}

//----------------------------------------------------------------------------
svtkIdType svtkImageData::FindCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkGenericCell* svtkNotUsed(gencell), svtkIdType svtkNotUsed(cellId), double tol2, int& subId,
  double pcoords[3], double* weights)
{
  return this->FindCell(x, nullptr, 0, tol2, subId, pcoords, weights);
}

//----------------------------------------------------------------------------
svtkIdType svtkImageData::FindCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkIdType svtkNotUsed(cellId), double tol2, int& subId, double pcoords[3], double* weights)
{
  int idx[3];

  // Compute the voxel index
  if (this->ComputeStructuredCoordinates(x, idx, pcoords) == 0)
  {
    // If voxel index is out of bounds, check point "x" against the
    // bounds to see if within tolerance of the bounds.
    const int* extent = this->Extent;
    const double* spacing = this->Spacing;

    // Compute squared distance of point x from the boundary
    double dist2 = 0.0;

    for (int i = 0; i < 3; i++)
    {
      int minIdx = extent[i * 2];
      int maxIdx = extent[i * 2 + 1];

      if (idx[i] < minIdx)
      {
        double dist = (idx[i] + pcoords[i] - minIdx) * spacing[i];
        idx[i] = minIdx;
        pcoords[i] = 0.0;
        dist2 += dist * dist;
      }
      else if (idx[i] >= maxIdx)
      {
        double dist = (idx[i] + pcoords[i] - maxIdx) * spacing[i];
        if (maxIdx == minIdx)
        {
          idx[i] = minIdx;
          pcoords[i] = 0.0;
        }
        else
        {
          idx[i] = maxIdx - 1;
          pcoords[i] = 1.0;
        }
        dist2 += dist * dist;
      }
    }

    // Check squared distance against the tolerance
    if (dist2 > tol2)
    {
      return -1;
    }
  }

  if (weights)
  {
    // Shift parametric coordinates for XZ/YZ planes
    if (this->DataDescription == SVTK_XZ_PLANE)
    {
      pcoords[1] = pcoords[2];
      pcoords[2] = 0.0;
    }
    else if (this->DataDescription == SVTK_YZ_PLANE)
    {
      pcoords[0] = pcoords[1];
      pcoords[1] = pcoords[2];
      pcoords[2] = 0.0;
    }
    else if (this->DataDescription == SVTK_XY_PLANE)
    {
      pcoords[2] = 0.0;
    }
    svtkVoxel::InterpolationFunctions(pcoords, weights);
  }

  //
  //  From this location get the cell id
  //
  subId = 0;
  return this->ComputeCellId(idx);
}

//----------------------------------------------------------------------------
svtkCell* svtkImageData::FindAndGetCell(double x[3], svtkCell* svtkNotUsed(cell),
  svtkIdType svtkNotUsed(cellId), double tol2, int& subId, double pcoords[3], double* weights)
{
  svtkIdType cellId = this->FindCell(x, nullptr, 0, tol2, subId, pcoords, nullptr);

  if (cellId < 0)
  {
    return nullptr;
  }

  svtkCell* cell = this->GetCell(cellId);
  cell->InterpolateFunctions(pcoords, weights);

  return cell;
}

//----------------------------------------------------------------------------
int svtkImageData::GetCellType(svtkIdType svtkNotUsed(cellId))
{
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
      return SVTK_PIXEL;

    case SVTK_XYZ_GRID:
      return SVTK_VOXEL;

    default:
      svtkErrorMacro(<< "Bad data description!");
      return SVTK_EMPTY_CELL;
  }
}

//----------------------------------------------------------------------------
void svtkImageData::ComputeBounds()
{
  if (this->GetMTime() <= this->ComputeTime)
  {
    return;
  }
  const int* extent = this->Extent;

  if (extent[0] > extent[1] || extent[2] > extent[3] || extent[4] > extent[5])
  {
    svtkMath::UninitializeBounds(this->Bounds);
  }
  else
  {
    if (this->DirectionMatrix->IsIdentity())
    {
      // Direction is identity: bounds are easy to compute
      // with only origin and spacing
      const double* origin = this->Origin;
      const double* spacing = this->Spacing;
      int swapXBounds = (spacing[0] < 0); // 1 if true, 0 if false
      int swapYBounds = (spacing[1] < 0); // 1 if true, 0 if false
      int swapZBounds = (spacing[2] < 0); // 1 if true, 0 if false

      this->Bounds[0] = origin[0] + (extent[0 + swapXBounds] * spacing[0]);
      this->Bounds[2] = origin[1] + (extent[2 + swapYBounds] * spacing[1]);
      this->Bounds[4] = origin[2] + (extent[4 + swapZBounds] * spacing[2]);

      this->Bounds[1] = origin[0] + (extent[1 - swapXBounds] * spacing[0]);
      this->Bounds[3] = origin[1] + (extent[3 - swapYBounds] * spacing[1]);
      this->Bounds[5] = origin[2] + (extent[5 - swapZBounds] * spacing[2]);
    }
    else
    {
      // Direction isn't identity: use IndexToPhysical matrix
      // to determine the position of the dataset corners
      int iMin, iMax, jMin, jMax, kMin, kMax;
      iMin = extent[0];
      iMax = extent[1];
      jMin = extent[2];
      jMax = extent[3];
      kMin = extent[4];
      kMax = extent[5];
      int ijkCorners[8][3] = {
        { iMin, jMin, kMin },
        { iMax, jMin, kMin },
        { iMin, jMax, kMin },
        { iMax, jMax, kMin },
        { iMin, jMin, kMax },
        { iMax, jMin, kMax },
        { iMin, jMax, kMax },
        { iMax, jMax, kMax },
      };

      double xyz[3];
      double xMin, xMax, yMin, yMax, zMin, zMax;
      xMin = yMin = zMin = SVTK_DOUBLE_MAX;
      xMax = yMax = zMax = SVTK_DOUBLE_MIN;
      for (int* ijkCorner : ijkCorners)
      {
        this->TransformIndexToPhysicalPoint(ijkCorner, xyz);
        if (xyz[0] < xMin)
          xMin = xyz[0];
        if (xyz[0] > xMax)
          xMax = xyz[0];
        if (xyz[1] < yMin)
          yMin = xyz[1];
        if (xyz[1] > yMax)
          yMax = xyz[1];
        if (xyz[2] < zMin)
          zMin = xyz[2];
        if (xyz[2] > zMax)
          zMax = xyz[2];
      }
      this->Bounds[0] = xMin;
      this->Bounds[1] = xMax;
      this->Bounds[2] = yMin;
      this->Bounds[3] = yMax;
      this->Bounds[4] = zMin;
      this->Bounds[5] = zMax;
    }
  }
  this->ComputeTime.Modified();
}

//----------------------------------------------------------------------------
// Given structured coordinates (i,j,k) for a voxel cell, compute the eight
// gradient values for the voxel corners. The order in which the gradient
// vectors are arranged corresponds to the ordering of the voxel points.
// Gradient vector is computed by central differences (except on edges of
// volume where forward difference is used). The scalars s are the scalars
// from which the gradient is to be computed. This method will treat
// only 3D structured point datasets (i.e., volumes).
void svtkImageData::GetVoxelGradient(int i, int j, int k, svtkDataArray* s, svtkDataArray* g)
{
  double gv[3];
  int ii, jj, kk, idx = 0;

  for (kk = 0; kk < 2; kk++)
  {
    for (jj = 0; jj < 2; jj++)
    {
      for (ii = 0; ii < 2; ii++)
      {
        this->GetPointGradient(i + ii, j + jj, k + kk, s, gv);
        g->SetTuple(idx++, gv);
      }
    }
  }
}

//----------------------------------------------------------------------------
// Given structured coordinates (i,j,k) for a point in a structured point
// dataset, compute the gradient vector from the scalar data at that point.
// The scalars s are the scalars from which the gradient is to be computed.
// This method will treat structured point datasets of any dimension.
void svtkImageData::GetPointGradient(int i, int j, int k, svtkDataArray* s, double g[3])
{
  const double* ar = this->Spacing;
  double sp, sm;
  const int* extent = this->Extent;

  svtkIdType dims[3];
  this->GetDimensions(dims);
  svtkIdType ijsize = dims[0] * dims[1];

  // Adjust i,j,k to the start of the extent
  i -= extent[0];
  j -= extent[2];
  k -= extent[4];

  // Check for out-of-bounds
  if (i < 0 || i >= dims[0] || j < 0 || j >= dims[1] || k < 0 || k >= dims[2])
  {
    g[0] = g[1] = g[2] = 0.0;
    return;
  }

  // i-axis
  if (dims[0] == 1)
  {
    g[0] = 0.0;
  }
  else if (i == 0)
  {
    sp = s->GetComponent(i + 1 + j * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    g[0] = (sm - sp) / ar[0];
  }
  else if (i == (dims[0] - 1))
  {
    sp = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i - 1 + j * dims[0] + k * ijsize, 0);
    g[0] = (sm - sp) / ar[0];
  }
  else
  {
    sp = s->GetComponent(i + 1 + j * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i - 1 + j * dims[0] + k * ijsize, 0);
    g[0] = 0.5 * (sm - sp) / ar[0];
  }

  // j-axis
  if (dims[1] == 1)
  {
    g[1] = 0.0;
  }
  else if (j == 0)
  {
    sp = s->GetComponent(i + (j + 1) * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    g[1] = (sm - sp) / ar[1];
  }
  else if (j == (dims[1] - 1))
  {
    sp = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i + (j - 1) * dims[0] + k * ijsize, 0);
    g[1] = (sm - sp) / ar[1];
  }
  else
  {
    sp = s->GetComponent(i + (j + 1) * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i + (j - 1) * dims[0] + k * ijsize, 0);
    g[1] = 0.5 * (sm - sp) / ar[1];
  }

  // k-axis
  if (dims[2] == 1)
  {
    g[2] = 0.0;
  }
  else if (k == 0)
  {
    sp = s->GetComponent(i + j * dims[0] + (k + 1) * ijsize, 0);
    sm = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    g[2] = (sm - sp) / ar[2];
  }
  else if (k == (dims[2] - 1))
  {
    sp = s->GetComponent(i + j * dims[0] + k * ijsize, 0);
    sm = s->GetComponent(i + j * dims[0] + (k - 1) * ijsize, 0);
    g[2] = (sm - sp) / ar[2];
  }
  else
  {
    sp = s->GetComponent(i + j * dims[0] + (k + 1) * ijsize, 0);
    sm = s->GetComponent(i + j * dims[0] + (k - 1) * ijsize, 0);
    g[2] = 0.5 * (sm - sp) / ar[2];
  }

  // Apply direction transform to get in xyz coordinate system
  // Note: we already applied the spacing when handling the ijk
  // axis above, and do not need to translate by the origin
  // since this is a gradient computation
  this->DirectionMatrix->MultiplyPoint(g, g);
}

//----------------------------------------------------------------------------
// Set dimensions of structured points dataset.
void svtkImageData::SetDimensions(int i, int j, int k)
{
  this->SetExtent(0, i - 1, 0, j - 1, 0, k - 1);
}

//----------------------------------------------------------------------------
// Set dimensions of structured points dataset.
void svtkImageData::SetDimensions(const int dim[3])
{
  this->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);
}

//----------------------------------------------------------------------------
// Convenience function computes the structured coordinates for a point x[3].
// The voxel is specified by the array ijk[3], and the parametric coordinates
// in the cell are specified with pcoords[3]. The function returns a 0 if the
// point x is outside of the volume, and a 1 if inside the volume.
int svtkImageData::ComputeStructuredCoordinates(const double x[3], int ijk[3], double pcoords[3])
{
  // tolerance is needed for floating points error margin
  // (this is squared tolerance)
  const double tol2 = 1e-12;

  //
  //  Compute the ijk location
  //
  double doubleLoc[3];
  this->TransformPhysicalPointToContinuousIndex(x, doubleLoc);

  const int* extent = this->Extent;
  int isInBounds = 1;
  for (int i = 0; i < 3; i++)
  {
    // Floor for negative indexes.
    ijk[i] = svtkMath::Floor(doubleLoc[i]); // integer
    pcoords[i] = doubleLoc[i] - ijk[i];    // >= 0 and < 1

    int tmpInBounds = 0;
    int minExt = extent[i * 2];
    int maxExt = extent[i * 2 + 1];

    // check if data is one pixel thick as well as
    // low boundary check
    if (minExt == maxExt || ijk[i] < minExt)
    {
      double dist = doubleLoc[i] - minExt;
      if (dist * dist <= tol2)
      {
        pcoords[i] = 0.0;
        ijk[i] = minExt;
        tmpInBounds = 1;
      }
    }

    // high boundary check
    else if (ijk[i] >= maxExt)
    {
      double dist = doubleLoc[i] - maxExt;
      if (dist * dist <= tol2)
      {
        // make sure index is within the allowed cell index range
        pcoords[i] = 1.0;
        ijk[i] = maxExt - 1;
        tmpInBounds = 1;
      }
    }

    // else index is definitely within bounds
    else
    {
      tmpInBounds = 1;
    }

    // clear isInBounds if out of bounds for this dimension
    isInBounds = (isInBounds & tmpInBounds);
  }

  return isInBounds;
}

//----------------------------------------------------------------------------
void svtkImageData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  int idx;
  const double* direction = this->GetDirectionMatrix()->GetData();
  const int* dims = this->GetDimensions();
  const int* extent = this->Extent;

  os << indent << "Spacing: (" << this->Spacing[0] << ", " << this->Spacing[1] << ", "
     << this->Spacing[2] << ")\n";
  os << indent << "Origin: (" << this->Origin[0] << ", " << this->Origin[1] << ", "
     << this->Origin[2] << ")\n";
  os << indent << "Direction: (" << direction[0];
  for (idx = 1; idx < 9; ++idx)
  {
    os << ", " << direction[idx];
  }
  os << ")\n";
  os << indent << "Dimensions: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")\n";
  os << indent << "Increments: (" << this->Increments[0] << ", " << this->Increments[1] << ", "
     << this->Increments[2] << ")\n";
  os << indent << "Extent: (" << extent[0];
  for (idx = 1; idx < 6; ++idx)
  {
    os << ", " << extent[idx];
  }
  os << ")\n";
}

//----------------------------------------------------------------------------
void svtkImageData::SetNumberOfScalarComponents(int num, svtkInformation* meta_data)
{
  svtkDataObject::SetPointDataActiveScalarInfo(meta_data, -1, num);
}

//----------------------------------------------------------------------------
bool svtkImageData::HasNumberOfScalarComponents(svtkInformation* meta_data)
{
  svtkInformation* scalarInfo = svtkDataObject::GetActiveFieldInformation(
    meta_data, FIELD_ASSOCIATION_POINTS, svtkDataSetAttributes::SCALARS);
  if (!scalarInfo)
  {
    return false;
  }
  return scalarInfo->Has(FIELD_NUMBER_OF_COMPONENTS()) != 0;
}

//----------------------------------------------------------------------------
int svtkImageData::GetNumberOfScalarComponents(svtkInformation* meta_data)
{
  svtkInformation* scalarInfo = svtkDataObject::GetActiveFieldInformation(
    meta_data, FIELD_ASSOCIATION_POINTS, svtkDataSetAttributes::SCALARS);
  if (scalarInfo && scalarInfo->Has(FIELD_NUMBER_OF_COMPONENTS()))
  {
    return scalarInfo->Get(FIELD_NUMBER_OF_COMPONENTS());
  }
  return 1;
}

//----------------------------------------------------------------------------
int svtkImageData::GetNumberOfScalarComponents()
{
  svtkDataArray* scalars = this->GetPointData()->GetScalars();
  if (scalars)
  {
    return scalars->GetNumberOfComponents();
  }
  return 1;
}

//----------------------------------------------------------------------------
svtkIdType* svtkImageData::GetIncrements()
{
  // Make sure the increments are up to date. The filter bypass and update
  // mechanism make it tricky to update the increments anywhere other than here
  this->ComputeIncrements();

  return this->Increments;
}

//----------------------------------------------------------------------------
svtkIdType* svtkImageData::GetIncrements(svtkDataArray* scalars)
{
  // Make sure the increments are up to date. The filter bypass and update
  // mechanism make it tricky to update the increments anywhere other than here
  this->ComputeIncrements(scalars);

  return this->Increments;
}

//----------------------------------------------------------------------------
void svtkImageData::GetIncrements(svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ)
{
  svtkIdType inc[3];
  this->ComputeIncrements(inc);
  incX = inc[0];
  incY = inc[1];
  incZ = inc[2];
}

//----------------------------------------------------------------------------
void svtkImageData::GetIncrements(
  svtkDataArray* scalars, svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ)
{
  svtkIdType inc[3];
  this->ComputeIncrements(scalars, inc);
  incX = inc[0];
  incY = inc[1];
  incZ = inc[2];
}

//----------------------------------------------------------------------------
void svtkImageData::GetIncrements(svtkIdType inc[3])
{
  this->ComputeIncrements(inc);
}

//----------------------------------------------------------------------------
void svtkImageData::GetIncrements(svtkDataArray* scalars, svtkIdType inc[3])
{
  this->ComputeIncrements(scalars, inc);
}

//----------------------------------------------------------------------------
void svtkImageData::GetContinuousIncrements(
  int extent[6], svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ)
{
  this->GetContinuousIncrements(this->GetPointData()->GetScalars(), extent, incX, incY, incZ);
}
//----------------------------------------------------------------------------
void svtkImageData::GetContinuousIncrements(
  svtkDataArray* scalars, int extent[6], svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ)
{
  int e0, e1, e2, e3;

  incX = 0;
  const int* selfExtent = this->Extent;

  e0 = extent[0];
  if (e0 < selfExtent[0])
  {
    e0 = selfExtent[0];
  }
  e1 = extent[1];
  if (e1 > selfExtent[1])
  {
    e1 = selfExtent[1];
  }
  e2 = extent[2];
  if (e2 < selfExtent[2])
  {
    e2 = selfExtent[2];
  }
  e3 = extent[3];
  if (e3 > selfExtent[3])
  {
    e3 = selfExtent[3];
  }

  // Make sure the increments are up to date
  svtkIdType inc[3];
  this->ComputeIncrements(scalars, inc);

  incY = inc[1] - (e1 - e0 + 1) * inc[0];
  incZ = inc[2] - (e3 - e2 + 1) * inc[1];
}

//----------------------------------------------------------------------------
// This method computes the increments from the MemoryOrder and the extent.
// This version assumes we are using the Active Scalars
void svtkImageData::ComputeIncrements(svtkIdType inc[3])
{
  this->ComputeIncrements(this->GetPointData()->GetScalars(), inc);
}

//----------------------------------------------------------------------------
// This method computes the increments from the MemoryOrder and the extent.
void svtkImageData::ComputeIncrements(svtkDataArray* scalars, svtkIdType inc[3])
{
  if (!scalars)
  {
    svtkErrorMacro("No Scalar Field has been specified - assuming 1 component!");
    this->ComputeIncrements(1, inc);
  }
  else
  {
    this->ComputeIncrements(scalars->GetNumberOfComponents(), inc);
  }
}
//----------------------------------------------------------------------------
// This method computes the increments from the MemoryOrder and the extent.
void svtkImageData::ComputeIncrements(int numberOfComponents, svtkIdType inc[3])
{
  int idx;
  svtkIdType incr = numberOfComponents;
  const int* extent = this->Extent;

  for (idx = 0; idx < 3; ++idx)
  {
    inc[idx] = incr;
    incr *= (extent[idx * 2 + 1] - extent[idx * 2] + 1);
  }
}

//----------------------------------------------------------------------------
template <class TIn, class TOut>
void svtkImageDataConvertScalar(TIn* in, TOut* out)
{
  *out = static_cast<TOut>(*in);
}

//----------------------------------------------------------------------------
double svtkImageData::GetScalarComponentAsDouble(int x, int y, int z, int comp)
{
  // Check the component index.
  if (comp < 0 || comp >= this->GetNumberOfScalarComponents())
  {
    svtkErrorMacro("Bad component index " << comp);
    return 0.0;
  }

  // Get a pointer to the scalar tuple.
  void* ptr = this->GetScalarPointer(x, y, z);
  if (!ptr)
  {
    // An error message was already generated by GetScalarPointer.
    return 0.0;
  }
  double result = 0.0;

  // Convert the scalar type.
  int scalarType = this->GetPointData()->GetScalars()->GetDataType();
  switch (scalarType)
  {
    svtkTemplateMacro(svtkImageDataConvertScalar(static_cast<SVTK_TT*>(ptr) + comp, &result));
    default:
    {
      svtkErrorMacro("Unknown Scalar type " << scalarType);
    }
  }

  return result;
}

//----------------------------------------------------------------------------
void svtkImageData::SetScalarComponentFromDouble(int x, int y, int z, int comp, double value)
{
  // Check the component index.
  if (comp < 0 || comp >= this->GetNumberOfScalarComponents())
  {
    svtkErrorMacro("Bad component index " << comp);
    return;
  }

  // Get a pointer to the scalar tuple.
  void* ptr = this->GetScalarPointer(x, y, z);
  if (!ptr)
  {
    // An error message was already generated by GetScalarPointer.
    return;
  }

  // Convert the scalar type.
  int scalarType = this->GetPointData()->GetScalars()->GetDataType();
  switch (scalarType)
  {
    svtkTemplateMacro(svtkImageDataConvertScalar(&value, static_cast<SVTK_TT*>(ptr) + comp));
    default:
    {
      svtkErrorMacro("Unknown Scalar type " << scalarType);
    }
  }
}

//----------------------------------------------------------------------------
float svtkImageData::GetScalarComponentAsFloat(int x, int y, int z, int comp)
{
  return this->GetScalarComponentAsDouble(x, y, z, comp);
}

//----------------------------------------------------------------------------
void svtkImageData::SetScalarComponentFromFloat(int x, int y, int z, int comp, float value)
{
  this->SetScalarComponentFromDouble(x, y, z, comp, value);
}

//----------------------------------------------------------------------------
// This Method returns a pointer to a location in the svtkImageData.
// Coordinates are in pixel units and are relative to the whole
// image origin.
void* svtkImageData::GetScalarPointer(int x, int y, int z)
{
  int tmp[3];
  tmp[0] = x;
  tmp[1] = y;
  tmp[2] = z;
  return this->GetScalarPointer(tmp);
}

//----------------------------------------------------------------------------
// This Method returns a pointer to a location in the svtkImageData.
// Coordinates are in pixel units and are relative to the whole
// image origin.
void* svtkImageData::GetScalarPointerForExtent(int extent[6])
{
  int tmp[3];
  tmp[0] = extent[0];
  tmp[1] = extent[2];
  tmp[2] = extent[4];
  return this->GetScalarPointer(tmp);
}

//----------------------------------------------------------------------------
void* svtkImageData::GetScalarPointer(int coordinate[3])
{
  svtkDataArray* scalars = this->GetPointData()->GetScalars();

  // Make sure the array has been allocated.
  if (scalars == nullptr)
  {
    // svtkDebugMacro("Allocating scalars in ImageData");
    // abort();
    // this->AllocateScalars();
    // scalars = this->PointData->GetScalars();
    return nullptr;
  }

  const int* extent = this->Extent;
  // error checking: since most access will be from pointer arithmetic.
  // this should not waste much time.
  for (int idx = 0; idx < 3; ++idx)
  {
    if (coordinate[idx] < extent[idx * 2] || coordinate[idx] > extent[idx * 2 + 1])
    {
      svtkErrorMacro(<< "GetScalarPointer: Pixel (" << coordinate[0] << ", " << coordinate[1] << ", "
                    << coordinate[2] << ") not in memory.\n Current extent= (" << extent[0] << ", "
                    << extent[1] << ", " << extent[2] << ", " << extent[3] << ", " << extent[4]
                    << ", " << extent[5] << ")");
      return nullptr;
    }
  }

  return this->GetArrayPointer(scalars, coordinate);
}

//----------------------------------------------------------------------------
// This method returns a pointer to the origin of the svtkImageData.
void* svtkImageData::GetScalarPointer()
{
  if (this->PointData->GetScalars() == nullptr)
  {
    // svtkDebugMacro("Allocating scalars in ImageData");
    // abort();
    // this->AllocateScalars();
    return nullptr;
  }
  return this->PointData->GetScalars()->GetVoidPointer(0);
}

//----------------------------------------------------------------------------
void svtkImageData::SetScalarType(int type, svtkInformation* meta_data)
{
  svtkDataObject::SetPointDataActiveScalarInfo(meta_data, type, -1);
}

//----------------------------------------------------------------------------
int svtkImageData::GetScalarType()
{
  svtkDataArray* scalars = this->GetPointData()->GetScalars();
  if (!scalars)
  {
    return SVTK_DOUBLE;
  }
  return scalars->GetDataType();
}

//----------------------------------------------------------------------------
bool svtkImageData::HasScalarType(svtkInformation* meta_data)
{
  svtkInformation* scalarInfo = svtkDataObject::GetActiveFieldInformation(
    meta_data, FIELD_ASSOCIATION_POINTS, svtkDataSetAttributes::SCALARS);
  if (!scalarInfo)
  {
    return false;
  }

  return scalarInfo->Has(FIELD_ARRAY_TYPE()) != 0;
}

//----------------------------------------------------------------------------
int svtkImageData::GetScalarType(svtkInformation* meta_data)
{
  svtkInformation* scalarInfo = svtkDataObject::GetActiveFieldInformation(
    meta_data, FIELD_ASSOCIATION_POINTS, svtkDataSetAttributes::SCALARS);
  if (scalarInfo)
  {
    return scalarInfo->Get(FIELD_ARRAY_TYPE());
  }
  return SVTK_DOUBLE;
}

//----------------------------------------------------------------------------
void svtkImageData::AllocateScalars(svtkInformation* pipeline_info)
{
  int newType = SVTK_DOUBLE;
  int newNumComp = 1;

  if (pipeline_info)
  {
    svtkInformation* scalarInfo = svtkDataObject::GetActiveFieldInformation(
      pipeline_info, FIELD_ASSOCIATION_POINTS, svtkDataSetAttributes::SCALARS);
    if (scalarInfo)
    {
      newType = scalarInfo->Get(FIELD_ARRAY_TYPE());
      if (scalarInfo->Has(FIELD_NUMBER_OF_COMPONENTS()))
      {
        newNumComp = scalarInfo->Get(FIELD_NUMBER_OF_COMPONENTS());
      }
    }
  }

  this->AllocateScalars(newType, newNumComp);
}

//----------------------------------------------------------------------------
void svtkImageData::AllocateScalars(int dataType, int numComponents)
{
  svtkDataArray* scalars;

  // if the scalar type has not been set then we have a problem
  if (dataType == SVTK_VOID)
  {
    svtkErrorMacro("Attempt to allocate scalars before scalar type was set!.");
    return;
  }

  const int* extent = this->Extent;
  // Use svtkIdType to avoid overflow on large images
  svtkIdType dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
  svtkIdType imageSize = dims[0] * dims[1] * dims[2];

  // if we currently have scalars then just adjust the size
  scalars = this->PointData->GetScalars();
  if (scalars && scalars->GetDataType() == dataType && scalars->GetReferenceCount() == 1)
  {
    scalars->SetNumberOfComponents(numComponents);
    scalars->SetNumberOfTuples(imageSize);
    // Since the execute method will be modifying the scalars
    // directly.
    scalars->Modified();
    return;
  }

  // allocate the new scalars
  scalars = svtkDataArray::CreateDataArray(dataType);
  scalars->SetNumberOfComponents(numComponents);
  scalars->SetName("ImageScalars");

  // allocate enough memory
  scalars->SetNumberOfTuples(imageSize);

  this->PointData->SetScalars(scalars);
  scalars->Delete();
}

//----------------------------------------------------------------------------
int svtkImageData::GetScalarSize(svtkInformation* meta_data)
{
  return svtkDataArray::GetDataTypeSize(this->GetScalarType(meta_data));
}

int svtkImageData::GetScalarSize()
{
  svtkDataArray* scalars = this->GetPointData()->GetScalars();
  if (!scalars)
  {
    return svtkDataArray::GetDataTypeSize(SVTK_DOUBLE);
  }
  return svtkDataArray::GetDataTypeSize(scalars->GetDataType());
}

//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
template <class IT, class OT>
void svtkImageDataCastExecute(
  svtkImageData* inData, IT* inPtr, svtkImageData* outData, OT* outPtr, int outExt[6])
{
  int idxR, idxY, idxZ;
  int maxY, maxZ;
  svtkIdType inIncX, inIncY, inIncZ;
  svtkIdType outIncX, outIncY, outIncZ;
  int rowLength;

  // find the region to loop over
  rowLength = (outExt[1] - outExt[0] + 1) * inData->GetNumberOfScalarComponents();
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];

  // Get increments to march through data
  inData->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);

  // Loop through output pixels
  for (idxZ = 0; idxZ <= maxZ; idxZ++)
  {
    for (idxY = 0; idxY <= maxY; idxY++)
    {
      for (idxR = 0; idxR < rowLength; idxR++)
      {
        // Pixel operation
        *outPtr = static_cast<OT>(*inPtr);
        outPtr++;
        inPtr++;
      }
      outPtr += outIncY;
      inPtr += inIncY;
    }
    outPtr += outIncZ;
    inPtr += inIncZ;
  }
}

//----------------------------------------------------------------------------
template <class T>
void svtkImageDataCastExecute(svtkImageData* inData, T* inPtr, svtkImageData* outData, int outExt[6])
{
  void* outPtr = outData->GetScalarPointerForExtent(outExt);

  if (outPtr == nullptr)
  {
    svtkGenericWarningMacro("Scalars not allocated.");
    return;
  }

  int scalarType = outData->GetPointData()->GetScalars()->GetDataType();
  switch (scalarType)
  {
    svtkTemplateMacro(svtkImageDataCastExecute(
      inData, static_cast<T*>(inPtr), outData, static_cast<SVTK_TT*>(outPtr), outExt));
    default:
      svtkGenericWarningMacro("Execute: Unknown output ScalarType");
      return;
  }
}

//----------------------------------------------------------------------------
// This method is passed a input and output region, and executes the filter
// algorithm to fill the output from the input.
// It just executes a switch statement to call the correct function for
// the regions data types.
void svtkImageData::CopyAndCastFrom(svtkImageData* inData, int extent[6])
{
  void* inPtr = inData->GetScalarPointerForExtent(extent);

  if (inPtr == nullptr)
  {
    svtkErrorMacro("Scalars not allocated.");
    return;
  }

  int scalarType = inData->GetPointData()->GetScalars()->GetDataType();
  switch (scalarType)
  {
    svtkTemplateMacro(svtkImageDataCastExecute(inData, static_cast<SVTK_TT*>(inPtr), this, extent));
    default:
      svtkErrorMacro(<< "Execute: Unknown input ScalarType");
      return;
  }
}

//----------------------------------------------------------------------------
void svtkImageData::Crop(const int* updateExtent)
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

  int nExt[6];
  int idxX, idxY, idxZ;
  int maxX, maxY, maxZ;
  svtkIdType outId, inId, inIdY, inIdZ, incZ, incY;
  svtkImageData* newImage;
  svtkIdType numPts, numCells, tmp;
  const int* extent = this->Extent;

  // If extents already match, then we need to do nothing.
  if (extent[0] == updateExtent[0] && extent[1] == updateExtent[1] &&
    extent[2] == updateExtent[2] && extent[3] == updateExtent[3] && extent[4] == updateExtent[4] &&
    extent[5] == updateExtent[5])
  {
    return;
  }

  // Take the intersection of the two extent so that
  // we are not asking for more than the extent.
  memcpy(nExt, updateExtent, 6 * sizeof(int));
  if (nExt[0] < extent[0])
  {
    nExt[0] = extent[0];
  }
  if (nExt[1] > extent[1])
  {
    nExt[1] = extent[1];
  }
  if (nExt[2] < extent[2])
  {
    nExt[2] = extent[2];
  }
  if (nExt[3] > extent[3])
  {
    nExt[3] = extent[3];
  }
  if (nExt[4] < extent[4])
  {
    nExt[4] = extent[4];
  }
  if (nExt[5] > extent[5])
  {
    nExt[5] = extent[5];
  }

  // If the extents are the same just return.
  if (extent[0] == nExt[0] && extent[1] == nExt[1] && extent[2] == nExt[2] &&
    extent[3] == nExt[3] && extent[4] == nExt[4] && extent[5] == nExt[5])
  {
    svtkDebugMacro("Extents already match.");
    return;
  }

  // How many point/cells.
  numPts = (nExt[1] - nExt[0] + 1) * (nExt[3] - nExt[2] + 1) * (nExt[5] - nExt[4] + 1);
  // Conditional are to handle 3d, 2d, and even 1d images.
  tmp = nExt[1] - nExt[0];
  if (tmp <= 0)
  {
    tmp = 1;
  }
  numCells = tmp;
  tmp = nExt[3] - nExt[2];
  if (tmp <= 0)
  {
    tmp = 1;
  }
  numCells *= tmp;
  tmp = nExt[5] - nExt[4];
  if (tmp <= 0)
  {
    tmp = 1;
  }
  numCells *= tmp;

  // Create a new temporary image.
  newImage = svtkImageData::New();
  newImage->SetExtent(nExt);
  svtkPointData* npd = newImage->GetPointData();
  svtkCellData* ncd = newImage->GetCellData();
  npd->CopyAllocate(this->PointData, numPts);
  ncd->CopyAllocate(this->CellData, numCells);

  // Loop through outData points
  incY = extent[1] - extent[0] + 1;
  incZ = (extent[3] - extent[2] + 1) * incY;
  outId = 0;
  inIdZ = incZ * (nExt[4] - extent[4]) + incY * (nExt[2] - extent[2]) + (nExt[0] - extent[0]);

  for (idxZ = nExt[4]; idxZ <= nExt[5]; idxZ++)
  {
    inIdY = inIdZ;
    for (idxY = nExt[2]; idxY <= nExt[3]; idxY++)
    {
      inId = inIdY;
      for (idxX = nExt[0]; idxX <= nExt[1]; idxX++)
      {
        npd->CopyData(this->PointData, inId, outId);
        ++inId;
        ++outId;
      }
      inIdY += incY;
    }
    inIdZ += incZ;
  }

  // Loop through outData cells
  // Have to handle the 2d and 1d cases.
  maxX = nExt[1];
  maxY = nExt[3];
  maxZ = nExt[5];
  if (maxX == nExt[0])
  {
    ++maxX;
  }
  if (maxY == nExt[2])
  {
    ++maxY;
  }
  if (maxZ == nExt[4])
  {
    ++maxZ;
  }
  incY = extent[1] - extent[0];
  incZ = (extent[3] - extent[2]) * incY;
  outId = 0;
  inIdZ = incZ * (nExt[4] - extent[4]) + incY * (nExt[2] - extent[2]) + (nExt[0] - extent[0]);
  for (idxZ = nExt[4]; idxZ < maxZ; idxZ++)
  {
    inIdY = inIdZ;
    for (idxY = nExt[2]; idxY < maxY; idxY++)
    {
      inId = inIdY;
      for (idxX = nExt[0]; idxX < maxX; idxX++)
      {
        ncd->CopyData(this->CellData, inId, outId);
        ++inId;
        ++outId;
      }
      inIdY += incY;
    }
    inIdZ += incZ;
  }

  this->PointData->ShallowCopy(npd);
  this->CellData->ShallowCopy(ncd);
  this->SetExtent(nExt);
  newImage->Delete();
}

//----------------------------------------------------------------------------
double svtkImageData::GetScalarTypeMin(svtkInformation* meta_data)
{
  return svtkDataArray::GetDataTypeMin(this->GetScalarType(meta_data));
}

//----------------------------------------------------------------------------
double svtkImageData::GetScalarTypeMin()
{
  return svtkDataArray::GetDataTypeMin(this->GetScalarType());
}

//----------------------------------------------------------------------------
double svtkImageData::GetScalarTypeMax(svtkInformation* meta_data)
{
  return svtkDataArray::GetDataTypeMax(this->GetScalarType(meta_data));
}

//----------------------------------------------------------------------------
double svtkImageData::GetScalarTypeMax()
{
  return svtkDataArray::GetDataTypeMax(this->GetScalarType());
}

//----------------------------------------------------------------------------
void svtkImageData::SetExtent(int x1, int x2, int y1, int y2, int z1, int z2)
{
  int ext[6];
  ext[0] = x1;
  ext[1] = x2;
  ext[2] = y1;
  ext[3] = y2;
  ext[4] = z1;
  ext[5] = z2;
  this->SetExtent(ext);
}

//----------------------------------------------------------------------------
void svtkImageData::SetDataDescription(int desc)
{
  if (desc == this->DataDescription)
  {
    return;
  }

  this->DataDescription = desc;

  if (this->Vertex)
  {
    this->Vertex->Delete();
    this->Vertex = nullptr;
  }
  if (this->Line)
  {
    this->Line->Delete();
    this->Line = nullptr;
  }
  if (this->Pixel)
  {
    this->Pixel->Delete();
    this->Pixel = nullptr;
  }
  if (this->Voxel)
  {
    this->Voxel->Delete();
    this->Voxel = nullptr;
  }
  switch (this->DataDescription)
  {
    case SVTK_SINGLE_POINT:
      this->Vertex = svtkVertex::New();
      break;

    case SVTK_X_LINE:
    case SVTK_Y_LINE:
    case SVTK_Z_LINE:
      this->Line = svtkLine::New();
      break;

    case SVTK_XY_PLANE:
    case SVTK_YZ_PLANE:
    case SVTK_XZ_PLANE:
      this->Pixel = svtkPixel::New();
      break;

    case SVTK_XYZ_GRID:
      this->Voxel = svtkVoxel::New();
      break;
  }
}

//----------------------------------------------------------------------------
void svtkImageData::SetExtent(int* extent)
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

  this->SetDataDescription(description);

  this->Modified();
}

//----------------------------------------------------------------------------
int* svtkImageData::GetDimensions()
{
  this->GetDimensions(this->Dimensions);
  return this->Dimensions;
}

//----------------------------------------------------------------------------
void svtkImageData::GetDimensions(int* dOut)
{
  const int* extent = this->Extent;
  dOut[0] = extent[1] - extent[0] + 1;
  dOut[1] = extent[3] - extent[2] + 1;
  dOut[2] = extent[5] - extent[4] + 1;
}

#if SVTK_ID_TYPE_IMPL != SVTK_INT
//----------------------------------------------------------------------------
void svtkImageData::GetDimensions(svtkIdType dims[3])
{
  // Use svtkIdType to avoid overflow on large images
  const int* extent = this->Extent;
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;
}
#endif

//----------------------------------------------------------------------------
void svtkImageData::SetAxisUpdateExtent(
  int idx, int min, int max, const int* updateExtent, int* axisUpdateExtent)
{
  if (idx > 2)
  {
    svtkWarningMacro("illegal axis!");
    return;
  }

  memcpy(axisUpdateExtent, updateExtent, 6 * sizeof(int));
  if (axisUpdateExtent[idx * 2] != min)
  {
    axisUpdateExtent[idx * 2] = min;
  }
  if (axisUpdateExtent[idx * 2 + 1] != max)
  {
    axisUpdateExtent[idx * 2 + 1] = max;
  }
}

//----------------------------------------------------------------------------
void svtkImageData::GetAxisUpdateExtent(int idx, int& min, int& max, const int* updateExtent)
{
  if (idx > 2)
  {
    svtkWarningMacro("illegal axis!");
    return;
  }

  min = updateExtent[idx * 2];
  max = updateExtent[idx * 2 + 1];
}

//----------------------------------------------------------------------------
unsigned long svtkImageData::GetActualMemorySize()
{
  return this->svtkDataSet::GetActualMemorySize();
}

//----------------------------------------------------------------------------
void svtkImageData::ShallowCopy(svtkDataObject* dataObject)
{
  svtkImageData* imageData = svtkImageData::SafeDownCast(dataObject);

  if (imageData != nullptr)
  {
    this->InternalImageDataCopy(imageData);
  }

  // Do superclass
  this->svtkDataSet::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkImageData::DeepCopy(svtkDataObject* dataObject)
{
  svtkImageData* imageData = svtkImageData::SafeDownCast(dataObject);

  if (imageData != nullptr)
  {
    this->InternalImageDataCopy(imageData);
  }

  // Do superclass
  this->svtkDataSet::DeepCopy(dataObject);
}

//----------------------------------------------------------------------------
// This copies all the local variables (but not objects).
void svtkImageData::InternalImageDataCopy(svtkImageData* src)
{
  int idx;

  // this->SetScalarType(src->GetScalarType());
  // this->SetNumberOfScalarComponents(src->GetNumberOfScalarComponents());
  for (idx = 0; idx < 3; ++idx)
  {
    this->Dimensions[idx] = src->Dimensions[idx];
    this->Increments[idx] = src->Increments[idx];
    this->Origin[idx] = src->Origin[idx];
    this->Spacing[idx] = src->Spacing[idx];
  }
  this->DirectionMatrix->DeepCopy(src->DirectionMatrix);
  this->ComputeTransforms();
  this->SetExtent(src->GetExtent());
}

//----------------------------------------------------------------------------
svtkIdType svtkImageData::GetNumberOfCells()
{
  svtkIdType nCells = 1;
  int i;
  const int* extent = this->Extent;

  svtkIdType dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;

  for (i = 0; i < 3; i++)
  {
    if (dims[i] == 0)
    {
      return 0;
    }
    if (dims[i] > 1)
    {
      nCells *= (dims[i] - 1);
    }
  }

  return nCells;
}

//============================================================================
// Starting to make some more general methods that deal with any array
// (not just scalars).
//============================================================================

//----------------------------------------------------------------------------
// This Method returns a pointer to a location in the svtkImageData.
// Coordinates are in pixel units and are relative to the whole
// image origin.
void svtkImageData::GetArrayIncrements(svtkDataArray* array, svtkIdType increments[3])
{
  const int* extent = this->Extent;
  // We could store tuple increments and just
  // multiply by the number of components...
  increments[0] = array->GetNumberOfComponents();
  increments[1] = increments[0] * (extent[1] - extent[0] + 1);
  increments[2] = increments[1] * (extent[3] - extent[2] + 1);
}

//----------------------------------------------------------------------------
void* svtkImageData::GetArrayPointerForExtent(svtkDataArray* array, int extent[6])
{
  int tmp[3];
  tmp[0] = extent[0];
  tmp[1] = extent[2];
  tmp[2] = extent[4];
  return this->GetArrayPointer(array, tmp);
}

//----------------------------------------------------------------------------
// This Method returns a pointer to a location in the svtkImageData.
// Coordinates are in pixel units and are relative to the whole
// image origin.
void* svtkImageData::GetArrayPointer(svtkDataArray* array, int coordinate[3])
{
  svtkIdType incs[3];
  svtkIdType idx;

  if (array == nullptr)
  {
    return nullptr;
  }

  const int* extent = this->Extent;
  // error checking: since most accesses will be from pointer arithmetic.
  // this should not waste much time.
  for (idx = 0; idx < 3; ++idx)
  {
    if (coordinate[idx] < extent[idx * 2] || coordinate[idx] > extent[idx * 2 + 1])
    {
      svtkErrorMacro(<< "GetPointer: Pixel (" << coordinate[0] << ", " << coordinate[1] << ", "
                    << coordinate[2] << ") not in current extent: (" << extent[0] << ", "
                    << extent[1] << ", " << extent[2] << ", " << extent[3] << ", " << extent[4]
                    << ", " << extent[5] << ")");
      return nullptr;
    }
  }

  // compute the index of the vector.
  this->GetArrayIncrements(array, incs);
  idx = ((coordinate[0] - extent[0]) * incs[0] + (coordinate[1] - extent[2]) * incs[1] +
    (coordinate[2] - extent[4]) * incs[2]);
  // I could check to see if the array has the correct number
  // of tuples for the extent, but that would be an extra multiply.
  if (idx < 0 || idx > array->GetMaxId())
  {
    svtkErrorMacro("Coordinate (" << coordinate[0] << ", " << coordinate[1] << ", " << coordinate[2]
                                 << ") out side of array (max = " << array->GetMaxId());
    return nullptr;
  }

  return array->GetVoidPointer(idx);
}

//----------------------------------------------------------------------------
void svtkImageData::ComputeInternalExtent(int* intExt, int* tgtExt, int* bnds)
{
  int i;
  const int* extent = this->Extent;
  for (i = 0; i < 3; ++i)
  {
    intExt[i * 2] = tgtExt[i * 2];
    if (intExt[i * 2] - bnds[i * 2] < extent[i * 2])
    {
      intExt[i * 2] = extent[i * 2] + bnds[i * 2];
    }
    intExt[i * 2 + 1] = tgtExt[i * 2 + 1];
    if (intExt[i * 2 + 1] + bnds[i * 2 + 1] > extent[i * 2 + 1])
    {
      intExt[i * 2 + 1] = extent[i * 2 + 1] - bnds[i * 2 + 1];
    }
  }
}

//----------------------------------------------------------------------------
svtkImageData* svtkImageData::GetData(svtkInformation* info)
{
  return info ? svtkImageData::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkImageData* svtkImageData::GetData(svtkInformationVector* v, int i)
{
  return svtkImageData::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkImageData::SetSpacing(double i, double j, double k)
{
  svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Spacing to (" << i << ","
                << j << "," << k << ")");
  if ((this->Spacing[0] != i) || (this->Spacing[1] != j) || (this->Spacing[2] != k))
  {
    this->Spacing[0] = i;
    this->Spacing[1] = j;
    this->Spacing[2] = k;
    this->ComputeTransforms();
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkImageData::SetSpacing(const double ijk[3])
{
  this->SetSpacing(ijk[0], ijk[1], ijk[2]);
}

//----------------------------------------------------------------------------
void svtkImageData::SetOrigin(double i, double j, double k)
{
  svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Origin to (" << i << "," << j
                << "," << k << ")");
  if ((this->Origin[0] != i) || (this->Origin[1] != j) || (this->Origin[2] != k))
  {
    this->Origin[0] = i;
    this->Origin[1] = j;
    this->Origin[2] = k;
    this->ComputeTransforms();
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkImageData::SetOrigin(const double ijk[3])
{
  this->SetOrigin(ijk[0], ijk[1], ijk[2]);
}

//----------------------------------------------------------------------------
void svtkImageData::SetDirectionMatrix(svtkMatrix3x3* m)
{
  svtkMTimeType lastModified = this->GetMTime();
  svtkSetObjectBodyMacro(DirectionMatrix, svtkMatrix3x3, m);
  if (lastModified < this->GetMTime())
  {
    this->ComputeTransforms();
  }
}

//----------------------------------------------------------------------------
void svtkImageData::SetDirectionMatrix(const double elements[9])
{
  this->SetDirectionMatrix(elements[0], elements[1], elements[2], elements[3], elements[4],
    elements[5], elements[6], elements[7], elements[8]);
}

//----------------------------------------------------------------------------
void svtkImageData::SetDirectionMatrix(double e00, double e01, double e02, double e10, double e11,
  double e12, double e20, double e21, double e22)
{
  svtkMatrix3x3* m3 = this->DirectionMatrix;
  svtkMTimeType lastModified = m3->GetMTime();

  m3->SetElement(0, 0, e00);
  m3->SetElement(0, 1, e01);
  m3->SetElement(0, 2, e02);
  m3->SetElement(1, 0, e10);
  m3->SetElement(1, 1, e11);
  m3->SetElement(1, 2, e12);
  m3->SetElement(2, 0, e20);
  m3->SetElement(2, 1, e21);
  m3->SetElement(2, 2, e22);

  if (lastModified < m3->GetMTime())
  {
    this->ComputeTransforms();
    this->Modified();
  }
}

//----------------------------------------------------------------------------
template <typename T1, typename T2>
inline static void TransformCoordinates(
  T1 input0, T1 input1, T1 input2, T2 output[3], svtkMatrix4x4* m4)
{
  double* mdata = m4->GetData();
  output[0] = mdata[0] * input0 + mdata[1] * input1 + mdata[2] * input2 + mdata[3];
  output[1] = mdata[4] * input0 + mdata[5] * input1 + mdata[6] * input2 + mdata[7];
  output[2] = mdata[8] * input0 + mdata[9] * input1 + mdata[10] * input2 + mdata[11];
}

// must pass the inverse matrix
template <typename T1, typename T2>
inline static void TransformNormal(T1 input0, T1 input1, T1 input2, T2 output[3], svtkMatrix4x4* m4)
{
  double* mdata = m4->GetData();
  output[0] = mdata[0] * input0 + mdata[4] * input1 + mdata[8] * input2;
  output[1] = mdata[1] * input0 + mdata[5] * input1 + mdata[9] * input2;
  output[2] = mdata[2] * input0 + mdata[6] * input1 + mdata[10] * input2;
}

// useful for when the ImageData is not available but the information
// spacing, origin, direction are
void svtkImageData::TransformContinuousIndexToPhysicalPoint(double i, double j, double k,
  double const origin[3], double const spacing[3], double const direction[9], double xyz[3])
{
  for (int c = 0; c < 3; ++c)
  {
    xyz[c] = i * spacing[0] * direction[c * 3] + j * spacing[1] * direction[c * 3 + 1] +
      k * spacing[2] * direction[c * 3 + 2] + origin[c];
  }
}

//----------------------------------------------------------------------------
void svtkImageData::TransformContinuousIndexToPhysicalPoint(
  double i, double j, double k, double xyz[3])
{
  TransformCoordinates<double, double>(i, j, k, xyz, this->IndexToPhysicalMatrix);
}

//----------------------------------------------------------------------------
void svtkImageData::TransformContinuousIndexToPhysicalPoint(const double ijk[3], double xyz[3])
{

  TransformCoordinates<double, double>(ijk[0], ijk[1], ijk[2], xyz, this->IndexToPhysicalMatrix);
}

//----------------------------------------------------------------------------
void svtkImageData::TransformIndexToPhysicalPoint(int i, int j, int k, double xyz[3])
{
  TransformCoordinates<int, double>(i, j, k, xyz, this->IndexToPhysicalMatrix);
}

//----------------------------------------------------------------------------
void svtkImageData::TransformIndexToPhysicalPoint(const int ijk[3], double xyz[3])
{
  TransformCoordinates<int, double>(ijk[0], ijk[1], ijk[2], xyz, this->IndexToPhysicalMatrix);
}

//----------------------------------------------------------------------------
void svtkImageData::TransformPhysicalPointToContinuousIndex(
  double x, double y, double z, double ijk[3])
{
  TransformCoordinates<double, double>(x, y, z, ijk, this->PhysicalToIndexMatrix);
}
//----------------------------------------------------------------------------
void svtkImageData::TransformPhysicalPointToContinuousIndex(const double xyz[3], double ijk[3])
{
  TransformCoordinates<double, double>(xyz[0], xyz[1], xyz[2], ijk, this->PhysicalToIndexMatrix);
}

//----------------------------------------------------------------------------
void svtkImageData::TransformPhysicalNormalToContinuousIndex(const double xyz[3], double ijk[3])
{
  TransformNormal<double, double>(xyz[0], xyz[1], xyz[2], ijk, this->IndexToPhysicalMatrix);
}

void svtkImageData::TransformPhysicalPlaneToContinuousIndex(
  double const normal[4], double xnormal[4])
{
  // transform the normal, note the inverse matrix is passed in
  TransformNormal<double, double>(
    normal[0], normal[1], normal[2], xnormal, this->IndexToPhysicalMatrix);
  svtkMath::Normalize(xnormal);

  // transform the point
  double newPt[3];
  TransformCoordinates<double, double>(-normal[3] * normal[0], -normal[3] * normal[1],
    -normal[3] * normal[2], newPt, this->PhysicalToIndexMatrix);

  // recompute plane eqn
  xnormal[3] = -xnormal[0] * newPt[0] - xnormal[1] * newPt[1] - xnormal[2] * newPt[2];
}

//----------------------------------------------------------------------------
void svtkImageData::ComputeTransforms()
{
  svtkMatrix4x4* m4 = svtkMatrix4x4::New();
  if (this->DirectionMatrix->IsIdentity())
  {
    m4->Zero();
    m4->SetElement(0, 0, this->Spacing[0]);
    m4->SetElement(1, 1, this->Spacing[1]);
    m4->SetElement(2, 2, this->Spacing[2]);
    m4->SetElement(3, 3, 1);
  }
  else
  {
    const double* m3 = this->DirectionMatrix->GetData();
    m4->SetElement(0, 0, m3[0] * this->Spacing[0]);
    m4->SetElement(0, 1, m3[1] * this->Spacing[1]);
    m4->SetElement(0, 2, m3[2] * this->Spacing[2]);
    m4->SetElement(1, 0, m3[3] * this->Spacing[0]);
    m4->SetElement(1, 1, m3[4] * this->Spacing[1]);
    m4->SetElement(1, 2, m3[5] * this->Spacing[2]);
    m4->SetElement(2, 0, m3[6] * this->Spacing[0]);
    m4->SetElement(2, 1, m3[7] * this->Spacing[1]);
    m4->SetElement(2, 2, m3[8] * this->Spacing[2]);
    m4->SetElement(3, 0, 0);
    m4->SetElement(3, 1, 0);
    m4->SetElement(3, 2, 0);
    m4->SetElement(3, 3, 1);
  }
  m4->SetElement(0, 3, this->Origin[0]);
  m4->SetElement(1, 3, this->Origin[1]);
  m4->SetElement(2, 3, this->Origin[2]);

  this->IndexToPhysicalMatrix->DeepCopy(m4);
  svtkMatrix4x4::Invert(m4, this->PhysicalToIndexMatrix);
  m4->Delete();
}

//----------------------------------------------------------------------------
void svtkImageData::ComputeIndexToPhysicalMatrix(
  double const origin[3], double const spacing[3], double const direction[9], double result[16])
{
  for (int i = 0; i < 3; ++i)
  {
    result[i * 4] = direction[i * 3] * spacing[0];
    result[i * 4 + 1] = direction[i * 3 + 1] * spacing[1];
    result[i * 4 + 2] = direction[i * 3 + 2] * spacing[2];
  }

  result[3] = origin[0];
  result[7] = origin[1];
  result[11] = origin[2];
  result[12] = 0;
  result[13] = 0;
  result[14] = 0;
  result[15] = 1;
}
