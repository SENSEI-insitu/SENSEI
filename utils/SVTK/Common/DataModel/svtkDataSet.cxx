/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataSet.h"

#include "svtkBezierCurve.h"
#include "svtkBezierHexahedron.h"
#include "svtkBezierQuadrilateral.h"
#include "svtkBezierTetra.h"
#include "svtkBezierTriangle.h"
#include "svtkBezierWedge.h"
#include "svtkCallbackCommand.h"
#include "svtkCell.h"
#include "svtkCellData.h"
#include "svtkCellTypes.h"
#include "svtkDataSetCellIterator.h"
#include "svtkDoubleArray.h"
#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkLagrangeHexahedron.h"
#include "svtkLagrangeQuadrilateral.h"
#include "svtkLagrangeWedge.h"
#include "svtkMath.h"
#include "svtkPointData.h"
#include "svtkSmartPointer.h"
#include "svtkStructuredData.h"

#include <cmath>

//----------------------------------------------------------------------------
// Constructor with default bounds (0,1, 0,1, 0,1).
svtkDataSet::svtkDataSet()
{
  svtkMath::UninitializeBounds(this->Bounds);
  // Observer for updating the cell/point ghost arrays pointers
  this->DataObserver = svtkCallbackCommand::New();
  this->DataObserver->SetCallback(&svtkDataSet::OnDataModified);
  this->DataObserver->SetClientData(this);

  this->PointData = svtkPointData::New();
  this->PointGhostArray = nullptr;
  this->PointGhostArrayCached = false;
  // when point data is modified, we update the point data ghost array cache
  this->PointData->AddObserver(svtkCommand::ModifiedEvent, this->DataObserver);

  this->CellData = svtkCellData::New();
  this->CellGhostArray = nullptr;
  this->CellGhostArrayCached = false;
  // when cell data is modified, we update the cell data ghost array cache
  this->CellData->AddObserver(svtkCommand::ModifiedEvent, this->DataObserver);

  this->ScalarRange[0] = 0.0;
  this->ScalarRange[1] = 1.0;
}

//----------------------------------------------------------------------------
svtkDataSet::~svtkDataSet()
{
  this->PointData->RemoveObserver(this->DataObserver);
  this->PointData->Delete();

  this->CellData->RemoveObserver(this->DataObserver);
  this->CellData->Delete();

  this->DataObserver->Delete();
}

//----------------------------------------------------------------------------
void svtkDataSet::Initialize()
{
  // We don't modify ourselves because the "ReleaseData" methods depend upon
  // no modification when initialized.
  svtkDataObject::Initialize();

  this->CellData->Initialize();
  this->PointData->Initialize();
}

//----------------------------------------------------------------------------
void svtkDataSet::CopyAttributes(svtkDataSet* ds)
{
  this->GetPointData()->PassData(ds->GetPointData());
  this->GetCellData()->PassData(ds->GetCellData());
  this->GetFieldData()->PassData(ds->GetFieldData());
}

//----------------------------------------------------------------------------
svtkCellIterator* svtkDataSet::NewCellIterator()
{
  svtkDataSetCellIterator* iter = svtkDataSetCellIterator::New();
  iter->SetDataSet(this);
  return iter;
}

//----------------------------------------------------------------------------
// Compute the data bounding box from data points.
void svtkDataSet::ComputeBounds()
{
  int j;
  svtkIdType i;
  double* x;

  if (this->GetMTime() > this->ComputeTime)
  {
    if (this->GetNumberOfPoints())
    {
      x = this->GetPoint(0);
      this->Bounds[0] = this->Bounds[1] = x[0];
      this->Bounds[2] = this->Bounds[3] = x[1];
      this->Bounds[4] = this->Bounds[5] = x[2];
      for (i = 1; i < this->GetNumberOfPoints(); i++)
      {
        x = this->GetPoint(i);
        for (j = 0; j < 3; j++)
        {
          if (x[j] < this->Bounds[2 * j])
          {
            this->Bounds[2 * j] = x[j];
          }
          if (x[j] > this->Bounds[2 * j + 1])
          {
            this->Bounds[2 * j + 1] = x[j];
          }
        }
      }
    }
    else
    {
      svtkMath::UninitializeBounds(this->Bounds);
    }
    this->ComputeTime.Modified();
  }
}

//----------------------------------------------------------------------------
// Description:
// Compute the range of the scalars and cache it into ScalarRange
// only if the cache became invalid (ScalarRangeComputeTime).
void svtkDataSet::ComputeScalarRange()
{
  if (this->GetMTime() > this->ScalarRangeComputeTime)
  {
    svtkDataArray *ptScalars, *cellScalars;
    ptScalars = this->PointData->GetScalars();
    cellScalars = this->CellData->GetScalars();

    if (ptScalars && cellScalars)
    {
      double r1[2], r2[2];
      ptScalars->GetRange(r1, 0);
      cellScalars->GetRange(r2, 0);
      this->ScalarRange[0] = (r1[0] < r2[0] ? r1[0] : r2[0]);
      this->ScalarRange[1] = (r1[1] > r2[1] ? r1[1] : r2[1]);
    }
    else if (ptScalars)
    {
      ptScalars->GetRange(this->ScalarRange, 0);
    }
    else if (cellScalars)
    {
      cellScalars->GetRange(this->ScalarRange, 0);
    }
    else
    {
      this->ScalarRange[0] = 0.0;
      this->ScalarRange[1] = 1.0;
    }
    this->ScalarRangeComputeTime.Modified();
  }
}

//----------------------------------------------------------------------------
void svtkDataSet::GetScalarRange(double range[2])
{
  this->ComputeScalarRange();
  range[0] = this->ScalarRange[0];
  range[1] = this->ScalarRange[1];
}

//----------------------------------------------------------------------------
double* svtkDataSet::GetScalarRange()
{
  this->ComputeScalarRange();
  return this->ScalarRange;
}

//----------------------------------------------------------------------------
// Return a pointer to the geometry bounding box in the form
// (xmin,xmax, ymin,ymax, zmin,zmax).
double* svtkDataSet::GetBounds()
{
  this->ComputeBounds();
  return this->Bounds;
}

//----------------------------------------------------------------------------
void svtkDataSet::GetBounds(double bounds[6])
{
  this->ComputeBounds();
  for (int i = 0; i < 6; i++)
  {
    bounds[i] = this->Bounds[i];
  }
}

//----------------------------------------------------------------------------
// Get the center of the bounding box.
double* svtkDataSet::GetCenter()
{
  this->ComputeBounds();
  for (int i = 0; i < 3; i++)
  {
    this->Center[i] = (this->Bounds[2 * i + 1] + this->Bounds[2 * i]) / 2.0;
  }
  return this->Center;
}

//----------------------------------------------------------------------------
void svtkDataSet::GetCenter(double center[3])
{
  this->ComputeBounds();
  for (int i = 0; i < 3; i++)
  {
    center[i] = (this->Bounds[2 * i + 1] + this->Bounds[2 * i]) / 2.0;
  }
}

//----------------------------------------------------------------------------
// Return the length of the diagonal of the bounding box.
double svtkDataSet::GetLength()
{
  if (this->GetNumberOfPoints() == 0)
  {
    return 0;
  }

  double diff, l = 0.0;
  int i;

  this->ComputeBounds();
  for (i = 0; i < 3; i++)
  {
    diff = static_cast<double>(this->Bounds[2 * i + 1]) - static_cast<double>(this->Bounds[2 * i]);
    l += diff * diff;
  }
  diff = sqrt(l);
  return diff;
}

//----------------------------------------------------------------------------
svtkMTimeType svtkDataSet::GetMTime()
{
  svtkMTimeType mtime, result;

  result = svtkDataObject::GetMTime();

  mtime = this->PointData->GetMTime();
  result = (mtime > result ? mtime : result);

  mtime = this->CellData->GetMTime();
  return (mtime > result ? mtime : result);
}

//----------------------------------------------------------------------------
svtkCell* svtkDataSet::FindAndGetCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2,
  int& subId, double pcoords[3], double* weights)
{
  svtkIdType newCell = this->FindCell(x, cell, cellId, tol2, subId, pcoords, weights);
  if (newCell >= 0)
  {
    cell = this->GetCell(newCell);
  }
  else
  {
    return nullptr;
  }
  return cell;
}

//----------------------------------------------------------------------------
void svtkDataSet::GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds)
{
  svtkIdType i, numPts;
  svtkIdList* otherCells = svtkIdList::New();
  otherCells->Allocate(SVTK_CELL_SIZE);

  // load list with candidate cells, remove current cell
  this->GetPointCells(ptIds->GetId(0), cellIds);
  cellIds->DeleteId(cellId);

  // now perform multiple intersections on list
  if (cellIds->GetNumberOfIds() > 0)
  {
    for (numPts = ptIds->GetNumberOfIds(), i = 1; i < numPts; i++)
    {
      this->GetPointCells(ptIds->GetId(i), otherCells);
      cellIds->IntersectWith(*otherCells);
    }
  }

  otherCells->Delete();
}

//----------------------------------------------------------------------------
void svtkDataSet::GetCellTypes(svtkCellTypes* types)
{
  svtkIdType cellId, numCells = this->GetNumberOfCells();
  unsigned char type;

  types->Reset();
  for (cellId = 0; cellId < numCells; cellId++)
  {
    type = this->GetCellType(cellId);
    if (!types->IsType(type))
    {
      types->InsertNextType(type);
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataSet::SetCellOrderAndRationalWeights(svtkIdType cellId, svtkGenericCell* cell)
{
  switch (cell->GetCellType())
  {
    // Set the degree for Lagrange elements
    case SVTK_LAGRANGE_QUADRILATERAL:
    {
      svtkHigherOrderQuadrilateral* cellBezier =
        dynamic_cast<svtkHigherOrderQuadrilateral*>(cell->GetRepresentativeCell());
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1]);
      }
      else
      {
        svtkIdType numPts = cell->PointIds->GetNumberOfIds();
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }
      break;
    }
    case SVTK_LAGRANGE_WEDGE:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkHigherOrderWedge* cellBezier =
        dynamic_cast<svtkHigherOrderWedge*>(cell->GetRepresentativeCell());
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1], degs[2], numPts);
      }
      else
      {
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }
      break;
    }
    case SVTK_LAGRANGE_HEXAHEDRON:
    {
      svtkHigherOrderHexahedron* cellBezier =
        dynamic_cast<svtkHigherOrderHexahedron*>(cell->GetRepresentativeCell());
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1], degs[2]);
      }
      else
      {
        svtkIdType numPts = cell->PointIds->GetNumberOfIds();
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }
      break;
    }

    // Set the degree and rational weights for Bezier elements
    case SVTK_BEZIER_QUADRILATERAL:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierQuadrilateral* cellBezier =
        dynamic_cast<svtkBezierQuadrilateral*>(cell->GetRepresentativeCell());

      // Set the degrees
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1]);
      }
      else
      {
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }

      // Set the weights
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }
    case SVTK_BEZIER_HEXAHEDRON:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierHexahedron* cellBezier =
        dynamic_cast<svtkBezierHexahedron*>(cell->GetRepresentativeCell());

      // Set the degrees
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1], degs[2]);
      }
      else
      {
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }

      // Set the weights
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }
    case SVTK_BEZIER_WEDGE:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierWedge* cellBezier = dynamic_cast<svtkBezierWedge*>(cell->GetRepresentativeCell());

      // Set the degrees
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        cellBezier->SetOrder(degs[0], degs[1], degs[2], numPts);
      }
      else
      {
        cellBezier->SetUniformOrderFromNumPoints(numPts);
      }

      // Set the weights
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }

    case SVTK_BEZIER_CURVE:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierCurve* cellBezier = dynamic_cast<svtkBezierCurve*>(cell->GetRepresentativeCell());
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }
    case SVTK_BEZIER_TRIANGLE:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierTriangle* cellBezier =
        dynamic_cast<svtkBezierTriangle*>(cell->GetRepresentativeCell());
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }
    case SVTK_BEZIER_TETRAHEDRON:
    {
      svtkIdType numPts = cell->PointIds->GetNumberOfIds();
      svtkBezierTetra* cellBezier = dynamic_cast<svtkBezierTetra*>(cell->GetRepresentativeCell());
      cellBezier->SetRationalWeightsFromPointData(GetPointData(), numPts);
      break;
    }
    default:
      break;
  }
}

//----------------------------------------------------------------------------
// Default implementation. This is very slow way to compute this information.
// Subclasses should override this method for efficiency.
void svtkDataSet::GetCellBounds(svtkIdType cellId, double bounds[6])
{
  svtkGenericCell* cell = svtkGenericCell::New();

  this->GetCell(cellId, cell);
  cell->GetBounds(bounds);
  cell->Delete();
}

//----------------------------------------------------------------------------
void svtkDataSet::Squeeze()
{
  this->CellData->Squeeze();
  this->PointData->Squeeze();
}

//----------------------------------------------------------------------------
unsigned long svtkDataSet::GetActualMemorySize()
{
  unsigned long size = this->svtkDataObject::GetActualMemorySize();
  size += this->PointData->GetActualMemorySize();
  size += this->CellData->GetActualMemorySize();
  return size;
}

//----------------------------------------------------------------------------
void svtkDataSet::ShallowCopy(svtkDataObject* dataObject)
{
  svtkDataSet* dataSet = svtkDataSet::SafeDownCast(dataObject);

  if (dataSet != nullptr)
  {
    this->InternalDataSetCopy(dataSet);
    this->CellData->ShallowCopy(dataSet->GetCellData());
    this->PointData->ShallowCopy(dataSet->GetPointData());
  }
  // Do superclass
  this->svtkDataObject::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkDataSet::DeepCopy(svtkDataObject* dataObject)
{
  svtkDataSet* dataSet = svtkDataSet::SafeDownCast(dataObject);

  if (dataSet != nullptr)
  {
    this->InternalDataSetCopy(dataSet);
    this->CellData->DeepCopy(dataSet->GetCellData());
    this->PointData->DeepCopy(dataSet->GetPointData());
  }

  // Do superclass
  this->svtkDataObject::DeepCopy(dataObject);
}

//----------------------------------------------------------------------------
// This copies all the local variables (but not objects).
void svtkDataSet::InternalDataSetCopy(svtkDataSet* src)
{
  int idx;

  this->ScalarRangeComputeTime = src->ScalarRangeComputeTime;
  this->ScalarRange[0] = src->ScalarRange[0];
  this->ScalarRange[1] = src->ScalarRange[1];

  this->ComputeTime = src->ComputeTime;
  for (idx = 0; idx < 3; ++idx)
  {
    this->Bounds[2 * idx] = src->Bounds[2 * idx];
    this->Bounds[2 * idx + 1] = src->Bounds[2 * idx + 1];
  }
}

//----------------------------------------------------------------------------
int svtkDataSet::CheckAttributes()
{
  svtkIdType numPts, numCells;
  int numArrays, idx;
  svtkAbstractArray* array;
  svtkIdType numTuples;
  const char* name;

  numArrays = this->GetPointData()->GetNumberOfArrays();
  if (numArrays > 0)
  {
    // This call can be expensive.
    numPts = this->GetNumberOfPoints();
    for (idx = 0; idx < numArrays; ++idx)
    {
      array = this->GetPointData()->GetAbstractArray(idx);
      numTuples = array->GetNumberOfTuples();
      name = array->GetName();
      if (name == nullptr)
      {
        name = "";
      }
      if (numTuples < numPts)
      {
        svtkErrorMacro("Point array " << name << " with " << array->GetNumberOfComponents()
                                     << " components, only has " << numTuples
                                     << " tuples but there are " << numPts << " points");
        return 1;
      }
      if (numTuples > numPts)
      {
        svtkWarningMacro("Point array " << name << " with " << array->GetNumberOfComponents()
                                       << " components, has " << numTuples
                                       << " tuples but there are only " << numPts << " points");
      }
    }
  }

  numArrays = this->GetCellData()->GetNumberOfArrays();
  if (numArrays > 0)
  {
    // This call can be expensive.
    numCells = this->GetNumberOfCells();

    for (idx = 0; idx < numArrays; ++idx)
    {
      array = this->GetCellData()->GetAbstractArray(idx);
      numTuples = array->GetNumberOfTuples();
      name = array->GetName();
      if (name == nullptr)
      {
        name = "";
      }
      if (numTuples < numCells)
      {
        svtkErrorMacro("Cell array " << name << " with " << array->GetNumberOfComponents()
                                    << " components, has only " << numTuples
                                    << " tuples but there are " << numCells << " cells");
        return 1;
      }
      if (numTuples > numCells)
      {
        svtkWarningMacro("Cell array " << name << " with " << array->GetNumberOfComponents()
                                      << " components, has " << numTuples
                                      << " tuples but there are only " << numCells << " cells");
      }
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
void svtkDataSet::GenerateGhostArray(int zeroExt[6], bool cellOnly)
{
  // Make sure this is a structured data set.
  if (this->GetExtentType() != SVTK_3D_EXTENT)
  {
    return;
  }

  int extent[6];
  this->Information->Get(svtkDataObject::DATA_EXTENT(), extent);

  int i, j, k, di, dj, dk, dist;

  bool sameExtent = true;
  for (i = 0; i < 6; i++)
  {
    if (extent[i] != zeroExt[i])
    {
      sameExtent = false;
      break;
    }
  }
  if (sameExtent)
  {
    return;
  }

  svtkIdType index = 0;

  // ---- POINTS ----

  if (!cellOnly)
  {
    svtkSmartPointer<svtkUnsignedCharArray> ghostPoints = svtkArrayDownCast<svtkUnsignedCharArray>(
      this->PointData->GetArray(svtkDataSetAttributes::GhostArrayName()));
    if (!ghostPoints)
    {
      ghostPoints.TakeReference(svtkUnsignedCharArray::New());
      ghostPoints->SetName(svtkDataSetAttributes::GhostArrayName());
      ghostPoints->SetNumberOfTuples(svtkStructuredData::GetNumberOfPoints(extent));
      ghostPoints->FillValue(0);
      this->PointData->AddArray(ghostPoints);
    }

    // Loop through the points in this image.
    for (k = extent[4]; k <= extent[5]; ++k)
    {
      dk = 0;
      if (k < zeroExt[4])
      {
        dk = zeroExt[4] - k;
      }
      if (k > zeroExt[5])
      { // Special case for last tile.
        dk = k - zeroExt[5] + 1;
      }
      for (j = extent[2]; j <= extent[3]; ++j)
      {
        dj = 0;
        if (j < zeroExt[2])
        {
          dj = zeroExt[2] - j;
        }
        if (j > zeroExt[3])
        { // Special case for last tile.
          dj = j - zeroExt[3] + 1;
        }
        for (i = extent[0]; i <= extent[1]; ++i)
        {
          di = 0;
          if (i < zeroExt[0])
          {
            di = zeroExt[0] - i;
          }
          if (i > zeroExt[1])
          { // Special case for last tile.
            di = i - zeroExt[1] + 1;
          }
          // Compute Manhatten distance.
          dist = di;
          if (dj > dist)
          {
            dist = dj;
          }
          if (dk > dist)
          {
            dist = dk;
          }
          unsigned char value = ghostPoints->GetValue(index);
          if (dist > 0)
          {
            value |= svtkDataSetAttributes::DUPLICATEPOINT;
          }
          ghostPoints->SetValue(index, value);
          index++;
        }
      }
    }
  }

  // ---- CELLS ----

  svtkSmartPointer<svtkUnsignedCharArray> ghostCells = svtkArrayDownCast<svtkUnsignedCharArray>(
    this->CellData->GetArray(svtkDataSetAttributes::GhostArrayName()));
  if (!ghostCells)
  {
    ghostCells.TakeReference(svtkUnsignedCharArray::New());
    ghostCells->SetName(svtkDataSetAttributes::GhostArrayName());
    ghostCells->SetNumberOfTuples(svtkStructuredData::GetNumberOfCells(extent));
    ghostCells->FillValue(0);
    this->CellData->AddArray(ghostCells);
  }

  index = 0;

  // Loop through the cells in this image.
  // Cells may be 2d or 1d ... Treat all as 3D
  if (extent[0] == extent[1])
  {
    ++extent[1];
    ++zeroExt[1];
  }
  if (extent[2] == extent[3])
  {
    ++extent[3];
    ++zeroExt[3];
  }
  if (extent[4] == extent[5])
  {
    ++extent[5];
    ++zeroExt[5];
  }

  // Loop
  for (k = extent[4]; k < extent[5]; ++k)
  { // Determine the Manhatten distances to zero extent.
    dk = 0;
    if (k < zeroExt[4])
    {
      dk = zeroExt[4] - k;
    }
    if (k >= zeroExt[5])
    {
      dk = k - zeroExt[5] + 1;
    }
    for (j = extent[2]; j < extent[3]; ++j)
    {
      dj = 0;
      if (j < zeroExt[2])
      {
        dj = zeroExt[2] - j;
      }
      if (j >= zeroExt[3])
      {
        dj = j - zeroExt[3] + 1;
      }
      for (i = extent[0]; i < extent[1]; ++i)
      {
        di = 0;
        if (i < zeroExt[0])
        {
          di = zeroExt[0] - i;
        }
        if (i >= zeroExt[1])
        {
          di = i - zeroExt[1] + 1;
        }
        // Compute Manhatten distance.
        dist = di;
        if (dj > dist)
        {
          dist = dj;
        }
        if (dk > dist)
        {
          dist = dk;
        }
        unsigned char value = ghostCells->GetValue(index);
        if (dist > 0)
        {
          value |= svtkDataSetAttributes::DUPLICATECELL;
        }
        ghostCells->SetValue(index, value);
        index++;
      }
    }
  }
}

//----------------------------------------------------------------------------
svtkDataSet* svtkDataSet::GetData(svtkInformation* info)
{
  return info ? svtkDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkDataSet* svtkDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkDataSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
svtkFieldData* svtkDataSet::GetAttributesAsFieldData(int type)
{
  switch (type)
  {
    case POINT:
      return this->GetPointData();
    case CELL:
      return this->GetCellData();
  }
  return this->Superclass::GetAttributesAsFieldData(type);
}

//----------------------------------------------------------------------------
svtkIdType svtkDataSet::GetNumberOfElements(int type)
{
  switch (type)
  {
    case POINT:
      return this->GetNumberOfPoints();
    case CELL:
      return this->GetNumberOfCells();
  }
  return this->Superclass::GetNumberOfElements(type);
}

//----------------------------------------------------------------------------
void svtkDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number Of Points: " << this->GetNumberOfPoints() << "\n";
  os << indent << "Number Of Cells: " << this->GetNumberOfCells() << "\n";

  os << indent << "Cell Data:\n";
  this->CellData->PrintSelf(os, indent.GetNextIndent());

  os << indent << "Point Data:\n";
  this->PointData->PrintSelf(os, indent.GetNextIndent());

  const double* bounds = this->GetBounds();
  os << indent << "Bounds: \n";
  os << indent << "  Xmin,Xmax: (" << bounds[0] << ", " << bounds[1] << ")\n";
  os << indent << "  Ymin,Ymax: (" << bounds[2] << ", " << bounds[3] << ")\n";
  os << indent << "  Zmin,Zmax: (" << bounds[4] << ", " << bounds[5] << ")\n";
  os << indent << "Compute Time: " << this->ComputeTime.GetMTime() << "\n";
}

//----------------------------------------------------------------------------
bool svtkDataSet::HasAnyGhostPoints()
{
  return IsAnyBitSet(this->GetPointGhostArray(), svtkDataSetAttributes::DUPLICATEPOINT);
}

//----------------------------------------------------------------------------
bool svtkDataSet::HasAnyGhostCells()
{
  return IsAnyBitSet(this->GetCellGhostArray(), svtkDataSetAttributes::DUPLICATECELL);
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkDataSet::GetPointGhostArray()
{
  if (!this->PointGhostArrayCached)
  {
    this->PointGhostArray = svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetPointData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
    this->PointGhostArrayCached = true;
  }
  assert(this->PointGhostArray ==
    svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetPointData()->GetArray(svtkDataSetAttributes::GhostArrayName())));
  return this->PointGhostArray;
}

//----------------------------------------------------------------------------
void svtkDataSet::UpdatePointGhostArrayCache()
{
  this->PointGhostArray = svtkArrayDownCast<svtkUnsignedCharArray>(
    this->GetPointData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
  this->PointGhostArrayCached = true;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkDataSet::AllocatePointGhostArray()
{
  if (!this->GetPointGhostArray())
  {
    svtkUnsignedCharArray* ghosts = svtkUnsignedCharArray::New();
    ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
    ghosts->SetNumberOfComponents(1);
    ghosts->SetNumberOfTuples(this->GetNumberOfPoints());
    ghosts->FillValue(0);
    this->GetPointData()->AddArray(ghosts);
    ghosts->Delete();
    this->PointGhostArray = ghosts;
    this->PointGhostArrayCached = true;
  }
  return this->PointGhostArray;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkDataSet::GetCellGhostArray()
{
  if (!this->CellGhostArrayCached)
  {
    this->CellGhostArray = svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetCellData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
    this->CellGhostArrayCached = true;
  }
  assert(this->CellGhostArray ==
    svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetCellData()->GetArray(svtkDataSetAttributes::GhostArrayName())));
  return this->CellGhostArray;
}

//----------------------------------------------------------------------------
void svtkDataSet::UpdateCellGhostArrayCache()
{
  this->CellGhostArray = svtkArrayDownCast<svtkUnsignedCharArray>(
    this->GetCellData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
  this->CellGhostArrayCached = true;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkDataSet::AllocateCellGhostArray()
{
  if (!this->GetCellGhostArray())
  {
    svtkUnsignedCharArray* ghosts = svtkUnsignedCharArray::New();
    ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
    ghosts->SetNumberOfComponents(1);
    ghosts->SetNumberOfTuples(this->GetNumberOfCells());
    ghosts->FillValue(0);
    this->GetCellData()->AddArray(ghosts);
    ghosts->Delete();
    this->CellGhostArray = ghosts;
    this->CellGhostArrayCached = true;
  }
  return this->CellGhostArray;
}

//----------------------------------------------------------------------------
bool svtkDataSet::IsAnyBitSet(svtkUnsignedCharArray* a, int bitFlag)
{
  if (a)
  {
    for (svtkIdType i = 0; i < a->GetNumberOfTuples(); ++i)
    {
      if (a->GetValue(i) & bitFlag)
      {
        return true;
      }
    }
  }
  return false;
}

//----------------------------------------------------------------------------
void svtkDataSet::OnDataModified(svtkObject* source, unsigned long, void* clientdata, void*)
{
  // update the point/cell pointers to ghost data arrays.
  svtkDataSet* This = static_cast<svtkDataSet*>(clientdata);
  if (source == This->GetPointData())
  {
    This->UpdatePointGhostArrayCache();
  }
  else
  {
    assert(source == This->GetCellData());
    This->UpdateCellGhostArrayCache();
  }
}
