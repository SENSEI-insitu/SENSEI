/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetCellIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkDataSetCellIterator.h"

#include "svtkHyperTreeGrid.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkRectilinearGrid.h"

svtkStandardNewMacro(svtkDataSetCellIterator);

namespace
{
template <typename T>
void SetArrayType(T* grid, svtkPoints* points)
{
  // check all directions to see if any of them are doubles and
  // if they are we set the points data type to double. If the
  // data types are all the same then we set it to the common
  // data type. Otherwise we give up and just keep the default
  // float data type.
  int xType = -1, yType = -1, zType = -1;
  if (svtkDataArray* x = grid->GetXCoordinates())
  {
    xType = x->GetDataType();
    if (xType == SVTK_DOUBLE)
    {
      points->SetDataType(SVTK_DOUBLE);
      return;
    }
  }
  if (svtkDataArray* y = grid->GetYCoordinates())
  {
    yType = y->GetDataType();
    if (yType == SVTK_DOUBLE)
    {
      points->SetDataType(SVTK_DOUBLE);
      return;
    }
  }
  if (svtkDataArray* z = grid->GetZCoordinates())
  {
    zType = z->GetDataType();
    if (zType == SVTK_DOUBLE)
    {
      points->SetDataType(SVTK_DOUBLE);
      return;
    }
  }
  if (xType != -1 || yType != -1 || zType != -1)
  {
    if (xType == yType && xType == zType)
    {
      points->SetDataType(xType);
      return;
    }
    if (xType == -1)
    {
      if (yType == -1)
      {
        points->SetDataType(zType);
        return;
      }
      else if (zType == -1 || yType == zType)
      {
        points->SetDataType(yType);
        return;
      }
    }
    if (yType == -1)
    {
      if (xType == -1)
      {
        points->SetDataType(zType);
        return;
      }
      else if (zType == -1 || xType == zType)
      {
        points->SetDataType(xType);
        return;
      }
    }
    if (zType == -1)
    {
      if (xType == -1)
      {
        points->SetDataType(yType);
        return;
      }
      else if (yType == -1 || xType == yType)
      {
        points->SetDataType(xType);
        return;
      }
    }
  }

  // Set it to the default since it may have gotten set to something else
  points->SetDataType(SVTK_FLOAT);
}
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "DataSet: " << this->DataSet << endl;
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::SetDataSet(svtkDataSet* ds)
{
  this->DataSet = ds;
  this->CellId = 0;

  if (svtkRectilinearGrid* rg = svtkRectilinearGrid::SafeDownCast(ds))
  {
    SetArrayType(rg, this->Points);
  }
  else if (svtkHyperTreeGrid* htg = svtkHyperTreeGrid::SafeDownCast(ds))
  {
    SetArrayType(htg, this->Points);
  }
  else if (ds->IsA("svtkImageData"))
  {
    // ImageData Origin and Spacing are doubles so
    // the data type for this should also be double
    this->Points->SetDataType(SVTK_DOUBLE);
  }
}

//------------------------------------------------------------------------------
bool svtkDataSetCellIterator::IsDoneWithTraversal()
{
  return this->DataSet == nullptr || this->CellId >= this->DataSet->GetNumberOfCells();
}

//------------------------------------------------------------------------------
svtkIdType svtkDataSetCellIterator::GetCellId()
{
  return this->CellId;
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::IncrementToNextCell()
{
  ++this->CellId;
}

//------------------------------------------------------------------------------
svtkDataSetCellIterator::svtkDataSetCellIterator()
  : svtkCellIterator()
  , DataSet(nullptr)
  , CellId(0)
{
}

//------------------------------------------------------------------------------
svtkDataSetCellIterator::~svtkDataSetCellIterator() = default;

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::ResetToFirstCell()
{
  this->CellId = 0;
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::FetchCellType()
{
  this->CellType = this->DataSet->GetCellType(this->CellId);
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::FetchPointIds()
{
  this->DataSet->GetCellPoints(this->CellId, this->PointIds);
}

//------------------------------------------------------------------------------
void svtkDataSetCellIterator::FetchPoints()
{
  // This will fetch the point ids if needed:
  svtkIdList* pointIds = this->GetPointIds();

  svtkIdType numPoints = pointIds->GetNumberOfIds();
  svtkIdType* id = pointIds->GetPointer(0);

  this->Points->SetNumberOfPoints(numPoints);

  double point[3];
  for (int i = 0; i < numPoints; ++i)
  {
    this->DataSet->GetPoint(*id++, point);
    this->Points->SetPoint(i, point);
  }
}
