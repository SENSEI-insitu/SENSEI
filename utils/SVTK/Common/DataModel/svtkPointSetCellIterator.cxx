/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointSetCellIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkPointSetCellIterator.h"

#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkPointSet.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkPointSetCellIterator);

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "PointSet: " << this->PointSet << endl;
}

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::SetPointSet(svtkPointSet* ds)
{
  this->PointSet = ds;
  this->PointSetPoints = ds ? ds->GetPoints() : nullptr;
  this->CellId = 0;
  if (this->PointSetPoints)
  {
    this->Points->SetDataType(this->PointSetPoints->GetDataType());
  }
}

//------------------------------------------------------------------------------
bool svtkPointSetCellIterator::IsDoneWithTraversal()
{
  return this->PointSet == nullptr || this->CellId >= this->PointSet->GetNumberOfCells();
}

//------------------------------------------------------------------------------
svtkIdType svtkPointSetCellIterator::GetCellId()
{
  return this->CellId;
}

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::IncrementToNextCell()
{
  ++this->CellId;
}

//------------------------------------------------------------------------------
svtkPointSetCellIterator::svtkPointSetCellIterator()
  : svtkCellIterator()
  , PointSet(nullptr)
  , PointSetPoints(nullptr)
  , CellId(0)
{
}

//------------------------------------------------------------------------------
svtkPointSetCellIterator::~svtkPointSetCellIterator() = default;

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::ResetToFirstCell()
{
  this->CellId = 0;
}

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::FetchCellType()
{
  this->CellType = this->PointSet->GetCellType(this->CellId);
}

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::FetchPointIds()
{
  this->PointSet->GetCellPoints(this->CellId, this->PointIds);
}

//------------------------------------------------------------------------------
void svtkPointSetCellIterator::FetchPoints()
{
  svtkIdList* pointIds = this->GetPointIds();
  this->PointSetPoints->GetPoints(pointIds, this->Points);
}
