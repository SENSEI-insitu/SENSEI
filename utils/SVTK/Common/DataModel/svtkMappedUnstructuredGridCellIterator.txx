/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedUnstructuredGridCellIterator.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMappedUnstructuredGridCellIterator.h"

#include "svtkMappedUnstructuredGrid.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"

//------------------------------------------------------------------------------
template <class Implementation>
svtkMappedUnstructuredGridCellIterator<Implementation>*
svtkMappedUnstructuredGridCellIterator<Implementation>::New()
{
  SVTK_STANDARD_NEW_BODY(ThisType);
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Implementation:";
  if (this->Impl == nullptr)
  {
    os << " nullptr" << endl;
  }
  else
  {
    os << endl;
    this->Impl->PrintSelf(os, indent.GetNextIndent());
  }

  os << indent << "GridPoints:";
  if (this->GridPoints == nullptr)
  {
    os << " nullptr" << endl;
  }
  else
  {
    os << endl;
    this->GridPoints->PrintSelf(os, indent.GetNextIndent());
  }
}

//------------------------------------------------------------------------------
template <class Implementation>
bool svtkMappedUnstructuredGridCellIterator<Implementation>::IsDoneWithTraversal()
{
  return this->CellId >= this->NumberOfCells;
}

//------------------------------------------------------------------------------
template <class Implementation>
svtkIdType svtkMappedUnstructuredGridCellIterator<Implementation>::GetCellId()
{
  return this->CellId;
}

//------------------------------------------------------------------------------
template <class Implementation>
svtkMappedUnstructuredGridCellIterator<Implementation>::svtkMappedUnstructuredGridCellIterator()
  : Impl(nullptr)
  , GridPoints(nullptr)
  , CellId(0)
  , NumberOfCells(0)
{
}

//------------------------------------------------------------------------------
template <class Implementation>
svtkMappedUnstructuredGridCellIterator<Implementation>::~svtkMappedUnstructuredGridCellIterator()
{
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::ResetToFirstCell()
{
  this->CellId = 0;
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::IncrementToNextCell()
{
  ++this->CellId;
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::FetchCellType()
{
  this->CellType = this->Impl->GetCellType(this->CellId);
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::FetchPointIds()
{
  this->Impl->GetCellPoints(this->CellId, this->PointIds);
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::FetchPoints()
{
  this->GridPoints->GetPoints(this->GetPointIds(), this->Points);
}

//------------------------------------------------------------------------------
template <class Implementation>
void svtkMappedUnstructuredGridCellIterator<Implementation>::SetMappedUnstructuredGrid(
  svtkMappedUnstructuredGrid<ImplementationType, ThisType>* grid)
{
  this->Impl = grid->GetImplementation();
  this->GridPoints = grid->GetPoints();
  this->CellId = 0;
  this->NumberOfCells = grid->GetNumberOfCells();
  if (this->GridPoints)
  {
    this->Points->SetDataType(this->GridPoints->GetDataType());
  }
}
