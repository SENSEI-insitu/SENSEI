/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedUnstructuredGrid.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMappedUnstructuredGrid.h"

#include "svtkGenericCell.h"
#include <algorithm>

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::PrintSelf(
  ostream& os, svtkIndent indent)
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
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::CopyStructure(svtkDataSet* ds)
{
  if (ThisType* grid = ThisType::SafeDownCast(ds))
  {
    this->SetImplementation(grid->GetImplementation());
  }

  this->Superclass::CopyStructure(ds);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::ShallowCopy(svtkDataObject* src)
{
  if (ThisType* grid = ThisType::SafeDownCast(src))
  {
    this->SetImplementation(grid->GetImplementation());
  }

  this->Superclass::ShallowCopy(src);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkIdType svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetNumberOfCells()
{
  return this->Impl->GetNumberOfCells();
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkCell* svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetCell(svtkIdType cellId)
{
  this->GetCell(cellId, this->TempCell);
  return this->TempCell;
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetCell(
  svtkIdType cellId, svtkGenericCell* cell)
{
  cell->SetCellType(this->Impl->GetCellType(cellId));
  this->Impl->GetCellPoints(cellId, cell->PointIds);
  this->Points->GetPoints(cell->PointIds, cell->Points);

  if (cell->RequiresInitialization())
  {
    cell->Initialize();
  }
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
int svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetCellType(svtkIdType cellId)
{
  return this->Impl->GetCellType(cellId);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetCellPoints(
  svtkIdType cellId, svtkIdList* ptIds)
{
  this->Impl->GetCellPoints(cellId, ptIds);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkCellIterator* svtkMappedUnstructuredGrid<Implementation, CellIterator>::NewCellIterator()
{
  CellIteratorType* cellIterator = CellIteratorType::New();
  cellIterator->SetMappedUnstructuredGrid(this);
  return cellIterator;
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetPointCells(
  svtkIdType ptId, svtkIdList* cellIds)
{
  this->Impl->GetPointCells(ptId, cellIds);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
int svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetMaxCellSize()
{
  return this->Impl->GetMaxCellSize();
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetIdsOfCellsOfType(
  int type, svtkIdTypeArray* array)
{
  this->Impl->GetIdsOfCellsOfType(type, array);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
int svtkMappedUnstructuredGrid<Implementation, CellIterator>::IsHomogeneous()
{
  return this->Impl->IsHomogeneous();
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::Allocate(svtkIdType numCells, int)
{
  return this->Impl->Allocate(numCells);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkIdType svtkMappedUnstructuredGrid<Implementation, CellIterator>::InternalInsertNextCell(
  int type, svtkIdList* ptIds)
{
  return this->Impl->InsertNextCell(type, ptIds);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkIdType svtkMappedUnstructuredGrid<Implementation, CellIterator>::InternalInsertNextCell(
  int type, svtkIdType npts, const svtkIdType ptIds[])
{
  return this->Impl->InsertNextCell(type, npts, ptIds);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkIdType svtkMappedUnstructuredGrid<Implementation, CellIterator>::InternalInsertNextCell(
  int type, svtkIdType npts, const svtkIdType ptIds[], svtkIdType nfaces, const svtkIdType faces[])
{
  return this->Impl->InsertNextCell(type, npts, ptIds, nfaces, faces);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::InternalReplaceCell(
  svtkIdType cellId, int npts, const svtkIdType pts[])
{
  this->Impl->ReplaceCell(cellId, npts, pts);
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkMTimeType svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetMTime()
{
  return std::max(this->MTime.GetMTime(), this->Impl->GetMTime());
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkMappedUnstructuredGrid<Implementation, CellIterator>::svtkMappedUnstructuredGrid()
  : Impl(nullptr)
{
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
svtkMappedUnstructuredGrid<Implementation, CellIterator>::~svtkMappedUnstructuredGrid()
{
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
void svtkMappedUnstructuredGrid<Implementation, CellIterator>::SetImplementation(
  Implementation* impl)
{
  this->Impl = impl;
  this->Modified();
}

//------------------------------------------------------------------------------
template <class Implementation, class CellIterator>
Implementation* svtkMappedUnstructuredGrid<Implementation, CellIterator>::GetImplementation()
{
  return this->Impl;
}
