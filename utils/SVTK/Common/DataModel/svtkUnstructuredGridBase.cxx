/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGridBase.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkUnstructuredGridBase.h"

#include "svtkCellIterator.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkSmartPointer.h"

//----------------------------------------------------------------------------
svtkUnstructuredGridBase::svtkUnstructuredGridBase() = default;

//----------------------------------------------------------------------------
svtkUnstructuredGridBase::~svtkUnstructuredGridBase() = default;

//----------------------------------------------------------------------------
void svtkUnstructuredGridBase::DeepCopy(svtkDataObject* src)
{
  this->Superclass::DeepCopy(src);

  if (svtkDataSet* ds = svtkDataSet::SafeDownCast(src))
  {
    svtkSmartPointer<svtkCellIterator> cellIter =
      svtkSmartPointer<svtkCellIterator>::Take(ds->NewCellIterator());
    for (cellIter->InitTraversal(); !cellIter->IsDoneWithTraversal(); cellIter->GoToNextCell())
    {
      this->InsertNextCell(cellIter->GetCellType(), cellIter->GetNumberOfPoints(),
        cellIter->GetPointIds()->GetPointer(0), cellIter->GetNumberOfFaces(),
        cellIter->GetFaces()->GetPointer(1));
    }
  }
}

//----------------------------------------------------------------------------
svtkUnstructuredGridBase* svtkUnstructuredGridBase::GetData(svtkInformation* info)
{
  return svtkUnstructuredGridBase::SafeDownCast(info ? info->Get(DATA_OBJECT()) : nullptr);
}

//----------------------------------------------------------------------------
svtkUnstructuredGridBase* svtkUnstructuredGridBase::GetData(svtkInformationVector* v, int i)
{
  return svtkUnstructuredGridBase::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
svtkIdType svtkUnstructuredGridBase::InsertNextCell(int type, svtkIdType npts, const svtkIdType pts[])
{
  return this->InternalInsertNextCell(type, npts, pts);
}

//----------------------------------------------------------------------------
svtkIdType svtkUnstructuredGridBase::InsertNextCell(int type, svtkIdList* ptIds)
{
  return this->InternalInsertNextCell(type, ptIds);
}

//----------------------------------------------------------------------------
svtkIdType svtkUnstructuredGridBase::InsertNextCell(
  int type, svtkIdType npts, const svtkIdType pts[], svtkIdType nfaces, const svtkIdType faces[])
{
  return this->InternalInsertNextCell(type, npts, pts, nfaces, faces);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGridBase::ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[])
{
  this->InternalReplaceCell(cellId, npts, pts);
}
