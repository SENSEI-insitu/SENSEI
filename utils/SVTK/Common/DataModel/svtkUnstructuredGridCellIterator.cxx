/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGridCellIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkUnstructuredGridCellIterator.h"

#include "svtkCellArray.h"
#include "svtkCellArrayIterator.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnstructuredGrid.h"

#include <cassert>

svtkStandardNewMacro(svtkUnstructuredGridCellIterator);

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Cells)
  {
    os << indent << "Cells:\n";
    this->Cells->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "Cells: (none)" << endl;
  }

  if (this->Types)
  {
    os << indent << "Types:\n";
    this->Types->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "Types: (none)" << endl;
  }

  if (this->FaceConn)
  {
    os << indent << "FaceConn:\n";
    this->FaceConn->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "FaceConn: (none)" << endl;
  }

  if (this->FaceLocs)
  {
    os << indent << "FaceLocs:\n";
    this->FaceLocs->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "FaceLocs: (none)" << endl;
  }

  if (this->Coords)
  {
    os << indent << "Coords:\n";
    this->Coords->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "Coords: (none)" << endl;
  }
}

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::SetUnstructuredGrid(svtkUnstructuredGrid* ug)
{
  // If the unstructured grid has not been initialized yet, these may not exist:
  svtkUnsignedCharArray* cellTypeArray = ug ? ug->GetCellTypesArray() : nullptr;
  svtkCellArray* cellArray = ug ? ug->GetCells() : nullptr;
  svtkPoints* points = ug ? ug->GetPoints() : nullptr;

  if (points)
  {
    this->Points->SetDataType(points->GetDataType());
  }

  if (ug && cellTypeArray && cellArray && points)
  {
    this->Cells = svtk::TakeSmartPointer(cellArray->NewIterator());
    this->Cells->GoToFirstCell();

    this->Types = cellTypeArray;
    this->FaceConn = ug->GetFaces();
    this->FaceLocs = ug->GetFaceLocations();
    this->Coords = points;
  }
}

//------------------------------------------------------------------------------
bool svtkUnstructuredGridCellIterator::IsDoneWithTraversal()
{
  return this->Cells ? this->Cells->IsDoneWithTraversal() : true;
}

//------------------------------------------------------------------------------
svtkIdType svtkUnstructuredGridCellIterator::GetCellId()
{
  return this->Cells->GetCurrentCellId();
}

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::IncrementToNextCell()
{
  this->Cells->GoToNextCell();
}

//------------------------------------------------------------------------------
svtkUnstructuredGridCellIterator::svtkUnstructuredGridCellIterator() {}

//------------------------------------------------------------------------------
svtkUnstructuredGridCellIterator::~svtkUnstructuredGridCellIterator() = default;

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::ResetToFirstCell()
{
  if (this->Cells)
  {
    this->Cells->GoToFirstCell();
  }
}

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::FetchCellType()
{
  const svtkIdType cellId = this->Cells->GetCurrentCellId();
  this->CellType = this->Types->GetValue(cellId);
}

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::FetchPointIds()
{
  this->Cells->GetCurrentCell(this->PointIds);
}

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::FetchPoints()
{
  this->Coords->GetPoints(this->GetPointIds(), this->Points);
}

//------------------------------------------------------------------------------
// Given a pointer into a set of faces, traverse the faces and return the total
// number of ids (including size hints) in the face set.
namespace
{
inline svtkIdType FaceSetSize(const svtkIdType* begin)
{
  const svtkIdType* result = begin;
  svtkIdType numFaces = *(result++);
  while (numFaces-- > 0)
  {
    result += *result + 1;
  }
  return result - begin;
}
} // end anon namespace

//------------------------------------------------------------------------------
void svtkUnstructuredGridCellIterator::FetchFaces()
{
  if (this->FaceLocs)
  {
    const svtkIdType cellId = this->Cells->GetCurrentCellId();
    const svtkIdType faceLoc = this->FaceLocs->GetValue(cellId);
    const svtkIdType* faceSet = this->FaceConn->GetPointer(faceLoc);
    svtkIdType facesSize = FaceSetSize(faceSet);
    this->Faces->SetNumberOfIds(facesSize);
    svtkIdType* tmpPtr = this->Faces->GetPointer(0);
    std::copy_n(faceSet, facesSize, tmpPtr);
  }
  else
  {
    this->Faces->SetNumberOfIds(0);
  }
}
