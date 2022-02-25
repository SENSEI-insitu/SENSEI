/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellLinks.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCellLinks.h"

#include "svtkCellArray.h"
#include "svtkDataSet.h"
#include "svtkGenericCell.h"
#include "svtkObjectFactory.h"
#include "svtkPolyData.h"

#include <vector>

svtkStandardNewMacro(svtkCellLinks);

//----------------------------------------------------------------------------
svtkCellLinks::~svtkCellLinks()
{
  this->Type = svtkAbstractCellLinks::CELL_LINKS;
  this->Initialize();
}

//----------------------------------------------------------------------------
void svtkCellLinks::Initialize()
{
  if (this->Array != nullptr)
  {
    for (svtkIdType i = 0; i <= this->MaxId; i++)
    {
      delete[] this->Array[i].cells;
    }

    delete[] this->Array;
    this->Array = nullptr;
  }
}

//----------------------------------------------------------------------------
void svtkCellLinks::Allocate(svtkIdType sz, svtkIdType ext)
{
  static svtkCellLinks::Link linkInit = { 0, nullptr };

  this->Size = sz;
  delete[] this->Array;
  this->Array = new svtkCellLinks::Link[sz];
  this->Extend = ext;
  this->MaxId = -1;

  for (svtkIdType i = 0; i < sz; i++)
  {
    this->Array[i] = linkInit;
  }
}

//----------------------------------------------------------------------------
// Allocate memory for the list of lists of cell ids.
void svtkCellLinks::AllocateLinks(svtkIdType n)
{
  for (svtkIdType i = 0; i < n; i++)
  {
    this->Array[i].cells = new svtkIdType[this->Array[i].ncells];
  }
}

//----------------------------------------------------------------------------
// Reclaim any unused memory.
void svtkCellLinks::Squeeze()
{
  this->Resize(this->MaxId + 1);
}

//----------------------------------------------------------------------------
void svtkCellLinks::Reset()
{
  this->MaxId = -1;
}

//----------------------------------------------------------------------------
//
// Private function does "reallocate"
//
svtkCellLinks::Link* svtkCellLinks::Resize(svtkIdType sz)
{
  svtkIdType i;
  svtkCellLinks::Link* newArray;
  svtkIdType newSize;
  svtkCellLinks::Link linkInit = { 0, nullptr };

  if (sz >= this->Size)
  {
    newSize = this->Size + sz;
  }
  else
  {
    newSize = sz;
  }

  newArray = new svtkCellLinks::Link[newSize];

  for (i = 0; i < sz && i < this->Size; i++)
  {
    newArray[i] = this->Array[i];
  }

  for (i = this->Size; i < newSize; i++)
  {
    newArray[i] = linkInit;
  }

  this->Size = newSize;
  delete[] this->Array;
  this->Array = newArray;

  return this->Array;
}

//----------------------------------------------------------------------------
// Build the link list array.
void svtkCellLinks::BuildLinks(svtkDataSet* data)
{
  svtkIdType numPts = data->GetNumberOfPoints();
  svtkIdType numCells = data->GetNumberOfCells();
  int j;
  svtkIdType cellId;

  // fill out lists with number of references to cells
  std::vector<svtkIdType> linkLoc(numPts, 0);

  // Use fast path if polydata
  if (data->GetDataObjectType() == SVTK_POLY_DATA)
  {
    svtkIdType npts;
    const svtkIdType* pts;

    svtkPolyData* pdata = static_cast<svtkPolyData*>(data);
    // traverse data to determine number of uses of each point
    for (cellId = 0; cellId < numCells; cellId++)
    {
      pdata->GetCellPoints(cellId, npts, pts);
      for (j = 0; j < npts; j++)
      {
        this->IncrementLinkCount(pts[j]);
      }
    }

    // now allocate storage for the links
    this->AllocateLinks(numPts);
    this->MaxId = numPts - 1;

    for (cellId = 0; cellId < numCells; cellId++)
    {
      pdata->GetCellPoints(cellId, npts, pts);
      for (j = 0; j < npts; j++)
      {
        this->InsertCellReference(pts[j], (linkLoc[pts[j]])++, cellId);
      }
    }
  }

  else // any other type of dataset
  {
    svtkIdType numberOfPoints, ptId;
    svtkGenericCell* cell = svtkGenericCell::New();

    // traverse data to determine number of uses of each point
    for (cellId = 0; cellId < numCells; cellId++)
    {
      data->GetCell(cellId, cell);
      numberOfPoints = cell->GetNumberOfPoints();
      for (j = 0; j < numberOfPoints; j++)
      {
        this->IncrementLinkCount(cell->PointIds->GetId(j));
      }
    }

    // now allocate storage for the links
    this->AllocateLinks(numPts);
    this->MaxId = numPts - 1;

    for (cellId = 0; cellId < numCells; cellId++)
    {
      data->GetCell(cellId, cell);
      numberOfPoints = cell->GetNumberOfPoints();
      for (j = 0; j < numberOfPoints; j++)
      {
        ptId = cell->PointIds->GetId(j);
        this->InsertCellReference(ptId, (linkLoc[ptId])++, cellId);
      }
    }
    cell->Delete();
  } // end else
}

//----------------------------------------------------------------------------
// Insert a new point into the cell-links data structure. The size parameter
// is the initial size of the list.
svtkIdType svtkCellLinks::InsertNextPoint(int numLinks)
{
  if (++this->MaxId >= this->Size)
  {
    this->Resize(this->MaxId + 1);
  }
  this->Array[this->MaxId].cells = new svtkIdType[numLinks];
  return this->MaxId;
}

//----------------------------------------------------------------------------
unsigned long svtkCellLinks::GetActualMemorySize()
{
  svtkIdType size = 0;
  svtkIdType ptId;

  for (ptId = 0; ptId < (this->MaxId + 1); ptId++)
  {
    size += this->GetNcells(ptId);
  }

  size *= sizeof(int*);                                   // references to cells
  size += (this->MaxId + 1) * sizeof(svtkCellLinks::Link); // list of cell lists

  return static_cast<unsigned long>(ceil(size / 1024.0)); // kibibytes
}

//----------------------------------------------------------------------------
void svtkCellLinks::DeepCopy(svtkAbstractCellLinks* src)
{
  svtkCellLinks* clinks = static_cast<svtkCellLinks*>(src);
  this->Allocate(clinks->Size, clinks->Extend);
  memcpy(this->Array, clinks->Array, this->Size * sizeof(svtkCellLinks::Link));
  this->MaxId = clinks->MaxId;
}

//----------------------------------------------------------------------------
void svtkCellLinks::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Size: " << this->Size << "\n";
  os << indent << "MaxId: " << this->MaxId << "\n";
  os << indent << "Extend: " << this->Extend << "\n";
}
