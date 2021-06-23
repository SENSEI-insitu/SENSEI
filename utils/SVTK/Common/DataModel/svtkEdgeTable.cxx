/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEdgeTable.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkEdgeTable.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkVoidArray.h"

svtkStandardNewMacro(svtkEdgeTable);

//----------------------------------------------------------------------------
// Instantiate object based on maximum point id.
svtkEdgeTable::svtkEdgeTable()
{
  this->Table = nullptr;
  this->Attributes = nullptr;
  this->PointerAttributes = nullptr;
  this->Points = nullptr;

  this->TableMaxId = -1;
  this->TableSize = 0;

  this->Position[0] = 0;
  this->Position[1] = -1;
  this->NumberOfEdges = 0;
}

//----------------------------------------------------------------------------
// Free memory and return to instantiated state.
void svtkEdgeTable::Initialize()
{
  svtkIdType i;

  if (this->Table)
  {
    for (i = 0; i < this->TableSize; i++)
    {
      if (this->Table[i])
      {
        this->Table[i]->Delete();
      }
    }
    delete[] this->Table;
    this->Table = nullptr;
    this->TableMaxId = -1;

    if (this->StoreAttributes == 1)
    {
      for (i = 0; i < this->TableSize; i++)
      {
        if (this->Attributes[i])
        {
          this->Attributes[i]->Delete();
        }
      }
      delete[] this->Attributes;
      this->Attributes = nullptr;
    }
    else if (this->StoreAttributes == 2)
    {
      for (i = 0; i < this->TableSize; i++)
      {
        if (this->PointerAttributes[i])
        {
          this->PointerAttributes[i]->Delete();
        }
      }
      delete[] this->PointerAttributes;
      this->PointerAttributes = nullptr;
    }
  } // if table defined

  if (this->Points)
  {
    this->Points->Delete();
    this->Points = nullptr;
  }

  this->TableSize = 0;
  this->NumberOfEdges = 0;
}

//----------------------------------------------------------------------------
// Free memory and return to instantiated state.
void svtkEdgeTable::Reset()
{
  svtkIdType i;

  if (this->Table)
  {
    for (i = 0; i < this->TableSize; i++)
    {
      if (this->Table[i])
      {
        this->Table[i]->Reset();
      }
    }

    if (this->StoreAttributes == 1 && this->Attributes)
    {
      for (i = 0; i < this->TableSize; i++)
      {
        if (this->Attributes[i])
        {
          this->Attributes[i]->Reset();
        }
      }
    }
    else if (this->StoreAttributes == 2 && this->PointerAttributes)
    {
      for (i = 0; i < this->TableSize; i++)
      {
        if (this->PointerAttributes[i])
        {
          this->PointerAttributes[i]->Reset();
        }
      }
    }
  } // if table defined

  this->TableMaxId = -1;

  if (this->Points)
  {
    this->Points->Reset();
  }

  this->NumberOfEdges = 0;
}

//----------------------------------------------------------------------------
svtkEdgeTable::~svtkEdgeTable()
{
  this->Initialize();
}

//----------------------------------------------------------------------------
int svtkEdgeTable::InitEdgeInsertion(svtkIdType numPoints, int storeAttributes)
{
  svtkIdType i;

  numPoints = (numPoints < 1 ? 1 : numPoints);

  // Discard old memory if not enough has been previously allocated
  this->StoreAttributes = storeAttributes;
  this->TableMaxId = -1;

  if (numPoints > this->TableSize)
  {
    this->Initialize();
    this->Table = new svtkIdList*[numPoints];
    for (i = 0; i < numPoints; i++)
    {
      this->Table[i] = nullptr;
    }

    if (this->StoreAttributes == 1)
    {
      this->Attributes = new svtkIdList*[numPoints];
      for (i = 0; i < numPoints; i++)
      {
        this->Attributes[i] = nullptr;
      }
    }
    else if (this->StoreAttributes == 2)
    {
      this->PointerAttributes = new svtkVoidArray*[numPoints];
      for (i = 0; i < numPoints; i++)
      {
        this->PointerAttributes[i] = nullptr;
      }
    }
    this->TableSize = numPoints;
  }

  // Otherwise, reuse the old memory
  else
  {
    this->Reset();
  }

  this->Position[0] = 0;
  this->Position[1] = -1;

  this->NumberOfEdges = 0;

  return 1;
}

//----------------------------------------------------------------------------
// Return non-negative if edge (p1,p2) is an edge; otherwise -1.
svtkIdType svtkEdgeTable::IsEdge(svtkIdType p1, svtkIdType p2)
{
  svtkIdType index, search;

  if (p1 < p2)
  {
    index = p1;
    search = p2;
  }
  else
  {
    index = p2;
    search = p1;
  }

  if (index > this->TableMaxId || this->Table[index] == nullptr)
  {
    return (-1);
  }
  else
  {
    svtkIdType loc;
    if ((loc = this->Table[index]->IsId(search)) == (-1))
    {
      return (-1);
    }
    else
    {
      if (this->StoreAttributes == 1)
      {
        return this->Attributes[index]->GetId(loc);
      }
      else
      {
        return 1;
      }
    }
  }
}

//----------------------------------------------------------------------------
// Return non-negative if edge (p1,p2) is an edge; otherwise -1.
void svtkEdgeTable::IsEdge(svtkIdType p1, svtkIdType p2, void*& ptr)
{
  svtkIdType index, search;

  if (p1 < p2)
  {
    index = p1;
    search = p2;
  }
  else
  {
    index = p2;
    search = p1;
  }

  if (index > this->TableMaxId || this->Table[index] == nullptr)
  {
    ptr = nullptr;
  }
  else
  {
    svtkIdType loc;
    if ((loc = this->Table[index]->IsId(search)) == (-1))
    {
      ptr = nullptr;
    }
    else
    {
      if (this->StoreAttributes == 2)
      {
        ptr = this->PointerAttributes[index]->GetVoidPointer(loc);
      }
      else
      {
        ptr = nullptr;
      }
    }
  }
}

//----------------------------------------------------------------------------
// Insert the edge (p1,p2) into the table. It is the user's responsibility to
// check if the edge has already been inserted.
svtkIdType svtkEdgeTable::InsertEdge(svtkIdType p1, svtkIdType p2)
{
  svtkIdType index, search;

  if (p1 < p2)
  {
    index = p1;
    search = p2;
  }
  else
  {
    index = p2;
    search = p1;
  }

  if (index >= this->TableSize)
  {
    this->Resize(index + 1);
  }

  if (index > this->TableMaxId)
  {
    this->TableMaxId = index;
  }

  if (this->Table[index] == nullptr)
  {
    this->Table[index] = svtkIdList::New();
    this->Table[index]->Allocate(6, 12);
    if (this->StoreAttributes == 1)
    {
      if (this->Attributes[index])
      {
        this->Attributes[index]->Delete();
      }
      this->Attributes[index] = svtkIdList::New();
      this->Attributes[index]->Allocate(6, 12);
    }
  }

  this->Table[index]->InsertNextId(search);
  if (this->StoreAttributes == 1)
  {
    this->Attributes[index]->InsertNextId(this->NumberOfEdges);
  }
  this->NumberOfEdges++;

  return (this->NumberOfEdges - 1);
}

//----------------------------------------------------------------------------
void svtkEdgeTable::InsertEdge(svtkIdType p1, svtkIdType p2, svtkIdType attributeId)
{
  svtkIdType index, search;

  if (p1 < p2)
  {
    index = p1;
    search = p2;
  }
  else
  {
    index = p2;
    search = p1;
  }

  if (index >= this->TableSize)
  {
    this->Resize(index + 1);
  }

  if (index > this->TableMaxId)
  {
    this->TableMaxId = index;
  }

  if (this->Table[index] == nullptr)
  {
    this->Table[index] = svtkIdList::New();
    this->Table[index]->Allocate(6, 12);
    if (this->StoreAttributes == 1)
    {
      this->Attributes[index] = svtkIdList::New();
      this->Attributes[index]->Allocate(6, 12);
    }
  }

  this->NumberOfEdges++;
  this->Table[index]->InsertNextId(search);
  if (this->StoreAttributes)
  {
    this->Attributes[index]->InsertNextId(attributeId);
  }
}

//----------------------------------------------------------------------------
void svtkEdgeTable::InsertEdge(svtkIdType p1, svtkIdType p2, void* ptr)
{
  svtkIdType index, search;

  if (p1 < p2)
  {
    index = p1;
    search = p2;
  }
  else
  {
    index = p2;
    search = p1;
  }

  if (index >= this->TableSize)
  {
    this->Resize(index + 1);
  }

  if (index > this->TableMaxId)
  {
    this->TableMaxId = index;
  }

  if (this->Table[index] == nullptr)
  {
    this->Table[index] = svtkIdList::New();
    this->Table[index]->Allocate(6, 12);
    if (this->StoreAttributes == 2)
    {
      this->PointerAttributes[index] = svtkVoidArray::New();
      this->PointerAttributes[index]->Allocate(6, 12);
    }
  }

  this->NumberOfEdges++;
  this->Table[index]->InsertNextId(search);
  if (this->StoreAttributes == 2)
  {
    this->PointerAttributes[index]->InsertNextVoidPointer(ptr);
  }
}

//----------------------------------------------------------------------------
// Initialize traversal of edges in table.
void svtkEdgeTable::InitTraversal()
{
  this->Position[0] = 0;
  this->Position[1] = -1;
}

// Traverse list of edges in table. Return the edge as (p1,p2), where p1 and
// p2 are point id's. Method return value is <0 if the list is exhausted;
// otherwise a valid id >=0. The value of p1 is guaranteed to be <= p2. The
// return value is an id that can be used for accessing attributes.
svtkIdType svtkEdgeTable::GetNextEdge(svtkIdType& p1, svtkIdType& p2)
{
  for (; this->Position[0] <= this->TableMaxId; this->Position[0]++, this->Position[1] = (-1))
  {
    if (this->Table[this->Position[0]] != nullptr &&
      ++this->Position[1] < this->Table[this->Position[0]]->GetNumberOfIds())
    {
      p1 = this->Position[0];
      p2 = this->Table[this->Position[0]]->GetId(this->Position[1]);
      if (this->StoreAttributes == 1)
      {
        return this->Attributes[this->Position[0]]->GetId(this->Position[1]);
      }
      else
      {
        return (-1);
      }
    }
  }

  return (-1);
}

//----------------------------------------------------------------------------
// Traverse list of edges in table. Return the edge as (p1,p2), where p1 and
// p2 are point id's. The value of p1 is guaranteed to be <= p2. The
// return value is either 1 for success or 0 if the list is exhausted.
int svtkEdgeTable::GetNextEdge(svtkIdType& p1, svtkIdType& p2, void*& ptr)
{
  for (; this->Position[0] <= this->TableMaxId; this->Position[0]++, this->Position[1] = (-1))
  {
    if (this->Table[this->Position[0]] != nullptr &&
      ++this->Position[1] < this->Table[this->Position[0]]->GetNumberOfIds())
    {
      p1 = this->Position[0];
      p2 = this->Table[this->Position[0]]->GetId(this->Position[1]);
      if (this->StoreAttributes == 2)
      {
        this->IsEdge(p1, p2, ptr);
      }
      else
      {
        ptr = nullptr;
      }
      return 1;
    }
  }
  return 0;
}

svtkIdList** svtkEdgeTable::Resize(svtkIdType sz)
{
  svtkIdList** newTableArray;
  svtkIdList** newAttributeArray;
  svtkVoidArray** newPointerAttributeArray;
  svtkIdType newSize, i;
  svtkIdType extend = this->TableSize / 2 + 1;

  if (sz >= this->TableSize)
  {
    newSize = this->TableSize + extend * (((sz - this->TableSize) / extend) + 1);
  }
  else
  {
    newSize = sz;
  }

  sz = (sz < this->TableSize ? sz : this->TableSize);
  newTableArray = new svtkIdList*[newSize];
  memcpy(newTableArray, this->Table, sz * sizeof(svtkIdList*));
  for (i = sz; i < newSize; i++)
  {
    newTableArray[i] = nullptr;
  }
  this->TableSize = newSize;
  delete[] this->Table;
  this->Table = newTableArray;

  if (this->StoreAttributes == 1)
  {
    newAttributeArray = new svtkIdList*[newSize];
    memcpy(newAttributeArray, this->Attributes, sz * sizeof(svtkIdList*));
    for (i = sz; i < newSize; i++)
    {
      newAttributeArray[i] = nullptr;
    }
    delete[] this->Attributes;
    this->Attributes = newAttributeArray;
  }
  else if (this->StoreAttributes == 2)
  {
    newPointerAttributeArray = new svtkVoidArray*[newSize];
    memcpy(newPointerAttributeArray, this->Attributes, sz * sizeof(svtkVoidArray*));
    for (i = sz; i < newSize; i++)
    {
      newPointerAttributeArray[i] = nullptr;
    }
    delete[] this->PointerAttributes;
    this->PointerAttributes = newPointerAttributeArray;
  }

  return this->Table;
}

//----------------------------------------------------------------------------
int svtkEdgeTable::InitPointInsertion(svtkPoints* newPts, svtkIdType estSize)
{
  // Initialize
  if (this->Table)
  {
    this->Initialize();
  }
  if (newPts == nullptr)
  {
    svtkErrorMacro(<< "Must define points for point insertion");
    return 0;
  }
  if (this->Points != nullptr)
  {
    this->Points->Delete();
  }
  // Set up the edge insertion
  this->InitEdgeInsertion(estSize, 1);

  this->Points = newPts;
  this->Points->Register(this);

  return 1;
}

//----------------------------------------------------------------------------
int svtkEdgeTable::InsertUniquePoint(svtkIdType p1, svtkIdType p2, double x[3], svtkIdType& ptId)
{
  svtkIdType loc = this->IsEdge(p1, p2);

  if (loc != -1)
  {
    ptId = loc;
    return 0;
  }
  else
  {
    ptId = this->InsertEdge(p1, p2);
    this->Points->InsertPoint(ptId, x);
    return 1;
  }
}

//----------------------------------------------------------------------------
void svtkEdgeTable::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "NumberOfEdges: " << this->GetNumberOfEdges() << "\n";
}
