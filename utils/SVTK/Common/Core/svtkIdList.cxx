/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdList.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkSMPTools.h" //for parallel sort

svtkStandardNewMacro(svtkIdList);

//----------------------------------------------------------------------------
svtkIdList::svtkIdList()
{
  this->NumberOfIds = 0;
  this->Size = 0;
  this->Ids = nullptr;
}

//----------------------------------------------------------------------------
svtkIdList::~svtkIdList()
{
  delete[] this->Ids;
}

//----------------------------------------------------------------------------
void svtkIdList::Initialize()
{
  delete[] this->Ids;
  this->Ids = nullptr;
  this->NumberOfIds = 0;
  this->Size = 0;
}

//----------------------------------------------------------------------------
int svtkIdList::Allocate(const svtkIdType sz, const int svtkNotUsed(strategy))
{
  if (sz > this->Size)
  {
    this->Initialize();
    this->Size = (sz > 0 ? sz : 1);
    if ((this->Ids = new svtkIdType[this->Size]) == nullptr)
    {
      return 0;
    }
  }
  this->NumberOfIds = 0;
  return 1;
}

//----------------------------------------------------------------------------
void svtkIdList::SetNumberOfIds(const svtkIdType number)
{
  this->Allocate(number, 0);
  this->NumberOfIds = number;
}

//----------------------------------------------------------------------------
svtkIdType svtkIdList::InsertUniqueId(const svtkIdType svtkid)
{
  for (svtkIdType i = 0; i < this->NumberOfIds; i++)
  {
    if (svtkid == this->Ids[i])
    {
      return i;
    }
  }

  return this->InsertNextId(svtkid);
}

//----------------------------------------------------------------------------
svtkIdType* svtkIdList::WritePointer(const svtkIdType i, const svtkIdType number)
{
  svtkIdType newSize = i + number;
  if (newSize > this->Size)
  {
    this->Resize(newSize);
  }
  if (newSize > this->NumberOfIds)
  {
    this->NumberOfIds = newSize;
  }
  return this->Ids + i;
}

//----------------------------------------------------------------------------
void svtkIdList::SetArray(svtkIdType* array, svtkIdType size)
{
  delete[] this->Ids;
  this->Ids = array;
  this->NumberOfIds = size;
  this->Size = size;
}

//----------------------------------------------------------------------------
void svtkIdList::DeleteId(svtkIdType svtkid)
{
  svtkIdType i = 0;

  // while loop is necessary to delete all occurrences of svtkid
  while (i < this->NumberOfIds)
  {
    for (; i < this->NumberOfIds; i++)
    {
      if (this->Ids[i] == svtkid)
      {
        break;
      }
    }

    // if found; replace current id with last
    if (i < this->NumberOfIds)
    {
      this->SetId(i, this->Ids[this->NumberOfIds - 1]);
      this->NumberOfIds--;
    }
  }
}

//----------------------------------------------------------------------------
void svtkIdList::DeepCopy(svtkIdList* ids)
{
  this->SetNumberOfIds(ids->NumberOfIds);
  if (ids->NumberOfIds > 0)
  {
    std::copy(ids->Ids, ids->Ids + ids->NumberOfIds, this->Ids);
  }
  this->Squeeze();
}

//----------------------------------------------------------------------------
svtkIdType* svtkIdList::Resize(const svtkIdType sz)
{
  svtkIdType* newIds;
  svtkIdType newSize;

  if (sz > this->Size)
  {
    newSize = this->Size + sz;
  }
  else if (sz == this->Size)
  {
    return this->Ids;
  }
  else
  {
    newSize = sz;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return nullptr;
  }

  if ((newIds = new svtkIdType[newSize]) == nullptr)
  {
    svtkErrorMacro(<< "Cannot allocate memory\n");
    return nullptr;
  }

  if (this->NumberOfIds > newSize)
  {
    this->NumberOfIds = newSize;
  }

  if (this->Ids)
  {
    memcpy(newIds, this->Ids,
      static_cast<size_t>(sz < this->Size ? sz : this->Size) * sizeof(svtkIdType));
    delete[] this->Ids;
  }

  this->Size = newSize;
  this->Ids = newIds;
  return this->Ids;
}

//----------------------------------------------------------------------------
#define SVTK_TMP_ARRAY_SIZE 500
// Intersect this list with another svtkIdList. Updates current list according
// to result of intersection operation.
void svtkIdList::IntersectWith(svtkIdList* otherIds)
{
  // Fast method due to Dr. Andreas Mueller of ISE Integrated Systems
  // Engineering (CH).
  svtkIdType thisNumIds = this->GetNumberOfIds();

  if (thisNumIds <= SVTK_TMP_ARRAY_SIZE)
  { // Use fast method if we can fit in temporary storage
    svtkIdType thisIds[SVTK_TMP_ARRAY_SIZE];
    svtkIdType i, svtkid;

    for (i = 0; i < thisNumIds; i++)
    {
      thisIds[i] = this->GetId(i);
    }
    for (this->Reset(), i = 0; i < thisNumIds; i++)
    {
      svtkid = thisIds[i];
      if (otherIds->IsId(svtkid) != (-1))
      {
        this->InsertNextId(svtkid);
      }
    }
  }
  else
  { // use slower method for extreme cases
    svtkIdType* thisIds = new svtkIdType[thisNumIds];
    svtkIdType i, svtkid;

    for (i = 0; i < thisNumIds; i++)
    {
      *(thisIds + i) = this->GetId(i);
    }
    for (this->Reset(), i = 0; i < thisNumIds; i++)
    {
      svtkid = *(thisIds + i);
      if (otherIds->IsId(svtkid) != (-1))
      {
        this->InsertNextId(svtkid);
      }
    }
    delete[] thisIds;
  }
}
#undef SVTK_TMP_ARRAY_SIZE

//----------------------------------------------------------------------------
void svtkIdList::Sort()
{
  if (this->Ids == nullptr || this->NumberOfIds < 2)
  {
    return;
  }
  svtkSMPTools::Sort(this->Ids, this->Ids + this->NumberOfIds);
}

//----------------------------------------------------------------------------
void svtkIdList::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number of Ids: " << this->NumberOfIds << "\n";
}
