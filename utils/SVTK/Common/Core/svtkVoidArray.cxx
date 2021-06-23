/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVoidArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkVoidArray.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkVoidArray);

typedef void* voidPtr;

// Instantiate object.
svtkVoidArray::svtkVoidArray()
  : NumberOfPointers(0)
  , Size(0)
  , Array(nullptr)
{
}

svtkVoidArray::~svtkVoidArray()
{
  delete[] this->Array;
}

// Allocate memory for this array. Delete old storage only if necessary.
svtkTypeBool svtkVoidArray::Allocate(svtkIdType sz, svtkIdType svtkNotUsed(ext))
{
  if (sz > this->Size || this->Array != nullptr)
  {
    delete[] this->Array;

    this->Size = (sz > 0 ? sz : 1);
    if ((this->Array = new voidPtr[this->Size]) == nullptr)
    {
      return 0;
    }
  }

  this->NumberOfPointers = 0;

  return 1;
}

// Release storage and reset array to initial state.
void svtkVoidArray::Initialize()
{
  delete[] this->Array;
  this->Array = nullptr;
  this->Size = 0;
  this->NumberOfPointers = 0;
}

// Deep copy of another void array.
void svtkVoidArray::DeepCopy(svtkVoidArray* va)
{
  // Do nothing on a nullptr input.
  if (va == nullptr)
  {
    return;
  }

  if (this != va)
  {
    delete[] this->Array;

    this->NumberOfPointers = va->NumberOfPointers;
    this->Size = va->Size;

    this->Array = new voidPtr[this->Size];
    memcpy(this->Array, va->GetVoidPointer(0), this->Size * sizeof(void*));
  }
}

void** svtkVoidArray::WritePointer(svtkIdType id, svtkIdType number)
{
  svtkIdType newSize = id + number;
  if (newSize > this->Size)
  {
    this->ResizeAndExtend(newSize);
  }
  if (newSize > this->NumberOfPointers)
  {
    this->NumberOfPointers = newSize;
  }
  return this->Array + id;
}

void svtkVoidArray::InsertVoidPointer(svtkIdType id, void* p)
{
  if (id >= this->Size)
  {
    if (!this->ResizeAndExtend(id + 1))
    {
      return;
    }
  }
  this->Array[id] = p;
  if (id >= this->NumberOfPointers)
  {
    this->NumberOfPointers = id + 1;
  }
}

svtkIdType svtkVoidArray::InsertNextVoidPointer(void* p)
{
  this->InsertVoidPointer(this->NumberOfPointers, p);
  return this->NumberOfPointers - 1;
}

// Protected function does "reallocate"
//
void** svtkVoidArray::ResizeAndExtend(svtkIdType sz)
{
  void** newArray;
  svtkIdType newSize;

  if (sz > this->Size)
  {
    newSize = this->Size + sz;
  }
  else if (sz == this->Size)
  {
    return this->Array;
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

  if ((newArray = new voidPtr[newSize]) == nullptr)
  {
    svtkErrorMacro(<< "Cannot allocate memory\n");
    return nullptr;
  }

  memcpy(newArray, this->Array, (sz < this->Size ? sz : this->Size) * sizeof(voidPtr));

  if (newSize < this->Size)
  {
    this->NumberOfPointers = newSize;
  }
  this->Size = newSize;
  delete[] this->Array;
  this->Array = newArray;

  return this->Array;
}

void svtkVoidArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Array)
  {
    os << indent << "Array: " << this->Array << "\n";
  }
  else
  {
    os << indent << "Array: (null)\n";
  }
}
