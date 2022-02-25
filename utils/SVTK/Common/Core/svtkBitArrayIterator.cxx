/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBitArrayIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBitArrayIterator.h"

#include "svtkBitArray.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkBitArrayIterator);
svtkCxxSetObjectMacro(svtkBitArrayIterator, Array, svtkBitArray);
//-----------------------------------------------------------------------------
svtkBitArrayIterator::svtkBitArrayIterator()
{
  this->Array = nullptr;
  this->Tuple = nullptr;
  this->TupleSize = 0;
}

//-----------------------------------------------------------------------------
svtkBitArrayIterator::~svtkBitArrayIterator()
{
  this->SetArray(nullptr);
  delete[] this->Tuple;
}

//-----------------------------------------------------------------------------
void svtkBitArrayIterator::Initialize(svtkAbstractArray* a)
{
  svtkBitArray* b = svtkArrayDownCast<svtkBitArray>(a);
  if (!b && a)
  {
    svtkErrorMacro("svtkBitArrayIterator can iterate only over svtkBitArray.");
    return;
  }
  this->SetArray(b);
}

//-----------------------------------------------------------------------------
svtkAbstractArray* svtkBitArrayIterator::GetArray()
{
  return this->Array;
}

//-----------------------------------------------------------------------------
int* svtkBitArrayIterator::GetTuple(svtkIdType id)
{
  if (!this->Array)
  {
    return nullptr;
  }

  svtkIdType numComps = this->Array->GetNumberOfComponents();
  if (this->TupleSize < numComps)
  {
    this->TupleSize = static_cast<int>(numComps);
    delete[] this->Tuple;
    this->Tuple = new int[this->TupleSize];
  }
  svtkIdType loc = id * numComps;
  for (int j = 0; j < numComps; j++)
  {
    this->Tuple[j] = this->Array->GetValue(loc + j);
  }
  return this->Tuple;
}

//-----------------------------------------------------------------------------
int svtkBitArrayIterator::GetValue(svtkIdType id)
{
  if (this->Array)
  {
    return this->Array->GetValue(id);
  }
  svtkErrorMacro("Array Iterator not initialized.");
  return 0;
}

//-----------------------------------------------------------------------------
void svtkBitArrayIterator::SetValue(svtkIdType id, int value)
{
  if (this->Array)
  {
    this->Array->SetValue(id, value);
  }
}

//-----------------------------------------------------------------------------
svtkIdType svtkBitArrayIterator::GetNumberOfTuples()
{
  if (this->Array)
  {
    return this->Array->GetNumberOfTuples();
  }
  return 0;
}
//-----------------------------------------------------------------------------
svtkIdType svtkBitArrayIterator::GetNumberOfValues()
{
  if (this->Array)
  {
    return this->Array->GetNumberOfTuples() * this->Array->GetNumberOfComponents();
  }
  return 0;
}
//-----------------------------------------------------------------------------
int svtkBitArrayIterator::GetNumberOfComponents()
{
  if (this->Array)
  {
    return this->Array->GetNumberOfComponents();
  }
  return 0;
}

//-----------------------------------------------------------------------------
int svtkBitArrayIterator::GetDataType() const
{
  if (this->Array)
  {
    return this->Array->GetDataType();
  }
  return 0;
}
//-----------------------------------------------------------------------------
int svtkBitArrayIterator::GetDataTypeSize() const
{
  if (this->Array)
  {
    return this->Array->GetDataTypeSize();
  }
  return 0;
}

//-----------------------------------------------------------------------------
void svtkBitArrayIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
