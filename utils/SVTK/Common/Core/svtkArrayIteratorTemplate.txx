/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayIteratorTemplate.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkArrayIteratorTemplate_txx
#define svtkArrayIteratorTemplate_txx

#include "svtkArrayIteratorTemplate.h"

#include "svtkAbstractArray.h"
#include "svtkObjectFactory.h"

//-----------------------------------------------------------------------------
template <class T>
svtkArrayIteratorTemplate<T>* svtkArrayIteratorTemplate<T>::New()
{
  SVTK_STANDARD_NEW_BODY(svtkArrayIteratorTemplate<T>);
}

template <class T>
svtkCxxSetObjectMacro(svtkArrayIteratorTemplate<T>, Array, svtkAbstractArray);

//-----------------------------------------------------------------------------
template <class T>
svtkArrayIteratorTemplate<T>::svtkArrayIteratorTemplate()
{
  this->Array = nullptr;
  this->Pointer = nullptr;
}

//-----------------------------------------------------------------------------
template <class T>
svtkArrayIteratorTemplate<T>::~svtkArrayIteratorTemplate()
{
  this->SetArray(nullptr);
  this->Pointer = nullptr;
}

//-----------------------------------------------------------------------------
template <class T>
void svtkArrayIteratorTemplate<T>::Initialize(svtkAbstractArray* a)
{
  this->SetArray(a);
  this->Pointer = nullptr;
  if (this->Array)
  {
    this->Pointer = static_cast<T*>(this->Array->GetVoidPointer(0));
  }
}

//-----------------------------------------------------------------------------
template <class T>
svtkIdType svtkArrayIteratorTemplate<T>::GetNumberOfTuples()
{
  if (this->Array)
  {
    return this->Array->GetNumberOfTuples();
  }
  return 0;
}

//-----------------------------------------------------------------------------
template <class T>
svtkIdType svtkArrayIteratorTemplate<T>::GetNumberOfValues()
{
  if (this->Array)
  {
    return (this->Array->GetNumberOfTuples() * this->Array->GetNumberOfComponents());
  }
  return 0;
}

//-----------------------------------------------------------------------------
template <class T>
int svtkArrayIteratorTemplate<T>::GetNumberOfComponents()
{
  if (this->Array)
  {
    return this->Array->GetNumberOfComponents();
  }
  return 0;
}

//-----------------------------------------------------------------------------
template <class T>
T* svtkArrayIteratorTemplate<T>::GetTuple(svtkIdType id)
{
  return &this->Pointer[id * this->Array->GetNumberOfComponents()];
}

//-----------------------------------------------------------------------------
template <class T>
int svtkArrayIteratorTemplate<T>::GetDataType() const
{
  return this->Array->GetDataType();
}

//-----------------------------------------------------------------------------
template <class T>
int svtkArrayIteratorTemplate<T>::GetDataTypeSize() const
{
  return this->Array->GetDataTypeSize();
}

//-----------------------------------------------------------------------------
template <class T>
void svtkArrayIteratorTemplate<T>::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Array: ";
  if (this->Array)
  {
    os << "\n";
    this->Array->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << "(none)"
       << "\n";
  }
}

#endif
