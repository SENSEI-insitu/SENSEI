/*==============================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedDataArray.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

==============================================================================*/

#ifndef svtkMappedDataArray_txx
#define svtkMappedDataArray_txx

#include "svtkMappedDataArray.h"

#include "svtkVariant.h" // for svtkVariant

//------------------------------------------------------------------------------
template <class Scalar>
svtkMappedDataArray<Scalar>::svtkMappedDataArray()
{
  this->TemporaryScalarPointer = nullptr;
  this->TemporaryScalarPointerSize = 0;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkMappedDataArray<Scalar>::~svtkMappedDataArray()
{
  delete[] this->TemporaryScalarPointer;
  this->TemporaryScalarPointer = nullptr;
  this->TemporaryScalarPointerSize = 0;
}

//------------------------------------------------------------------------------
template <class Scalar>
void* svtkMappedDataArray<Scalar>::GetVoidPointer(svtkIdType id)
{
  svtkWarningMacro(<< "GetVoidPointer called. This is very expensive for "
                     "svtkMappedDataArray subclasses, since the scalar array must "
                     "be generated for each call. Consider using "
                     "a svtkTypedDataArrayIterator instead.");
  size_t numValues = this->NumberOfComponents * this->GetNumberOfTuples();

  if (this->TemporaryScalarPointer && this->TemporaryScalarPointerSize != numValues)
  {
    delete[] this->TemporaryScalarPointer;
    this->TemporaryScalarPointer = nullptr;
    this->TemporaryScalarPointerSize = 0;
  }

  if (!this->TemporaryScalarPointer)
  {
    this->TemporaryScalarPointer = new Scalar[numValues];
    this->TemporaryScalarPointerSize = numValues;
  }

  this->ExportToVoidPointer(static_cast<void*>(this->TemporaryScalarPointer));

  return static_cast<void*>(this->TemporaryScalarPointer + id);
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::ExportToVoidPointer(void* voidPtr)
{
  svtkTypedDataArrayIterator<Scalar> begin(this, 0);
  svtkTypedDataArrayIterator<Scalar> end =
    begin + (this->NumberOfComponents * this->GetNumberOfTuples());

  Scalar* ptr = static_cast<Scalar*>(voidPtr);

  while (begin != end)
  {
    *ptr = *begin;
    ++begin;
    ++ptr;
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::SetVoidArray(void*, svtkIdType, int)
{
  svtkErrorMacro(<< "SetVoidArray not supported for svtkMappedDataArray "
                   "subclasses.");
  return;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::SetVoidArray(void*, svtkIdType, int, int)
{
  svtkErrorMacro(<< "SetVoidArray not supported for svtkMappedDataArray "
                   "subclasses.");
  return;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::DataChanged()
{
  if (!this->TemporaryScalarPointer)
  {
    svtkWarningMacro(<< "DataChanged called, but no scalar pointer available.");
    return;
  }

  svtkTypedDataArrayIterator<Scalar> begin(this, 0);
  svtkTypedDataArrayIterator<Scalar> end = begin + this->TemporaryScalarPointerSize;

  Scalar* ptr = this->TemporaryScalarPointer;

  while (begin != end)
  {
    *begin = *ptr;
    ++begin;
    ++ptr;
  }

  this->Modified();
}

//------------------------------------------------------------------------------
template <class Scalar>
inline svtkMappedDataArray<Scalar>* svtkMappedDataArray<Scalar>::FastDownCast(
  svtkAbstractArray* source)
{
  if (source)
  {
    switch (source->GetArrayType())
    {
      case svtkAbstractArray::MappedDataArray:
        if (svtkDataTypesCompare(source->GetDataType(), svtkTypeTraits<Scalar>::SVTK_TYPE_ID))
        {
          return static_cast<svtkMappedDataArray<Scalar>*>(source);
        }
      default:
        break;
    }
  }
  return nullptr;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "TemporaryScalarPointer: " << this->TemporaryScalarPointer << "\n";
  os << indent << "TemporaryScalarPointerSize: " << this->TemporaryScalarPointerSize << "\n";
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkMappedDataArray<Scalar>::Modified()
{
  this->svtkTypedDataArray<Scalar>::Modified();

  if (this->TemporaryScalarPointer == nullptr)
  {
    return;
  }

  delete[] this->TemporaryScalarPointer;
  this->TemporaryScalarPointer = nullptr;
  this->TemporaryScalarPointerSize = 0;
}

#endif // svtkMappedDataArray_txx
