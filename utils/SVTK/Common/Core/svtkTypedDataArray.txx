/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypedDataArray.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkTypedDataArray_txx
#define svtkTypedDataArray_txx

#include "svtkTypedDataArray.h"

//------------------------------------------------------------------------------
template <typename Scalar>
svtkTypedDataArray<Scalar>::svtkTypedDataArray()
{
}

//------------------------------------------------------------------------------
template <typename Scalar>
svtkTypedDataArray<Scalar>::~svtkTypedDataArray()
{
}

//------------------------------------------------------------------------------
template <typename Scalar>
bool svtkTypedDataArray<Scalar>::AllocateTuples(svtkIdType)
{
  svtkErrorMacro(<< "This method is not preferred for svtkTypedDataArray "
                   "implementations. Either add an appropriate implementation, or "
                   "use Allocate instead.");
  return false;
}

//------------------------------------------------------------------------------
template <typename Scalar>
bool svtkTypedDataArray<Scalar>::ReallocateTuples(svtkIdType)
{
  svtkErrorMacro(<< "This method is not preferred for svtkTypedDataArray "
                   "implementations. Either add an appropriate implementation, or "
                   "use Resize instead.");
  return false;
}

//------------------------------------------------------------------------------
template <typename Scalar>
inline int svtkTypedDataArray<Scalar>::GetDataType() const
{
  return svtkTypeTraits<Scalar>::SVTK_TYPE_ID;
}

//------------------------------------------------------------------------------
template <typename Scalar>
inline int svtkTypedDataArray<Scalar>::GetDataTypeSize() const
{
  return static_cast<int>(sizeof(Scalar));
}

//------------------------------------------------------------------------------
template <typename Scalar>
inline typename svtkTypedDataArray<Scalar>::ValueType svtkTypedDataArray<Scalar>::GetTypedComponent(
  svtkIdType tupleIdx, int comp) const
{
  return this->GetValue(tupleIdx * this->NumberOfComponents + comp);
}

//------------------------------------------------------------------------------
template <typename Scalar>
inline void svtkTypedDataArray<Scalar>::SetTypedComponent(svtkIdType tupleIdx, int comp, ValueType v)
{
  this->SetValue(tupleIdx * this->NumberOfComponents + comp, v);
}

//------------------------------------------------------------------------------
template <typename Scalar>
inline svtkTypedDataArray<Scalar>* svtkTypedDataArray<Scalar>::FastDownCast(svtkAbstractArray* source)
{
  if (source)
  {
    switch (source->GetArrayType())
    {
      case svtkAbstractArray::TypedDataArray:
      case svtkAbstractArray::MappedDataArray:
        if (svtkDataTypesCompare(source->GetDataType(), svtkTypeTraits<Scalar>::SVTK_TYPE_ID))
        {
          return static_cast<svtkTypedDataArray<Scalar>*>(source);
        }
        break;
    }
  }
  return nullptr;
}

#endif // svtkTypedDataArray_txx
