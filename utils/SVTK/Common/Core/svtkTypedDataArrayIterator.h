/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypedDataArrayIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkTypedDataArrayIterator
 * @brief   STL-style random access iterator for
 * svtkTypedDataArrays.
 *
 *
 * svtkTypedDataArrayIterator provides an STL-style iterator that can be used to
 * interact with instances of svtkTypedDataArray. It is intended to provide an
 * alternative to using svtkDataArray::GetVoidPointer() that only uses
 * svtkTypedDataArray API functions to retrieve values. It is especially helpful
 * for safely iterating through subclasses of svtkMappedDataArray, which may
 * not use the same memory layout as a typical svtkDataArray.
 *
 * NOTE: This class has been superceded by the newer svtkGenericDataArray and
 * svtkArrayDispatch mechanism.
 */

#ifndef svtkTypedDataArrayIterator_h
#define svtkTypedDataArrayIterator_h

#include <iterator> // For iterator traits

#include "svtkTypedDataArray.h" // For svtkTypedDataArray

template <class Scalar>
class svtkTypedDataArrayIterator
{
public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef Scalar value_type;
  typedef std::ptrdiff_t difference_type;
  typedef Scalar& reference;
  typedef Scalar* pointer;

  svtkTypedDataArrayIterator()
    : Data(nullptr)
    , Index(0)
  {
  }

  explicit svtkTypedDataArrayIterator(svtkTypedDataArray<Scalar>* arr, const svtkIdType index = 0)
    : Data(arr)
    , Index(index)
  {
  }

  svtkTypedDataArrayIterator(const svtkTypedDataArrayIterator& o)
    : Data(o.Data)
    , Index(o.Index)
  {
  }

  svtkTypedDataArrayIterator& operator=(svtkTypedDataArrayIterator<Scalar> o)
  {
    std::swap(this->Data, o.Data);
    std::swap(this->Index, o.Index);
    return *this;
  }

  bool operator==(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index == o.Index;
  }

  bool operator!=(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index != o.Index;
  }

  bool operator>(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index > o.Index;
  }

  bool operator>=(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index >= o.Index;
  }

  bool operator<(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index < o.Index;
  }

  bool operator<=(const svtkTypedDataArrayIterator<Scalar>& o) const
  {
    return this->Data == o.Data && this->Index <= o.Index;
  }

  Scalar& operator*() { return this->Data->GetValueReference(this->Index); }

  Scalar* operator->() const { return &this->Data->GetValueReference(this->Index); }

  Scalar& operator[](const difference_type& n)
  {
    return this->Data->GetValueReference(this->Index + n);
  }

  svtkTypedDataArrayIterator& operator++()
  {
    ++this->Index;
    return *this;
  }

  svtkTypedDataArrayIterator& operator--()
  {
    --this->Index;
    return *this;
  }

  svtkTypedDataArrayIterator operator++(int)
  {
    return svtkTypedDataArrayIterator(this->Data, this->Index++);
  }

  svtkTypedDataArrayIterator operator--(int)
  {
    return svtkTypedDataArrayIterator(this->Data, this->Index--);
  }

  svtkTypedDataArrayIterator operator+(const difference_type& n) const
  {
    return svtkTypedDataArrayIterator(this->Data, this->Index + n);
  }

  svtkTypedDataArrayIterator operator-(const difference_type& n) const
  {
    return svtkTypedDataArrayIterator(this->Data, this->Index - n);
  }

  difference_type operator-(const svtkTypedDataArrayIterator& other) const
  {
    return this->Index - other.Index;
  }

  svtkTypedDataArrayIterator& operator+=(const difference_type& n)
  {
    this->Index += n;
    return *this;
  }

  svtkTypedDataArrayIterator& operator-=(const difference_type& n)
  {
    this->Index -= n;
    return *this;
  }

private:
  svtkTypedDataArray<Scalar>* Data;
  svtkIdType Index;
};

#endif // svtkTypedDataArrayIterator_h

// SVTK-HeaderTest-Exclude: svtkTypedDataArrayIterator.h
