/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAtomic.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAtomic.h"

namespace detail
{

svtkTypeInt64 AtomicOps<8>::AddAndFetch(svtkTypeInt64* ref, svtkTypeInt64 val)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  {
    (*ref) += val;
    result = *ref;
  }
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::SubAndFetch(svtkTypeInt64* ref, svtkTypeInt64 val)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  {
    (*ref) -= val;
    result = *ref;
  }
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::PreIncrement(svtkTypeInt64* ref)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  result = ++(*ref);
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::PreDecrement(svtkTypeInt64* ref)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  result = --(*ref);
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::PostIncrement(svtkTypeInt64* ref)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  result = (*ref)++;
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::PostDecrement(svtkTypeInt64* ref)
{
  svtkTypeInt64 result;
#pragma omp atomic capture
  result = (*ref)--;
#pragma omp flush
  return result;
}

svtkTypeInt64 AtomicOps<8>::Load(const svtkTypeInt64* ref)
{
  svtkTypeInt64 result;
#pragma omp flush
#pragma omp atomic read
  result = *ref;
  return result;
}

void AtomicOps<8>::Store(svtkTypeInt64* ref, svtkTypeInt64 val)
{
#pragma omp atomic write
  *ref = val;
#pragma omp flush
}

svtkTypeInt32 AtomicOps<4>::AddAndFetch(svtkTypeInt32* ref, svtkTypeInt32 val)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  {
    (*ref) += val;
    result = *ref;
  }
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::SubAndFetch(svtkTypeInt32* ref, svtkTypeInt32 val)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  {
    (*ref) -= val;
    result = *ref;
  }
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::PreIncrement(svtkTypeInt32* ref)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  result = ++(*ref);
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::PreDecrement(svtkTypeInt32* ref)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  result = --(*ref);
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::PostIncrement(svtkTypeInt32* ref)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  result = (*ref)++;
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::PostDecrement(svtkTypeInt32* ref)
{
  svtkTypeInt32 result;
#pragma omp atomic capture
  result = (*ref)--;
#pragma omp flush
  return result;
}

svtkTypeInt32 AtomicOps<4>::Load(const svtkTypeInt32* ref)
{
  svtkTypeInt32 result;
#pragma omp flush
#pragma omp atomic read
  result = *ref;
  return result;
}

void AtomicOps<4>::Store(svtkTypeInt32* ref, svtkTypeInt32 val)
{
#pragma omp atomic write
  *ref = val;
#pragma omp flush
}

}
