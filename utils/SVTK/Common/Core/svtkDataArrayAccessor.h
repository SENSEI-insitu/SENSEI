/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayAccessor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkDataArrayAccessor
 * @brief   Efficient templated access to svtkDataArray.
 *
 * @warning svtkDataArrayAccessor has been replaced by the much easier to use
 * range facilities svtk::DataArrayTupleRange and svtk::DataArrayValueRange,
 * defined in svtkDataArrayRange.h. This accessor class shouldn't need to be
 * used directly.
 *
 * svtkDataArrayAccessor provides access to data stored in a svtkDataArray. It
 * is intended to be used in conjunction with svtkArrayDispatcher.
 *
 * A more detailed description of this class and related tools can be found
 * \ref SVTK-7-1-ArrayDispatch "here".
 *
 * The goal of this helper template is to allow developers to write a single
 * templated worker function that will generates code to use the efficient typed
 * APIs provided by svtkGenericDataArray when the array type is known, but
 * fallback to the slower svtkDataArray virtual double API if needed.
 *
 * This can be used to reduce template-explosion issues by restricting the
 * svtkArrayDispatch call to only dispatch a few common cases/array types, and
 * route all other arrays through a slower implementation using svtkDataArray's
 * API. With svtkDataArrayAccessor, a single templated worker function can be
 * used to generate both.
 *
 * Note that while this interface provides both component-wise and
 * tuple access, the tuple methods are discouraged as they are
 * significantly slower as they copy data into a temporary array, and
 * prevent many advanced compiler optimizations that are possible when
 * using the component accessors. In other words, prefer the methods
 * that operate on a single component instead of an entire tuple when
 * performance matters.
 *
 * A standard usage pattern of this class would be:
 *
 * @code
 * // svtkArrayDispatch worker struct:
 * struct Worker
 * {
 *   // Templated worker function:
 *   template <typename ArrayT>
 *   void operator()(ArrayT *array)
 *   {
 *     // The accessor:
 *     svtkDataArrayAccessor<ArrayT> accessor(array);
 *     // The data type used by ArrayT's API, use this for
 *     // temporary/intermediate results:
 *     typedef typename svtkDataArrayAccessor<ArrayT>::APIType APIType;
 *
 *     // Do work using accessor to set/get values....
 *   }
 * };
 *
 * // Usage:
 * Worker worker;
 * svtkDataArray *array = ...;
 * if (!svtkArrayDispatch::Dispatch::Execute(array, worker))
 *   {
 *   // Dispatch failed: unknown array type. Fallback to svtkDataArray API:
 *   worker(array);
 *   }
 * @endcode
 *
 * We define Worker::operator() as the templated worker function, restricting
 * all data accesses to go through the 'accessor' object (methods like
 * GetNumberOfTuples, GetNumberOfComponents, etc should be called on the array
 * itself).
 *
 * This worker is passed into an array dispatcher, which tests 'array' to see
 * if it can figure out the array subclass. If it does, Worker is instantiated
 * with ArrayT set to the array's subclass, resulting in efficient code. If
 * Dispatch::Execute returns false (meaning the array type is not recognized),
 * the worker is executed using the svtkDataArray pointer. While slower, this
 * ensures that less-common cases will still be handled -- all from one worker
 * function template.
 *
 * .SEE ALSO
 * svtkArrayDispatch svtk::DataArrayValueRange svtk::DataArrayTupleRange
 */

#include "svtkDataArray.h"
#include "svtkGenericDataArray.h"

#ifndef svtkDataArrayAccessor_h
#define svtkDataArrayAccessor_h

#ifndef __SVTK_WRAP__

// Generic form for all (non-bit) svtkDataArray subclasses.
template <typename ArrayT>
struct svtkDataArrayAccessor
{
  typedef ArrayT ArrayType;
  typedef typename ArrayType::ValueType APIType;

  ArrayType* Array;

  svtkDataArrayAccessor(ArrayType* array)
    : Array(array)
  {
  }

  SVTK_ALWAYS_INLINE
  APIType Get(svtkIdType tupleIdx, int compIdx) const
  {
    return this->Array->GetTypedComponent(tupleIdx, compIdx);
  }

  SVTK_ALWAYS_INLINE
  void Set(svtkIdType tupleIdx, int compIdx, APIType val) const
  {
    this->Array->SetTypedComponent(tupleIdx, compIdx, val);
  }

  SVTK_ALWAYS_INLINE
  void Insert(svtkIdType tupleIdx, int compIdx, APIType val) const
  {
    this->Array->InsertTypedComponent(tupleIdx, compIdx, val);
  }

  SVTK_ALWAYS_INLINE
  void Get(svtkIdType tupleIdx, APIType* tuple) const
  {
    this->Array->GetTypedTuple(tupleIdx, tuple);
  }

  SVTK_ALWAYS_INLINE
  void Set(svtkIdType tupleIdx, const APIType* tuple) const
  {
    this->Array->SetTypedTuple(tupleIdx, tuple);
  }

  SVTK_ALWAYS_INLINE
  void Insert(svtkIdType tupleIdx, const APIType* tuple) const
  {
    this->Array->InsertTypedTuple(tupleIdx, tuple);
  }
};

// Specialization for svtkDataArray.
template <>
struct svtkDataArrayAccessor<svtkDataArray>
{
  typedef svtkDataArray ArrayType;
  typedef double APIType;

  ArrayType* Array;

  svtkDataArrayAccessor(ArrayType* array)
    : Array(array)
  {
  }

  SVTK_ALWAYS_INLINE
  APIType Get(svtkIdType tupleIdx, int compIdx) const
  {
    return this->Array->GetComponent(tupleIdx, compIdx);
  }

  SVTK_ALWAYS_INLINE
  void Set(svtkIdType tupleIdx, int compIdx, APIType val) const
  {
    this->Array->SetComponent(tupleIdx, compIdx, val);
  }

  SVTK_ALWAYS_INLINE
  void Insert(svtkIdType tupleIdx, int compIdx, APIType val) const
  {
    this->Array->InsertComponent(tupleIdx, compIdx, val);
  }

  SVTK_ALWAYS_INLINE
  void Get(svtkIdType tupleIdx, APIType* tuple) const { this->Array->GetTuple(tupleIdx, tuple); }

  SVTK_ALWAYS_INLINE
  void Set(svtkIdType tupleIdx, const APIType* tuple) const
  {
    this->Array->SetTuple(tupleIdx, tuple);
  }

  SVTK_ALWAYS_INLINE
  void Insert(svtkIdType tupleIdx, const APIType* tuple) const
  {
    this->Array->InsertTuple(tupleIdx, tuple);
  }
};

#endif

#endif // svtkDataArrayAccessor_h
// SVTK-HeaderTest-Exclude: svtkDataArrayAccessor.h
