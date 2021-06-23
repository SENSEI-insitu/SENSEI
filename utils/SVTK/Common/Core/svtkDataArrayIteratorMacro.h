/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayIteratorMacro.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @def   svtkDataArrayIteratorMacro
 * @brief   (deprecated) A macro for obtaining iterators to
 * svtkDataArray data when the array implementation and type are unknown.
 *
 * @deprecated This macro is deprecated and should not be used any longer. See
 * svtkArrayDispatch.h and svtkDataArrayRange.h for replacements.
 *
 * See svtkTemplateMacro.
 * This macro is similar, but defines several additional typedefs and variables
 * for safely iterating through data in a svtkAbstractArray container:
 *  - svtkDAValueType is typedef'd to the array's element value type.
 *  - svtkDAContainerType is typedef'd to the most derived class of
 *    svtkAbstractArray for which a suitable iterator has been found.
 *  - svtkDAIteratorType is typedef'd to the most suitable iterator type found.
 *  - svtkDABegin is an object of svtkDAIteratorType that points to the first
 *    component of the first tuple in the array.
 *  - svtkDAEnd is an object of svtkDAIteratorType that points to the element
 *    *after* the last component of the last tuple in the array.
 * The primary advantage to using this macro is that arrays with non-standard
 * memory layouts will be safely handled, and dangerous calls to GetVoidPointer
 * are avoided.
 * For arrays with > 1 component, the iterator will proceed through all
 * components of a tuple before moving on to the next tuple.
 * This matches the memory layout of the standard svtkDataArray subclasses such
 * as svtkFloatArray.
 *
 * For the standard svtkDataArray implementations (which are subclasses of
 * svtkAOSDataArrayTemplate), the iterators will simply be pointers to the raw
 * memory of the array.
 * This allows very fast iteration when possible, and permits compiler
 * optimizations in the standard template library to occur (such as reducing
 * std::copy to memmove).
 *
 * For arrays that are subclasses of svtkTypedDataArray, a
 * svtkTypedDataArrayIterator is used.
 * Such iterators safely traverse the array using API calls and have
 * pointer-like semantics, but add about a 35% performance overhead compared
 * with iterating over the raw memory (measured by summing a svtkFloatArray
 * containing 10M values on GCC 4.8.1 with -O3 optimization using both iterator
 * types -- see TestDataArrayIterators).
 *
 * For arrays that are not subclasses of svtkTypedDataArray, there is no reliably
 * safe way to iterate over the array elements.
 * In such cases, this macro performs the legacy behavior of casting
 * svtkAbstractArray::GetVoidPointer(...) to svtkDAValueType* to create the
 * iterators.
 *
 * To use this macro, create a templated worker function:
 *
 * template <class Iterator>
 * void myFunc(Iterator begin, Iterator end, ...) {...}
 *
 * and then call the svtkDataArrayIteratorMacro inside of a switch statement
 * using the above objects and typedefs as needed:
 *
 * svtkAbstractArray *someArray = ...;
 * switch (someArray->GetDataType())
 *   {
 *   svtkDataArrayIteratorMacro(someArray, myFunc(svtkDABegin, svtkDAEnd, ...));
 *   }
 *
 * @sa
 * svtkArrayDispatch svtkGenericDataArray
 * svtkTemplateMacro svtkTypedDataArrayIterator
 */

#ifndef svtkDataArrayIteratorMacro_h
#define svtkDataArrayIteratorMacro_h

#ifndef SVTK_LEGACY_REMOVE

#include "svtkAOSDataArrayTemplate.h" // For classes referred to in the macro
#include "svtkSetGet.h"               // For svtkTemplateMacro
#include "svtkTypedDataArray.h"       // For classes referred to in the macro

// Silence 'unused typedef' warnings on GCC.
// use of the typedef in question depends on the macro
// argument _call and thus should not be removed.
#if defined(__GNUC__)
#define _svtkDAIMUnused __attribute__((unused))
#else
#define _svtkDAIMUnused
#endif

#define svtkDataArrayIteratorMacro(_array, _call)                                                   \
  svtkTemplateMacro(                                                                                \
    svtkAbstractArray* _aa(_array); if (svtkAOSDataArrayTemplate<SVTK_TT>* _dat =                     \
                                         svtkAOSDataArrayTemplate<SVTK_TT>::FastDownCast(_aa)) {     \
      typedef SVTK_TT svtkDAValueType;                                                               \
      typedef svtkAOSDataArrayTemplate<svtkDAValueType> svtkDAContainerType;                          \
      typedef svtkDAContainerType::Iterator svtkDAIteratorType;                                      \
      svtkDAIteratorType svtkDABegin(_dat->Begin());                                                 \
      svtkDAIteratorType svtkDAEnd(_dat->End());                                                     \
      (void)svtkDABegin; /* Prevent warnings when unused */                                         \
      (void)svtkDAEnd;                                                                              \
      _call;                                                                                       \
    } else if (svtkTypedDataArray<SVTK_TT>* _tda = svtkTypedDataArray<SVTK_TT>::FastDownCast(_aa)) {   \
      typedef SVTK_TT svtkDAValueType;                                                               \
      typedef svtkTypedDataArray<svtkDAValueType> svtkDAContainerType;                                \
      typedef svtkDAContainerType::Iterator svtkDAIteratorType;                                      \
      svtkDAIteratorType svtkDABegin(_tda->Begin());                                                 \
      svtkDAIteratorType svtkDAEnd(_tda->End());                                                     \
      (void)svtkDABegin;                                                                            \
      (void)svtkDAEnd;                                                                              \
      _call;                                                                                       \
    } else {                                                                                       \
      /* This is not ideal, as no explicit iterator has been declared. */                          \
      /* Cast the void pointer and hope for the best!                  */                          \
      typedef SVTK_TT svtkDAValueType;                                                               \
      typedef svtkAbstractArray svtkDAContainerType _svtkDAIMUnused;                                  \
      typedef svtkDAValueType* svtkDAIteratorType;                                                   \
      svtkDAIteratorType svtkDABegin = static_cast<svtkDAIteratorType>(_aa->GetVoidPointer(0));       \
      svtkDAIteratorType svtkDAEnd = svtkDABegin + _aa->GetMaxId() + 1;                               \
      (void)svtkDABegin;                                                                            \
      (void)svtkDAEnd;                                                                              \
      _call;                                                                                       \
    })

#endif // legacy remove

#endif // svtkDataArrayIteratorMacro_h

// SVTK-HeaderTest-Exclude: svtkDataArrayIteratorMacro.h
