/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImageIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImageIterator
 * @brief   a simple image iterator
 *
 * This is a simple image iterator that can be used to iterate over an
 * image. This should be used internally by Filter writers.
 *
 * @sa
 * svtkImageData svtkImageProgressIterator
 */

#ifndef svtkImageIterator_h
#define svtkImageIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSystemIncludes.h"
class svtkImageData;

template <class DType>
class SVTKCOMMONDATAMODEL_EXPORT svtkImageIterator
{
public:
  typedef DType* SpanIterator;

  /**
   * Default empty constructor, useful only when creating an array of iterators
   * You need to call Initialize afterward
   */
  svtkImageIterator();

  /**
   * Create an image iterator for a given image data and a given extent
   */
  svtkImageIterator(svtkImageData* id, int* ext);

  /**
   * Initialize the image iterator for a given image data, and given extent
   */
  void Initialize(svtkImageData* id, int* ext);

  /**
   * Move the iterator to the next span
   */
  void NextSpan();

  /**
   * Return an iterator (pointer) for the span
   */
  SpanIterator BeginSpan() { return this->Pointer; }

  /**
   * Return an iterator (pointer) for the end of the span
   */
  SpanIterator EndSpan() { return this->SpanEndPointer; }

  /**
   * Test if the end of the extent has been reached
   */
  svtkTypeBool IsAtEnd() { return (this->Pointer >= this->EndPointer); }

protected:
  DType* Pointer;
  DType* SpanEndPointer;
  DType* SliceEndPointer;
  DType* EndPointer;
  svtkIdType Increments[3];
  svtkIdType ContinuousIncrements[3];
};

#ifndef svtkImageIterator_cxx
#ifdef _MSC_VER
#pragma warning(push)
// The following is needed when the svtkImageIterator is declared
// dllexport and is used from another class in svtkCommonCore
#pragma warning(disable : 4910) // extern and dllexport incompatible
#endif
svtkExternTemplateMacro(extern template class SVTKCOMMONDATAMODEL_EXPORT svtkImageIterator);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif

#endif
// SVTK-HeaderTest-Exclude: svtkImageIterator.h
