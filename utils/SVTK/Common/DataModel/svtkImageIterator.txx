/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImageIterator.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkImageIterator_txx
#define svtkImageIterator_txx

#include "svtkImageData.h"
#include "svtkImageIterator.h"

//----------------------------------------------------------------------------
template <class DType>
svtkImageIterator<DType>::svtkImageIterator()
{
  this->Pointer = nullptr;
  this->EndPointer = nullptr;
  this->SpanEndPointer = nullptr;
  this->SliceEndPointer = nullptr;
}

//----------------------------------------------------------------------------
template <class DType>
void svtkImageIterator<DType>::Initialize(svtkImageData* id, int* ext)
{
  this->Pointer = static_cast<DType*>(id->GetScalarPointerForExtent(ext));
  id->GetIncrements(this->Increments[0], this->Increments[1], this->Increments[2]);
  id->GetContinuousIncrements(ext, this->ContinuousIncrements[0], this->ContinuousIncrements[1],
    this->ContinuousIncrements[2]);
  this->EndPointer =
    static_cast<DType*>(id->GetScalarPointer(ext[1], ext[3], ext[5])) + this->Increments[0];

  // if the extent is empty then the end pointer should equal the beg pointer
  if (ext[1] < ext[0] || ext[3] < ext[2] || ext[5] < ext[4])
  {
    this->EndPointer = this->Pointer;
  }

  this->SpanEndPointer = this->Pointer + this->Increments[0] * (ext[1] - ext[0] + 1);
  this->SliceEndPointer = this->Pointer + this->Increments[1] * (ext[3] - ext[2] + 1);
}

//----------------------------------------------------------------------------
template <class DType>
svtkImageIterator<DType>::svtkImageIterator(svtkImageData* id, int* ext)
{
  this->Initialize(id, ext);
}

//----------------------------------------------------------------------------
template <class DType>
void svtkImageIterator<DType>::NextSpan()
{
  this->Pointer += this->Increments[1];
  this->SpanEndPointer += this->Increments[1];
  if (this->Pointer >= this->SliceEndPointer)
  {
    this->Pointer += this->ContinuousIncrements[2];
    this->SpanEndPointer += this->ContinuousIncrements[2];
    this->SliceEndPointer += this->Increments[2];
  }
}

#endif
