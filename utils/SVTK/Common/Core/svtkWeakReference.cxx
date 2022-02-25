/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWeakReference.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkWeakReference.h"
#include "svtkObjectFactory.h"
#include "svtkWeakPointer.h"

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkWeakReference);

//----------------------------------------------------------------------------
svtkWeakReference::svtkWeakReference() = default;

//----------------------------------------------------------------------------
svtkWeakReference::~svtkWeakReference() = default;

//----------------------------------------------------------------------------
void svtkWeakReference::Set(svtkObject* object)
{
  this->Object = object;
}

//----------------------------------------------------------------------------
svtkObject* svtkWeakReference::Get()
{
  return this->Object;
}
