/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedLongLongArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Instantiate superclass first to give the template a DLL interface.
#define SVTK_AOS_DATA_ARRAY_TEMPLATE_INSTANTIATING
#include "svtkAOSDataArrayTemplate.txx"
SVTK_AOS_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned long long);

#include "svtkUnsignedLongLongArray.h"

#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkUnsignedLongLongArray);

//----------------------------------------------------------------------------
svtkUnsignedLongLongArray::svtkUnsignedLongLongArray() = default;

//----------------------------------------------------------------------------
svtkUnsignedLongLongArray::~svtkUnsignedLongLongArray() = default;

//----------------------------------------------------------------------------
void svtkUnsignedLongLongArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->RealSuperclass::PrintSelf(os, indent);
}
