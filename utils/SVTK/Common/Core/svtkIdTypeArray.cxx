/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdTypeArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// We never need to instantiate svtkAOSDataArrayTemplate<svtkIdType> or
// svtkArrayIteratorTemplate<svtkIdType> because they are instantiated
// by the corresponding array for its native type.  Therefore this
// code should not be uncommented and is here for reference:
//   #define SVTK_AOS_DATA_ARRAY_TEMPLATE_INSTANTIATING
//   #include "svtkAOSDataArrayTemplate.txx"
//   SVTK_AOS_DATA_ARRAY_TEMPLATE_INSTANTIATE(svtkIdType);

#include "svtkIdTypeArray.h"

#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkIdTypeArray);

//----------------------------------------------------------------------------
svtkIdTypeArray::svtkIdTypeArray() = default;

//----------------------------------------------------------------------------
svtkIdTypeArray::~svtkIdTypeArray() = default;

//----------------------------------------------------------------------------
void svtkIdTypeArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->RealSuperclass::PrintSelf(os, indent);
}
