/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractElectronicData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAbstractElectronicData.h"

//----------------------------------------------------------------------------
svtkAbstractElectronicData::svtkAbstractElectronicData()
  : Padding(0.0)
{
}

//----------------------------------------------------------------------------
svtkAbstractElectronicData::~svtkAbstractElectronicData() = default;

//----------------------------------------------------------------------------
void svtkAbstractElectronicData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Padding: " << this->Padding << "\n";
}

//----------------------------------------------------------------------------
void svtkAbstractElectronicData::DeepCopy(svtkDataObject* obj)
{
  svtkAbstractElectronicData* aed = svtkAbstractElectronicData::SafeDownCast(obj);
  if (!aed)
  {
    svtkErrorMacro("Can only deep copy from svtkAbstractElectronicData "
                  "or subclass.");
    return;
  }

  // Call superclass
  this->Superclass::DeepCopy(aed);

  // Copy ivars
  this->Padding = aed->Padding;
}
