/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericAttribute.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkGenericAttribute - Objects that manage some attribute data.
// .SECTION Description

#include "svtkGenericAttribute.h"
#include <cassert>

//---------------------------------------------------------------------------
svtkGenericAttribute::svtkGenericAttribute() = default;

//---------------------------------------------------------------------------
svtkGenericAttribute::~svtkGenericAttribute() = default;

//---------------------------------------------------------------------------
void svtkGenericAttribute::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Name: " << this->GetName() << endl;
  os << indent << "Number of components: " << this->GetNumberOfComponents() << endl;
  os << indent << "Centering: ";

  switch (this->GetCentering())
  {
    case svtkPointCentered:
      os << "on points";
      break;
    case svtkCellCentered:
      os << "on cells";
      break;
    case svtkBoundaryCentered:
      os << "on boundaries";
      break;
    default:
      assert("check: Impossible case" && 0);
      break;
  }
  os << endl;
}
