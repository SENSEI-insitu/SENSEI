/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTestNewVar.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkTestNewVar.h"
#include "svtkPoints2D.h"

#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkTestNewVar);

svtkTestNewVar::svtkTestNewVar() = default;

svtkTestNewVar::~svtkTestNewVar() = default;

svtkIdType svtkTestNewVar::GetPointsRefCount()
{
  // Note - this is valid until class destruction and then Delete() will be
  // called on the Data object, decrementing its reference count.
  return this->Points->GetReferenceCount();
}

svtkObject* svtkTestNewVar::GetPoints()
{
  return this->Points.GetPointer();
}

svtkObject* svtkTestNewVar::GetPoints2()
{
  return this->Points;
}

void svtkTestNewVar::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Points: " << endl;
  this->Points->PrintSelf(os, indent.GetNextIndent());
}
