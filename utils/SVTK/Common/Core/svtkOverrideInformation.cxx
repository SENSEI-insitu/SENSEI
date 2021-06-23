/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOverrideInformation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkOverrideInformation.h"

#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkOverrideInformation);
svtkCxxSetObjectMacro(svtkOverrideInformation, ObjectFactory, svtkObjectFactory);

svtkOverrideInformation::svtkOverrideInformation()
{
  this->ClassOverrideName = nullptr;
  this->ClassOverrideWithName = nullptr;
  this->Description = nullptr;
  this->ObjectFactory = nullptr;
}

svtkOverrideInformation::~svtkOverrideInformation()
{
  delete[] this->ClassOverrideName;
  delete[] this->ClassOverrideWithName;
  delete[] this->Description;
  if (this->ObjectFactory)
  {
    this->ObjectFactory->Delete();
  }
}

void svtkOverrideInformation::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Override: ";
  if (this->ClassOverrideName && this->ClassOverrideWithName && this->Description)
  {
    os << this->ClassOverrideName << "\nWith: " << this->ClassOverrideWithName
       << "\nDescription: " << this->Description;
  }
  else
  {
    os << "(none)\n";
  }

  os << indent << "From Factory:\n";
  if (this->ObjectFactory)
  {
    this->ObjectFactory->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    svtkIndent n = indent.GetNextIndent();
    os << n << "(none)\n";
  }
}
