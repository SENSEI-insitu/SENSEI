/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFunctionSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkFunctionSet.h"

svtkFunctionSet::svtkFunctionSet()
{
  this->NumFuncs = 0;
  this->NumIndepVars = 0;
}

void svtkFunctionSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Number of functions: " << this->NumFuncs << "\n";
  os << indent << "Number of independent variables: " << this->NumIndepVars << "\n";
}
