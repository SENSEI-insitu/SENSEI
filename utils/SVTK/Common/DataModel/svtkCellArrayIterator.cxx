/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellArrayIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellArrayIterator.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkCellArrayIterator);

void svtkCellArrayIterator::PrintSelf(std::ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "CurrentCellId: " << this->CurrentCellId << "\n";
  os << indent << "CellArray: " << this->CellArray.Get() << "\n";
}
