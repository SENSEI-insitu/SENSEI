/*=========================================================================

 Program:   Visualization Toolkit
 Module:    svtkHierarchicalBoxDataIterator.cxx

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

 =========================================================================*/
#include "svtkHierarchicalBoxDataIterator.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkHierarchicalBoxDataIterator);

svtkHierarchicalBoxDataIterator::svtkHierarchicalBoxDataIterator() = default;

//------------------------------------------------------------------------------
svtkHierarchicalBoxDataIterator::~svtkHierarchicalBoxDataIterator() = default;

//------------------------------------------------------------------------------
void svtkHierarchicalBoxDataIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
