/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkNonMergingPointLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkNonMergingPointLocator.h"

#include "svtkObjectFactory.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkNonMergingPointLocator);

//----------------------------------------------------------------------------
int svtkNonMergingPointLocator::InsertUniquePoint(const double x[3], svtkIdType& ptId)
{
  ptId = this->Points->InsertNextPoint(x);
  return 1;
}

//----------------------------------------------------------------------------
void svtkNonMergingPointLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
