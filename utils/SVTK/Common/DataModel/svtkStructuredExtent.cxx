/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStructuredExtent.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkStructuredExtent.h"

#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkStructuredExtent);
//----------------------------------------------------------------------------
svtkStructuredExtent::svtkStructuredExtent() = default;

//----------------------------------------------------------------------------
svtkStructuredExtent::~svtkStructuredExtent() = default;

//----------------------------------------------------------------------------
void svtkStructuredExtent::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
