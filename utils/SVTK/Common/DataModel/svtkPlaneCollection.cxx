/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPlaneCollection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPlaneCollection.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkPlaneCollection);

svtkPlane* svtkPlaneCollection::GetNextPlane(svtkCollectionSimpleIterator& cookie)
{
  return static_cast<svtkPlane*>(this->GetNextItemAsObject(cookie));
}
