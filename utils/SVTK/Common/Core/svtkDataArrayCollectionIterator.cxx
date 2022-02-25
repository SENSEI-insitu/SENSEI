/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayCollectionIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataArrayCollectionIterator.h"
#include "svtkDataArray.h"
#include "svtkDataArrayCollection.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkDataArrayCollectionIterator);

//----------------------------------------------------------------------------
svtkDataArrayCollectionIterator::svtkDataArrayCollectionIterator() = default;

//----------------------------------------------------------------------------
svtkDataArrayCollectionIterator::~svtkDataArrayCollectionIterator() = default;

//----------------------------------------------------------------------------
void svtkDataArrayCollectionIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkDataArrayCollectionIterator::SetCollection(svtkCollection* c)
{
  if (c)
  {
    this->Superclass::SetCollection(svtkDataArrayCollection::SafeDownCast(c));
    if (!this->Collection)
    {
      svtkErrorMacro("svtkDataArrayCollectionIterator cannot traverse a " << c->GetClassName());
    }
  }
  else
  {
    this->Superclass::SetCollection(nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkDataArrayCollectionIterator::SetCollection(svtkDataArrayCollection* c)
{
  this->Superclass::SetCollection(c);
}

//----------------------------------------------------------------------------
svtkDataArray* svtkDataArrayCollectionIterator::GetDataArray()
{
  return static_cast<svtkDataArray*>(this->GetCurrentObject());
}
