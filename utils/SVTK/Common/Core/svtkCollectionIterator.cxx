/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCollectionIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCollectionIterator.h"
#include "svtkCollection.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkCollectionIterator);

//----------------------------------------------------------------------------
svtkCollectionIterator::svtkCollectionIterator()
{
  this->Element = nullptr;
  this->Collection = nullptr;
}

//----------------------------------------------------------------------------
svtkCollectionIterator::~svtkCollectionIterator()
{
  this->SetCollection(nullptr);
}

//----------------------------------------------------------------------------
void svtkCollectionIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->Collection)
  {
    os << indent << "Collection: " << this->Collection << "\n";
  }
  else
  {
    os << indent << "Collection: (none)\n";
  }
}

//----------------------------------------------------------------------------
void svtkCollectionIterator::SetCollection(svtkCollection* collection)
{
  svtkSetObjectBodyMacro(Collection, svtkCollection, collection);
  this->GoToFirstItem();
}

//----------------------------------------------------------------------------
void svtkCollectionIterator::GoToFirstItem()
{
  if (this->Collection)
  {
    this->Element = this->Collection->Top;
  }
  else
  {
    this->Element = nullptr;
  }
}

//----------------------------------------------------------------------------
void svtkCollectionIterator::GoToNextItem()
{
  if (this->Element)
  {
    this->Element = this->Element->Next;
  }
}

//----------------------------------------------------------------------------
int svtkCollectionIterator::IsDoneWithTraversal()
{
  return (this->Element ? 0 : 1);
}

//----------------------------------------------------------------------------
svtkObject* svtkCollectionIterator::GetCurrentObject()
{
  if (this->Element)
  {
    return this->Element->Item;
  }
  return nullptr;
}
