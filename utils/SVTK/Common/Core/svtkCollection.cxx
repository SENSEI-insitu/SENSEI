/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCollection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCollection.h"

#include "svtkCollectionIterator.h"
#include "svtkGarbageCollector.h"
#include "svtkObjectFactory.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

svtkStandardNewMacro(svtkCollection);

// Construct with empty list.
svtkCollection::svtkCollection()
{
  this->NumberOfItems = 0;
  this->Top = nullptr;
  this->Bottom = nullptr;
  this->Current = nullptr;
}

// Destructor for the svtkCollection class. This removes all
// objects from the collection.
svtkCollection::~svtkCollection()
{
  this->RemoveAllItems();
}

// protected function to delete an element. Internal use only.
void svtkCollection::DeleteElement(svtkCollectionElement* e)
{
  if (e->Item != nullptr)
  {
    e->Item->UnRegister(this);
  }
  delete e;
}

// protected function to remove an element. Internal use only.
void svtkCollection::RemoveElement(svtkCollectionElement* elem, svtkCollectionElement* prev)
{
  assert(elem);
  if (prev)
  {
    prev->Next = elem->Next;
  }
  else
  {
    this->Top = elem->Next;
  }

  if (!elem->Next)
  {
    this->Bottom = prev;
  }

  if (this->Current == elem)
  {
    this->Current = elem->Next;
  }

  this->NumberOfItems--;
  this->DeleteElement(elem);
}

// Add an object to the bottom of the list. Does not prevent duplicate entries.
void svtkCollection::AddItem(svtkObject* a)
{
  svtkCollectionElement* elem;

  elem = new svtkCollectionElement;

  if (!this->Top)
  {
    this->Top = elem;
  }
  else
  {
    this->Bottom->Next = elem;
  }
  this->Bottom = elem;

  a->Register(this);
  elem->Item = a;
  elem->Next = nullptr;

  this->Modified();

  this->NumberOfItems++;
}

// Insert an object into the list. There must be at least one
// entry pre-existing.
void svtkCollection::InsertItem(int i, svtkObject* a)
{
  if (i >= this->NumberOfItems || !this->Top)
  {
    return;
  }

  svtkCollectionElement* elem;

  elem = new svtkCollectionElement;
  svtkCollectionElement* curr = this->Top;

  if (i < 0)
  {
    this->Top = elem;
    elem->Next = curr;
  }
  else
  {
    svtkCollectionElement* next = curr->Next;

    int j = 0;
    while (j != i)
    {
      curr = next;
      next = curr->Next;
      j++;
    }

    curr->Next = elem;
    if (curr == this->Bottom)
    {
      this->Bottom = elem;
    }
    else
    {
      elem->Next = next;
    }
  }

  a->Register(this);
  elem->Item = a;

  this->Modified();

  this->NumberOfItems++;
}

// Remove an object from the list. Removes the first object found, not
// all occurrences. If no object found, list is unaffected.  See warning
// in description of RemoveItem(int).
void svtkCollection::RemoveItem(svtkObject* a)
{
  if (!this->Top)
  {
    return;
  }

  svtkCollectionElement* prev = nullptr;
  svtkCollectionElement* elem = this->Top;
  for (int i = 0; i < this->NumberOfItems; i++)
  {
    if (elem->Item == a)
    {
      this->RemoveElement(elem, prev);
      this->Modified();
      return;
    }
    else
    {
      prev = elem;
      elem = elem->Next;
    }
  }
}

// Remove all objects from the list.
void svtkCollection::RemoveAllItems()
{
  // Don't modify if collection is empty
  if (this->NumberOfItems == 0)
  {
    return;
  }

  while (this->NumberOfItems)
  {
    this->RemoveElement(this->Top, nullptr);
  }

  this->Modified();
}

// Search for an object and return location in list. If location == 0,
// object was not found.
int svtkCollection::IsItemPresent(svtkObject* a)
{
  int i;
  svtkCollectionElement* elem;

  if (!this->Top)
  {
    return 0;
  }

  elem = this->Top;
  for (i = 0; i < this->NumberOfItems; i++)
  {
    if (elem->Item == a)
    {
      return i + 1;
    }
    else
    {
      elem = elem->Next;
    }
  }

  return 0;
}

void svtkCollection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number Of Items: " << this->NumberOfItems << "\n";
}

// Get the i'th item in the collection. nullptr is returned if i is out
// of range
svtkObject* svtkCollection::GetItemAsObject(int i)
{
  svtkCollectionElement* elem = this->Top;

  if (i < 0)
  {
    return nullptr;
  }

  if (i == this->NumberOfItems - 1)
  {
    // optimize for the special case where we're looking for the last elem
    elem = this->Bottom;
  }
  else
  {
    while (elem != nullptr && i > 0)
    {
      elem = elem->Next;
      i--;
    }
  }
  if (elem != nullptr)
  {
    return elem->Item;
  }
  else
  {
    return nullptr;
  }
}

// Replace the i'th item in the collection with a
void svtkCollection::ReplaceItem(int i, svtkObject* a)
{
  svtkCollectionElement* elem;

  if (i < 0 || i >= this->NumberOfItems)
  {
    return;
  }

  elem = this->Top;
  if (i == this->NumberOfItems - 1)
  {
    elem = this->Bottom;
  }
  else
  {
    for (int j = 0; j < i; j++, elem = elem->Next)
    {
    }
  }

  // Take care of reference counting
  if (elem->Item != nullptr)
  {
    elem->Item->UnRegister(this);
  }
  a->Register(this);

  // j == i
  elem->Item = a;

  this->Modified();
}

// Remove the i'th item in the list.
// Be careful if using this function during traversal of the list using
// GetNextItemAsObject (or GetNextItem in derived class).  The list WILL
// be shortened if a valid index is given!  If this->Current is equal to the
// element being removed, have it point to then next element in the list.
void svtkCollection::RemoveItem(int i)
{
  svtkCollectionElement *elem, *prev;

  if (i < 0 || i >= this->NumberOfItems)
  {
    return;
  }

  elem = this->Top;
  prev = nullptr;
  for (int j = 0; j < i; j++)
  {
    prev = elem;
    elem = elem->Next;
  }

  this->RemoveElement(elem, prev);
  this->Modified();
}

svtkCollectionIterator* svtkCollection::NewIterator()
{
  svtkCollectionIterator* it = svtkCollectionIterator::New();
  it->SetCollection(this);
  return it;
}

//----------------------------------------------------------------------------
void svtkCollection::Register(svtkObjectBase* o)
{
  this->RegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkCollection::UnRegister(svtkObjectBase* o)
{
  this->UnRegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkCollection::ReportReferences(svtkGarbageCollector* collector)
{
  this->Superclass::ReportReferences(collector);
  for (svtkCollectionElement* elem = this->Top; elem; elem = elem->Next)
  {
    svtkGarbageCollectorReport(collector, elem->Item, "Element");
  }
}
