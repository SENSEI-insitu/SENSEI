/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCollection
 * @brief   create and manipulate ordered lists of objects
 *
 * svtkCollection is a general object for creating and manipulating lists
 * of objects. The lists are ordered and allow duplicate entries.
 * svtkCollection also serves as a base class for lists of specific types
 * of objects.
 *
 * @sa
 * svtkActorCollection svtkAssemblyPaths svtkDataSetCollection
 * svtkImplicitFunctionCollection svtkLightCollection svtkPolyDataCollection
 * svtkRenderWindowCollection svtkRendererCollection
 * svtkStructuredPointsCollection svtkTransformCollection svtkVolumeCollection
 */

#ifndef svtkCollection_h
#define svtkCollection_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkCollectionElement //;prevents pick-up by man page generator
{
public:
  svtkCollectionElement()
    : Item(nullptr)
    , Next(nullptr)
  {
  }
  svtkObject* Item;
  svtkCollectionElement* Next;
};
typedef void* svtkCollectionSimpleIterator;

class svtkCollectionIterator;

class SVTKCOMMONCORE_EXPORT svtkCollection : public svtkObject
{
public:
  svtkTypeMacro(svtkCollection, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct with empty list.
   */
  static svtkCollection* New();

  /**
   * Add an object to the bottom of the list. Does not prevent duplicate entries.
   */
  void AddItem(svtkObject*);

  /**
   * Insert item into the list after the i'th item. Does not prevent duplicate entries.
   * If i < 0 the item is placed at the top of the list.
   */
  void InsertItem(int i, svtkObject*);

  /**
   * Replace the i'th item in the collection with another item.
   */
  void ReplaceItem(int i, svtkObject*);

  /**
   * Remove the i'th item in the list.
   * Be careful if using this function during traversal of the list using
   * GetNextItemAsObject (or GetNextItem in derived class).  The list WILL
   * be shortened if a valid index is given!  If this->Current is equal to the
   * element being removed, have it point to then next element in the list.
   */
  void RemoveItem(int i);

  /**
   * Remove an object from the list. Removes the first object found, not
   * all occurrences. If no object found, list is unaffected.  See warning
   * in description of RemoveItem(int).
   */
  void RemoveItem(svtkObject*);

  /**
   * Remove all objects from the list.
   */
  void RemoveAllItems();

  /**
   * Search for an object and return location in list. If the return value is
   * 0, the object was not found. If the object was found, the location is
   * the return value-1.
   */
  int IsItemPresent(svtkObject* a);

  /**
   * Return the number of objects in the list.
   */
  int GetNumberOfItems() { return this->NumberOfItems; }

  /**
   * Initialize the traversal of the collection. This means the data pointer
   * is set at the beginning of the list.
   */
  void InitTraversal() { this->Current = this->Top; }

  /**
   * A reentrant safe way to iterate through a collection.
   * Just pass the same cookie value around each time
   */
  void InitTraversal(svtkCollectionSimpleIterator& cookie)
  {
    cookie = static_cast<svtkCollectionSimpleIterator>(this->Top);
  }

  /**
   * Get the next item in the collection. nullptr is returned if the collection
   * is exhausted.
   */
  svtkObject* GetNextItemAsObject();

  /**
   * Get the i'th item in the collection. nullptr is returned if i is out
   * of range
   */
  svtkObject* GetItemAsObject(int i);

  /**
   * A reentrant safe way to get the next object as a collection. Just pass the
   * same cookie back and forth.
   */
  svtkObject* GetNextItemAsObject(svtkCollectionSimpleIterator& cookie);

  /**
   * Get an iterator to traverse the objects in this collection.
   */
  SVTK_NEWINSTANCE svtkCollectionIterator* NewIterator();

  //@{
  /**
   * Participate in garbage collection.
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

protected:
  svtkCollection();
  ~svtkCollection() override;

  virtual void RemoveElement(svtkCollectionElement* element, svtkCollectionElement* previous);
  virtual void DeleteElement(svtkCollectionElement*);
  int NumberOfItems;
  svtkCollectionElement* Top;
  svtkCollectionElement* Bottom;
  svtkCollectionElement* Current;

  friend class svtkCollectionIterator;

  // See svtkGarbageCollector.h:
  void ReportReferences(svtkGarbageCollector* collector) override;

private:
  svtkCollection(const svtkCollection&) = delete;
  void operator=(const svtkCollection&) = delete;
};

inline svtkObject* svtkCollection::GetNextItemAsObject()
{
  svtkCollectionElement* elem = this->Current;

  if (elem != nullptr)
  {
    this->Current = elem->Next;
    return elem->Item;
  }
  else
  {
    return nullptr;
  }
}

inline svtkObject* svtkCollection::GetNextItemAsObject(void*& cookie)
{
  svtkCollectionElement* elem = static_cast<svtkCollectionElement*>(cookie);

  if (elem != nullptr)
  {
    cookie = static_cast<void*>(elem->Next);
    return elem->Item;
  }
  else
  {
    return nullptr;
  }
}

#endif
