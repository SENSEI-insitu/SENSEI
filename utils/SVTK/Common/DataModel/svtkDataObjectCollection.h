/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataObjectCollection
 * @brief   maintain an unordered list of data objects
 *
 * svtkDataObjectCollection is an object that creates and manipulates ordered
 * lists of data objects. See also svtkCollection and subclasses.
 */

#ifndef svtkDataObjectCollection_h
#define svtkDataObjectCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkDataObject.h" // Needed for inline methods

class SVTKCOMMONDATAMODEL_EXPORT svtkDataObjectCollection : public svtkCollection
{
public:
  static svtkDataObjectCollection* New();
  svtkTypeMacro(svtkDataObjectCollection, svtkCollection);

  /**
   * Add a data object to the bottom of the list.
   */
  void AddItem(svtkDataObject* ds) { this->svtkCollection::AddItem(ds); }

  /**
   * Get the next data object in the list.
   */
  svtkDataObject* GetNextItem() { return static_cast<svtkDataObject*>(this->GetNextItemAsObject()); }

  /**
   * Get the ith data object in the list.
   */
  svtkDataObject* GetItem(int i) { return static_cast<svtkDataObject*>(this->GetItemAsObject(i)); }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkDataObject* GetNextDataObject(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkDataObject*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkDataObjectCollection() {}
  ~svtkDataObjectCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkDataObjectCollection(const svtkDataObjectCollection&) = delete;
  void operator=(const svtkDataObjectCollection&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkDataObjectCollection.h
