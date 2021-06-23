/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdListCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIdListCollection
 * @brief   maintain an ordered list of IdList objects
 *
 * svtkIdListCollection is an object that creates and manipulates lists of
 * IdLists. See also svtkCollection and subclasses.
 */

#ifndef svtkIdListCollection_h
#define svtkIdListCollection_h

#include "svtkCollection.h"
#include "svtkCommonCoreModule.h" // For export macro

#include "svtkIdList.h" // Needed for inline methods

class SVTKCOMMONCORE_EXPORT svtkIdListCollection : public svtkCollection
{
public:
  static svtkIdListCollection* New();
  svtkTypeMacro(svtkIdListCollection, svtkCollection);

  /**
   * Add an IdList to the bottom of the list.
   */
  void AddItem(svtkIdList* ds) { this->svtkCollection::AddItem(ds); }

  /**
   * Get the next IdList in the list.
   */
  svtkIdList* GetNextItem() { return static_cast<svtkIdList*>(this->GetNextItemAsObject()); }

  /**
   * Get the ith IdList in the list.
   */
  svtkIdList* GetItem(int i) { return static_cast<svtkIdList*>(this->GetItemAsObject(i)); }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkIdList* GetNextIdList(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkIdList*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkIdListCollection() {}
  ~svtkIdListCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkIdListCollection(const svtkIdListCollection&) = delete;
  void operator=(const svtkIdListCollection&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkIdListCollection.h
