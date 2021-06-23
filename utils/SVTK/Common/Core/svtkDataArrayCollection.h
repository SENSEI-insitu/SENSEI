/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataArrayCollection
 * @brief   maintain an ordered list of dataarray objects
 *
 * svtkDataArrayCollection is an object that creates and manipulates lists of
 * datasets. See also svtkCollection and subclasses.
 */

#ifndef svtkDataArrayCollection_h
#define svtkDataArrayCollection_h

#include "svtkCollection.h"
#include "svtkCommonCoreModule.h" // For export macro

#include "svtkDataArray.h" // Needed for inline methods

class SVTKCOMMONCORE_EXPORT svtkDataArrayCollection : public svtkCollection
{
public:
  static svtkDataArrayCollection* New();
  svtkTypeMacro(svtkDataArrayCollection, svtkCollection);

  /**
   * Add a dataarray to the bottom of the list.
   */
  void AddItem(svtkDataArray* ds) { this->svtkCollection::AddItem(ds); }

  /**
   * Get the next dataarray in the list.
   */
  svtkDataArray* GetNextItem() { return static_cast<svtkDataArray*>(this->GetNextItemAsObject()); }

  /**
   * Get the ith dataarray in the list.
   */
  svtkDataArray* GetItem(int i) { return static_cast<svtkDataArray*>(this->GetItemAsObject(i)); }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkDataArray* GetNextDataArray(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkDataArray*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkDataArrayCollection() {}
  ~svtkDataArrayCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkDataArrayCollection(const svtkDataArrayCollection&) = delete;
  void operator=(const svtkDataArrayCollection&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkDataArrayCollection.h
