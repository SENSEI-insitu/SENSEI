/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObjectFactoryCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkObjectFactoryCollection
 * @brief   maintain a list of object factories
 *
 * svtkObjectFactoryCollection is an object that creates and manipulates
 * ordered lists of objects of type svtkObjectFactory.
 *
 * @sa
 * svtkCollection svtkObjectFactory
 */

#ifndef svtkObjectFactoryCollection_h
#define svtkObjectFactoryCollection_h

#include "svtkCollection.h"
#include "svtkCommonCoreModule.h" // For export macro

#include "svtkObjectFactory.h" // Needed for inline methods

class SVTKCOMMONCORE_EXPORT svtkObjectFactoryCollection : public svtkCollection
{
public:
  svtkTypeMacro(svtkObjectFactoryCollection, svtkCollection);
  static svtkObjectFactoryCollection* New();

  /**
   * Add an ObjectFactory the bottom of the list.
   */
  void AddItem(svtkObjectFactory* t) { this->svtkCollection::AddItem(t); }

  /**
   * Get the next ObjectFactory in the list. Return nullptr when the end of the
   * list is reached.
   */
  svtkObjectFactory* GetNextItem()
  {
    return static_cast<svtkObjectFactory*>(this->GetNextItemAsObject());
  }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkObjectFactory* GetNextObjectFactory(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkObjectFactory*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkObjectFactoryCollection() {}
  ~svtkObjectFactoryCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkObjectFactoryCollection(const svtkObjectFactoryCollection&) = delete;
  void operator=(const svtkObjectFactoryCollection&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkObjectFactoryCollection.h
