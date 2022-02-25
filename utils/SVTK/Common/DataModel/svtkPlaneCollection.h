/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPlaneCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPlaneCollection
 * @brief   maintain a list of planes
 *
 * svtkPlaneCollection is an object that creates and manipulates
 * lists of objects of type svtkPlane.
 * @sa
 * svtkCollection
 */

#ifndef svtkPlaneCollection_h
#define svtkPlaneCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkPlane.h" // Needed for inline methods

class SVTKCOMMONDATAMODEL_EXPORT svtkPlaneCollection : public svtkCollection
{
public:
  svtkTypeMacro(svtkPlaneCollection, svtkCollection);
  static svtkPlaneCollection* New();

  /**
   * Add a plane to the list.
   */
  void AddItem(svtkPlane*);

  /**
   * Get the next plane in the list.
   */
  svtkPlane* GetNextItem();

  /**
   * Get the ith plane in the list.
   */
  svtkPlane* GetItem(int i) { return static_cast<svtkPlane*>(this->GetItemAsObject(i)); }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkPlane* GetNextPlane(svtkCollectionSimpleIterator& cookie);

protected:
  svtkPlaneCollection() {}
  ~svtkPlaneCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkPlaneCollection(const svtkPlaneCollection&) = delete;
  void operator=(const svtkPlaneCollection&) = delete;
};

inline void svtkPlaneCollection::AddItem(svtkPlane* f)
{
  this->svtkCollection::AddItem(f);
}

inline svtkPlane* svtkPlaneCollection::GetNextItem()
{
  return static_cast<svtkPlane*>(this->GetNextItemAsObject());
}

#endif
// SVTK-HeaderTest-Exclude: svtkPlaneCollection.h
