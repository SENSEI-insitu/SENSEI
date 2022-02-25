/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTransformCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTransformCollection
 * @brief   maintain a list of transforms
 *
 *
 * svtkTransformCollection is an object that creates and manipulates lists of
 * objects of type svtkTransform.
 *
 * @sa
 * svtkCollection svtkTransform
 */

#ifndef svtkTransformCollection_h
#define svtkTransformCollection_h

#include "svtkCollection.h"
#include "svtkCommonTransformsModule.h" // For export macro

#include "svtkTransform.h" // Needed for inline methods

class SVTKCOMMONTRANSFORMS_EXPORT svtkTransformCollection : public svtkCollection
{
public:
  svtkTypeMacro(svtkTransformCollection, svtkCollection);
  static svtkTransformCollection* New();

  /**
   * Add a Transform to the list.
   */
  void AddItem(svtkTransform*);

  /**
   * Get the next Transform in the list. Return nullptr when the end of the
   * list is reached.
   */
  svtkTransform* GetNextItem();

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkTransform* GetNextTransform(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkTransform*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkTransformCollection() {}
  ~svtkTransformCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkTransformCollection(const svtkTransformCollection&) = delete;
  void operator=(const svtkTransformCollection&) = delete;
};

//----------------------------------------------------------------------------
inline void svtkTransformCollection::AddItem(svtkTransform* t)
{
  this->svtkCollection::AddItem(t);
}

//----------------------------------------------------------------------------
inline svtkTransform* svtkTransformCollection::GetNextItem()
{
  return static_cast<svtkTransform*>(this->GetNextItemAsObject());
}

#endif
// SVTK-HeaderTest-Exclude: svtkTransformCollection.h
