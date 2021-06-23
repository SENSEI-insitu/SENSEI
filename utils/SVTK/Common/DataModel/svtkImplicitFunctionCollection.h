/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitFunctionCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitFunctionCollection
 * @brief   maintain a list of implicit functions
 *
 * svtkImplicitFunctionCollection is an object that creates and manipulates
 * lists of objects of type svtkImplicitFunction.
 * @sa
 * svtkCollection svtkPlaneCollection
 */

#ifndef svtkImplicitFunctionCollection_h
#define svtkImplicitFunctionCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkImplicitFunction.h" // Needed for inline methods

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitFunctionCollection : public svtkCollection
{
public:
  svtkTypeMacro(svtkImplicitFunctionCollection, svtkCollection);
  static svtkImplicitFunctionCollection* New();

  /**
   * Add an implicit function to the list.
   */
  void AddItem(svtkImplicitFunction*);

  /**
   * Get the next implicit function in the list.
   */
  svtkImplicitFunction* GetNextItem();

  //@{
  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkImplicitFunction* GetNextImplicitFunction(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkImplicitFunction*>(this->GetNextItemAsObject(cookie));
  }
  //@}

protected:
  svtkImplicitFunctionCollection() {}
  ~svtkImplicitFunctionCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkImplicitFunctionCollection(const svtkImplicitFunctionCollection&) = delete;
  void operator=(const svtkImplicitFunctionCollection&) = delete;
};

inline void svtkImplicitFunctionCollection::AddItem(svtkImplicitFunction* f)
{
  this->svtkCollection::AddItem(f);
}

inline svtkImplicitFunction* svtkImplicitFunctionCollection::GetNextItem()
{
  return static_cast<svtkImplicitFunction*>(this->GetNextItemAsObject());
}

#endif
// SVTK-HeaderTest-Exclude: svtkImplicitFunctionCollection.h
