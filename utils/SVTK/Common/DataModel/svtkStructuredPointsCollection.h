/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStructuredPointsCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStructuredPointsCollection
 * @brief   maintain a list of structured points data objects
 *
 * svtkStructuredPointsCollection is an object that creates and manipulates
 * ordered lists of structured points datasets. See also svtkCollection and
 * subclasses.
 */

#ifndef svtkStructuredPointsCollection_h
#define svtkStructuredPointsCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkStructuredPoints.h"      // Needed for static cast

class SVTKCOMMONDATAMODEL_EXPORT svtkStructuredPointsCollection : public svtkCollection
{
public:
  static svtkStructuredPointsCollection* New();
  svtkTypeMacro(svtkStructuredPointsCollection, svtkCollection);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Add a pointer to a svtkStructuredPoints to the bottom of the list.
   */
  void AddItem(svtkStructuredPoints* ds) { this->svtkCollection::AddItem(ds); }

  /**
   * Get the next item in the collection. nullptr is returned if the collection
   * is exhausted.
   */
  svtkStructuredPoints* GetNextItem()
  {
    return static_cast<svtkStructuredPoints*>(this->GetNextItemAsObject());
  }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkStructuredPoints* GetNextStructuredPoints(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkStructuredPoints*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkStructuredPointsCollection() {}
  ~svtkStructuredPointsCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkStructuredPointsCollection(const svtkStructuredPointsCollection&) = delete;
  void operator=(const svtkStructuredPointsCollection&) = delete;
};

#endif
