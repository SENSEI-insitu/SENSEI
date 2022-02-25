/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataSetCollection
 * @brief   maintain an unordered list of dataset objects
 *
 * svtkDataSetCollection is an object that creates and manipulates ordered
 * lists of datasets. See also svtkCollection and subclasses.
 */

#ifndef svtkDataSetCollection_h
#define svtkDataSetCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkDataSet.h" // Needed for inline methods.

class SVTKCOMMONDATAMODEL_EXPORT svtkDataSetCollection : public svtkCollection
{
public:
  static svtkDataSetCollection* New();
  svtkTypeMacro(svtkDataSetCollection, svtkCollection);

  /**
   * Add a dataset to the bottom of the list.
   */
  void AddItem(svtkDataSet* ds) { this->svtkCollection::AddItem(ds); }

  //@{
  /**
   * Get the next dataset in the list.
   */
  svtkDataSet* GetNextItem() { return static_cast<svtkDataSet*>(this->GetNextItemAsObject()); }
  svtkDataSet* GetNextDataSet() { return static_cast<svtkDataSet*>(this->GetNextItemAsObject()); }
  //@}

  //@{
  /**
   * Get the ith dataset in the list.
   */
  svtkDataSet* GetItem(int i) { return static_cast<svtkDataSet*>(this->GetItemAsObject(i)); }
  svtkDataSet* GetDataSet(int i) { return static_cast<svtkDataSet*>(this->GetItemAsObject(i)); }
  //@}

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkDataSet* GetNextDataSet(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkDataSet*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkDataSetCollection() {}
  ~svtkDataSetCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkDataSetCollection(const svtkDataSetCollection&) = delete;
  void operator=(const svtkDataSetCollection&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkDataSetCollection.h
