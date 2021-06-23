/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyDataCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyDataCollection
 * @brief   maintain a list of polygonal data objects
 *
 * svtkPolyDataCollection is an object that creates and manipulates ordered
 * lists of datasets of type svtkPolyData.
 *
 * @sa
 * svtkDataSetCollection svtkCollection
 */

#ifndef svtkPolyDataCollection_h
#define svtkPolyDataCollection_h

#include "svtkCollection.h"
#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkPolyData.h" // Needed for static cast

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyDataCollection : public svtkCollection
{
public:
  static svtkPolyDataCollection* New();
  svtkTypeMacro(svtkPolyDataCollection, svtkCollection);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Add a poly data to the bottom of the list.
   */
  void AddItem(svtkPolyData* pd) { this->svtkCollection::AddItem(pd); }

  /**
   * Get the next poly data in the list.
   */
  svtkPolyData* GetNextItem() { return static_cast<svtkPolyData*>(this->GetNextItemAsObject()); }

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkPolyData* GetNextPolyData(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkPolyData*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkPolyDataCollection() {}
  ~svtkPolyDataCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkPolyDataCollection(const svtkPolyDataCollection&) = delete;
  void operator=(const svtkPolyDataCollection&) = delete;
};

#endif
