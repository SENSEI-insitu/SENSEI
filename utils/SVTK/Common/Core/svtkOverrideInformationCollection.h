/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOverrideInformationCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOverrideInformationCollection
 * @brief   maintain a list of override information objects
 *
 * svtkOverrideInformationCollection is an object that creates and manipulates
 * lists of objects of type svtkOverrideInformation.
 * @sa
 * svtkCollection
 */

#ifndef svtkOverrideInformationCollection_h
#define svtkOverrideInformationCollection_h

#include "svtkCollection.h"
#include "svtkCommonCoreModule.h" // For export macro

#include "svtkOverrideInformation.h" // Needed for inline methods

class SVTKCOMMONCORE_EXPORT svtkOverrideInformationCollection : public svtkCollection
{
public:
  svtkTypeMacro(svtkOverrideInformationCollection, svtkCollection);
  static svtkOverrideInformationCollection* New();

  /**
   * Add a OverrideInformation to the list.
   */
  void AddItem(svtkOverrideInformation*);

  /**
   * Get the next OverrideInformation in the list.
   */
  svtkOverrideInformation* GetNextItem();

  /**
   * Reentrant safe way to get an object in a collection. Just pass the
   * same cookie back and forth.
   */
  svtkOverrideInformation* GetNextOverrideInformation(svtkCollectionSimpleIterator& cookie)
  {
    return static_cast<svtkOverrideInformation*>(this->GetNextItemAsObject(cookie));
  }

protected:
  svtkOverrideInformationCollection() {}
  ~svtkOverrideInformationCollection() override {}

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(svtkObject* o) { this->svtkCollection::AddItem(o); }

private:
  svtkOverrideInformationCollection(const svtkOverrideInformationCollection&) = delete;
  void operator=(const svtkOverrideInformationCollection&) = delete;
};

inline void svtkOverrideInformationCollection::AddItem(svtkOverrideInformation* f)
{
  this->svtkCollection::AddItem(f);
}

inline svtkOverrideInformation* svtkOverrideInformationCollection::GetNextItem()
{
  return static_cast<svtkOverrideInformation*>(this->GetNextItemAsObject());
}

#endif
// SVTK-HeaderTest-Exclude: svtkOverrideInformationCollection.h
