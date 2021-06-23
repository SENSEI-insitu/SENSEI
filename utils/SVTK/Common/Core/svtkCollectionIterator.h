/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCollectionIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCollectionIterator
 * @brief   iterator through a svtkCollection.
 *
 * svtkCollectionIterator provides an alternative way to traverse
 * through the objects in a svtkCollection.  Unlike the collection's
 * built in interface, this allows multiple iterators to
 * simultaneously traverse the collection.  If items are removed from
 * the collection, only the iterators currently pointing to those
 * items are invalidated.  Other iterators will still continue to
 * function normally.
 */

#ifndef svtkCollectionIterator_h
#define svtkCollectionIterator_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkCollection;
class svtkCollectionElement;

class SVTKCOMMONCORE_EXPORT svtkCollectionIterator : public svtkObject
{
public:
  svtkTypeMacro(svtkCollectionIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkCollectionIterator* New();

  //@{
  /**
   * Set/Get the collection over which to iterate.
   */
  virtual void SetCollection(svtkCollection*);
  svtkGetObjectMacro(Collection, svtkCollection);
  //@}

  /**
   * Position the iterator at the first item in the collection.
   */
  void InitTraversal() { this->GoToFirstItem(); }

  /**
   * Position the iterator at the first item in the collection.
   */
  void GoToFirstItem();

  /**
   * Move the iterator to the next item in the collection.
   */
  void GoToNextItem();

  /**
   * Test whether the iterator is currently positioned at a valid item.
   * Returns 1 for yes, 0 for no.
   */
  int IsDoneWithTraversal();

  /**
   * Get the item at the current iterator position.  Valid only when
   * IsDoneWithTraversal() returns 1.
   */
  svtkObject* GetCurrentObject();

protected:
  svtkCollectionIterator();
  ~svtkCollectionIterator() override;

  // The collection over which we are iterating.
  svtkCollection* Collection;

  // The current iterator position.
  svtkCollectionElement* Element;

  svtkObject* GetObjectInternal();

private:
  svtkCollectionIterator(const svtkCollectionIterator&) = delete;
  void operator=(const svtkCollectionIterator&) = delete;
};

#endif
