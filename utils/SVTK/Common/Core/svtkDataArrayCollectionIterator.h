/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayCollectionIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataArrayCollectionIterator
 * @brief   iterator through a svtkDataArrayCollection.
 *
 * svtkDataArrayCollectionIterator provides an implementation of
 * svtkCollectionIterator which allows the items to be retrieved with
 * the proper subclass pointer type for svtkDataArrayCollection.
 */

#ifndef svtkDataArrayCollectionIterator_h
#define svtkDataArrayCollectionIterator_h

#include "svtkCollectionIterator.h"
#include "svtkCommonCoreModule.h" // For export macro

class svtkDataArray;
class svtkDataArrayCollection;

class SVTKCOMMONCORE_EXPORT svtkDataArrayCollectionIterator : public svtkCollectionIterator
{
public:
  svtkTypeMacro(svtkDataArrayCollectionIterator, svtkCollectionIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkDataArrayCollectionIterator* New();

  //@{
  /**
   * Set the collection over which to iterate.
   */
  void SetCollection(svtkCollection*) override;
  void SetCollection(svtkDataArrayCollection*);
  //@}

  /**
   * Get the item at the current iterator position.  Valid only when
   * IsDoneWithTraversal() returns 1.
   */
  svtkDataArray* GetDataArray();

protected:
  svtkDataArrayCollectionIterator();
  ~svtkDataArrayCollectionIterator() override;

private:
  svtkDataArrayCollectionIterator(const svtkDataArrayCollectionIterator&) = delete;
  void operator=(const svtkDataArrayCollectionIterator&) = delete;
};

#endif
