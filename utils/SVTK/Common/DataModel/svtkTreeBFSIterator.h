/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeBFSIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkTreeBFSIterator
 * @brief   breadth first search iterator through a svtkTree
 *
 *
 * svtkTreeBFSIterator performs a breadth first search traversal of a tree.
 *
 * After setting up the iterator, the normal mode of operation is to
 * set up a <code>while(iter->HasNext())</code> loop, with the statement
 * <code>svtkIdType vertex = iter->Next()</code> inside the loop.
 *
 * @par Thanks:
 * Thanks to David Doria for submitting this class.
 */

#ifndef svtkTreeBFSIterator_h
#define svtkTreeBFSIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkTreeIterator.h"

class svtkTreeBFSIteratorInternals;
class svtkIntArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkTreeBFSIterator : public svtkTreeIterator
{
public:
  static svtkTreeBFSIterator* New();
  svtkTypeMacro(svtkTreeBFSIterator, svtkTreeIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

protected:
  svtkTreeBFSIterator();
  ~svtkTreeBFSIterator() override;

  void Initialize() override;
  svtkIdType NextInternal() override;

  svtkTreeBFSIteratorInternals* Internals;
  svtkIntArray* Color;

  enum ColorType
  {
    WHITE,
    GRAY,
    BLACK
  };

private:
  svtkTreeBFSIterator(const svtkTreeBFSIterator&) = delete;
  void operator=(const svtkTreeBFSIterator&) = delete;
};

#endif
