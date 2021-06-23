/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeDFSIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/
/**
 * @class   svtkTreeDFSIterator
 * @brief   depth first iterator through a svtkGraph
 *
 *
 * svtkTreeDFSIterator performs a depth first search traversal of a tree.
 *
 * First, you must set the tree on which you are going to iterate, and then
 * optionally set the starting vertex and mode. The mode is either
 * DISCOVER (default), in which case vertices are visited as they are first
 * reached, or FINISH, in which case vertices are visited when they are
 * done, i.e. all adjacent vertices have been discovered already.
 *
 * After setting up the iterator, the normal mode of operation is to
 * set up a <code>while(iter->HasNext())</code> loop, with the statement
 * <code>svtkIdType vertex = iter->Next()</code> inside the loop.
 */

#ifndef svtkTreeDFSIterator_h
#define svtkTreeDFSIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkTreeIterator.h"

class svtkTreeDFSIteratorInternals;
class svtkIntArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkTreeDFSIterator : public svtkTreeIterator
{
public:
  static svtkTreeDFSIterator* New();
  svtkTypeMacro(svtkTreeDFSIterator, svtkTreeIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  enum ModeType
  {
    DISCOVER,
    FINISH
  };

  //@{
  /**
   * Set the visit mode of the iterator.  Mode can be
   * DISCOVER (0): Order by discovery time
   * FINISH   (1): Order by finish time
   * Default is DISCOVER.
   * Use DISCOVER for top-down algorithms where parents need to be processed before children.
   * Use FINISH for bottom-up algorithms where children need to be processed before parents.
   */
  void SetMode(int mode);
  svtkGetMacro(Mode, int);
  //@}

protected:
  svtkTreeDFSIterator();
  ~svtkTreeDFSIterator() override;

  void Initialize() override;
  svtkIdType NextInternal() override;

  int Mode;
  svtkIdType CurRoot;
  svtkTreeDFSIteratorInternals* Internals;
  svtkIntArray* Color;

  enum ColorType
  {
    WHITE,
    GRAY,
    BLACK
  };

private:
  svtkTreeDFSIterator(const svtkTreeDFSIterator&) = delete;
  void operator=(const svtkTreeDFSIterator&) = delete;
};

#endif
