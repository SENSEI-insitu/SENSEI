/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkTreeIterator
 * @brief   Abstract class for iterator over a svtkTree.
 *
 *
 * The base class for tree iterators svtkTreeBFSIterator and svtkTreeDFSIterator.
 *
 * After setting up the iterator, the normal mode of operation is to
 * set up a <code>while(iter->HasNext())</code> loop, with the statement
 * <code>svtkIdType vertex = iter->Next()</code> inside the loop.
 *
 * @sa
 * svtkTreeBFSIterator svtkTreeDFSIterator
 */

#ifndef svtkTreeIterator_h
#define svtkTreeIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkTree;

class SVTKCOMMONDATAMODEL_EXPORT svtkTreeIterator : public svtkObject
{
public:
  svtkTypeMacro(svtkTreeIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set/get the graph to iterate over.
   */
  void SetTree(svtkTree* graph);
  svtkGetMacro(Tree, svtkTree*);
  //@}

  //@{
  /**
   * The start vertex of the traversal.
   * The tree iterator will only iterate over the subtree rooted at vertex.
   * If not set (or set to a negative value), starts at the root of the tree.
   */
  void SetStartVertex(svtkIdType vertex);
  svtkGetMacro(StartVertex, svtkIdType);
  //@}

  /**
   * The next vertex visited in the graph.
   */
  svtkIdType Next();

  /**
   * Return true when all vertices have been visited.
   */
  bool HasNext();

  /**
   * Reset the iterator to its start vertex.
   */
  void Restart();

protected:
  svtkTreeIterator();
  ~svtkTreeIterator() override;

  virtual void Initialize() = 0;
  virtual svtkIdType NextInternal() = 0;

  svtkTree* Tree;
  svtkIdType StartVertex;
  svtkIdType NextId;

private:
  svtkTreeIterator(const svtkTreeIterator&) = delete;
  void operator=(const svtkTreeIterator&) = delete;
};

#endif
