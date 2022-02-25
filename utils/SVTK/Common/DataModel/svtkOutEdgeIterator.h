/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOutEdgeIterator.h

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
 * @class   svtkOutEdgeIterator
 * @brief   Iterates through all outgoing edges from a vertex.
 *
 *
 * svtkOutEdgeIterator iterates through all edges whose source is a particular
 * vertex. Instantiate this class directly and call Initialize() to traverse
 * the vertex of a graph. Alternately, use GetInEdges() on the graph to
 * initialize the iterator. it->Next() returns a svtkOutEdgeType structure,
 * which contains Id, the edge's id, and Target, the edge's target vertex.
 *
 * @sa
 * svtkGraph svtkInEdgeIterator
 */

#ifndef svtkOutEdgeIterator_h
#define svtkOutEdgeIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkGraph.h" // For edge type definitions

class svtkGraphEdge;

class SVTKCOMMONDATAMODEL_EXPORT svtkOutEdgeIterator : public svtkObject
{
public:
  static svtkOutEdgeIterator* New();
  svtkTypeMacro(svtkOutEdgeIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Initialize the iterator with a graph and vertex.
   */
  void Initialize(svtkGraph* g, svtkIdType v);

  //@{
  /**
   * Get the graph and vertex associated with this iterator.
   */
  svtkGetObjectMacro(Graph, svtkGraph);
  svtkGetMacro(Vertex, svtkIdType);
  //@}

  //@{
  /**
   * Returns the next edge in the graph.
   */
  inline svtkOutEdgeType Next()
  {
    svtkOutEdgeType e = *this->Current;
    ++this->Current;
    return e;
  }
  //@}

  /**
   * Just like Next(), but
   * returns heavy-weight svtkGraphEdge object instead of
   * the svtkEdgeType struct, for use with wrappers.
   * The graph edge is owned by this iterator, and changes
   * after each call to NextGraphEdge().
   */
  svtkGraphEdge* NextGraphEdge();

  /**
   * Whether this iterator has more edges.
   */
  bool HasNext() { return this->Current != this->End; }

protected:
  svtkOutEdgeIterator();
  ~svtkOutEdgeIterator() override;

  /**
   * Protected method for setting the graph used
   * by Initialize().
   */
  virtual void SetGraph(svtkGraph* graph);

  svtkGraph* Graph;
  const svtkOutEdgeType* Current;
  const svtkOutEdgeType* End;
  svtkIdType Vertex;
  svtkGraphEdge* GraphEdge;

private:
  svtkOutEdgeIterator(const svtkOutEdgeIterator&) = delete;
  void operator=(const svtkOutEdgeIterator&) = delete;
};

#endif
