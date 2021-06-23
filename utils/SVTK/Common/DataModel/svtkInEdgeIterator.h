/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInEdgeIterator.h

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
 * @class   svtkInEdgeIterator
 * @brief   Iterates through all incoming edges to a vertex.
 *
 *
 * svtkInEdgeIterator iterates through all edges whose target is a particular
 * vertex. Instantiate this class directly and call Initialize() to traverse
 * the vertex of a graph. Alternately, use GetInEdges() on the graph to
 * initialize the iterator. it->Next() returns a svtkInEdgeType structure,
 * which contains Id, the edge's id, and Source, the edge's source vertex.
 *
 * @sa
 * svtkGraph svtkOutEdgeIterator
 */

#ifndef svtkInEdgeIterator_h
#define svtkInEdgeIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkGraph.h" // For edge type definitions

class svtkGraphEdge;

class SVTKCOMMONDATAMODEL_EXPORT svtkInEdgeIterator : public svtkObject
{
public:
  static svtkInEdgeIterator* New();
  svtkTypeMacro(svtkInEdgeIterator, svtkObject);
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
  inline svtkInEdgeType Next()
  {
    svtkInEdgeType e = *this->Current;
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
  svtkInEdgeIterator();
  ~svtkInEdgeIterator() override;

  /**
   * Protected method for setting the graph used
   * by Initialize().
   */
  virtual void SetGraph(svtkGraph* graph);

  svtkGraph* Graph;
  const svtkInEdgeType* Current;
  const svtkInEdgeType* End;
  svtkIdType Vertex;
  svtkGraphEdge* GraphEdge;

private:
  svtkInEdgeIterator(const svtkInEdgeIterator&) = delete;
  void operator=(const svtkInEdgeIterator&) = delete;
};

#endif
