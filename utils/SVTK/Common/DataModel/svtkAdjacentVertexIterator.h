/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAdjacentVertexIterator.h

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
 * @class   svtkAdjacentVertexIterator
 * @brief   Iterates through adjacent vertices in a graph.
 *
 *
 * svtkAdjacentVertexIterator iterates through all vertices adjacent to a
 * vertex, i.e. the vertices which may be reached by traversing an out edge
 * of the source vertex. Use graph->GetAdjacentVertices(v, it) to initialize
 * the iterator.
 *
 *
 *
 */

#ifndef svtkAdjacentVertexIterator_h
#define svtkAdjacentVertexIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkGraph.h" // For edge type definitions

class svtkGraphEdge;

class SVTKCOMMONDATAMODEL_EXPORT svtkAdjacentVertexIterator : public svtkObject
{
public:
  static svtkAdjacentVertexIterator* New();
  svtkTypeMacro(svtkAdjacentVertexIterator, svtkObject);
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
  svtkIdType Next()
  {
    svtkOutEdgeType e = *this->Current;
    ++this->Current;
    return e.Target;
  }
  //@}

  /**
   * Whether this iterator has more edges.
   */
  bool HasNext() { return this->Current != this->End; }

protected:
  svtkAdjacentVertexIterator();
  ~svtkAdjacentVertexIterator() override;

  /**
   * Protected method for setting the graph used
   * by Initialize().
   */
  virtual void SetGraph(svtkGraph* graph);

  svtkGraph* Graph;
  const svtkOutEdgeType* Current;
  const svtkOutEdgeType* End;
  svtkIdType Vertex;

private:
  svtkAdjacentVertexIterator(const svtkAdjacentVertexIterator&) = delete;
  void operator=(const svtkAdjacentVertexIterator&) = delete;
};

#endif
