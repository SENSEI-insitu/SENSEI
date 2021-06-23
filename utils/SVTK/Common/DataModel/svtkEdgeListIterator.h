/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEdgeListIterator.h

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
 * @class   svtkEdgeListIterator
 * @brief   Iterates through all edges in a graph.
 *
 *
 * svtkEdgeListIterator iterates through all the edges in a graph, by traversing
 * the adjacency list for each vertex. You may instantiate this class directly
 * and call SetGraph() to traverse a certain graph. You may also call the graph's
 * GetEdges() method to set up the iterator for a certain graph.
 *
 * Note that this class does NOT guarantee that the edges will be processed in
 * order of their ids (i.e. it will not necessarily return edge 0, then edge 1,
 * etc.).
 *
 * @sa
 * svtkGraph
 */

#ifndef svtkEdgeListIterator_h
#define svtkEdgeListIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkGraph;
class svtkGraphEdge;

struct svtkEdgeType;
struct svtkOutEdgeType;

class SVTKCOMMONDATAMODEL_EXPORT svtkEdgeListIterator : public svtkObject
{
public:
  static svtkEdgeListIterator* New();
  svtkTypeMacro(svtkEdgeListIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkGetObjectMacro(Graph, svtkGraph);
  virtual void SetGraph(svtkGraph* graph);

  /**
   * Returns the next edge in the graph.
   */
  svtkEdgeType Next();

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
  bool HasNext();

protected:
  svtkEdgeListIterator();
  ~svtkEdgeListIterator() override;

  void Increment();

  svtkGraph* Graph;
  const svtkOutEdgeType* Current;
  const svtkOutEdgeType* End;
  svtkIdType Vertex;
  bool Directed;
  svtkGraphEdge* GraphEdge;

private:
  svtkEdgeListIterator(const svtkEdgeListIterator&) = delete;
  void operator=(const svtkEdgeListIterator&) = delete;
};

#endif
