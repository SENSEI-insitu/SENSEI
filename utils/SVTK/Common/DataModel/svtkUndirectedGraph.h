/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUndirectedGraph.h

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
 * @class   svtkUndirectedGraph
 * @brief   An undirected graph.
 *
 *
 * svtkUndirectedGraph is a collection of vertices along with a collection of
 * undirected edges (they connect two vertices in no particular order).
 * ShallowCopy(), DeepCopy(), CheckedShallowCopy(), CheckedDeepCopy()
 * accept instances of svtkUndirectedGraph and svtkMutableUndirectedGraph.
 * GetOutEdges(v, it) and GetInEdges(v, it) return the same list of edges,
 * which is the list of all edges which have a v as an endpoint.
 * GetInDegree(v), GetOutDegree(v) and GetDegree(v) all return the full
 * degree of vertex v.
 *
 * svtkUndirectedGraph is read-only. To create an undirected graph,
 * use an instance of svtkMutableUndirectedGraph, then you may set the
 * structure to a svtkUndirectedGraph using ShallowCopy().
 *
 * @sa
 * svtkGraph svtkMutableUndirectedGraph
 */

#ifndef svtkUndirectedGraph_h
#define svtkUndirectedGraph_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkGraph.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkUndirectedGraph : public svtkGraph
{
public:
  static svtkUndirectedGraph* New();
  svtkTypeMacro(svtkUndirectedGraph, svtkGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_UNDIRECTED_GRAPH; }

  /**
   * Returns the full degree of the vertex.
   */
  svtkIdType GetInDegree(svtkIdType v) override;

  /**
   * Random-access method for retrieving the in edges of a vertex.
   * For an undirected graph, this is the same as the out edges.
   */
  svtkInEdgeType GetInEdge(svtkIdType v, svtkIdType i) override;

  /**
   * Random-access method for retrieving incoming edges to vertex v.
   * The method fills the svtkGraphEdge instance with the id, source, and
   * target of the edge. This method is provided for wrappers,
   * GetInEdge(svtkIdType, svtkIdType) is preferred.
   */
  void GetInEdge(svtkIdType v, svtkIdType i, svtkGraphEdge* e) override
  {
    this->Superclass::GetInEdge(v, i, e);
  }

  //@{
  /**
   * Retrieve a graph from an information vector.
   */
  static svtkUndirectedGraph* GetData(svtkInformation* info);
  static svtkUndirectedGraph* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Initialize the iterator to get the incoming edges to a vertex.
   * For an undirected graph, this is all incident edges.
   */
  void GetInEdges(svtkIdType v, svtkInEdgeIterator* it) override { Superclass::GetInEdges(v, it); }

  /**
   * Check the structure, and accept it if it is a valid
   * undirected graph. This is public to allow
   * the ToDirected/UndirectedGraph to work.
   */
  bool IsStructureValid(svtkGraph* g) override;

protected:
  svtkUndirectedGraph();
  ~svtkUndirectedGraph() override;

  /**
   * For iterators, returns the same edge list as GetOutEdges().
   */
  void GetInEdges(svtkIdType v, const svtkInEdgeType*& edges, svtkIdType& nedges) override;

private:
  svtkUndirectedGraph(const svtkUndirectedGraph&) = delete;
  void operator=(const svtkUndirectedGraph&) = delete;
};

#endif
