/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMutableUndirectedGraph.h

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
 * @class   svtkMutableUndirectedGraph
 * @brief   An editable undirected graph.
 *
 *
 * svtkMutableUndirectedGraph is an undirected graph with additional functions
 * for adding vertices and edges. ShallowCopy(), DeepCopy(), CheckedShallowCopy(),
 * and CheckedDeepCopy() will succeed when the argument is a svtkUndirectedGraph
 * or svtkMutableUndirectedGraph.
 *
 * @sa
 * svtkUndirectedGraph svtkGraph
 */

#ifndef svtkMutableUndirectedGraph_h
#define svtkMutableUndirectedGraph_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkUndirectedGraph.h"

class svtkEdgeListIterator;
class svtkGraphEdge;

class SVTKCOMMONDATAMODEL_EXPORT svtkMutableUndirectedGraph : public svtkUndirectedGraph
{
public:
  static svtkMutableUndirectedGraph* New();
  svtkTypeMacro(svtkMutableUndirectedGraph, svtkUndirectedGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocates space for the specified number of vertices in the graph's
   * internal data structures.
   * The previous number of vertices is returned on success and -1
   * is returned on failure.

   * This has no effect on the number of vertex coordinate tuples or
   * vertex attribute tuples allocated; you are responsible for
   * guaranteeing these match.
   * Also, this call is not implemented for distributed-memory graphs since
   * the semantics are unclear; calling this function on a graph with a
   * non-nullptr DistributedGraphHelper will generate an error message,
   * no allocation will be performed, and a value of -1 will be returned.
   */
  virtual svtkIdType SetNumberOfVertices(svtkIdType numVerts);

  /**
   * Adds a vertex to the graph and returns the index of the new vertex.

   * \note In a distributed graph (i.e. a graph whose DistributedHelper
   * is non-null), this routine cannot be used to add a vertex
   * if the vertices in the graph have pedigree IDs, because this routine
   * will always add the vertex locally, which may conflict with the
   * proper location of the vertex based on the distribution of the
   * pedigree IDs.
   */
  svtkIdType AddVertex();

  /**
   * Adds a vertex to the graph with associated properties defined in
   * \p propertyArr and returns the index of the new vertex.
   * The number and order of values in \p propertyArr must match up with the
   * arrays in the vertex data retrieved by GetVertexData().

   * If a vertex with the given pedigree ID already exists, its properties will be
   * overwritten with the properties in \p propertyArr and the existing
   * vertex index will be returned.

   * \note In a distributed graph (i.e. a graph whose DistributedHelper
   * is non-null) the vertex added or found might not be local. In this case,
   * AddVertex will wait until the vertex can be added or found
   * remotely, so that the proper vertex index can be returned. If you
   * don't actually need to use the vertex index, consider calling
   * LazyAddVertex, which provides better performance by eliminating
   * the delays associated with returning the vertex index.
   */
  svtkIdType AddVertex(svtkVariantArray* propertyArr);

  /**
   * Adds a vertex with the given \p pedigreeID to the graph and
   * returns the index of the new vertex.

   * If a vertex with the given pedigree ID already exists,
   * the existing vertex index will be returned.

   * \note In a distributed graph (i.e. a graph whose DistributedHelper
   * is non-null) the vertex added or found might not be local. In this case,
   * AddVertex will wait until the vertex can be added or found
   * remotely, so that the proper vertex index can be returned. If you
   * don't actually need to use the vertex index, consider calling
   * LazyAddVertex, which provides better performance by eliminating
   * the delays associated with returning the vertex index.
   */
  svtkIdType AddVertex(const svtkVariant& pedigreeId);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u and \p v are vertex indices,
   * and returns a \p svtkEdgeType structure describing that edge.

   * \p svtkEdgeType contains fields for \p Source vertex index,
   * \p Target vertex index, and edge index \p Id.
   */
  svtkEdgeType AddEdge(svtkIdType u, svtkIdType v);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u and \p v are vertex indices,
   * with associated properties defined in \p propertyArr
   * and returns a \p svtkEdgeType structure describing that edge.

   * The number and order of values in \p propertyArr must match up with the
   * arrays in the edge data retrieved by GetEdgeData().

   * \p svtkEdgeType contains fields for \p Source vertex index,
   * \p Target vertex index, and edge index \p Id.
   */
  svtkEdgeType AddEdge(svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u is a vertex pedigree ID and \p v is a vertex index,
   * and returns a \p svtkEdgeType structure describing that edge.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * \p svtkEdgeType contains fields for \p Source vertex index,
   * \p Target vertex index, and edge index \p Id.
   */
  svtkEdgeType AddEdge(const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Adds a directed edge from \p u to \p v,
   * where \p u is a vertex index and \p v is a vertex pedigree ID,
   * and returns a \p svtkEdgeType structure describing that edge.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * \p svtkEdgeType contains fields for \p Source vertex index,
   * \p Target vertex index, and edge index \p Id.
   */
  svtkEdgeType AddEdge(svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Adds a directed edge from \p u to \p v,
   * where \p u and \p v are vertex pedigree IDs,
   * and returns a \p svtkEdgeType structure describing that edge.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * \p svtkEdgeType contains fields for \p Source vertex index,
   * \p Target vertex index, and edge index \p Id.
   */
  svtkEdgeType AddEdge(
    const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Adds a vertex to the graph.

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddVertex();

  /**
   * Adds a vertex to the graph with associated properties defined in
   * \p propertyArr.
   * The number and order of values in \p propertyArr must match up with the
   * arrays in the vertex data retrieved by GetVertexData().

   * If a vertex with the given pedigree ID already exists, its properties will be
   * overwritten with the properties in \p propertyArr.

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddVertex(svtkVariantArray* propertyArr);

  /**
   * Adds a vertex with the given \p pedigreeID to the graph.

   * If a vertex with the given pedigree ID already exists,
   * no operation is performed.

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddVertex(const svtkVariant& pedigreeId);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u and \p v are vertex indices.

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddEdge(svtkIdType u, svtkIdType v);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u and \p v are vertex indices.

   * The number and order of values in
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddEdge(svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u is a vertex pedigree ID and \p v is a vertex index.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddEdge(const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u is a vertex index and \p v is a vertex pedigree ID.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddEdge(svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Adds an undirected edge from \p u to \p v,
   * where \p u and \p v are vertex pedigree IDs.

   * The number and order of values in the optional parameter
   * \p propertyArr must match up with the arrays in the edge data
   * retrieved by GetEdgeData().

   * This method is lazily evaluated for distributed graphs (i.e. graphs
   * whose DistributedHelper is non-null) the next time Synchronize is
   * called on the helper.
   */
  void LazyAddEdge(
    const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr = nullptr);

  /**
   * Variant of AddEdge() that returns a heavyweight \p svtkGraphEdge object.
   * The graph owns the reference of the edge and will replace
   * its contents on the next call to AddGraphEdge().

   * \note This is a less efficient method for use with wrappers.
   * In C++ you should use the faster AddEdge().
   */
  svtkGraphEdge* AddGraphEdge(svtkIdType u, svtkIdType v);

  /**
   * Removes the vertex from the graph along with any connected edges.
   * Note: This invalidates the last vertex index, which is reassigned to v.
   */
  void RemoveVertex(svtkIdType v);

  /**
   * Removes the edge from the graph.
   * Note: This invalidates the last edge index, which is reassigned to e.
   */
  void RemoveEdge(svtkIdType e);

  /**
   * Removes a collection of vertices from the graph along with any connected edges.
   */
  void RemoveVertices(svtkIdTypeArray* arr);

  /**
   * Removes a collection of edges from the graph.
   */
  void RemoveEdges(svtkIdTypeArray* arr);

protected:
  svtkMutableUndirectedGraph();
  ~svtkMutableUndirectedGraph() override;

  /**
   * Graph edge that is reused of AddGraphEdge calls.
   */
  svtkGraphEdge* GraphEdge;

private:
  svtkMutableUndirectedGraph(const svtkMutableUndirectedGraph&) = delete;
  void operator=(const svtkMutableUndirectedGraph&) = delete;
};

#endif
