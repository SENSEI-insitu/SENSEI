/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTree.h

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
 * @class   svtkTree
 * @brief   A rooted tree data structure.
 *
 *
 * svtkTree is a connected directed graph with no cycles. A tree is a type of
 * directed graph, so works with all graph algorithms.
 *
 * svtkTree is a read-only data structure.
 * To construct a tree, create an instance of svtkMutableDirectedGraph.
 * Add vertices and edges with AddVertex() and AddEdge(). You may alternately
 * start by adding a single vertex as the root then call graph->AddChild(parent)
 * which adds a new vertex and connects the parent to the child.
 * The tree MUST have all edges in the proper direction, from parent to child.
 * After building the tree, call tree->CheckedShallowCopy(graph) to copy the
 * structure into a svtkTree. This method will return false if the graph is
 * an invalid tree.
 *
 * svtkTree provides some convenience methods for obtaining the parent and
 * children of a vertex, for finding the root, and determining if a vertex
 * is a leaf (a vertex with no children).
 *
 * @sa
 * svtkDirectedGraph svtkMutableDirectedGraph svtkGraph
 */

#ifndef svtkTree_h
#define svtkTree_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDirectedAcyclicGraph.h"

class svtkIdTypeArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkTree : public svtkDirectedAcyclicGraph
{
public:
  static svtkTree* New();
  svtkTypeMacro(svtkTree, svtkDirectedAcyclicGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_TREE; }

  //@{
  /**
   * Get the root vertex of the tree.
   */
  svtkGetMacro(Root, svtkIdType);
  //@}

  /**
   * Get the number of children of a vertex.
   */
  svtkIdType GetNumberOfChildren(svtkIdType v) { return this->GetOutDegree(v); }

  /**
   * Get the i-th child of a parent vertex.
   */
  svtkIdType GetChild(svtkIdType v, svtkIdType i);

  /**
   * Get the child vertices of a vertex.
   * This is a convenience method that functions exactly like
   * GetAdjacentVertices.
   */
  void GetChildren(svtkIdType v, svtkAdjacentVertexIterator* it) { this->GetAdjacentVertices(v, it); }

  /**
   * Get the parent of a vertex.
   */
  svtkIdType GetParent(svtkIdType v);

  /**
   * Get the edge connecting the vertex to its parent.
   */
  svtkEdgeType GetParentEdge(svtkIdType v);

  /**
   * Get the level of the vertex in the tree.  The root vertex has level 0.
   * Returns -1 if the vertex id is < 0 or greater than the number of vertices
   * in the tree.
   */
  svtkIdType GetLevel(svtkIdType v);

  /**
   * Return whether the vertex is a leaf (i.e. it has no children).
   */
  bool IsLeaf(svtkIdType vertex);

  //@{
  /**
   * Retrieve a graph from an information vector.
   */
  static svtkTree* GetData(svtkInformation* info);
  static svtkTree* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Reorder the children of a parent vertex.
   * The children array must contain all the children of parent,
   * just in a different order.
   * This does not change the topology of the tree.
   */
  virtual void ReorderChildren(svtkIdType parent, svtkIdTypeArray* children);

protected:
  svtkTree();
  ~svtkTree() override;

  /**
   * Check the storage, and accept it if it is a valid
   * tree.
   */
  bool IsStructureValid(svtkGraph* g) override;

  /**
   * The root of the tree.
   */
  svtkIdType Root;

private:
  svtkTree(const svtkTree&) = delete;
  void operator=(const svtkTree&) = delete;
};

#endif
