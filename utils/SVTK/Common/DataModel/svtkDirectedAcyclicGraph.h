/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDirectedAcyclicGraph.h

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
 * @class   svtkDirectedAcyclicGraph
 * @brief   A rooted tree data structure.
 *
 *
 * svtkDirectedAcyclicGraph is a connected directed graph with no cycles. A tree is a type of
 * directed graph, so works with all graph algorithms.
 *
 * svtkDirectedAcyclicGraph is a read-only data structure.
 * To construct a tree, create an instance of svtkMutableDirectedGraph.
 * Add vertices and edges with AddVertex() and AddEdge(). You may alternately
 * start by adding a single vertex as the root then call graph->AddChild(parent)
 * which adds a new vertex and connects the parent to the child.
 * The tree MUST have all edges in the proper direction, from parent to child.
 * After building the tree, call tree->CheckedShallowCopy(graph) to copy the
 * structure into a svtkDirectedAcyclicGraph. This method will return false if the graph is
 * an invalid tree.
 *
 * svtkDirectedAcyclicGraph provides some convenience methods for obtaining the parent and
 * children of a vertex, for finding the root, and determining if a vertex
 * is a leaf (a vertex with no children).
 *
 * @sa
 * svtkDirectedGraph svtkMutableDirectedGraph svtkGraph
 */

#ifndef svtkDirectedAcyclicGraph_h
#define svtkDirectedAcyclicGraph_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDirectedGraph.h"

class svtkIdTypeArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkDirectedAcyclicGraph : public svtkDirectedGraph
{
public:
  static svtkDirectedAcyclicGraph* New();
  svtkTypeMacro(svtkDirectedAcyclicGraph, svtkDirectedGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_DIRECTED_ACYCLIC_GRAPH; }

  //@{
  /**
   * Retrieve a graph from an information vector.
   */
  static svtkDirectedAcyclicGraph* GetData(svtkInformation* info);
  static svtkDirectedAcyclicGraph* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkDirectedAcyclicGraph();
  ~svtkDirectedAcyclicGraph() override;

  /**
   * Check the storage, and accept it if it is a valid
   * tree.
   */
  bool IsStructureValid(svtkGraph* g) override;

private:
  svtkDirectedAcyclicGraph(const svtkDirectedAcyclicGraph&) = delete;
  void operator=(const svtkDirectedAcyclicGraph&) = delete;
};

#endif
