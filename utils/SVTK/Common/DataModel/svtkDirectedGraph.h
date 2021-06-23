/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDirectedGraph.h

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
 * @class   svtkDirectedGraph
 * @brief   A directed graph.
 *
 *
 * svtkDirectedGraph is a collection of vertices along with a collection of
 * directed edges (edges that have a source and target). ShallowCopy()
 * and DeepCopy() (and CheckedShallowCopy(), CheckedDeepCopy())
 * accept instances of svtkTree and svtkMutableDirectedGraph.
 *
 * svtkDirectedGraph is read-only. To create an undirected graph,
 * use an instance of svtkMutableDirectedGraph, then you may set the
 * structure to a svtkDirectedGraph using ShallowCopy().
 *
 * @sa
 * svtkGraph svtkMutableDirectedGraph
 */

#ifndef svtkDirectedGraph_h
#define svtkDirectedGraph_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkGraph.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkDirectedGraph : public svtkGraph
{
public:
  static svtkDirectedGraph* New();
  svtkTypeMacro(svtkDirectedGraph, svtkGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_DIRECTED_GRAPH; }

  //@{
  /**
   * Retrieve a graph from an information vector.
   */
  static svtkDirectedGraph* GetData(svtkInformation* info);
  static svtkDirectedGraph* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Check the storage, and accept it if it is a valid
   * undirected graph. This is public to allow
   * the ToDirected/UndirectedGraph to work.
   */
  bool IsStructureValid(svtkGraph* g) override;

protected:
  svtkDirectedGraph();
  ~svtkDirectedGraph() override;

private:
  svtkDirectedGraph(const svtkDirectedGraph&) = delete;
  void operator=(const svtkDirectedGraph&) = delete;
};

#endif
