/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDirectedAcyclicGraph.cxx

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

#include "svtkDirectedAcyclicGraph.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkOutEdgeIterator.h"
#include "svtkSmartPointer.h"

#include <vector>

svtkStandardNewMacro(svtkDirectedAcyclicGraph);
//----------------------------------------------------------------------------
svtkDirectedAcyclicGraph::svtkDirectedAcyclicGraph() = default;

//----------------------------------------------------------------------------
svtkDirectedAcyclicGraph::~svtkDirectedAcyclicGraph() = default;

//----------------------------------------------------------------------------
svtkDirectedAcyclicGraph* svtkDirectedAcyclicGraph::GetData(svtkInformation* info)
{
  return info ? svtkDirectedAcyclicGraph::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkDirectedAcyclicGraph* svtkDirectedAcyclicGraph::GetData(svtkInformationVector* v, int i)
{
  return svtkDirectedAcyclicGraph::GetData(v->GetInformationObject(i));
}

enum
{
  DFS_WHITE,
  DFS_GRAY,
  DFS_BLACK
};

//----------------------------------------------------------------------------
static bool svtkDirectedAcyclicGraphDFSVisit(
  svtkGraph* g, svtkIdType u, std::vector<int> color, svtkOutEdgeIterator* adj)
{
  color[u] = DFS_GRAY;
  g->GetOutEdges(u, adj);
  while (adj->HasNext())
  {
    svtkOutEdgeType e = adj->Next();
    svtkIdType v = e.Target;
    if (color[v] == DFS_WHITE)
    {
      if (!svtkDirectedAcyclicGraphDFSVisit(g, v, color, adj))
      {
        return false;
      }
    }
    else if (color[v] == DFS_GRAY)
    {
      return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
bool svtkDirectedAcyclicGraph::IsStructureValid(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }
  if (svtkDirectedAcyclicGraph::SafeDownCast(g))
  {
    return true;
  }

  // Empty graph is a valid DAG.
  if (g->GetNumberOfVertices() == 0)
  {
    return true;
  }

  // A directed graph is acyclic iff a depth-first search of
  // the graph yields no back edges.
  // (from Introduction to Algorithms.
  // Cormen, Leiserson, Rivest, p. 486).
  svtkIdType numVerts = g->GetNumberOfVertices();
  std::vector<int> color(numVerts, DFS_WHITE);
  svtkSmartPointer<svtkOutEdgeIterator> adj = svtkSmartPointer<svtkOutEdgeIterator>::New();
  for (svtkIdType s = 0; s < numVerts; ++s)
  {
    if (color[s] == DFS_WHITE)
    {
      if (!svtkDirectedAcyclicGraphDFSVisit(g, s, color, adj))
      {
        return false;
      }
    }
  }
  return true;
}

//----------------------------------------------------------------------------
void svtkDirectedAcyclicGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
