/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUndirectedGraph.cxx

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

#include "svtkUndirectedGraph.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkOutEdgeIterator.h"
#include "svtkSmartPointer.h"

#include <vector>

//----------------------------------------------------------------------------
// class svtkUndirectedGraph
//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkUndirectedGraph);
//----------------------------------------------------------------------------
svtkUndirectedGraph::svtkUndirectedGraph() = default;

//----------------------------------------------------------------------------
svtkUndirectedGraph::~svtkUndirectedGraph() = default;

//----------------------------------------------------------------------------
void svtkUndirectedGraph::GetInEdges(svtkIdType v, const svtkInEdgeType*& edges, svtkIdType& nedges)
{
  const svtkOutEdgeType* outEdges;
  this->GetOutEdges(v, outEdges, nedges);
  edges = reinterpret_cast<const svtkInEdgeType*>(outEdges);
}

//----------------------------------------------------------------------------
svtkInEdgeType svtkUndirectedGraph::GetInEdge(svtkIdType v, svtkIdType i)
{
  svtkOutEdgeType oe = this->GetOutEdge(v, i);
  svtkInEdgeType ie(oe.Target, oe.Id);
  return ie;
}

//----------------------------------------------------------------------------
svtkIdType svtkUndirectedGraph::GetInDegree(svtkIdType v)
{
  return this->GetOutDegree(v);
}

//----------------------------------------------------------------------------
svtkUndirectedGraph* svtkUndirectedGraph::GetData(svtkInformation* info)
{
  return info ? svtkUndirectedGraph::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkUndirectedGraph* svtkUndirectedGraph::GetData(svtkInformationVector* v, int i)
{
  return svtkUndirectedGraph::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
bool svtkUndirectedGraph::IsStructureValid(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }

  if (svtkUndirectedGraph::SafeDownCast(g))
  {
    return true;
  }

  // Verify that there are no in edges and that each edge
  // appears in exactly two edge lists.
  // Loop edges should be in exactly one edge list.
  std::vector<svtkIdType> place(g->GetNumberOfEdges(), -1);
  std::vector<svtkIdType> count(g->GetNumberOfEdges(), 0);
  svtkSmartPointer<svtkOutEdgeIterator> outIter = svtkSmartPointer<svtkOutEdgeIterator>::New();
  for (svtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
  {
    if (g->GetInDegree(v) > 0)
    {
      return false;
    }
    g->GetOutEdges(v, outIter);
    while (outIter->HasNext())
    {
      svtkOutEdgeType e = outIter->Next();
      if (place[e.Id] == v)
      {
        return false;
      }
      place[e.Id] = v;
      count[e.Id]++;
      // Count loops twice so they should all have count == 2
      if (v == e.Target)
      {
        count[e.Id]++;
      }
    }
  }
  for (svtkIdType i = 0; i < g->GetNumberOfEdges(); ++i)
  {
    if (count[i] != 2)
    {
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
void svtkUndirectedGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
