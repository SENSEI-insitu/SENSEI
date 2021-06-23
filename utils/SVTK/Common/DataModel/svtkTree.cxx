/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTree.cxx

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

#include "svtkTree.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkOutEdgeIterator.h"
#include "svtkSmartPointer.h"

#include <vector>

svtkStandardNewMacro(svtkTree);
//----------------------------------------------------------------------------
svtkTree::svtkTree()
{
  this->Root = -1;
}

//----------------------------------------------------------------------------
svtkTree::~svtkTree() = default;

//----------------------------------------------------------------------------
svtkIdType svtkTree::GetChild(svtkIdType v, svtkIdType i)
{
  const svtkOutEdgeType* edges;
  svtkIdType nedges;
  this->GetOutEdges(v, edges, nedges);
  if (i < nedges)
  {
    return edges[i].Target;
  }
  return -1;
}

//----------------------------------------------------------------------------
svtkIdType svtkTree::GetParent(svtkIdType v)
{
  const svtkInEdgeType* edges;
  svtkIdType nedges;
  this->GetInEdges(v, edges, nedges);
  if (nedges > 0)
  {
    return edges[0].Source;
  }
  return -1;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkTree::GetParentEdge(svtkIdType v)
{
  const svtkInEdgeType* edges;
  svtkIdType nedges;
  this->GetInEdges(v, edges, nedges);
  if (nedges > 0)
  {
    return svtkEdgeType(edges[0].Source, v, edges[0].Id);
  }
  return svtkEdgeType();
}

//----------------------------------------------------------------------------
svtkIdType svtkTree::GetLevel(svtkIdType vertex)
{
  if (vertex < 0 || vertex >= this->GetNumberOfVertices())
  {
    return -1;
  }
  svtkIdType level = 0;
  while (vertex != this->Root)
  {
    vertex = this->GetParent(vertex);
    level++;
  }
  return level;
}

//----------------------------------------------------------------------------
bool svtkTree::IsLeaf(svtkIdType vertex)
{
  return (this->GetNumberOfChildren(vertex) == 0);
}

//----------------------------------------------------------------------------
svtkTree* svtkTree::GetData(svtkInformation* info)
{
  return info ? svtkTree::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkTree* svtkTree::GetData(svtkInformationVector* v, int i)
{
  return svtkTree::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
bool svtkTree::IsStructureValid(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }

  svtkTree* tree = svtkTree::SafeDownCast(g);
  if (tree)
  {
    // Since a tree has the additional root property, we need
    // to set that here.
    this->Root = tree->Root;
    return true;
  }

  // Empty graph is a valid tree.
  if (g->GetNumberOfVertices() == 0)
  {
    this->Root = -1;
    return true;
  }

  // A tree must have one more vertex than its number of edges.
  if (g->GetNumberOfEdges() != g->GetNumberOfVertices() - 1)
  {
    return false;
  }

  // Find the root and fail if there is more than one.
  svtkIdType root = -1;
  for (svtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
  {
    svtkIdType indeg = g->GetInDegree(v);
    if (indeg > 1)
    {
      // No tree vertex should have in degree > 1, so fail.
      return false;
    }
    else if (indeg == 0 && root == -1)
    {
      // We found our first root.
      root = v;
    }
    else if (indeg == 0)
    {
      // We already found a root, so fail.
      return false;
    }
  }
  if (root < 0)
  {
    return false;
  }

  // Make sure the tree is connected with no cycles.
  std::vector<bool> visited(g->GetNumberOfVertices(), false);
  std::vector<svtkIdType> stack;
  stack.push_back(root);
  svtkSmartPointer<svtkOutEdgeIterator> outIter = svtkSmartPointer<svtkOutEdgeIterator>::New();
  while (!stack.empty())
  {
    svtkIdType v = stack.back();
    stack.pop_back();
    visited[v] = true;
    g->GetOutEdges(v, outIter);
    while (outIter->HasNext())
    {
      svtkIdType id = outIter->Next().Target;
      if (!visited[id])
      {
        stack.push_back(id);
      }
      else
      {
        return false;
      }
    }
  }
  for (svtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
  {
    if (!visited[v])
    {
      return false;
    }
  }

  // Since a tree has the additional root property, we need
  // to set that here.
  this->Root = root;

  return true;
}

//----------------------------------------------------------------------------
void svtkTree::ReorderChildren(svtkIdType parent, svtkIdTypeArray* children)
{
  this->ReorderOutVertices(parent, children);
}

//----------------------------------------------------------------------------
void svtkTree::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Root: " << this->Root << endl;
}
