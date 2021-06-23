/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDirectedGraph.cxx

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

#include "svtkDirectedGraph.h"

#include "svtkInEdgeIterator.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkOutEdgeIterator.h"
#include "svtkSmartPointer.h"

#include <vector>

//----------------------------------------------------------------------------
// class svtkDirectedGraph
//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkDirectedGraph);
//----------------------------------------------------------------------------
svtkDirectedGraph::svtkDirectedGraph() = default;

//----------------------------------------------------------------------------
svtkDirectedGraph::~svtkDirectedGraph() = default;

//----------------------------------------------------------------------------
svtkDirectedGraph* svtkDirectedGraph::GetData(svtkInformation* info)
{
  return info ? svtkDirectedGraph::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkDirectedGraph* svtkDirectedGraph::GetData(svtkInformationVector* v, int i)
{
  return svtkDirectedGraph::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
bool svtkDirectedGraph::IsStructureValid(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }
  if (svtkDirectedGraph::SafeDownCast(g))
  {
    return true;
  }

  // Verify that each edge appears in exactly one in and one out edge list.
  std::vector<bool> in(g->GetNumberOfEdges(), false);
  std::vector<bool> out(g->GetNumberOfEdges(), false);
  svtkSmartPointer<svtkInEdgeIterator> inIter = svtkSmartPointer<svtkInEdgeIterator>::New();
  svtkSmartPointer<svtkOutEdgeIterator> outIter = svtkSmartPointer<svtkOutEdgeIterator>::New();
  for (svtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
  {
    g->GetInEdges(v, inIter);
    while (inIter->HasNext())
    {
      svtkIdType id = inIter->Next().Id;
      if (in[id])
      {
        return false;
      }
      in[id] = true;
    }
    g->GetOutEdges(v, outIter);
    while (outIter->HasNext())
    {
      svtkIdType id = outIter->Next().Id;
      if (out[id])
      {
        return false;
      }
      out[id] = true;
    }
  }
  for (svtkIdType i = 0; i < g->GetNumberOfEdges(); ++i)
  {
    if (in[i] == false || out[i] == false)
    {
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
void svtkDirectedGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
