/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOutEdgeIterator.cxx

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

#include "svtkOutEdgeIterator.h"

#include "svtkGraph.h"
#include "svtkGraphEdge.h"
#include "svtkObjectFactory.h"

svtkCxxSetObjectMacro(svtkOutEdgeIterator, Graph, svtkGraph);
svtkStandardNewMacro(svtkOutEdgeIterator);
//----------------------------------------------------------------------------
svtkOutEdgeIterator::svtkOutEdgeIterator()
{
  this->Vertex = 0;
  this->Current = nullptr;
  this->End = nullptr;
  this->Graph = nullptr;
  this->GraphEdge = nullptr;
}

//----------------------------------------------------------------------------
svtkOutEdgeIterator::~svtkOutEdgeIterator()
{
  if (this->Graph)
  {
    this->Graph->Delete();
  }
  if (this->GraphEdge)
  {
    this->GraphEdge->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkOutEdgeIterator::Initialize(svtkGraph* graph, svtkIdType v)
{
  this->SetGraph(graph);
  this->Vertex = v;
  svtkIdType nedges;
  this->Graph->GetOutEdges(this->Vertex, this->Current, nedges);
  this->End = this->Current + nedges;
}

//----------------------------------------------------------------------------
svtkGraphEdge* svtkOutEdgeIterator::NextGraphEdge()
{
  svtkOutEdgeType e = this->Next();
  if (!this->GraphEdge)
  {
    this->GraphEdge = svtkGraphEdge::New();
  }
  this->GraphEdge->SetSource(this->Vertex);
  this->GraphEdge->SetTarget(e.Target);
  this->GraphEdge->SetId(e.Id);
  return this->GraphEdge;
}

//----------------------------------------------------------------------------
void svtkOutEdgeIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Graph: " << (this->Graph ? "" : "(null)") << endl;
  if (this->Graph)
  {
    this->Graph->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "Vertex: " << this->Vertex << endl;
}
