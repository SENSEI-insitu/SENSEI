/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAdjacentVertexIterator.cxx

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

#include "svtkAdjacentVertexIterator.h"

#include "svtkGraph.h"
#include "svtkObjectFactory.h"

svtkCxxSetObjectMacro(svtkAdjacentVertexIterator, Graph, svtkGraph);
svtkStandardNewMacro(svtkAdjacentVertexIterator);
//----------------------------------------------------------------------------
svtkAdjacentVertexIterator::svtkAdjacentVertexIterator()
{
  this->Vertex = 0;
  this->Current = nullptr;
  this->End = nullptr;
  this->Graph = nullptr;
}

//----------------------------------------------------------------------------
svtkAdjacentVertexIterator::~svtkAdjacentVertexIterator()
{
  if (this->Graph)
  {
    this->Graph->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkAdjacentVertexIterator::Initialize(svtkGraph* graph, svtkIdType v)
{
  this->SetGraph(graph);
  this->Vertex = v;
  svtkIdType nedges;
  this->Graph->GetOutEdges(this->Vertex, this->Current, nedges);
  this->End = this->Current + nedges;
}

//----------------------------------------------------------------------------
void svtkAdjacentVertexIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Graph: " << (this->Graph ? "" : "(null)") << endl;
  if (this->Graph)
  {
    this->Graph->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "Vertex: " << this->Vertex << endl;
}
