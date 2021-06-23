/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEdgeListIterator.cxx

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

#include "svtkEdgeListIterator.h"

#include "svtkDirectedGraph.h"
#include "svtkDistributedGraphHelper.h"
#include "svtkGraph.h"
#include "svtkGraphEdge.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkEdgeListIterator);
//----------------------------------------------------------------------------
svtkEdgeListIterator::svtkEdgeListIterator()
{
  this->Vertex = 0;
  this->Current = nullptr;
  this->End = nullptr;
  this->Graph = nullptr;
  this->Directed = false;
  this->GraphEdge = nullptr;
}

//----------------------------------------------------------------------------
svtkEdgeListIterator::~svtkEdgeListIterator()
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
void svtkEdgeListIterator::SetGraph(svtkGraph* graph)
{
  svtkSetObjectBodyMacro(Graph, svtkGraph, graph);
  this->Current = nullptr;
  this->End = nullptr;
  if (this->Graph && this->Graph->GetNumberOfEdges() > 0)
  {
    this->Directed = (svtkDirectedGraph::SafeDownCast(this->Graph) != nullptr);
    this->Vertex = 0;
    svtkIdType lastVertex = this->Graph->GetNumberOfVertices();

    int myRank = -1;
    svtkDistributedGraphHelper* helper = this->Graph->GetDistributedGraphHelper();
    if (helper)
    {
      myRank = this->Graph->GetInformation()->Get(svtkDataObject::DATA_PIECE_NUMBER());
      this->Vertex = helper->MakeDistributedId(myRank, this->Vertex);
      lastVertex = helper->MakeDistributedId(myRank, lastVertex);
    }

    // Find a vertex with nonzero out degree.
    while (this->Vertex < lastVertex && this->Graph->GetOutDegree(this->Vertex) == 0)
    {
      ++this->Vertex;
    }
    if (this->Vertex < lastVertex)
    {
      svtkIdType nedges;
      this->Graph->GetOutEdges(this->Vertex, this->Current, nedges);
      this->End = this->Current + nedges;
      // If it is undirected, skip edges that are non-local or
      // entirely-local edges whose source is greater than the target.
      if (!this->Directed)
      {
        while (this->Current != nullptr &&
          ( // Skip non-local edges.
            (helper && helper->GetEdgeOwner(this->Current->Id) != myRank)
            // Skip entirely-local edges where Source > Target
            || (((helper && myRank == helper->GetVertexOwner(this->Current->Target)) || !helper) &&
                 this->Vertex > this->Current->Target)))
        {
          this->Increment();
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
svtkEdgeType svtkEdgeListIterator::Next()
{
  // First, determine the current item.
  svtkEdgeType e(this->Vertex, this->Current->Target, this->Current->Id);

  // Next, increment the iterator.
  this->Increment();
  // If it is undirected, skip edges that are non-local or
  // entirely-local edges whose source is greater than the target.
  if (!this->Directed)
  {
    int myRank = -1;
    svtkDistributedGraphHelper* helper = this->Graph->GetDistributedGraphHelper();

    if (helper)
    {
      myRank = this->Graph->GetInformation()->Get(svtkDataObject::DATA_PIECE_NUMBER());
    }

    while (this->Current != nullptr &&
      ( // Skip non-local edges.
        (helper && helper->GetEdgeOwner(this->Current->Id) != myRank)
        // Skip entirely-local edges where Source > Target
        || (((helper && myRank == helper->GetVertexOwner(this->Current->Target)) || !helper) &&
             this->Vertex > this->Current->Target)))
    {
      this->Increment();
    }
  }

  // Return the current item.
  return e;
}

//----------------------------------------------------------------------------
svtkGraphEdge* svtkEdgeListIterator::NextGraphEdge()
{
  svtkEdgeType e = this->Next();
  if (!this->GraphEdge)
  {
    this->GraphEdge = svtkGraphEdge::New();
  }
  this->GraphEdge->SetSource(e.Source);
  this->GraphEdge->SetTarget(e.Target);
  this->GraphEdge->SetId(e.Id);
  return this->GraphEdge;
}

//----------------------------------------------------------------------------
void svtkEdgeListIterator::Increment()
{
  if (!this->Graph)
  {
    return;
  }

  svtkIdType lastVertex = this->Graph->GetNumberOfVertices();

  svtkDistributedGraphHelper* helper = this->Graph->GetDistributedGraphHelper();
  if (helper)
  {
    int myRank = this->Graph->GetInformation()->Get(svtkDataObject::DATA_PIECE_NUMBER());
    this->Vertex = helper->MakeDistributedId(myRank, this->Vertex);
    lastVertex = helper->MakeDistributedId(myRank, lastVertex);
  }

  ++this->Current;
  if (this->Current == this->End)
  {
    // Find a vertex with nonzero out degree.
    ++this->Vertex;
    while (this->Vertex < lastVertex && this->Graph->GetOutDegree(this->Vertex) == 0)
    {
      ++this->Vertex;
    }

    // If there is another vertex with out edges, get its edges.
    // Otherwise, signal that we have reached the end of the iterator.
    if (this->Vertex < lastVertex)
    {
      svtkIdType nedges;
      this->Graph->GetOutEdges(this->Vertex, this->Current, nedges);
      this->End = this->Current + nedges;
    }
    else
    {
      this->Current = nullptr;
    }
  }
}

//----------------------------------------------------------------------------
bool svtkEdgeListIterator::HasNext()
{
  return (this->Current != nullptr);
}

//----------------------------------------------------------------------------
void svtkEdgeListIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Graph: " << (this->Graph ? "" : "(null)") << endl;
  if (this->Graph)
  {
    this->Graph->PrintSelf(os, indent.GetNextIndent());
  }
}
