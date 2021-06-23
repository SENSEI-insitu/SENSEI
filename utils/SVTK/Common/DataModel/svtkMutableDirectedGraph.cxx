/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMutableDirectedGraph.cxx

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

#include "svtkMutableDirectedGraph.h"

#include "svtkDataSetAttributes.h"
#include "svtkGraphEdge.h"
#include "svtkGraphInternals.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
// class svtkMutableDirectedGraph
//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkMutableDirectedGraph);
//----------------------------------------------------------------------------
svtkMutableDirectedGraph::svtkMutableDirectedGraph()
{
  this->GraphEdge = svtkGraphEdge::New();
}

//----------------------------------------------------------------------------
svtkMutableDirectedGraph::~svtkMutableDirectedGraph()
{
  this->GraphEdge->Delete();
}

//----------------------------------------------------------------------------
svtkIdType svtkMutableDirectedGraph::SetNumberOfVertices(svtkIdType numVerts)
{
  svtkIdType retval = -1;

  if (this->GetDistributedGraphHelper())
  {
    svtkWarningMacro("SetNumberOfVertices will not work on distributed graphs.");
    return retval;
  }

  retval = static_cast<svtkIdType>(this->Internals->Adjacency.size());
  this->Internals->Adjacency.resize(numVerts);
  return retval;
}

//----------------------------------------------------------------------------
svtkIdType svtkMutableDirectedGraph::AddVertex()
{
  if (this->Internals->UsingPedigreeIds && this->GetDistributedGraphHelper() != nullptr)
  {
    svtkErrorMacro("Adding vertex without a pedigree ID into a distributed graph that uses pedigree "
                  "IDs to name vertices");
  }

  return this->AddVertex(nullptr);
}
//----------------------------------------------------------------------------
svtkIdType svtkMutableDirectedGraph::AddVertex(svtkVariantArray* propertyArr)
{
  if (this->GetVertexData()->GetPedigreeIds() != nullptr)
  {
    this->Internals->UsingPedigreeIds = true;
  }

  svtkIdType vertex;
  this->AddVertexInternal(propertyArr, &vertex);
  return vertex;
}

//----------------------------------------------------------------------------
svtkIdType svtkMutableDirectedGraph::AddVertex(const svtkVariant& pedigreeId)
{
  this->Internals->UsingPedigreeIds = true;

  svtkIdType vertex;
  this->AddVertexInternal(pedigreeId, &vertex);
  return vertex;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableDirectedGraph::AddEdge(svtkIdType u, svtkIdType v)
{
  return this->AddEdge(u, v, nullptr);
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableDirectedGraph::AddEdge(svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr)
{
  svtkEdgeType e;
  this->AddEdgeInternal(u, v, true, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableDirectedGraph::AddEdge(
  const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, true, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableDirectedGraph::AddEdge(
  svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, true, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableDirectedGraph::AddEdge(
  const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, true, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddVertex()
{
  if (this->Internals->UsingPedigreeIds && this->GetDistributedGraphHelper() != nullptr)
  {
    svtkErrorMacro("Adding vertex without a pedigree ID into a distributed graph that uses pedigree "
                  "IDs to name vertices");
  }

  this->LazyAddVertex(nullptr);
}
//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddVertex(svtkVariantArray* propertyArr)
{
  if (this->GetVertexData()->GetPedigreeIds() != nullptr)
  {
    this->Internals->UsingPedigreeIds = true;
  }

  this->AddVertexInternal(propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddVertex(const svtkVariant& pedigreeId)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddVertexInternal(pedigreeId, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddEdge(svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->AddEdgeInternal(u, v, true, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddEdge(
  const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, true, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddEdge(
  svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, true, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::LazyAddEdge(
  const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, true, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
svtkGraphEdge* svtkMutableDirectedGraph::AddGraphEdge(svtkIdType u, svtkIdType v)
{
  svtkEdgeType e = this->AddEdge(u, v);
  this->GraphEdge->SetSource(e.Source);
  this->GraphEdge->SetTarget(e.Target);
  this->GraphEdge->SetId(e.Id);
  return this->GraphEdge;
}

//----------------------------------------------------------------------------
svtkIdType svtkMutableDirectedGraph::AddChild(svtkIdType parent, svtkVariantArray* propertyArr /* = 0*/)
{
  svtkIdType v = this->AddVertex();
  this->AddEdge(parent, v, propertyArr);
  return v;
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::RemoveVertex(svtkIdType v)
{
  this->RemoveVertexInternal(v, true);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::RemoveEdge(svtkIdType e)
{
  this->RemoveEdgeInternal(e, true);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::RemoveVertices(svtkIdTypeArray* arr)
{
  this->RemoveVerticesInternal(arr, true);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::RemoveEdges(svtkIdTypeArray* arr)
{
  this->RemoveEdgesInternal(arr, true);
}

//----------------------------------------------------------------------------
void svtkMutableDirectedGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
