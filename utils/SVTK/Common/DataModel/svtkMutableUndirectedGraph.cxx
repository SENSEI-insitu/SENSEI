/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMutableUndirectedGraph.cxx

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

#include "svtkMutableUndirectedGraph.h"

#include "svtkDataSetAttributes.h"
#include "svtkGraphEdge.h"
#include "svtkGraphInternals.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
// class svtkMutableUndirectedGraph
//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkMutableUndirectedGraph);
//----------------------------------------------------------------------------
svtkMutableUndirectedGraph::svtkMutableUndirectedGraph()
{
  this->GraphEdge = svtkGraphEdge::New();
}

//----------------------------------------------------------------------------
svtkMutableUndirectedGraph::~svtkMutableUndirectedGraph()
{
  this->GraphEdge->Delete();
}

//----------------------------------------------------------------------------
svtkIdType svtkMutableUndirectedGraph::SetNumberOfVertices(svtkIdType numVerts)
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
svtkIdType svtkMutableUndirectedGraph::AddVertex()
{
  if (this->Internals->UsingPedigreeIds && this->GetDistributedGraphHelper() != nullptr)
  {
    svtkErrorMacro("Adding vertex without a pedigree ID into a distributed graph that uses pedigree "
                  "IDs to name vertices");
  }

  return this->AddVertex(nullptr);
}
//----------------------------------------------------------------------------
svtkIdType svtkMutableUndirectedGraph::AddVertex(svtkVariantArray* propertyArr)
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
svtkIdType svtkMutableUndirectedGraph::AddVertex(const svtkVariant& pedigreeId)
{
  this->Internals->UsingPedigreeIds = true;

  svtkIdType vertex;
  this->AddVertexInternal(pedigreeId, &vertex);
  return vertex;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableUndirectedGraph::AddEdge(svtkIdType u, svtkIdType v)
{
  return this->AddEdge(u, v, nullptr);
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableUndirectedGraph::AddEdge(
  svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr)
{
  svtkEdgeType e;
  this->AddEdgeInternal(u, v, false, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableUndirectedGraph::AddEdge(
  const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, false, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableUndirectedGraph::AddEdge(
  svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, false, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
svtkEdgeType svtkMutableUndirectedGraph::AddEdge(
  const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  svtkEdgeType e;
  this->AddEdgeInternal(u, v, false, propertyArr, &e);
  return e;
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddVertex()
{
  if (this->Internals->UsingPedigreeIds && this->GetDistributedGraphHelper() != nullptr)
  {
    svtkErrorMacro("Adding vertex without a pedigree ID into a distributed graph that uses pedigree "
                  "IDs to name vertices");
  }

  this->LazyAddVertex(nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddVertex(svtkVariantArray* propertyArr)
{
  if (this->GetVertexData()->GetPedigreeIds() != nullptr)
  {
    this->Internals->UsingPedigreeIds = true;
  }

  this->AddVertexInternal(propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddVertex(const svtkVariant& pedigreeId)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddVertexInternal(pedigreeId, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddEdge(svtkIdType u, svtkIdType v)
{
  this->LazyAddEdge(u, v, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddEdge(svtkIdType u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->AddEdgeInternal(u, v, false, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddEdge(
  const svtkVariant& u, svtkIdType v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, false, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddEdge(
  svtkIdType u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, false, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::LazyAddEdge(
  const svtkVariant& u, const svtkVariant& v, svtkVariantArray* propertyArr)
{
  this->Internals->UsingPedigreeIds = true;

  this->AddEdgeInternal(u, v, false, propertyArr, nullptr);
}

//----------------------------------------------------------------------------
svtkGraphEdge* svtkMutableUndirectedGraph::AddGraphEdge(svtkIdType u, svtkIdType v)
{
  svtkEdgeType e = this->AddEdge(u, v);
  this->GraphEdge->SetSource(e.Source);
  this->GraphEdge->SetTarget(e.Target);
  this->GraphEdge->SetId(e.Id);
  return this->GraphEdge;
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::RemoveVertex(svtkIdType v)
{
  this->RemoveVertexInternal(v, false);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::RemoveEdge(svtkIdType e)
{
  this->RemoveEdgeInternal(e, false);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::RemoveVertices(svtkIdTypeArray* arr)
{
  this->RemoveVerticesInternal(arr, false);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::RemoveEdges(svtkIdTypeArray* arr)
{
  this->RemoveEdgesInternal(arr, false);
}

//----------------------------------------------------------------------------
void svtkMutableUndirectedGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
