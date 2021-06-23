/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGraph.cxx

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

#include "svtkGraph.h"

#include "svtkAdjacentVertexIterator.h"
#include "svtkCellArray.h"
#include "svtkDataSetAttributes.h"
#include "svtkDirectedGraph.h"
#include "svtkDistributedGraphHelper.h"
#include "svtkEdgeListIterator.h"
#include "svtkGraphEdge.h"
#include "svtkGraphInternals.h"
#include "svtkIdTypeArray.h"
#include "svtkInEdgeIterator.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkMath.h"
#include "svtkMutableDirectedGraph.h"
#include "svtkMutableUndirectedGraph.h"
#include "svtkObjectFactory.h"
#include "svtkOutEdgeIterator.h"
#include "svtkPoints.h"
#include "svtkSmartPointer.h"
#include "svtkStringArray.h"
#include "svtkUndirectedGraph.h"
#include "svtkVariantArray.h"
#include "svtkVertexListIterator.h"

#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

double svtkGraph::DefaultPoint[3] = { 0, 0, 0 };

//----------------------------------------------------------------------------
// private class svtkGraphEdgePoints
//----------------------------------------------------------------------------
class svtkGraphEdgePoints : public svtkObject
{
public:
  static svtkGraphEdgePoints* New();
  svtkTypeMacro(svtkGraphEdgePoints, svtkObject);
  std::vector<std::vector<double> > Storage;

protected:
  svtkGraphEdgePoints() = default;
  ~svtkGraphEdgePoints() override = default;

private:
  svtkGraphEdgePoints(const svtkGraphEdgePoints&) = delete;
  void operator=(const svtkGraphEdgePoints&) = delete;
};
svtkStandardNewMacro(svtkGraphEdgePoints);

//----------------------------------------------------------------------------
// class svtkGraph
//----------------------------------------------------------------------------
svtkCxxSetObjectMacro(svtkGraph, Points, svtkPoints);
svtkCxxSetObjectMacro(svtkGraph, Internals, svtkGraphInternals);
svtkCxxSetObjectMacro(svtkGraph, EdgePoints, svtkGraphEdgePoints);
svtkCxxSetObjectMacro(svtkGraph, EdgeList, svtkIdTypeArray);
//----------------------------------------------------------------------------
svtkGraph::svtkGraph()
{
  this->VertexData = svtkDataSetAttributes::New();
  this->EdgeData = svtkDataSetAttributes::New();
  this->Points = nullptr;
  svtkMath::UninitializeBounds(this->Bounds);

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_PIECES_EXTENT);
  this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);

  this->Internals = svtkGraphInternals::New();
  this->DistributedHelper = nullptr;
  this->EdgePoints = nullptr;
  this->EdgeList = nullptr;
}

//----------------------------------------------------------------------------
svtkGraph::~svtkGraph()
{
  this->VertexData->Delete();
  this->EdgeData->Delete();
  if (this->Points)
  {
    this->Points->Delete();
  }
  this->Internals->Delete();
  if (this->DistributedHelper)
  {
    this->DistributedHelper->Delete();
  }
  if (this->EdgeList)
  {
    this->EdgeList->Delete();
  }
  if (this->EdgePoints)
  {
    this->EdgePoints->Delete();
  }
}

//----------------------------------------------------------------------------
double* svtkGraph::GetPoint(svtkIdType ptId)
{
  if (this->Points)
  {
    return this->Points->GetPoint(ptId);
  }
  return this->DefaultPoint;
}

//----------------------------------------------------------------------------
void svtkGraph::GetPoint(svtkIdType ptId, double x[3])
{
  if (this->Points)
  {
    svtkIdType index = ptId;
    if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
    {
      int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
      if (myRank != helper->GetVertexOwner(ptId))
      {
        svtkErrorMacro("svtkGraph cannot retrieve a point for a non-local vertex");
        return;
      }

      index = helper->GetVertexIndex(ptId);
    }

    this->Points->GetPoint(index, x);
  }
  else
  {
    for (int i = 0; i < 3; i++)
    {
      x[i] = this->DefaultPoint[i];
    }
  }
}

//----------------------------------------------------------------------------
svtkPoints* svtkGraph::GetPoints()
{
  if (!this->Points)
  {
    this->Points = svtkPoints::New();
  }
  if (this->Points->GetNumberOfPoints() != this->GetNumberOfVertices())
  {
    this->Points->SetNumberOfPoints(this->GetNumberOfVertices());
    for (svtkIdType i = 0; i < this->GetNumberOfVertices(); i++)
    {
      this->Points->SetPoint(i, 0, 0, 0);
    }
  }
  return this->Points;
}

//----------------------------------------------------------------------------
void svtkGraph::ComputeBounds()
{
  if (this->Points)
  {
    if (this->GetMTime() >= this->ComputeTime)
    {
      const double* bounds = this->Points->GetBounds();
      for (int i = 0; i < 6; i++)
      {
        this->Bounds[i] = bounds[i];
      }
      // TODO: how to compute the bounds for a distributed graph?
      this->ComputeTime.Modified();
    }
  }
}

//----------------------------------------------------------------------------
double* svtkGraph::GetBounds()
{
  this->ComputeBounds();
  return this->Bounds;
}

//----------------------------------------------------------------------------
void svtkGraph::GetBounds(double bounds[6])
{
  this->ComputeBounds();
  for (int i = 0; i < 6; i++)
  {
    bounds[i] = this->Bounds[i];
  }
}

//----------------------------------------------------------------------------
svtkMTimeType svtkGraph::GetMTime()
{
  svtkMTimeType doTime = svtkDataObject::GetMTime();

  if (this->VertexData->GetMTime() > doTime)
  {
    doTime = this->VertexData->GetMTime();
  }
  if (this->EdgeData->GetMTime() > doTime)
  {
    doTime = this->EdgeData->GetMTime();
  }
  if (this->Points)
  {
    if (this->Points->GetMTime() > doTime)
    {
      doTime = this->Points->GetMTime();
    }
  }

  return doTime;
}

//----------------------------------------------------------------------------
void svtkGraph::Initialize()
{
  this->ForceOwnership();
  Superclass::Initialize();
  this->EdgeData->Initialize();
  this->VertexData->Initialize();
  this->Internals->NumberOfEdges = 0;
  this->Internals->Adjacency.clear();
  if (this->EdgePoints)
  {
    this->EdgePoints->Storage.clear();
  }
}

//----------------------------------------------------------------------------
void svtkGraph::GetOutEdges(svtkIdType v, svtkOutEdgeIterator* it)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the out edges for non-local vertex " << v);
      return;
    }
  }

  if (it)
  {
    it->Initialize(this, v);
  }
}

//----------------------------------------------------------------------------
svtkOutEdgeType svtkGraph::GetOutEdge(svtkIdType v, svtkIdType i)
{
  svtkIdType index = v;
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the out edges for non-local vertex " << v);
      return svtkOutEdgeType();
    }
    index = helper->GetVertexIndex(v);
  }

  if (i < this->GetOutDegree(v))
  {
    return this->Internals->Adjacency[index].OutEdges[i];
  }
  svtkErrorMacro("Out edge index out of bounds");
  return svtkOutEdgeType();
}

//----------------------------------------------------------------------------
void svtkGraph::GetOutEdge(svtkIdType v, svtkIdType i, svtkGraphEdge* e)
{
  svtkOutEdgeType oe = this->GetOutEdge(v, i);
  e->SetId(oe.Id);
  e->SetSource(v);
  e->SetTarget(oe.Target);
}

//----------------------------------------------------------------------------
void svtkGraph::GetOutEdges(svtkIdType v, const svtkOutEdgeType*& edges, svtkIdType& nedges)
{
  svtkIdType index = v;
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the out edges for non-local vertex " << v);
      return;
    }

    index = helper->GetVertexIndex(v);
  }

  nedges = static_cast<svtkIdType>(this->Internals->Adjacency[index].OutEdges.size());
  if (nedges > 0)
  {
    edges = &(this->Internals->Adjacency[index].OutEdges[0]);
  }
  else
  {
    edges = nullptr;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetOutDegree(svtkIdType v)
{
  svtkIdType index = v;
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot determine the out degree for a non-local vertex");
      return 0;
    }

    index = helper->GetVertexIndex(v);
  }
  return static_cast<svtkIdType>(this->Internals->Adjacency[index].OutEdges.size());
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetDegree(svtkIdType v)
{
  svtkIdType index = v;

  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot determine the degree for a non-local vertex");
      return 0;
    }

    index = helper->GetVertexIndex(v);
  }

  return static_cast<svtkIdType>(this->Internals->Adjacency[index].InEdges.size() +
    this->Internals->Adjacency[index].OutEdges.size());
}

//----------------------------------------------------------------------------
void svtkGraph::GetInEdges(svtkIdType v, svtkInEdgeIterator* it)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the in edges for a non-local vertex");
      return;
    }
  }

  if (it)
  {
    it->Initialize(this, v);
  }
}

//----------------------------------------------------------------------------
svtkInEdgeType svtkGraph::GetInEdge(svtkIdType v, svtkIdType i)
{
  svtkIdType index = v;
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the in edges for a non-local vertex");
      return svtkInEdgeType();
    }
    index = helper->GetVertexIndex(v);
  }

  if (i < this->GetInDegree(v))
  {
    return this->Internals->Adjacency[index].InEdges[i];
  }
  svtkErrorMacro("In edge index out of bounds");
  return svtkInEdgeType();
}

//----------------------------------------------------------------------------
void svtkGraph::GetInEdge(svtkIdType v, svtkIdType i, svtkGraphEdge* e)
{
  svtkInEdgeType ie = this->GetInEdge(v, i);
  e->SetId(ie.Id);
  e->SetSource(ie.Source);
  e->SetTarget(v);
}

//----------------------------------------------------------------------------
void svtkGraph::GetInEdges(svtkIdType v, const svtkInEdgeType*& edges, svtkIdType& nedges)
{
  svtkIdType index = v;

  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the in edges for a non-local vertex");
      return;
    }

    index = helper->GetVertexIndex(v);
  }

  nedges = static_cast<svtkIdType>(this->Internals->Adjacency[index].InEdges.size());
  if (nedges > 0)
  {
    edges = &(this->Internals->Adjacency[index].InEdges[0]);
  }
  else
  {
    edges = nullptr;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetInDegree(svtkIdType v)
{
  svtkIdType index = v;

  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot determine the in degree for a non-local vertex");
      return 0;
    }

    index = helper->GetVertexIndex(v);
  }

  return static_cast<svtkIdType>(this->Internals->Adjacency[index].InEdges.size());
}

//----------------------------------------------------------------------------
void svtkGraph::GetAdjacentVertices(svtkIdType v, svtkAdjacentVertexIterator* it)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot retrieve the adjacent vertices for a non-local vertex");
      return;
    }
  }

  if (it)
  {
    it->Initialize(this, v);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::GetEdges(svtkEdgeListIterator* it)
{
  if (it)
  {
    it->SetGraph(this);
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetNumberOfEdges()
{
  return this->Internals->NumberOfEdges;
}

//----------------------------------------------------------------------------
void svtkGraph::GetVertices(svtkVertexListIterator* it)
{
  if (it)
  {
    it->SetGraph(this);
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetNumberOfVertices()
{
  return static_cast<svtkIdType>(this->Internals->Adjacency.size());
}

//----------------------------------------------------------------------------
void svtkGraph::SetDistributedGraphHelper(svtkDistributedGraphHelper* helper)
{
  if (this->DistributedHelper)
    this->DistributedHelper->AttachToGraph(nullptr);

  this->DistributedHelper = helper;
  if (this->DistributedHelper)
  {
    this->DistributedHelper->Register(this);
    this->DistributedHelper->AttachToGraph(this);
  }
}

//----------------------------------------------------------------------------
svtkDistributedGraphHelper* svtkGraph::GetDistributedGraphHelper()
{
  return this->DistributedHelper;
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::FindVertex(const svtkVariant& pedigreeId)
{
  svtkAbstractArray* pedigrees = this->GetVertexData()->GetPedigreeIds();
  if (pedigrees == nullptr)
  {
    return -1;
  }

  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    svtkIdType myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (helper->GetVertexOwnerByPedigreeId(pedigreeId) != myRank)
    {
      // The vertex is remote; ask the distributed graph helper to find it.
      return helper->FindVertex(pedigreeId);
    }

    svtkIdType result = pedigrees->LookupValue(pedigreeId);
    if (result == -1)
    {
      return -1;
    }
    return helper->MakeDistributedId(myRank, result);
  }

  return pedigrees->LookupValue(pedigreeId);
}

//----------------------------------------------------------------------------
bool svtkGraph::CheckedShallowCopy(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }
  bool valid = this->IsStructureValid(g);
  if (valid)
  {
    this->CopyInternal(g, false);
  }
  return valid;
}

//----------------------------------------------------------------------------
bool svtkGraph::CheckedDeepCopy(svtkGraph* g)
{
  if (!g)
  {
    return false;
  }
  bool valid = this->IsStructureValid(g);
  if (valid)
  {
    this->CopyInternal(g, true);
  }
  return valid;
}

//----------------------------------------------------------------------------
void svtkGraph::ShallowCopy(svtkDataObject* obj)
{
  svtkGraph* g = svtkGraph::SafeDownCast(obj);
  if (!g)
  {
    svtkErrorMacro("Can only shallow copy from svtkGraph subclass.");
    return;
  }
  bool valid = this->IsStructureValid(g);
  if (valid)
  {
    this->CopyInternal(g, false);
  }
  else
  {
    svtkErrorMacro("Invalid graph structure for this type of graph.");
  }
}

//----------------------------------------------------------------------------
void svtkGraph::DeepCopy(svtkDataObject* obj)
{
  svtkGraph* g = svtkGraph::SafeDownCast(obj);
  if (!g)
  {
    svtkErrorMacro("Can only shallow copy from svtkGraph subclass.");
    return;
  }
  bool valid = this->IsStructureValid(g);
  if (valid)
  {
    this->CopyInternal(g, true);
  }
  else
  {
    svtkErrorMacro("Invalid graph structure for this type of graph.");
  }
}

//----------------------------------------------------------------------------
void svtkGraph::CopyStructure(svtkGraph* g)
{
  // Copy on write.
  this->SetInternals(g->Internals);
  if (g->Points)
  {
    if (!this->Points)
    {
      this->Points = svtkPoints::New();
    }
    this->Points->ShallowCopy(g->Points);
  }
  else if (this->Points)
  {
    this->Points->Delete();
    this->Points = nullptr;
  }

  // Propagate information used by distributed graphs
  this->Information->Set(
    svtkDataObject::DATA_PIECE_NUMBER(), g->Information->Get(svtkDataObject::DATA_PIECE_NUMBER()));
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(),
    g->Information->Get(svtkDataObject::DATA_NUMBER_OF_PIECES()));
}

//----------------------------------------------------------------------------
void svtkGraph::CopyInternal(svtkGraph* g, bool deep)
{
  if (deep)
  {
    svtkDataObject::DeepCopy(g);
  }
  else
  {
    svtkDataObject::ShallowCopy(g);
  }
  if (g->DistributedHelper)
  {
    if (!this->DistributedHelper)
    {
      this->SetDistributedGraphHelper(g->DistributedHelper->Clone());
    }
  }
  else if (this->DistributedHelper)
  {
    this->SetDistributedGraphHelper(nullptr);
  }

  // Copy on write.
  this->SetInternals(g->Internals);

  if (deep)
  {
    this->EdgeData->DeepCopy(g->EdgeData);
    this->VertexData->DeepCopy(g->VertexData);
    this->DeepCopyEdgePoints(g);
  }
  else
  {
    this->EdgeData->ShallowCopy(g->EdgeData);
    this->VertexData->ShallowCopy(g->VertexData);
    this->ShallowCopyEdgePoints(g);
  }

  // Copy points
  if (g->Points && deep)
  {
    if (!this->Points)
    {
      this->Points = svtkPoints::New();
    }
    this->Points->DeepCopy(g->Points);
  }
  else
  {
    this->SetPoints(g->Points);
  }

  // Copy edge list
  this->Internals->NumberOfEdges = g->Internals->NumberOfEdges;
  if (g->EdgeList && deep)
  {
    if (!this->EdgeList)
    {
      this->EdgeList = svtkIdTypeArray::New();
    }
    this->EdgeList->DeepCopy(g->EdgeList);
  }
  else
  {
    this->SetEdgeList(g->EdgeList);
    if (g->EdgeList)
    {
      this->BuildEdgeList();
    }
  }

  // Propagate information used by distributed graphs
  this->Information->Set(
    svtkDataObject::DATA_PIECE_NUMBER(), g->Information->Get(svtkDataObject::DATA_PIECE_NUMBER()));
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(),
    g->Information->Get(svtkDataObject::DATA_NUMBER_OF_PIECES()));
}

//----------------------------------------------------------------------------
void svtkGraph::Squeeze()
{
  if (this->Points)
  {
    this->Points->Squeeze();
  }
  this->EdgeData->Squeeze();
  this->VertexData->Squeeze();
}

//----------------------------------------------------------------------------
unsigned long svtkGraph::GetActualMemorySize()
{
  unsigned long size = this->Superclass::GetActualMemorySize();
  size += this->EdgeData->GetActualMemorySize();
  size += this->VertexData->GetActualMemorySize();
  if (this->Points)
  {
    size += this->Points->GetActualMemorySize();
  }
  return size;
}

//----------------------------------------------------------------------------
svtkGraph* svtkGraph::GetData(svtkInformation* info)
{
  return info ? svtkGraph::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkGraph* svtkGraph::GetData(svtkInformationVector* v, int i)
{
  return svtkGraph::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetSourceVertex(svtkIdType e)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      if (e != this->Internals->LastRemoteEdgeId)
      {
        helper->FindEdgeSourceAndTarget(
          e, &this->Internals->LastRemoteEdgeSource, &this->Internals->LastRemoteEdgeTarget);
      }

      return this->Internals->LastRemoteEdgeSource;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e >= this->GetNumberOfEdges())
  {
    svtkErrorMacro("Edge index out of range.");
    return -1;
  }
  if (!this->EdgeList)
  {
    this->BuildEdgeList();
  }
  return this->EdgeList->GetValue(2 * e);
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetTargetVertex(svtkIdType e)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      if (e != this->Internals->LastRemoteEdgeId)
      {
        this->Internals->LastRemoteEdgeId = e;
        helper->FindEdgeSourceAndTarget(
          e, &this->Internals->LastRemoteEdgeSource, &this->Internals->LastRemoteEdgeTarget);
      }

      return this->Internals->LastRemoteEdgeTarget;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e >= this->GetNumberOfEdges())
  {
    svtkErrorMacro("Edge index out of range.");
    return -1;
  }
  if (!this->EdgeList)
  {
    this->BuildEdgeList();
  }
  return this->EdgeList->GetValue(2 * e + 1);
}

//----------------------------------------------------------------------------
void svtkGraph::BuildEdgeList()
{
  if (this->EdgeList)
  {
    this->EdgeList->SetNumberOfTuples(this->GetNumberOfEdges());
  }
  else
  {
    this->EdgeList = svtkIdTypeArray::New();
    this->EdgeList->SetNumberOfComponents(2);
    this->EdgeList->SetNumberOfTuples(this->GetNumberOfEdges());
  }
  svtkEdgeListIterator* it = svtkEdgeListIterator::New();
  this->GetEdges(it);
  while (it->HasNext())
  {
    svtkEdgeType e = it->Next();
    this->EdgeList->SetValue(2 * e.Id, e.Source);
    this->EdgeList->SetValue(2 * e.Id + 1, e.Target);
  }
  it->Delete();
}

//----------------------------------------------------------------------------
void svtkGraph::SetEdgePoints(svtkIdType e, svtkIdType npts, const double pts[])
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot set edge points for a non-local vertex");
      return;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return;
  }
  if (!this->EdgePoints)
  {
    this->EdgePoints = svtkGraphEdgePoints::New();
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  this->EdgePoints->Storage[e].clear();
  for (svtkIdType i = 0; i < 3 * npts; ++i, ++pts)
  {
    this->EdgePoints->Storage[e].push_back(*pts);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::GetEdgePoints(svtkIdType e, svtkIdType& npts, double*& pts)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot retrieve edge points for a non-local vertex");
      return;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return;
  }
  if (!this->EdgePoints)
  {
    npts = 0;
    pts = nullptr;
    return;
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  npts = static_cast<svtkIdType>(this->EdgePoints->Storage[e].size() / 3);
  if (npts > 0)
  {
    pts = &this->EdgePoints->Storage[e][0];
  }
  else
  {
    pts = nullptr;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetNumberOfEdgePoints(svtkIdType e)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot retrieve edge points for a non-local vertex");
      return 0;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return 0;
  }
  if (!this->EdgePoints)
  {
    return 0;
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  return static_cast<svtkIdType>(this->EdgePoints->Storage[e].size() / 3);
}

//----------------------------------------------------------------------------
double* svtkGraph::GetEdgePoint(svtkIdType e, svtkIdType i)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot receive edge points for a non-local vertex");
      return nullptr;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return nullptr;
  }
  if (!this->EdgePoints)
  {
    this->EdgePoints = svtkGraphEdgePoints::New();
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  svtkIdType npts = static_cast<svtkIdType>(this->EdgePoints->Storage[e].size() / 3);
  if (i >= npts)
  {
    svtkErrorMacro("Edge point index out of range.");
    return nullptr;
  }
  return &this->EdgePoints->Storage[e][3 * i];
}

//----------------------------------------------------------------------------
void svtkGraph::SetEdgePoint(svtkIdType e, svtkIdType i, const double x[3])
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot set edge points for a non-local vertex");
      return;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return;
  }
  if (!this->EdgePoints)
  {
    this->EdgePoints = svtkGraphEdgePoints::New();
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  svtkIdType npts = static_cast<svtkIdType>(this->EdgePoints->Storage[e].size() / 3);
  if (i >= npts)
  {
    svtkErrorMacro("Edge point index out of range.");
    return;
  }
  for (int c = 0; c < 3; ++c)
  {
    this->EdgePoints->Storage[e][3 * i + c] = x[c];
  }
}

//----------------------------------------------------------------------------
void svtkGraph::ClearEdgePoints(svtkIdType e)
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot clear edge points for a non-local vertex");
      return;
    }

    e = helper->GetEdgeIndex(e);
  }

  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return;
  }
  if (!this->EdgePoints)
  {
    this->EdgePoints = svtkGraphEdgePoints::New();
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  this->EdgePoints->Storage[e].clear();
}

//----------------------------------------------------------------------------
void svtkGraph::AddEdgePoint(svtkIdType e, const double x[3])
{
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetEdgeOwner(e))
    {
      svtkErrorMacro("svtkGraph cannot set edge points for a non-local vertex");
      return;
    }

    e = helper->GetEdgeIndex(e);
  }
  if (e < 0 || e > this->Internals->NumberOfEdges)
  {
    svtkErrorMacro("Invalid edge id.");
    return;
  }
  if (!this->EdgePoints)
  {
    this->EdgePoints = svtkGraphEdgePoints::New();
  }
  std::vector<std::vector<double> >::size_type numEdges = this->Internals->NumberOfEdges;
  if (this->EdgePoints->Storage.size() < numEdges)
  {
    this->EdgePoints->Storage.resize(numEdges);
  }
  for (int c = 0; c < 3; ++c)
  {
    this->EdgePoints->Storage[e].push_back(x[c]);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::ShallowCopyEdgePoints(svtkGraph* g)
{
  this->SetEdgePoints(g->EdgePoints);
}

//----------------------------------------------------------------------------
void svtkGraph::DeepCopyEdgePoints(svtkGraph* g)
{
  if (g->EdgePoints)
  {
    if (!this->EdgePoints)
    {
      this->EdgePoints = svtkGraphEdgePoints::New();
    }
    this->EdgePoints->Storage = g->EdgePoints->Storage;
  }
  else
  {
    this->SetEdgePoints(nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::GetInducedEdges(svtkIdTypeArray* verts, svtkIdTypeArray* edges)
{
  edges->Initialize();
  if (this->GetDistributedGraphHelper())
  {
    svtkErrorMacro("Cannot get induced edges on a distributed graph.");
    return;
  }
  svtkSmartPointer<svtkEdgeListIterator> edgeIter = svtkSmartPointer<svtkEdgeListIterator>::New();
  this->GetEdges(edgeIter);
  while (edgeIter->HasNext())
  {
    svtkEdgeType e = edgeIter->Next();
    if (verts->LookupValue(e.Source) >= 0 && verts->LookupValue(e.Target) >= 0)
    {
      edges->InsertNextValue(e.Id);
    }
  }
}

//----------------------------------------------------------------------------
void svtkGraph::AddVertexInternal(svtkVariantArray* propertyArr, svtkIdType* vertex)
{
  this->ForceOwnership();

  svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper();

  if (propertyArr) // Add/replace vertex properties if passed in
  {
    svtkAbstractArray* peds = this->GetVertexData()->GetPedigreeIds();
    // If the properties include pedigreeIds, we need to see if this
    // pedigree already exists and, if so, simply update its properties.
    if (peds)
    {
      // Get the array index associated with pedIds.
      svtkIdType pedIdx = this->GetVertexData()->SetPedigreeIds(peds);
      svtkVariant pedigreeId = propertyArr->GetValue(pedIdx);
      if (helper)
      {
        svtkIdType myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
        if (helper->GetVertexOwnerByPedigreeId(pedigreeId) != myRank)
        {
          helper->AddVertexInternal(propertyArr, vertex);
          return;
        }
      }

      svtkIdType vertexIndex = this->FindVertex(pedigreeId);

      // FindVertex returns distributed ids for parallel graphs, must account
      // for this prior to the range check.
      if (helper)
      {
        vertexIndex = helper->GetVertexIndex(vertexIndex);
      }
      if (vertexIndex != -1 && vertexIndex < this->GetNumberOfVertices())
      {
        for (int iprop = 0; iprop < propertyArr->GetNumberOfValues(); iprop++)
        {
          svtkAbstractArray* arr = this->GetVertexData()->GetAbstractArray(iprop);
          arr->InsertVariantValue(vertexIndex, propertyArr->GetValue(iprop));
        }
        if (vertex)
        {
          *vertex = vertexIndex;
        }
        return;
      }

      this->Internals->Adjacency.push_back(svtkVertexAdjacencyList()); // Add a new (local) vertex
      svtkIdType index = static_cast<svtkIdType>(this->Internals->Adjacency.size() - 1);

      svtkDataSetAttributes* vertexData = this->GetVertexData();
      int numProps = propertyArr->GetNumberOfValues();
      assert(numProps == vertexData->GetNumberOfArrays());
      for (int iprop = 0; iprop < numProps; iprop++)
      {
        svtkAbstractArray* arr = vertexData->GetAbstractArray(iprop);
        arr->InsertVariantValue(index, propertyArr->GetValue(iprop));
      }
    } // end if (peds)
    //----------------------------------------------------------------
    else // We have propArr, but not pedIds - just add the propArr
    {
      this->Internals->Adjacency.push_back(svtkVertexAdjacencyList());
      svtkIdType index = static_cast<svtkIdType>(this->Internals->Adjacency.size() - 1);

      svtkDataSetAttributes* vertexData = this->GetVertexData();
      int numProps = propertyArr->GetNumberOfValues();
      assert(numProps == vertexData->GetNumberOfArrays());
      for (int iprop = 0; iprop < numProps; iprop++)
      {
        svtkAbstractArray* arr = vertexData->GetAbstractArray(iprop);
        arr->InsertVariantValue(index, propertyArr->GetValue(iprop));
      }
    }
  }
  else // No properties, just add a new vertex
  {
    this->Internals->Adjacency.push_back(svtkVertexAdjacencyList());
  }

  if (vertex)
  {
    if (helper)
    {
      *vertex =
        helper->MakeDistributedId(this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER()),
          static_cast<svtkIdType>(this->Internals->Adjacency.size() - 1));
    }
    else
    {
      *vertex = static_cast<svtkIdType>(this->Internals->Adjacency.size() - 1);
    }
  }
}
//----------------------------------------------------------------------------
void svtkGraph::AddVertexInternal(const svtkVariant& pedigreeId, svtkIdType* vertex)
{
  // Add vertex V, given a pedId:
  //   1) if a dist'd G and this proc doesn't own V, add it (via helper class) and RETURN.
  //   2) if V already exists for this pedId, RETURN it.
  //   3) add V locally and insert its pedId
  svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper();
  if (helper)
  {
    svtkIdType myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (helper->GetVertexOwnerByPedigreeId(pedigreeId) != myRank)
    {
      helper->AddVertexInternal(pedigreeId, vertex);
      return;
    }
  }

  svtkIdType vertexIndex = this->FindVertex(pedigreeId);

  // If we're on a distributed graph, FindVertex returns a distributed-id,
  // must account for that.
  if (helper)
  {
    vertexIndex = helper->GetVertexIndex(vertexIndex);
  }
  if (vertexIndex != -1 && vertexIndex < this->GetNumberOfVertices())
  {
    // We found this vertex; nothing more to do.
    if (vertex)
    {
      *vertex = vertexIndex;
    }
    return;
  }

  // Add the vertex locally
  this->ForceOwnership();
  svtkIdType v;
  this->AddVertexInternal(nullptr, &v);
  if (vertex)
  {
    *vertex = v;
  }

  // Add the pedigree ID of the vertex
  svtkAbstractArray* pedigrees = this->GetVertexData()->GetPedigreeIds();
  if (pedigrees == nullptr)
  {
    svtkErrorMacro("Added a vertex with a pedigree ID to a svtkGraph with no pedigree ID array");
    return;
  }

  svtkIdType index = v;
  if (helper)
  {
    index = helper->GetVertexIndex(v);
  }

  pedigrees->InsertVariantValue(index, pedigreeId);
}
//----------------------------------------------------------------------------
void svtkGraph::AddEdgeInternal(
  svtkIdType u, svtkIdType v, bool directed, svtkVariantArray* propertyArr, svtkEdgeType* edge)
{
  this->ForceOwnership();
  if (this->DistributedHelper)
  {
    this->DistributedHelper->AddEdgeInternal(u, v, directed, propertyArr, edge);
    return;
  }

  if (u >= this->GetNumberOfVertices() || v >= this->GetNumberOfVertices())
  {
    svtkErrorMacro(<< "Vertex index out of range");
    return;
  }

  svtkIdType edgeId = this->Internals->NumberOfEdges;
  svtkIdType edgeIndex = edgeId;
  this->Internals->NumberOfEdges++;
  this->Internals->Adjacency[u].OutEdges.push_back(svtkOutEdgeType(v, edgeId));
  if (directed)
  {
    this->Internals->Adjacency[v].InEdges.push_back(svtkInEdgeType(u, edgeId));
  }
  else if (u != v)
  {
    // Avoid storing self-loops twice in undirected graphs.
    this->Internals->Adjacency[v].OutEdges.push_back(svtkOutEdgeType(u, edgeId));
  }

  if (this->EdgeList)
  {
    this->EdgeList->InsertNextValue(u);
    this->EdgeList->InsertNextValue(v);
  }

  if (edge)
  {
    *edge = svtkEdgeType(u, v, edgeId);
  }

  if (propertyArr)
  {
    // Insert edge properties
    svtkDataSetAttributes* edgeData = this->GetEdgeData();
    int numProps = propertyArr->GetNumberOfValues();
    assert(numProps == edgeData->GetNumberOfArrays());
    for (int iprop = 0; iprop < numProps; iprop++)
    {
      svtkAbstractArray* array = edgeData->GetAbstractArray(iprop);
      array->InsertVariantValue(edgeIndex, propertyArr->GetValue(iprop));
    }
  }
}

//----------------------------------------------------------------------------
void svtkGraph::AddEdgeInternal(const svtkVariant& uPedigreeId, svtkIdType v, bool directed,
  svtkVariantArray* propertyArr, svtkEdgeType* edge)
{
  this->ForceOwnership();
  if (this->DistributedHelper)
  {
    this->DistributedHelper->AddEdgeInternal(uPedigreeId, v, directed, propertyArr, edge);
    return;
  }

  svtkIdType u;
  this->AddVertexInternal(uPedigreeId, &u);
  this->AddEdgeInternal(u, v, directed, propertyArr, edge);
}

//----------------------------------------------------------------------------
void svtkGraph::AddEdgeInternal(svtkIdType u, const svtkVariant& vPedigreeId, bool directed,
  svtkVariantArray* propertyArr, svtkEdgeType* edge)
{
  this->ForceOwnership();
  if (this->DistributedHelper)
  {
    this->DistributedHelper->AddEdgeInternal(u, vPedigreeId, directed, propertyArr, edge);
    return;
  }

  svtkIdType v;
  this->AddVertexInternal(vPedigreeId, &v);
  this->AddEdgeInternal(u, v, directed, propertyArr, edge);
}

//----------------------------------------------------------------------------
void svtkGraph::AddEdgeInternal(const svtkVariant& uPedigreeId, const svtkVariant& vPedigreeId,
  bool directed, svtkVariantArray* propertyArr, svtkEdgeType* edge)
{
  this->ForceOwnership();
  if (this->DistributedHelper)
  {
    this->DistributedHelper->AddEdgeInternal(uPedigreeId, vPedigreeId, directed, propertyArr, edge);
    return;
  }

  svtkIdType u, v;
  this->AddVertexInternal(uPedigreeId, &u);
  this->AddVertexInternal(vPedigreeId, &v);
  this->AddEdgeInternal(u, v, directed, propertyArr, edge);
}

//----------------------------------------------------------------------------
void svtkGraph::RemoveVertexInternal(svtkIdType v, bool directed)
{
  if (this->DistributedHelper)
  {
    svtkErrorMacro("Cannot remove vertices in a distributed graph.");
    return;
  }
  if (v < 0 || v >= this->GetNumberOfVertices())
  {
    return;
  }

  this->ForceOwnership();
  if (!this->EdgeList)
  {
    this->BuildEdgeList();
  }

  // Remove connected edges
  std::set<svtkIdType> edges;
  std::vector<svtkOutEdgeType>::iterator oi, oiEnd;
  oiEnd = this->Internals->Adjacency[v].OutEdges.end();
  for (oi = this->Internals->Adjacency[v].OutEdges.begin(); oi != oiEnd; ++oi)
  {
    edges.insert(oi->Id);
  }
  std::vector<svtkInEdgeType>::iterator ii, iiEnd;
  iiEnd = this->Internals->Adjacency[v].InEdges.end();
  for (ii = this->Internals->Adjacency[v].InEdges.begin(); ii != iiEnd; ++ii)
  {
    edges.insert(ii->Id);
  }
  std::set<svtkIdType>::reverse_iterator ei, eiEnd;
  eiEnd = edges.rend();
  for (ei = edges.rbegin(); ei != eiEnd; ++ei)
  {
    this->RemoveEdgeInternal(*ei, directed);
  }

  // Replace all occurrences of last vertex id with v
  svtkIdType lv = this->GetNumberOfVertices() - 1;
  this->Internals->Adjacency[v] = this->Internals->Adjacency[lv];
  oiEnd = this->Internals->Adjacency[v].OutEdges.end();
  for (oi = this->Internals->Adjacency[v].OutEdges.begin(); oi != oiEnd; ++oi)
  {
    if (oi->Target == lv)
    {
      oi->Target = v;
      this->EdgeList->SetValue(2 * oi->Id + 1, v);
      continue;
    }
    if (directed)
    {
      iiEnd = this->Internals->Adjacency[oi->Target].InEdges.end();
      for (ii = this->Internals->Adjacency[oi->Target].InEdges.begin(); ii != iiEnd; ++ii)
      {
        if (ii->Source == lv)
        {
          ii->Source = v;
          this->EdgeList->SetValue(2 * ii->Id + 0, v);
        }
      }
    }
    else
    {
      std::vector<svtkOutEdgeType>::iterator oi2, oi2End;
      oi2End = this->Internals->Adjacency[oi->Target].OutEdges.end();
      for (oi2 = this->Internals->Adjacency[oi->Target].OutEdges.begin(); oi2 != oi2End; ++oi2)
      {
        if (oi2->Target == lv)
        {
          oi2->Target = v;
          this->EdgeList->SetValue(2 * oi2->Id + 1, v);
        }
      }
    }
  }

  if (directed)
  {
    iiEnd = this->Internals->Adjacency[v].InEdges.end();
    for (ii = this->Internals->Adjacency[v].InEdges.begin(); ii != iiEnd; ++ii)
    {
      if (ii->Source == lv)
      {
        ii->Source = v;
        this->EdgeList->SetValue(2 * ii->Id + 0, v);
        continue;
      }
      oiEnd = this->Internals->Adjacency[ii->Source].OutEdges.end();
      for (oi = this->Internals->Adjacency[ii->Source].OutEdges.begin(); oi != oiEnd; ++oi)
      {
        if (oi->Target == lv)
        {
          oi->Target = v;
          this->EdgeList->SetValue(2 * oi->Id + 1, v);
        }
      }
    }
  }

  // Update properties
  svtkDataSetAttributes* vd = this->GetVertexData();
  for (int i = 0; i < vd->GetNumberOfArrays(); ++i)
  {
    svtkAbstractArray* arr = vd->GetAbstractArray(i);
    arr->SetTuple(v, lv, arr);
    arr->SetNumberOfTuples(lv);
  }

  // Update points
  if (this->Points)
  {
    double x[3];
    this->Points->GetPoint(lv, x);
    //    this->Points->GetPoint(lv);
    this->Points->SetPoint(v, x);
    this->Points->SetNumberOfPoints(lv);
  }

  this->Internals->Adjacency.pop_back();
}

//----------------------------------------------------------------------------
void svtkGraph::RemoveEdgeInternal(svtkIdType e, bool directed)
{
  if (this->DistributedHelper)
  {
    svtkErrorMacro("Cannot remove edges in a distributed graph.");
    return;
  }
  if (e < 0 || e >= this->GetNumberOfEdges())
  {
    return;
  }
  this->ForceOwnership();
  svtkIdType u = this->GetSourceVertex(e);
  svtkIdType v = this->GetTargetVertex(e);

  this->Internals->RemoveEdgeFromOutList(e, this->Internals->Adjacency[u].OutEdges);
  if (directed)
  {
    this->Internals->RemoveEdgeFromInList(e, this->Internals->Adjacency[v].InEdges);
  }
  else if (u != v)
  {
    this->Internals->RemoveEdgeFromOutList(e, this->Internals->Adjacency[v].OutEdges);
  }

  // Replace last edge id with e
  svtkIdType le = this->GetNumberOfEdges() - 1;
  svtkIdType lu = this->GetSourceVertex(le);
  svtkIdType lv = this->GetTargetVertex(le);
  this->Internals->ReplaceEdgeFromOutList(le, e, this->Internals->Adjacency[lu].OutEdges);
  if (directed)
  {
    this->Internals->ReplaceEdgeFromInList(le, e, this->Internals->Adjacency[lv].InEdges);
  }
  else if (lu != lv)
  {
    this->Internals->ReplaceEdgeFromOutList(le, e, this->Internals->Adjacency[lv].OutEdges);
  }

  // Update edge list
  this->EdgeList->SetValue(2 * e + 0, lu);
  this->EdgeList->SetValue(2 * e + 1, lv);
  this->EdgeList->SetNumberOfTuples(le);

  // Update properties
  svtkDataSetAttributes* ed = this->GetEdgeData();
  for (int i = 0; i < ed->GetNumberOfArrays(); ++i)
  {
    svtkAbstractArray* arr = ed->GetAbstractArray(i);
    arr->SetTuple(e, le, arr);
    arr->SetNumberOfTuples(le);
  }

  // Update edge points
  if (this->EdgePoints)
  {
    this->EdgePoints->Storage[e] = this->EdgePoints->Storage[le];
    this->EdgePoints->Storage.pop_back();
  }

  this->Internals->NumberOfEdges--;
}

//----------------------------------------------------------------------------
void svtkGraph::RemoveVerticesInternal(svtkIdTypeArray* arr, bool directed)
{
  if (this->DistributedHelper)
  {
    svtkErrorMacro("Cannot remove vertices in a distributed graph.");
    return;
  }
  if (!arr)
  {
    return;
  }

  // Sort
  svtkIdType* p = arr->GetPointer(0);
  svtkIdType numVert = arr->GetNumberOfTuples();
  std::sort(p, p + numVert);

  // Collect all edges to be removed
  std::set<svtkIdType> edges;
  for (svtkIdType vind = 0; vind < numVert; ++vind)
  {
    svtkIdType v = p[vind];
    std::vector<svtkOutEdgeType>::iterator oi, oiEnd;
    oiEnd = this->Internals->Adjacency[v].OutEdges.end();
    for (oi = this->Internals->Adjacency[v].OutEdges.begin(); oi != oiEnd; ++oi)
    {
      edges.insert(oi->Id);
    }
    std::vector<svtkInEdgeType>::iterator ii, iiEnd;
    iiEnd = this->Internals->Adjacency[v].InEdges.end();
    for (ii = this->Internals->Adjacency[v].InEdges.begin(); ii != iiEnd; ++ii)
    {
      edges.insert(ii->Id);
    }
  }

  // Remove edges in reverse index order
  std::set<svtkIdType>::reverse_iterator ei, eiEnd;
  eiEnd = edges.rend();
  for (ei = edges.rbegin(); ei != eiEnd; ++ei)
  {
    this->RemoveEdgeInternal(*ei, directed);
  }

  // Remove vertices in reverse index order
  for (svtkIdType vind = numVert - 1; vind >= 0; --vind)
  {
    this->RemoveVertexInternal(p[vind], directed);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::RemoveEdgesInternal(svtkIdTypeArray* arr, bool directed)
{
  if (this->DistributedHelper)
  {
    svtkErrorMacro("Cannot remove edges in a distributed graph.");
    return;
  }
  if (!arr)
  {
    return;
  }

  // Sort
  svtkIdType* p = arr->GetPointer(0);
  svtkIdType numEdges = arr->GetNumberOfTuples();
  std::sort(p, p + numEdges);

  // Edges vertices in reverse index order
  for (svtkIdType eind = numEdges - 1; eind >= 0; --eind)
  {
    this->RemoveEdgeInternal(p[eind], directed);
  }
}

//----------------------------------------------------------------------------
void svtkGraph::ReorderOutVertices(svtkIdType v, svtkIdTypeArray* vertices)
{
  svtkIdType index = v;
  if (svtkDistributedGraphHelper* helper = this->GetDistributedGraphHelper())
  {
    int myRank = this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
    if (myRank != helper->GetVertexOwner(v))
    {
      svtkErrorMacro("svtkGraph cannot reorder the out vertices for a non-local vertex");
      return;
    }

    index = helper->GetVertexIndex(v);
  }

  this->ForceOwnership();
  std::vector<svtkOutEdgeType> outEdges;
  std::vector<svtkOutEdgeType>::iterator it, itEnd;
  itEnd = this->Internals->Adjacency[index].OutEdges.end();
  for (svtkIdType i = 0; i < vertices->GetNumberOfTuples(); ++i)
  {
    svtkIdType vert = vertices->GetValue(i);
    // Find the matching edge
    for (it = this->Internals->Adjacency[index].OutEdges.begin(); it != itEnd; ++it)
    {
      if (it->Target == vert)
      {
        outEdges.push_back(*it);
        break;
      }
    }
  }
  if (outEdges.size() != this->Internals->Adjacency[index].OutEdges.size())
  {
    svtkErrorMacro("Invalid reorder list.");
    return;
  }
  this->Internals->Adjacency[index].OutEdges = outEdges;
}

//----------------------------------------------------------------------------
bool svtkGraph::IsSameStructure(svtkGraph* other)
{
  return (this->Internals == other->Internals);
}

//----------------------------------------------------------------------------
svtkGraphInternals* svtkGraph::GetGraphInternals(bool modifying)
{
  if (modifying)
  {
    this->ForceOwnership();
  }
  return this->Internals;
}

//----------------------------------------------------------------------------
void svtkGraph::ForceOwnership()
{
  // If the reference count == 1, we own it and can change it.
  // If the reference count > 1, we must make a copy to avoid
  // changing the structure of other graphs.
  if (this->Internals->GetReferenceCount() > 1)
  {
    svtkGraphInternals* internals = svtkGraphInternals::New();
    internals->Adjacency = this->Internals->Adjacency;
    internals->NumberOfEdges = this->Internals->NumberOfEdges;
    this->SetInternals(internals);
    internals->Delete();
  }
  if (this->EdgePoints && this->EdgePoints->GetReferenceCount() > 1)
  {
    svtkGraphEdgePoints* oldEdgePoints = this->EdgePoints;
    svtkGraphEdgePoints* edgePoints = svtkGraphEdgePoints::New();
    edgePoints->Storage = oldEdgePoints->Storage;
    this->EdgePoints = edgePoints;
    oldEdgePoints->Delete();
  }
}

//----------------------------------------------------------------------------
svtkFieldData* svtkGraph::GetAttributesAsFieldData(int type)
{
  switch (type)
  {
    case VERTEX:
      return this->GetVertexData();
    case EDGE:
      return this->GetEdgeData();
  }
  return this->Superclass::GetAttributesAsFieldData(type);
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetNumberOfElements(int type)
{
  switch (type)
  {
    case VERTEX:
      return this->GetNumberOfVertices();
    case EDGE:
      return this->GetNumberOfEdges();
  }
  return this->Superclass::GetNumberOfElements(type);
}

//----------------------------------------------------------------------------
void svtkGraph::Dump()
{
  cout << "vertex adjacency:" << endl;
  for (size_t v = 0; v < this->Internals->Adjacency.size(); ++v)
  {
    cout << v << " (out): ";
    for (size_t eind = 0; eind < this->Internals->Adjacency[v].OutEdges.size(); ++eind)
    {
      cout << "[" << this->Internals->Adjacency[v].OutEdges[eind].Id << ","
           << this->Internals->Adjacency[v].OutEdges[eind].Target << "]";
    }
    cout << " (in): ";
    for (size_t eind = 0; eind < this->Internals->Adjacency[v].InEdges.size(); ++eind)
    {
      cout << "[" << this->Internals->Adjacency[v].InEdges[eind].Id << ","
           << this->Internals->Adjacency[v].InEdges[eind].Source << "]";
    }
    cout << endl;
  }
  if (this->EdgeList)
  {
    cout << "edge list:" << endl;
    for (svtkIdType e = 0; e < this->EdgeList->GetNumberOfTuples(); ++e)
    {
      cout << e << ": (" << this->EdgeList->GetValue(2 * e + 0) << ","
           << this->EdgeList->GetValue(2 * e + 1) << ")" << endl;
    }
    cout << endl;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkGraph::GetEdgeId(svtkIdType a, svtkIdType b)
{
  // Check if there is an edge from b to a
  svtkSmartPointer<svtkInEdgeIterator> inEdgeIterator = svtkSmartPointer<svtkInEdgeIterator>::New();
  this->GetInEdges(a, inEdgeIterator);

  while (inEdgeIterator->HasNext())
  {
    svtkInEdgeType edge = inEdgeIterator->Next();
    if (edge.Source == b)
    {
      return edge.Id;
    }
  }

  // Check if there is an edge from a to b
  svtkSmartPointer<svtkOutEdgeIterator> outEdgeIterator = svtkSmartPointer<svtkOutEdgeIterator>::New();
  this->GetOutEdges(a, outEdgeIterator);

  while (outEdgeIterator->HasNext())
  {
    svtkOutEdgeType edge = outEdgeIterator->Next();
    if (edge.Target == b)
    {
      return edge.Id;
    }
  }

  return -1;
}

//----------------------------------------------------------------------------
bool svtkGraph::ToDirectedGraph(svtkDirectedGraph* g)
{
  // This function will convert a svtkUndirectedGraph to a
  // svtkDirectedGraph. It copies all of the data associated
  // with the graph by calling CopyInternal. Only one directed
  // edge is added for each input undirected edge.

  if (this->IsA("svtkDirectedGraph"))
  {
    // Return the status of CheckedShallowCopy
    return g->CheckedShallowCopy(this);
  }
  else if (this->IsA("svtkUndirectedGraph"))
  {
    svtkSmartPointer<svtkMutableDirectedGraph> m = svtkSmartPointer<svtkMutableDirectedGraph>::New();
    for (svtkIdType i = 0; i < this->GetNumberOfVertices(); i++)
    {
      m->AddVertex();
    }

    // Need to add edges in the same order by index.
    // svtkEdgeListIterator does not guarantee this, so we cannot use it.
    for (svtkIdType i = 0; i < this->GetNumberOfEdges(); i++)
    {
      m->AddEdge(this->GetSourceVertex(i), this->GetTargetVertex(i));
    }

    if (g->IsStructureValid(m))
    {
      // Force full copy from this, internals will be invalid
      g->CopyInternal(this, false);

      // Make internals valid
      g->SetInternals(m->Internals);
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }
}

//----------------------------------------------------------------------------
bool svtkGraph::ToUndirectedGraph(svtkUndirectedGraph* g)
{
  // This function will convert a svtkDirectedGraph to a
  // svtkUndirectedGraph. It copies all of the data associated
  // with the graph by calling CopyInternal

  if (this->IsA("svtkUndirectedGraph"))
  {
    // A normal CheckedShallowCopy will work fine.
    return g->CheckedShallowCopy(this);
  }
  else if (this->IsA("svtkDirectedGraph"))
  {
    svtkSmartPointer<svtkMutableUndirectedGraph> m =
      svtkSmartPointer<svtkMutableUndirectedGraph>::New();
    for (svtkIdType i = 0; i < this->GetNumberOfVertices(); i++)
    {
      m->AddVertex();
    }

    // Need to add edges in the same order by index.
    // svtkEdgeListIterator does not guarantee this, so we cannot use it.
    for (svtkIdType i = 0; i < this->GetNumberOfEdges(); i++)
    {
      m->AddEdge(this->GetSourceVertex(i), this->GetTargetVertex(i));
    }

    if (g->IsStructureValid(m))
    {
      // Force full copy from this, internals will be invalid
      g->CopyInternal(this, false);

      // Make internals valid
      g->SetInternals(m->Internals);

      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }
}

//----------------------------------------------------------------------------
void svtkGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "VertexData: " << (this->VertexData ? "" : "(none)") << endl;
  if (this->VertexData)
  {
    this->VertexData->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "EdgeData: " << (this->EdgeData ? "" : "(none)") << endl;
  if (this->EdgeData)
  {
    this->EdgeData->PrintSelf(os, indent.GetNextIndent());
  }
  if (this->Internals)
  {
    os << indent << "DistributedHelper: " << (this->DistributedHelper ? "" : "(none)") << endl;
    if (this->DistributedHelper)
    {
      this->DistributedHelper->PrintSelf(os, indent.GetNextIndent());
    }
  }
}

//----------------------------------------------------------------------------
// Supporting operators
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
bool operator==(svtkEdgeBase e1, svtkEdgeBase e2)
{
  return e1.Id == e2.Id;
}

//----------------------------------------------------------------------------
bool operator!=(svtkEdgeBase e1, svtkEdgeBase e2)
{
  return e1.Id != e2.Id;
}

//----------------------------------------------------------------------------
ostream& operator<<(ostream& out, svtkEdgeBase e)
{
  return out << e.Id;
}
