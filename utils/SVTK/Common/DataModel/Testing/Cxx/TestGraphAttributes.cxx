/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestGraphAttributes.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*----------------------------------------------------------------------------
 Copyright (c) Sandia Corporation
 See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.
----------------------------------------------------------------------------*/

#include "svtkAdjacentVertexIterator.h"
#include "svtkDataSetAttributes.h"
#include "svtkDirectedAcyclicGraph.h"
#include "svtkEdgeListIterator.h"
#include "svtkInEdgeIterator.h"
#include "svtkIntArray.h"
#include "svtkMutableDirectedGraph.h"
#include "svtkMutableUndirectedGraph.h"
#include "svtkOutEdgeIterator.h"
#include "svtkSmartPointer.h"
#include "svtkStringArray.h"
#include "svtkTree.h"
#include "svtkVariantArray.h"
#include "svtkVertexListIterator.h"

#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

void TestGraphAttribIterators(svtkGraph* g, int& errors)
{
  if (g->GetNumberOfVertices() != 10)
  {
    cerr << "ERROR: Wrong number of vertices." << endl;
    ++errors;
  }
  if (g->GetNumberOfEdges() != 9)
  {
    cerr << "ERROR: Wrong number of edges." << endl;
    ++errors;
  }
  SVTK_CREATE(svtkVertexListIterator, vertices);
  g->GetVertices(vertices);
  svtkIdType numVertices = 0;
  while (vertices->HasNext())
  {
    vertices->Next();
    ++numVertices;
  }
  if (numVertices != 10)
  {
    cerr << "ERROR: Vertex list iterator failed." << endl;
    ++errors;
  }
  SVTK_CREATE(svtkEdgeListIterator, edges);
  g->GetEdges(edges);
  svtkIdType numEdges = 0;
  while (edges->HasNext())
  {
    edges->Next();
    ++numEdges;
  }
  if (numEdges != 9)
  {
    cerr << "ERROR: Edge list iterator failed." << endl;
    ++errors;
  }
  numEdges = 0;
  SVTK_CREATE(svtkOutEdgeIterator, outEdges);
  g->GetVertices(vertices);
  while (vertices->HasNext())
  {
    svtkIdType v = vertices->Next();
    g->GetOutEdges(v, outEdges);
    while (outEdges->HasNext())
    {
      svtkOutEdgeType e = outEdges->Next();
      ++numEdges;
      // Count self-loops twice, to ensure all edges are counted twice.
      if (svtkUndirectedGraph::SafeDownCast(g) && v == e.Target)
      {
        ++numEdges;
      }
    }
  }
  if (svtkDirectedGraph::SafeDownCast(g) && numEdges != 9)
  {
    cerr << "ERROR: Out edge iterator failed." << endl;
    ++errors;
  }
  if (svtkUndirectedGraph::SafeDownCast(g) && numEdges != 18)
  {
    cerr << "ERROR: Undirected out edge iterator failed." << endl;
    ++errors;
  }
  numEdges = 0;
  SVTK_CREATE(svtkInEdgeIterator, inEdges);
  g->GetVertices(vertices);
  while (vertices->HasNext())
  {
    svtkIdType v = vertices->Next();
    g->GetInEdges(v, inEdges);
    while (inEdges->HasNext())
    {
      svtkInEdgeType e = inEdges->Next();
      ++numEdges;
      // Count self-loops twice, to ensure all edges are counted twice.
      if (svtkUndirectedGraph::SafeDownCast(g) && v == e.Source)
      {
        ++numEdges;
      }
    }
  }
  if (svtkDirectedGraph::SafeDownCast(g) && numEdges != 9)
  {
    cerr << "ERROR: In edge iterator failed." << endl;
    ++errors;
  }
  if (svtkUndirectedGraph::SafeDownCast(g) && numEdges != 18)
  {
    cerr << "ERROR: Undirected in edge iterator failed." << endl;
    ++errors;
  }
  numEdges = 0;
  SVTK_CREATE(svtkAdjacentVertexIterator, adjacent);
  g->GetVertices(vertices);
  while (vertices->HasNext())
  {
    svtkIdType v = vertices->Next();
    g->GetAdjacentVertices(v, adjacent);
    while (adjacent->HasNext())
    {
      svtkIdType u = adjacent->Next();
      ++numEdges;
      // Count self-loops twice, to ensure all edges are counted twice.
      if (svtkUndirectedGraph::SafeDownCast(g) && v == u)
      {
        ++numEdges;
      }
    }
  }
  if (svtkDirectedGraph::SafeDownCast(g) && numEdges != 9)
  {
    cerr << "ERROR: In edge iterator failed." << endl;
    ++errors;
  }
  if (svtkUndirectedGraph::SafeDownCast(g) && numEdges != 18)
  {
    cerr << "ERROR: Undirected in edge iterator failed." << endl;
    ++errors;
  }
}

int TestGraphAttributes(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  int errors = 0;

  SVTK_CREATE(svtkMutableDirectedGraph, mdgTree);

  SVTK_CREATE(svtkDirectedGraph, dg);
  SVTK_CREATE(svtkTree, t);

  //  Create some vertex property arrays
  SVTK_CREATE(svtkVariantArray, vertexPropertyArr);
  int numVertexProperties = 2;
  vertexPropertyArr->SetNumberOfValues(numVertexProperties);

  SVTK_CREATE(svtkStringArray, vertexProp0Array);
  vertexProp0Array->SetName("labels");
  mdgTree->GetVertexData()->AddArray(vertexProp0Array);

  SVTK_CREATE(svtkIntArray, vertexProp1Array);
  vertexProp1Array->SetName("weight");
  mdgTree->GetVertexData()->AddArray(vertexProp1Array);

  const char* vertexLabel[] = { "Dick", "Jane", "Sally", "Spot", "Puff" };

  const char* stringProp;
  int weight;

  for (svtkIdType i = 0; i < 10; ++i)
  {
    stringProp = vertexLabel[rand() % 5];
    weight = rand() % 10;
    //    cout << myRank <<" vertex "<< v <<","<< stringProp <<","<<weight<< endl;
    vertexPropertyArr->SetValue(0, stringProp);
    vertexPropertyArr->SetValue(1, weight);
    mdgTree->AddVertex(vertexPropertyArr);
  }

  // Create a valid tree.
  mdgTree->AddEdge(0, 1);
  mdgTree->AddEdge(0, 2);
  mdgTree->AddEdge(0, 3);
  mdgTree->AddEdge(1, 4);
  mdgTree->AddEdge(1, 5);
  mdgTree->AddEdge(2, 6);
  mdgTree->AddEdge(2, 7);
  mdgTree->AddEdge(3, 8);
  mdgTree->AddEdge(3, 9);

  cerr << "Testing graph conversions ..." << endl;
  if (!t->CheckedShallowCopy(mdgTree))
  {
    cerr << "ERROR: Cannot set valid tree." << endl;
    ++errors;
  }

  if (!dg->CheckedShallowCopy(mdgTree))
  {
    cerr << "ERROR: Cannot set valid directed graph." << endl;
    ++errors;
  }
  if (!dg->CheckedShallowCopy(t))
  {
    cerr << "ERROR: Cannot set tree to directed graph." << endl;
    ++errors;
  }

  cerr << "... done." << endl;

  cerr << "Testing basic graph structure ..." << endl;
  TestGraphAttribIterators(mdgTree, errors);
  TestGraphAttribIterators(dg, errors);
  TestGraphAttribIterators(t, errors);
  cerr << "... done." << endl;

  cerr << "Testing copy on write ..." << endl;
  if (!t->IsSameStructure(mdgTree))
  {
    cerr << "ERROR: Tree and directed graph should be sharing the same structure." << endl;
    ++errors;
  }
  mdgTree->AddVertex();
  if (t->IsSameStructure(mdgTree))
  {
    cerr << "ERROR: Tree and directed graph should not be sharing the same structure." << endl;
    ++errors;
  }
  if (t->GetNumberOfVertices() != 10)
  {
    cerr << "ERROR: Tree changed when modifying directed graph." << endl;
    ++errors;
  }
  cerr << "... done." << endl;

  return errors;
}
