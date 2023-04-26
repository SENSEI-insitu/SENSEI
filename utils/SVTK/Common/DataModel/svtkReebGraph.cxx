/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkReebGraph.cxx

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

#include "svtkReebGraph.h"

#include "svtkCell.h"
#include "svtkDataArray.h"
#include "svtkEdgeListIterator.h"
#include "svtkIdTypeArray.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkReebGraphSimplificationMetric.h"
#include "svtkUnstructuredGrid.h"
#include "svtkVariantArray.h"

#include <algorithm>
#include <map>
#include <queue>
#include <vector>

//----------------------------------------------------------------------------
// Contain all of the internal data structures, and macros, in the
// implementation.
namespace
{
//----------------------------------------------------------------------------
inline static bool svtkReebGraphVertexSoS(
  const std::pair<int, double>& v0, const std::pair<int, double>& v1)
{
  return ((v0.second < v1.second) || ((v0.second == v1.second) && (v0.first < v1.first)));
}
}

// INTERNAL MACROS ---------------------------------------------------------
#define svtkReebGraphSwapVars(type, var1, var2)                                                     \
  {                                                                                                \
    type tmp;                                                                                      \
    tmp = (var1);                                                                                  \
    (var1) = (var2);                                                                               \
    (var2) = tmp;                                                                                  \
  }

#define svtkReebGraphInitialStreamSize 1000

#define svtkReebGraphIsSmaller(myReebGraph, nodeId0, nodeId1, node0, node1)                         \
  ((node0->Value < node1->Value) || (node0->Value == node1->Value && (nodeId0) < (nodeId1)))

#define svtkReebGraphGetArcPersistence(rg, a)                                                       \
  ((this->GetNode(a->NodeId1)->Value - this->GetNode(a->NodeId0)->Value) /                         \
    (this->MaximumScalarValue - this->MinimumScalarValue))

#define svtkReebGraphIsHigherThan(rg, N0, N1, n0, n1)                                               \
  ((n0->Value > n1->Value) || (n0->Value == n1->Value && n0->VertexId > n1->VertexId))

// Note: usually this macro is called after the node has been finalized,
// otherwise the behaviour is undefined.
#define svtkReebGraphIsRegular(rg, n)                                                               \
  ((!(n)->IsCritical) &&                                                                           \
    ((n)->ArcDownId && !this->GetArc((n)->ArcDownId)->ArcDwId1 && (n)->ArcUpId &&                  \
      !this->GetArc((n)->ArcUpId)->ArcDwId0))

#define svtkReebGraphAddUpArc(rg, N, A)                                                             \
  {                                                                                                \
    svtkReebNode* n = this->GetNode(N);                                                             \
    svtkReebArc* a = this->GetArc(A);                                                               \
    a->ArcUpId0 = 0;                                                                               \
    a->ArcDwId0 = n->ArcUpId;                                                                      \
    if (n->ArcUpId)                                                                                \
      this->GetArc(n->ArcUpId)->ArcUpId0 = (A);                                                    \
    n->ArcUpId = (A);                                                                              \
  }

#define svtkReebGraphAddDownArc(rg, N, A)                                                           \
  {                                                                                                \
    svtkReebNode* n = this->GetNode(N);                                                             \
    svtkReebArc* a = this->GetArc(A);                                                               \
    a->ArcUpId1 = 0;                                                                               \
    a->ArcDwId1 = n->ArcDownId;                                                                    \
    if (n->ArcDownId)                                                                              \
      this->GetArc(n->ArcDownId)->ArcUpId1 = (A);                                                  \
    n->ArcDownId = (A);                                                                            \
  }

#define svtkReebGraphRemoveUpArc(rg, N, A)                                                          \
  {                                                                                                \
    svtkReebNode* n = this->GetNode(N);                                                             \
    svtkReebArc* a = this->GetArc(A);                                                               \
    if (a->ArcUpId0)                                                                               \
      this->GetArc(a->ArcUpId0)->ArcDwId0 = a->ArcDwId0;                                           \
    else                                                                                           \
      n->ArcUpId = a->ArcDwId0;                                                                    \
    if (a->ArcDwId0)                                                                               \
      this->GetArc(a->ArcDwId0)->ArcUpId0 = a->ArcUpId0;                                           \
  }

#define svtkReebGraphRemoveDownArc(rg, N, A)                                                        \
  {                                                                                                \
    svtkReebNode* n = this->GetNode(N);                                                             \
    svtkReebArc* a = this->GetArc(A);                                                               \
    if (a->ArcUpId1)                                                                               \
      this->GetArc(a->ArcUpId1)->ArcDwId1 = a->ArcDwId1;                                           \
    else                                                                                           \
      n->ArcDownId = a->ArcDwId1;                                                                  \
    if (a->ArcDwId1)                                                                               \
      this->GetArc(a->ArcDwId1)->ArcUpId1 = a->ArcUpId1;                                           \
  }

#ifndef svtkReebGraphMax
#define svtkReebGraphMax(a, b) (((a) >= (b)) ? (a) : (b))
#endif

#define svtkReebGraphStackPush(N)                                                                   \
  {                                                                                                \
    if (nstack == mstack)                                                                          \
    {                                                                                              \
      mstack = svtkReebGraphMax(128, mstack * 2);                                                   \
      int* oldstack = stack;                                                                       \
      stack = (int*)realloc(stack, sizeof(int) * mstack);                                          \
      if (!stack)                                                                                  \
      {                                                                                            \
        free(oldstack);                                                                            \
        assert(0 && "Ran out of memory");                                                          \
      }                                                                                            \
    }                                                                                              \
    stack[nstack++] = (N);                                                                         \
  }

#define svtkReebGraphStackSize() (nstack)

#define svtkReebGraphStackTop() (stack[nstack - 1])

#define svtkReebGraphStackPop() (--nstack)

//----------------------------------------------------------------------------
// PIMPLed classes...
class svtkReebGraph::Implementation
{
public:
  Implementation()
  {
    this->historyOn = false;

    this->MainNodeTable.Buffer = (svtkReebNode*)malloc(sizeof(svtkReebNode) * 2);
    this->MainArcTable.Buffer = (svtkReebArc*)malloc(sizeof(svtkReebArc) * 2);
    this->MainLabelTable.Buffer = (svtkReebLabel*)malloc(sizeof(svtkReebLabel) * 2);

    this->MainNodeTable.Size = 2;
    this->MainNodeTable.Number = 1; // the item "0" is blocked
    this->MainArcTable.Size = 2;
    this->MainArcTable.Number = 1;
    this->MainLabelTable.Size = 2;
    this->MainLabelTable.Number = 1;

    this->MainNodeTable.FreeZone = 1;
    // Clear node
    this->GetNode(1)->ArcUpId = ((int)-2);
    // Initialize DownArc
    this->GetNode(1)->ArcDownId = 0;
    this->MainArcTable.FreeZone = 1;
    // Clear Arc label 1
    this->GetArc(1)->LabelId1 = ((int)-2);
    // Initialize Arc label 0
    this->GetArc(1)->LabelId0 = 0;
    this->MainLabelTable.FreeZone = 1;
    // Clear label
    this->GetLabel(1)->HNext = ((int)-2);
    // Initialize Arc id
    this->GetLabel(1)->ArcId = 0;

    this->MinimumScalarValue = 0;
    this->MaximumScalarValue = 0;

    this->ArcNumber = 0;
    this->NodeNumber = 0;
    this->LoopNumber = 0;
    this->RemovedLoopNumber = 0;
    this->ArcLoopTable = nullptr;

    this->currentNodeId = 0;
    this->currentArcId = 0;

    // streaming support
    this->VertexMapSize = 0;
    this->VertexMapAllocatedSize = 0;
    this->TriangleVertexMapSize = 0;
    this->TriangleVertexMapAllocatedSize = 0;
  }
  ~Implementation()
  {
    free(this->MainNodeTable.Buffer);
    this->MainNodeTable.Buffer = nullptr;

    free(this->MainArcTable.Buffer);
    this->MainArcTable.Buffer = nullptr;

    free(this->MainLabelTable.Buffer);
    this->MainLabelTable.Buffer = nullptr;

    this->MainNodeTable.Size = this->MainNodeTable.Number = 0;
    this->MainArcTable.Size = this->MainArcTable.Number = 0;
    this->MainLabelTable.Size = this->MainLabelTable.Number = 0;

    this->MainNodeTable.FreeZone = 0;
    this->MainArcTable.FreeZone = 0;
    this->MainLabelTable.FreeZone = 0;

    if (this->ArcLoopTable)
      free(this->ArcLoopTable);

    if (this->VertexMapAllocatedSize)
      free(this->VertexMap);

    if (this->TriangleVertexMapAllocatedSize)
      free(this->TriangleVertexMap);
  }

  typedef unsigned long long svtkReebLabelTag;

  typedef struct _svtkReebCancellation
  {
    std::vector<std::pair<int, int> > removedArcs;
    std::vector<std::pair<int, int> > insertedArcs;
  } svtkReebCancellation;

  // Node structure
  typedef struct
  {
    svtkIdType VertexId;
    double Value;
    svtkIdType ArcDownId;
    svtkIdType ArcUpId;
    bool IsFinalized;
    bool IsCritical;
  } svtkReebNode;

  // Arc structure
  typedef struct
  {
    svtkIdType NodeId0, ArcUpId0, ArcDwId0;
    svtkIdType NodeId1, ArcUpId1, ArcDwId1;
    svtkIdType LabelId0, LabelId1;
  } svtkReebArc;

  // Label structure
  typedef struct
  {
    svtkIdType ArcId;
    svtkIdType HPrev, HNext; // "horizontal" (for a single arc)
    svtkReebLabelTag label;
    svtkIdType VPrev, VNext; // "vertical" (for a sequence of arcs)
  } svtkReebLabel;

  struct svtkReebPath
  {
    double SimplificationValue;
    int ArcNumber;
    svtkIdType* ArcTable;
    int NodeNumber;
    svtkIdType* NodeTable;

    inline bool operator<(struct svtkReebPath const& E) const
    {
      return !((SimplificationValue < E.SimplificationValue) ||
        (SimplificationValue == E.SimplificationValue && ArcNumber < E.ArcNumber) ||
        (SimplificationValue == E.SimplificationValue && ArcNumber == E.ArcNumber &&
          NodeTable[NodeNumber - 1] < E.NodeTable[E.NodeNumber - 1]));
      /*      return !((
              (MaximumScalarValue - MinimumScalarValue)
                < (E.MaximumScalarValue - E.MinimumScalarValue)) ||
                 ((MaximumScalarValue - MinimumScalarValue)
                   == (E.MaximumScalarValue-E.MinimumScalarValue)
                     && ArcNumber < E.ArcNumber) ||
                 ((MaximumScalarValue - MinimumScalarValue)
                   == (E.MaximumScalarValue - E.MinimumScalarValue)
                     && ArcNumber == E.ArcNumber
                       && NodeTable[NodeNumber - 1]<E.NodeTable[E.NodeNumber - 1])
               );*/
    }
  };

  struct
  {
    int Size, Number, FreeZone;
    svtkReebArc* Buffer;
  } MainArcTable;

  struct
  {
    int Size, Number, FreeZone;
    svtkReebNode* Buffer;
  } MainNodeTable;

  struct
  {
    int Size, Number, FreeZone;
    svtkReebLabel* Buffer;
  } MainLabelTable;

  svtkReebPath FindPath(
    svtkIdType arcId, double simplificationThreshold, svtkReebGraphSimplificationMetric* metric);

  // INTERNAL METHODS --------------------------------------------------------

  svtkIdType AddArc(svtkIdType nodeId0, svtkIdType nodeId1);

  // Description:
  // Get the node specified from the graph.
  svtkReebNode* GetNode(svtkIdType nodeId);

  // Description:
  // Get the arc specified from the graph.
  svtkReebArc* GetArc(svtkIdType arcId);

  // Description:
  // Get the Label specified from the graph.
  svtkReebLabel* GetLabel(svtkIdType labelId);

  // Description:
  // Collapse (consolidate) two nodes in the graph.
  void CollapseVertex(svtkIdType N, svtkReebNode* n);

  // Description:
  // Triggers customized code for simplification metric evaluation.
  double ComputeCustomMetric(svtkReebGraphSimplificationMetric* simplificationMetric, svtkReebArc* a);

  // Description:
  //  Add a monotonic path between nodes.
  svtkIdType AddPath(int nodeNumber, svtkIdType* nodeOffset, svtkReebLabelTag label);

  // Description:
  //   Add a vertex from the mesh to the Reeb graph.
  svtkIdType AddMeshVertex(svtkIdType vertexId, double scalar);

  // Description:
  //   Add a triangle from the mesh to the Reeb grpah.
  int AddMeshTriangle(
    svtkIdType vertex0Id, double f0, svtkIdType vertex1Id, double f1, svtkIdType vertex2Id, double f2);

  // Description:
  //   Add a tetrahedron from the mesh to the Reeb grpah.
  int AddMeshTetrahedron(svtkIdType vertex0Id, double f0, svtkIdType vertex1Id, double f1,
    svtkIdType vertex2Id, double f2, svtkIdType vertex3Id, double f3);

  // Description:
  // "Zip" the corresponding paths when the interior of a simplex is added to
  // the Reeb graph.
  void Collapse(svtkIdType startingNode, svtkIdType endingNode, svtkReebLabelTag startingLabel,
    svtkReebLabelTag endingLabel);

  // Description:
  // Finalize a vertex.
  void EndVertex(const svtkIdType N);

  // Description:
  // Remove an arc during filtering by persistence.
  void FastArcSimplify(svtkIdType arcId, int arcNumber, svtkIdType* arcTable);

  // Description:
  // Remove arcs below the provided persistence.
  int SimplifyBranches(
    double simplificationThreshold, svtkReebGraphSimplificationMetric* simplificationMetric);

  // Description:
  // Remove the loops below the provided persistence.
  int SimplifyLoops(
    double simplificationThreshold, svtkReebGraphSimplificationMetric* simplificationMetric);

  // Description:
  // Update the svtkMutableDirectedGraph internal structure after filtering, with
  // deg-2 nodes maintaining.
  int CommitSimplification();

  // Description:
  // Retrieve downwards labels.
  svtkIdType FindDwLabel(svtkIdType nodeId, svtkReebLabelTag label);

  // Description
  // Find greater arc (persistence-based simplification).
  svtkIdType FindGreater(svtkIdType nodeId, svtkIdType startingNodeId, svtkReebLabelTag label);

  // Description:
  // Find corresponding joining saddle node (persistence-based simplification).
  svtkIdType FindJoinNode(svtkIdType arcId, svtkReebLabelTag label, bool onePathOnly = false);

  // Description:
  // Find smaller arc (persistence-based simplification).
  svtkIdType FindLess(svtkIdType nodeId, svtkIdType startingNodeId, svtkReebLabelTag label);

  // Description:
  // Compute the loops in the Reeb graph.
  void FindLoops();

  // Description:
  // Find corresponding splitting saddle node (persistence-based
  // simplification).
  svtkIdType FindSplitNode(svtkIdType arcId, svtkReebLabelTag label, bool onePathOnly = false);

  // Description:
  // Retrieve upwards labels.
  svtkIdType FindUpLabel(svtkIdType nodeId, svtkReebLabelTag label);

  // Description:
  // Flush labels.
  void FlushLabels();

  // Description:
  // Resize the arc table.
  void ResizeMainArcTable(int newSize);

  // Description:
  // Resize the label table.
  void ResizeMainLabelTable(int newSize);

  // Description:
  // Resize the node table.
  void ResizeMainNodeTable(int newSize);

  // Description:
  // Set a label.
  void SetLabel(svtkIdType A, svtkReebLabelTag Label);

  // Description:
  // Simplify labels.
  void SimplifyLabels(
    const svtkIdType nodeId, svtkReebLabelTag onlyLabel = 0, bool goDown = true, bool goUp = true);

  // ACCESSORS

  // Description:
  // Returns the Id of the lower node of the arc specified by 'arcId'.
  svtkIdType GetArcDownNodeId(svtkIdType arcId);

  // Description:
  // Return the Id of the upper node of the arc specified by 'arcId'.
  svtkIdType GetArcUpNodeId(svtkIdType arcId);

  // Description:
  // Iterates forwards through the arcs of the Reeb graph.
  //
  // The first time this method is called, the first arc's Id will be returned.
  // When the last arc is reached, this method will keep on returning its Id at
  // each call. See 'GetPreviousArcId' to go back in the list.
  svtkIdType GetNextArcId();

  // Description:
  // Iterates forwards through the nodes of the Reeb graph.
  //
  // The first time this method is called, the first node's Id will be returned.
  // When the last node is reached, this method will keep on returning its Id at
  // each call. See 'GetPreviousNodeId' to go back in the list.
  svtkIdType GetNextNodeId();

  // Description:
  // Copy into 'arcIdList' the list of the down arcs' Ids, given a node
  // specified by 'nodeId'.
  void GetNodeDownArcIds(svtkIdType nodeId, svtkIdList* arcIdList);

  // Description:
  // Returns the scalar field value of the node specified by 'nodeId'.
  double GetNodeScalarValue(svtkIdType nodeId);

  // Description:
  // Copy into 'arcIdList' the list of the up arcs' Ids, given a node specified
  // by 'nodeId'.
  void GetNodeUpArcIds(svtkIdType nodeId, svtkIdList* arcIdList);

  // Description:
  // Returns the corresponding vertex Id (in the simplicial mesh, svtkPolyData),
  // given a node specified by 'nodeId'.
  svtkIdType GetNodeVertexId(svtkIdType nodeId);

  // Description:
  // Returns the number of arcs in the Reeb graph.
  int GetNumberOfArcs();

  // Description:
  // Returns the number of connected components of the Reeb graph.
  int GetNumberOfConnectedComponents();

  // Description:
  // Returns the number of nodes in the Reeb graph.
  int GetNumberOfNodes();

  // Description:
  // Returns the number of loops (cycles) in the Reeb graph.
  //
  // Notice that for closed PL 2-manifolds, this number equals the genus of the
  // manifold.
  //
  // Reference:
  // "Loops in Reeb graphs of 2-manifolds",
  // K. Cole-McLaughlin, H. Edelsbrunner, J. Harer, V. Natarajan, and V.
  // Pascucci,
  // ACM Symposium on Computational Geometry, pp. 344-350, 2003.
  int GetNumberOfLoops();

  // Description:
  // Iterates backwards through the arcs of the Reeb graph.
  //
  // When the first arc is reached, this method will keep on returning its Id at
  // each call. See 'GetNextArcId' to go forwards in the list.
  svtkIdType GetPreviousArcId();

  // Description:
  // Iterates backwards through the nodes of the Reeb graph.
  //
  // When the first node is reached, this method will keep on returning its Id
  // at each call. See 'GetNextNodeId' to go forwards in the list.
  svtkIdType GetPreviousNodeId();

  // Description:
  // Implementations of the stream classes to operate on the streams...
  int StreamTetrahedron(svtkIdType vertex0Id, double scalar0, svtkIdType vertex1Id, double scalar1,
    svtkIdType vertex2Id, double scalar2, svtkIdType vertex3Id, double scalar3);
  int StreamTriangle(svtkIdType vertex0Id, double scalar0, svtkIdType vertex1Id, double scalar1,
    svtkIdType vertex2Id, double scalar2);

  void DeepCopy(Implementation* src);

  // Description:
  // Data storage.
  std::map<int, int> VertexStream;
  std::vector<svtkReebCancellation> cancellationHistory;
  // Streaming support
  int VertexMapSize, VertexMapAllocatedSize, TriangleVertexMapSize, TriangleVertexMapAllocatedSize;
  bool historyOn;

  svtkIdType* VertexMap;
  int* TriangleVertexMap;

  double MinimumScalarValue, MaximumScalarValue;

  // Arcs and nodes
  int ArcNumber, NodeNumber;

  // Loops
  int LoopNumber, RemovedLoopNumber;
  svtkIdType* ArcLoopTable;

  // CC
  int ConnectedComponentNumber;

  std::map<int, double> ScalarField;

  svtkIdType currentNodeId, currentArcId;

  svtkDataSet* inputMesh;
  svtkDataArray* inputScalarField;

  svtkReebGraph* Parent;
};

//----------------------------------------------------------------------------
svtkReebGraph::Implementation::svtkReebNode* svtkReebGraph::Implementation::GetNode(svtkIdType nodeId)
{
  return (this->MainNodeTable.Buffer + nodeId);
}

//----------------------------------------------------------------------------
svtkReebGraph::Implementation::svtkReebArc* svtkReebGraph::Implementation::GetArc(svtkIdType arcId)
{
  return (this->MainArcTable.Buffer + arcId);
}

//----------------------------------------------------------------------------
svtkReebGraph::Implementation::svtkReebLabel* svtkReebGraph::Implementation::GetLabel(
  svtkIdType labelId)
{
  return (this->MainLabelTable.Buffer + labelId);
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::CollapseVertex(svtkIdType N, svtkReebNode* n)
{
  int Lb, Lnext, La;
  svtkReebLabel* lb;

  int _A0 = n->ArcDownId;
  int _A1 = n->ArcUpId;

  svtkReebArc* _a0 = this->GetArc(_A0);
  svtkReebArc* _a1 = this->GetArc(_A1);

  _a0->NodeId1 = _a1->NodeId1;
  _a0->ArcUpId1 = _a1->ArcUpId1;

  if (_a1->ArcUpId1)
    this->GetArc(_a1->ArcUpId1)->ArcDwId1 = _A0;

  _a0->ArcDwId1 = _a1->ArcDwId1;

  if (_a1->ArcDwId1)
    this->GetArc(_a1->ArcDwId1)->ArcUpId1 = _A0;

  if (this->GetNode(_a1->NodeId1)->ArcDownId == _A1)
    this->GetNode(_a1->NodeId1)->ArcDownId = _A0;

  for (Lb = _a1->LabelId0; Lb; Lb = Lnext)
  {
    lb = this->GetLabel(Lb);
    Lnext = lb->HNext;

    if (lb->VPrev)
    {
      La = lb->VPrev;
      this->GetLabel(La)->VNext = lb->VNext;
    }

    if (lb->VNext)
      this->GetLabel(lb->VNext)->VPrev = lb->VPrev;

    // delete the label...
    this->GetLabel(Lb)->HNext = ((int)-2);
    this->GetLabel(Lb)->ArcId = this->MainLabelTable.FreeZone;
    this->MainLabelTable.FreeZone = (Lb);
    --(this->MainLabelTable.Number);
  }

  // delete the arc from the graph...
  this->GetArc(_A1)->LabelId1 = ((int)-2);
  this->GetArc(_A1)->LabelId0 = this->MainArcTable.FreeZone;
  this->MainArcTable.FreeZone = (_A1);
  --(this->MainArcTable.Number);

  // delete the node from the graph...
  this->GetNode(N)->ArcUpId = ((int)-2);
  this->GetNode(N)->ArcDownId = this->MainNodeTable.FreeZone;
  this->MainNodeTable.FreeZone = (N);
  --(this->MainNodeTable.Number);
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::DeepCopy(Implementation* srcG)
{
  MinimumScalarValue = srcG->MinimumScalarValue;
  MaximumScalarValue = srcG->MaximumScalarValue;

  inputMesh = srcG->inputMesh;
  inputScalarField = srcG->inputScalarField;

  ArcNumber = srcG->ArcNumber;
  NodeNumber = srcG->NodeNumber;
  LoopNumber = srcG->LoopNumber;

  ScalarField = srcG->ScalarField;

  VertexStream = srcG->VertexStream;

  if (srcG->MainArcTable.Buffer)
  {
    MainArcTable.Size = srcG->MainArcTable.Size;
    MainArcTable.Number = srcG->MainArcTable.Number;
    free(this->MainArcTable.Buffer);
    this->MainArcTable.Buffer = (svtkReebArc*)malloc(sizeof(svtkReebArc) * srcG->MainArcTable.Size);

    memcpy(this->MainArcTable.Buffer, srcG->MainArcTable.Buffer,
      sizeof(svtkReebArc) * srcG->MainArcTable.Size);
  }

  if (srcG->MainNodeTable.Buffer)
  {
    MainNodeTable.Size = srcG->MainNodeTable.Size;
    MainNodeTable.Number = srcG->MainNodeTable.Number;
    // free existing buffer
    free(this->MainNodeTable.Buffer);
    this->MainNodeTable.Buffer =
      (svtkReebNode*)malloc(sizeof(svtkReebNode) * srcG->MainNodeTable.Size);

    memcpy(this->MainNodeTable.Buffer, srcG->MainNodeTable.Buffer,
      sizeof(svtkReebNode) * srcG->MainNodeTable.Size);
  }

  if (srcG->MainLabelTable.Buffer)
  {
    MainLabelTable.Size = srcG->MainLabelTable.Size;
    MainLabelTable.Number = srcG->MainLabelTable.Number;
    free(this->MainLabelTable.Buffer);

    this->MainLabelTable.Buffer =
      (svtkReebLabel*)malloc(sizeof(svtkReebLabel) * srcG->MainLabelTable.Size);
    memcpy(this->MainLabelTable.Buffer, srcG->MainLabelTable.Buffer,
      sizeof(svtkReebLabel) * srcG->MainLabelTable.Size);
  }

  if (srcG->ArcLoopTable)
  {
    this->ArcLoopTable = (svtkIdType*)malloc(sizeof(svtkIdType) * srcG->LoopNumber);
    memcpy(this->ArcLoopTable, srcG->ArcLoopTable, sizeof(svtkIdType) * srcG->LoopNumber);
  }

  if (srcG->VertexMapSize)
  {
    this->VertexMapSize = srcG->VertexMapSize;
    this->VertexMapAllocatedSize = srcG->VertexMapAllocatedSize;
    this->VertexMap = (svtkIdType*)malloc(sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    memcpy(this->VertexMap, srcG->VertexMap, sizeof(svtkIdType) * srcG->VertexMapAllocatedSize);
  }

  if (srcG->TriangleVertexMapSize)
  {
    this->TriangleVertexMapSize = srcG->TriangleVertexMapSize;
    this->TriangleVertexMapAllocatedSize = srcG->TriangleVertexMapAllocatedSize;
    this->TriangleVertexMap = (int*)malloc(sizeof(int) * this->TriangleVertexMapAllocatedSize);
    memcpy(this->TriangleVertexMap, srcG->TriangleVertexMap,
      sizeof(int) * srcG->TriangleVertexMapAllocatedSize);
  }
}

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkReebGraph);

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::SetLabel(svtkIdType arcId, svtkReebLabelTag Label)
{
  inputMesh = nullptr;

  ResizeMainLabelTable(1);

  // create a new label in the graph
  svtkIdType L = this->MainLabelTable.FreeZone;
  this->MainLabelTable.FreeZone = this->GetLabel(L)->ArcId;
  ++(this->MainLabelTable.Number);
  memset(this->GetLabel(L), 0, sizeof(svtkReebLabel));
  svtkReebLabel* l = this->GetLabel(L);

  l->HPrev = 0;
  l->HNext = 0;
  this->GetArc(arcId)->LabelId0 = L;
  this->GetArc(arcId)->LabelId1 = L;

  l->ArcId = arcId;
  l->label = Label;

  svtkIdType Lp = FindDwLabel(this->GetArc(arcId)->NodeId0, Label);
  svtkIdType Ln = FindUpLabel(this->GetArc(arcId)->NodeId1, Label);

  l->VPrev = Lp;
  if (Lp)
    this->GetLabel(Lp)->VNext = L;
  l->VNext = Ln;
  if (Ln)
    this->GetLabel(Ln)->VPrev = L;
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::FastArcSimplify(
  svtkIdType arcId, int svtkNotUsed(ArcNumber), svtkIdType* svtkNotUsed(arcTable))
{
  // Remove the arc which opens the loop
  svtkIdType nodeId0 = this->GetArc(arcId)->NodeId0;
  svtkIdType nodeId1 = this->GetArc(arcId)->NodeId1;

  svtkReebArc* A = this->GetArc(arcId);
  svtkReebArc* B = nullptr;
  int down, middle, up;

  if (historyOn)
  {
    if (A->ArcDwId0)
    {
      B = this->GetArc(A->ArcDwId0);

      down = this->GetNode(B->NodeId0)->VertexId;
      middle = this->GetNode(A->NodeId0)->VertexId;
      up = this->GetNode(B->NodeId1)->VertexId;

      svtkReebCancellation c;
      c.removedArcs.push_back(std::pair<int, int>(middle, up));
      c.insertedArcs.push_back(std::pair<int, int>(down, up));
      this->cancellationHistory.push_back(c);
    }
    if (A->ArcDwId1)
    {
      B = this->GetArc(A->ArcDwId1);

      down = this->GetNode(B->NodeId0)->VertexId;
      middle = this->GetNode(A->NodeId0)->VertexId;
      up = this->GetNode(A->NodeId1)->VertexId;

      svtkReebCancellation c;
      c.removedArcs.push_back(std::pair<int, int>(middle, up));
      c.insertedArcs.push_back(std::pair<int, int>(down, up));
      this->cancellationHistory.push_back(c);
    }
    if (A->ArcUpId0)
    {
      B = this->GetArc(A->ArcUpId0);

      down = this->GetNode(A->NodeId0)->VertexId;
      middle = this->GetNode(A->NodeId1)->VertexId;
      up = this->GetNode(B->NodeId1)->VertexId;

      svtkReebCancellation c;
      c.removedArcs.push_back(std::pair<int, int>(down, middle));
      c.insertedArcs.push_back(std::pair<int, int>(down, up));
      this->cancellationHistory.push_back(c);
    }
    if (A->ArcUpId1)
    {
      B = this->GetArc(A->ArcUpId1);

      down = this->GetNode(B->NodeId0)->VertexId;
      middle = this->GetNode(A->NodeId1)->VertexId;
      up = this->GetNode(B->NodeId1)->VertexId;

      svtkReebCancellation c;
      c.removedArcs.push_back(std::pair<int, int>(down, middle));
      c.insertedArcs.push_back(std::pair<int, int>(down, up));
      this->cancellationHistory.push_back(c);
    }
  }

  svtkReebGraphRemoveUpArc(this, nodeId0, arcId);
  svtkReebGraphRemoveDownArc(this, nodeId1, arcId);

  // delete the arc from the graph...
  this->GetArc(arcId)->LabelId1 = ((int)-2);
  this->GetArc(arcId)->LabelId0 = this->MainArcTable.FreeZone;
  this->MainArcTable.FreeZone = (arcId);
  --(this->MainArcTable.Number);
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindGreater(
  svtkIdType nodeId, svtkIdType startingNodeId, svtkReebLabelTag label)
{
  if (!this->GetNode(nodeId)->IsFinalized)
    return 0;

  // base case
  if (svtkReebGraphIsHigherThan(
        this, nodeId, startingNodeId, this->GetNode(nodeId), this->GetNode(startingNodeId)))
    return nodeId;

  // iterative case
  for (svtkIdType A = this->GetNode(nodeId)->ArcUpId; A; A = this->GetArc(A)->ArcDwId0)
  {
    svtkReebArc* a = this->GetArc(A);
    svtkIdType M = this->GetArc(A)->NodeId1;
    svtkReebNode* m = this->GetNode(M);

    if (a->LabelId0 || !m->IsFinalized) // other labels or not final node
    {
      continue;
    }

    if ((M = FindGreater(M, startingNodeId, label)))
    {
      if (label)
      {
        SetLabel(A, label);
      }
      return M;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindLess(
  svtkIdType nodeId, svtkIdType startingNodeId, svtkReebLabelTag label)
{
  if (!this->GetNode(nodeId)->IsFinalized)
    return 0;

  // base case
  if (svtkReebGraphIsSmaller(
        this, nodeId, startingNodeId, this->GetNode(nodeId), this->GetNode(startingNodeId)))
    return nodeId;

  // iterative case
  for (svtkIdType A = this->GetNode(nodeId)->ArcDownId; A; A = this->GetArc(A)->ArcDwId1)
  {
    svtkReebArc* a = this->GetArc(A);
    svtkIdType M = this->GetArc(A)->NodeId0;
    svtkReebNode* m = this->GetNode(M);

    if (a->LabelId0 || !m->IsFinalized) // other labels or not final node
      continue;

    if ((M = FindLess(M, startingNodeId, label)))
    {
      if (label)
      {
        SetLabel(A, label);
      }
      return M;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindJoinNode(
  svtkIdType arcId, svtkReebLabelTag label, bool onePathOnly)
{
  svtkIdType N = this->GetArc(arcId)->NodeId1;
  svtkIdType Ret, C;

  if (this->GetArc(arcId)->LabelId0 || !this->GetNode(N)->IsFinalized)
    // other labels or not final node
    return 0;

  if (onePathOnly && (this->GetArc(arcId)->ArcDwId0 || this->GetArc(arcId)->ArcUpId0))
    return 0;

  // base case
  if (this->GetArc(arcId)->ArcDwId1 || this->GetArc(arcId)->ArcUpId1)
  {
    if (label)
      SetLabel(arcId, label);
    return N;
  }

  for (C = this->GetNode(N)->ArcUpId; C; C = this->GetArc(C)->ArcDwId0)
  {
    Ret = FindJoinNode(C, label, onePathOnly);

    if (Ret)
    {
      if (label)
        SetLabel(arcId, label);
      return Ret;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindSplitNode(
  svtkIdType arcId, svtkReebLabelTag label, bool onePathOnly)
{
  svtkIdType N = this->GetArc(arcId)->NodeId0;
  svtkIdType Ret, C;

  if (this->GetArc(arcId)->LabelId0 || !this->GetNode(N)->IsFinalized)
    // other labels or not final node
    return 0;

  if (onePathOnly && (this->GetArc(arcId)->ArcDwId1 || this->GetArc(arcId)->ArcUpId1))
    return 0;

  // base case
  if (this->GetArc(arcId)->ArcDwId0 || this->GetArc(arcId)->ArcUpId0)
  {
    if (label)
      SetLabel(arcId, label);
    return N;
  }

  // iterative case
  for (C = this->GetNode(N)->ArcDownId; C; C = this->GetArc(C)->ArcDwId1)
  {
    Ret = FindSplitNode(C, label, onePathOnly);

    if (Ret)
    {
      if (label)
        SetLabel(arcId, label);
      return Ret;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkReebGraph::Implementation::svtkReebPath svtkReebGraph::Implementation::FindPath(
  svtkIdType arcId, double simplificationThreshold, svtkReebGraphSimplificationMetric* metric)
{
  svtkReebPath entry;
  std::priority_queue<svtkReebPath> pq;
  int size;

  svtkIdType N0 = this->GetArc(arcId)->NodeId0;
  svtkIdType N1 = this->GetArc(arcId)->NodeId1;

  char* Ntouch = nullptr;
  char* Atouch = nullptr;

  //  double simplificationValue = 0;
  if ((!inputMesh) || (!metric))
  {
    double f0 = this->GetNode(N0)->Value;
    double f1 = this->GetNode(N1)->Value;
    entry.SimplificationValue = (f1 - f0) / (this->MaximumScalarValue - this->MinimumScalarValue);
  }
  else
  {
    entry.SimplificationValue = ComputeCustomMetric(metric, this->GetArc(arcId));
  }

  // the arc itself has a good persistence
  if (simplificationThreshold && entry.SimplificationValue >= simplificationThreshold)
  {
  NOT_FOUND:
    if (Ntouch)
      free(Ntouch);
    if (Atouch)
      free(Atouch);
    svtkReebPath fake;
    memset(&fake, 0, sizeof(svtkReebPath));
    fake.SimplificationValue = 1;
    return fake;
  }

  Atouch = (char*)malloc(sizeof(char) * this->MainArcTable.Size);
  Ntouch = (char*)malloc(sizeof(char) * this->MainNodeTable.Size);
  memset(Atouch, 0, sizeof(char) * this->MainArcTable.Size);
  memset(Ntouch, 0, sizeof(char) * this->MainNodeTable.Size);

  Ntouch[N0] = 1;

  // I don't want to use the arc given by the user
  Atouch[arcId] = 1;

  entry.NodeNumber = 1;
  entry.NodeTable = new svtkIdType[1];
  entry.NodeTable[0] = N0;
  entry.ArcNumber = 0;
  entry.ArcTable = nullptr;
  pq.push(entry);

  while ((size = static_cast<int>(pq.size())))
  {
    entry = pq.top();
    pq.pop();

    int N = entry.NodeTable[entry.NodeNumber - 1];

    for (int dir = 0; dir <= 1; dir++)
    {
      for (int A = (!dir) ? this->GetNode(N)->ArcDownId : this->GetNode(N)->ArcUpId; A;
           A = (!dir) ? this->GetArc(A)->ArcDwId1 : this->GetArc(A)->ArcDwId0)
      {
        int M = (!dir) ? (this->GetArc(A)->NodeId0) : (this->GetArc(A)->NodeId1);

        if (Atouch[A])
          continue;
        Atouch[A] = 1;

        // already used (==there is a better path to reach M)
        if (Ntouch[M])
          continue;
        Ntouch[M] = 1;

        // found!!!
        if (M == N1)
        {
          // clear all the items in the priority queue
          while (!pq.empty())
          {
            svtkReebPath aux = pq.top();
            pq.pop();
            delete aux.ArcTable;
            delete aux.NodeTable;
          }

          free(Ntouch);
          free(Atouch);

          svtkIdType* tmp = new svtkIdType[entry.NodeNumber + 1];
          memcpy(tmp, entry.NodeTable, sizeof(svtkIdType) * entry.NodeNumber);
          tmp[entry.NodeNumber] = N1;
          delete[] entry.NodeTable;
          entry.NodeTable = tmp;
          entry.NodeNumber++;
          return entry;
        }

        if ((!inputMesh) || (!metric))
        {
          entry.SimplificationValue += svtkReebGraphGetArcPersistence(this, this->GetArc(A));
        }
        else
        {
          entry.SimplificationValue += ComputeCustomMetric(metric, this->GetArc(A));
        }

        // The loop persistence is greater than functionScale
        if (simplificationThreshold && entry.SimplificationValue >= simplificationThreshold)
          continue;

        svtkReebPath newentry;
        newentry.SimplificationValue = entry.SimplificationValue;
        newentry.ArcNumber = entry.ArcNumber + 1;
        newentry.ArcTable = new svtkIdType[newentry.ArcNumber];
        newentry.NodeNumber = entry.NodeNumber + 1;
        newentry.NodeTable = new svtkIdType[newentry.NodeNumber];
        if (entry.ArcNumber)
          memcpy(newentry.ArcTable, entry.ArcTable, sizeof(svtkIdType) * entry.ArcNumber);
        if (entry.NodeNumber)
          memcpy(newentry.NodeTable, entry.NodeTable, sizeof(svtkIdType) * entry.NodeNumber);

        newentry.ArcTable[entry.ArcNumber] = A;
        newentry.NodeTable[entry.NodeNumber] = M;
        pq.push(newentry);
      }
    }

    // finished with this entry
    delete entry.ArcTable;
    delete[] entry.NodeTable;
  }

  goto NOT_FOUND;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::SimplifyLoops(
  double simplificationThreshold, svtkReebGraphSimplificationMetric* simplificationMetric)
{

  if (!simplificationThreshold)
    return 0;

  // refresh information about ArcLoopTable
  this->FindLoops();

  int NumSimplified = 0;

  for (int n = 0; n < this->LoopNumber; n++)
  {
    int A = this->ArcLoopTable[n];

    if (this->GetArc(A)->LabelId1 == ((int)-2))
      continue;

    double simplificationValue = 0;
    if ((!inputMesh) || (!simplificationMetric))
    {
      svtkIdType N0 = this->GetArc(A)->NodeId0;
      svtkIdType N1 = this->GetArc(A)->NodeId1;
      double f0 = this->GetNode(N0)->Value;
      double f1 = this->GetNode(N1)->Value;
      simplificationValue = (f1 - f0) / (this->MaximumScalarValue - this->MinimumScalarValue);
    }
    else
    {
      simplificationValue = ComputeCustomMetric(simplificationMetric, this->GetArc(A));
    }

    if (simplificationValue >= simplificationThreshold)
      continue;

    svtkReebPath entry =
      this->FindPath(this->ArcLoopTable[n], simplificationThreshold, simplificationMetric);

    // too high for persistence
    if (entry.SimplificationValue >= simplificationThreshold)
      continue;

    // distribute its bucket to the loop and delete the arc
    this->FastArcSimplify(ArcLoopTable[n], entry.ArcNumber, entry.ArcTable);
    delete entry.ArcTable;
    delete entry.NodeTable;

    ++NumSimplified;
    CommitSimplification();
  }

  // check for regular points
  for (int N = 1; N < this->MainNodeTable.Size; N++)
  {
    if (this->GetNode(N)->ArcUpId == ((int)-2))
      continue;

    if (this->GetNode(N)->ArcDownId == 0 && this->GetNode(N)->ArcUpId == 0)
    {
      // delete the node from the graph...
      this->GetNode(N)->ArcUpId = ((int)-2);
      this->GetNode(N)->ArcDownId = this->MainNodeTable.FreeZone;
      this->MainNodeTable.FreeZone = (N);
      --(this->MainNodeTable.Number);
    }

    else if (svtkReebGraphIsRegular(this, this->GetNode(N)))
    {
      if (historyOn)
      {
        svtkReebNode* n = this->GetNode(N);

        int A0 = n->ArcDownId;
        int A1 = n->ArcUpId;

        svtkReebArc* a0 = this->GetArc(A0);
        svtkReebArc* a1 = this->GetArc(A1);
        svtkReebNode* downN = this->GetNode(a0->NodeId0);
        svtkReebNode* upN = this->GetNode(a1->NodeId1);

        int down, middle, up;
        down = downN->VertexId;
        middle = n->VertexId;
        up = upN->VertexId;

        svtkReebCancellation c;
        c.removedArcs.push_back(std::pair<int, int>(down, middle));
        c.removedArcs.push_back(std::pair<int, int>(middle, up));
        c.insertedArcs.push_back(std::pair<int, int>(down, up));

        this->cancellationHistory.push_back(c);
      }
      EndVertex(N);
    }
  }

  this->RemovedLoopNumber = NumSimplified;

  return NumSimplified;
}

//----------------------------------------------------------------------------
double svtkReebGraph::Implementation::ComputeCustomMetric(
  svtkReebGraphSimplificationMetric* simplificationMetric, svtkReebArc* a)
{
  int edgeId = -1, start = -1, end = -1;

  svtkDataArray* vertexInfo =
    svtkArrayDownCast<svtkDataArray>(this->Parent->GetVertexData()->GetAbstractArray("Vertex Ids"));
  if (!vertexInfo)
    return svtkReebGraphGetArcPersistence(this, a);

  svtkVariantArray* edgeInfo =
    svtkArrayDownCast<svtkVariantArray>(this->Parent->GetEdgeData()->GetAbstractArray("Vertex Ids"));
  if (!edgeInfo)
    return svtkReebGraphGetArcPersistence(this, a);

  svtkEdgeListIterator* eIt = svtkEdgeListIterator::New();
  this->Parent->GetEdges(eIt);

  do
  {
    svtkEdgeType e = eIt->Next();
    if (((*(vertexInfo->GetTuple(e.Source))) == GetNodeVertexId(a->NodeId0)) &&
      ((*(vertexInfo->GetTuple(e.Target))) == GetNodeVertexId(a->NodeId1)))
    {
      edgeId = e.Id;
      start = static_cast<int>(*(vertexInfo->GetTuple(e.Source)));
      end = static_cast<int>(*(vertexInfo->GetTuple(e.Target)));
      break;
    }
  } while (eIt->HasNext());
  eIt->Delete();

  svtkAbstractArray* vertexList = edgeInfo->GetPointer(edgeId)->ToArray();

  return simplificationMetric->ComputeMetric(inputMesh, inputScalarField, start, vertexList, end);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::SimplifyBranches(
  double simplificationThreshold, svtkReebGraphSimplificationMetric* simplificationMetric)
{
  static const svtkReebLabelTag RouteOld = 100;
  static const svtkReebLabelTag RouteNew = 200;
  int nstack, mstack = 0;
  int* stack = nullptr;

  if (!simplificationThreshold)
    return 0;

  int nsimp = 0;
  int cont = 0;
  const int step = 10000;
  bool redo;

  svtkDataSet* input = inputMesh;

REDO:
  nstack = 0;
  redo = false;

  for (int N = 1; N < this->MainNodeTable.Size; ++N)
  {
    if (this->GetNode(N)->ArcUpId == ((int)-2))
      continue;

    svtkReebNode* n = this->GetNode(N);

    // simplify atomic nodes
    if (!n->ArcDownId && !n->ArcUpId)
    {
      // delete the node from the graph...
      this->GetNode(N)->ArcUpId = ((int)-2);
      this->GetNode(N)->ArcDownId = this->MainNodeTable.FreeZone;
      this->MainNodeTable.FreeZone = (N);
      --(this->MainNodeTable.Number);
    }
    else if (!n->ArcDownId)
    {
      // insert into stack branches to simplify
      for (int _A_ = n->ArcUpId; _A_; _A_ = this->GetArc(_A_)->ArcDwId0)
      {
        svtkReebArc* _a_ = this->GetArc(_A_);
        if ((!inputMesh) || (!simplificationMetric))
        {
          if (svtkReebGraphGetArcPersistence(this, _a_) < simplificationThreshold)
          {
            svtkReebGraphStackPush(_A_);
          }
        }
        else
        {
          if (this->ComputeCustomMetric(simplificationMetric, _a_) < simplificationThreshold)
          {
            svtkReebGraphStackPush(_A_);
          }
        }
      }
    }
    else if (!n->ArcUpId)
    {
      // insert into stack branches to simplify
      for (int _A_ = n->ArcDownId; _A_; _A_ = this->GetArc(_A_)->ArcDwId1)
      {
        svtkReebArc* _a_ = this->GetArc(_A_);
        if (svtkReebGraphGetArcPersistence(this, _a_) < simplificationThreshold)
        {
          svtkReebGraphStackPush(_A_);
        }
      }
    }
  }

  while (svtkReebGraphStackSize())
  {
    int A = svtkReebGraphStackTop();
    svtkReebGraphStackPop();

    if (!--cont)
    {
      cont = step;
    }

    if (this->GetArc(A)->LabelId1 == ((int)-2))
      continue;

    cont++;

    svtkReebArc* arc = this->GetArc(A);

    int N = arc->NodeId0;
    int M = arc->NodeId1;

    if (this->GetNode(N)->ArcDownId && this->GetNode(M)->ArcUpId)
      continue;

    double persistence = svtkReebGraphGetArcPersistence(this, arc);

    // is the actual persistence (in percentage) greater than the applied filter?
    if (persistence >= simplificationThreshold)
      continue;

    int _A, Mdown = 0, Nup = 0, Ndown = 0, Mup = 0;

    // get the 'down' degree for M
    for (_A = this->GetNode(M)->ArcDownId; _A; _A = this->GetArc(_A)->ArcDwId1)
      ++Mdown;

    // Get the 'up' degree for N
    for (_A = this->GetNode(N)->ArcUpId; _A; _A = this->GetArc(_A)->ArcDwId0)
      ++Nup;

    // get the 'down' degree for N
    for (_A = this->GetNode(N)->ArcDownId; _A; _A = this->GetArc(_A)->ArcDwId1)
      ++Ndown;

    // get the 'up' degree for M
    for (_A = this->GetNode(M)->ArcUpId; _A; _A = this->GetArc(_A)->ArcDwId0)
      ++Mup;

    // isolated arc
    if (!Ndown && Nup == 1 && Mdown == 1 && !Mup)
    {
      svtkReebGraphRemoveUpArc(this, N, A);
      svtkReebGraphRemoveDownArc(this, M, A);

      // delete the arc from the graph...
      this->GetArc(A)->LabelId1 = ((int)-2);
      this->GetArc(A)->LabelId0 = this->MainArcTable.FreeZone;
      this->MainArcTable.FreeZone = (A);
      --(this->MainArcTable.Number);

      if (!(this->GetNode(N)->ArcUpId == ((int)-2)) &&
        svtkReebGraphIsRegular(this, this->GetNode(N)))
      {
        EndVertex(N);
      }
      if (!(this->GetNode(M)->ArcUpId == ((int)-2)) &&
        svtkReebGraphIsRegular(this, this->GetNode(M)))
      {
        EndVertex(M);
      }

      nsimp++;
      redo = true;
      continue;
    }

    int Down = 0;
    int Up = 0;

    bool simplified = false;

    // M is a maximum
    if (!Mup)
    {
      if ((Down = FindSplitNode(A, RouteOld)))
      {
        if ((Up = FindGreater(Down, M, RouteNew)))
        {
          SetLabel(AddArc(M, Up), RouteOld);
          Collapse(Down, Up, RouteOld, RouteNew);
          simplified = true;
        }
        else
        {
          this->SimplifyLabels(Down);
        }
      }
    }

    // N is a minimum
    if (!simplified && !Ndown)
    {
      if ((Up = FindJoinNode(A, RouteOld)))
      {
        if ((Down = FindLess(Up, N, RouteNew)))
        {
          SetLabel(AddArc(Down, N), RouteOld);
          Collapse(Down, Up, RouteOld, RouteNew);
          simplified = true;
        }
        else
        {
          this->SimplifyLabels(Up);
        }
      }
    }

    if (simplified)
    {
      if (!(this->GetNode(Down)->ArcUpId == ((int)-2)))
      {
        this->SimplifyLabels(Down);

        if (!this->GetNode(Down)->ArcDownId) // minimum
        {
          for (svtkIdType _A_ = this->GetNode(Down)->ArcUpId; _A_; _A_ = this->GetArc(_A_)->ArcDwId0)
          {
            svtkReebArc* _a_ = this->GetArc(_A_);
            if ((!inputMesh) || (!simplificationMetric))
            {
              if (svtkReebGraphGetArcPersistence(this, _a_) < simplificationThreshold)
              {
                svtkReebGraphStackPush(_A_);
              }
            }
            else
            {
              if (this->ComputeCustomMetric(simplificationMetric, _a_) < simplificationThreshold)
              {
                svtkReebGraphStackPush(_A_);
              }
            }
          }
        }
      }

      if (!(this->GetNode(Up)->ArcUpId == ((int)-2)))
      {
        this->SimplifyLabels(Up);

        if (!this->GetNode(Up)->ArcUpId)
        {
          for (int _A_ = this->GetNode(Up)->ArcDownId; _A_; _A_ = this->GetArc(_A_)->ArcDwId1)
          {
            svtkReebArc* _a_ = this->GetArc(_A_);
            if ((!inputMesh) || (!simplificationMetric))
            {
              if (svtkReebGraphGetArcPersistence(this, _a_) < simplificationThreshold)
              {
                svtkReebGraphStackPush(_A_);
              }
            }
            else
            {
              if (this->ComputeCustomMetric(simplificationMetric, _a_) < simplificationThreshold)
              {
                svtkReebGraphStackPush(_A_);
              }
            }
          }
        }
      }

      nsimp++;
      redo = true;
    }
    CommitSimplification();
  } // while

  if (redo)
    goto REDO;

  free(stack);

  inputMesh = input;

  return nsimp;
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::ResizeMainNodeTable(int newSize)
{
  int oldsize, i;

  if ((this->MainNodeTable.Size - this->MainNodeTable.Number) < newSize)
  {
    oldsize = this->MainNodeTable.Size;

    if (!this->MainNodeTable.Size)
      this->MainNodeTable.Size = newSize;
    while ((this->MainNodeTable.Size - this->MainNodeTable.Number) < newSize)
      this->MainNodeTable.Size <<= 1;

    this->MainNodeTable.Buffer = (svtkReebNode*)realloc(
      this->MainNodeTable.Buffer, sizeof(svtkReebNode) * this->MainNodeTable.Size);

    for (i = oldsize; i < this->MainNodeTable.Size - 1; i++)
    {
      this->GetNode(i)->ArcDownId = i + 1;
      this->GetNode(i)->ArcUpId = ((int)-2);
    }

    this->GetNode(i)->ArcDownId = this->MainNodeTable.FreeZone;
    this->GetNode(i)->ArcUpId = ((int)-2);
    this->MainNodeTable.FreeZone = oldsize;
  }
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::CommitSimplification()
{
  // now re-construct the graph with projected deg-2 nodes.
  std::vector<std::pair<std::pair<int, int>, std::vector<int> > > before, after;

  svtkEdgeListIterator* eIt = svtkEdgeListIterator::New();
  this->Parent->GetEdges(eIt);
  svtkVariantArray* edgeInfo =
    svtkArrayDownCast<svtkVariantArray>(this->Parent->GetEdgeData()->GetAbstractArray("Vertex Ids"));
  svtkDataArray* vertexInfo = this->Parent->GetVertexData()->GetArray("Vertex Ids");

  // avoids double projection
  int vertexNumber = vertexInfo->GetNumberOfTuples();
  std::vector<bool> segmentedVertices;

  do
  {
    std::pair<std::pair<int, int>, std::vector<int> > superArc;

    svtkEdgeType e = eIt->Next();
    svtkAbstractArray* vertexList = edgeInfo->GetPointer(e.Id)->ToArray();

    superArc.first.first = (int)*(vertexInfo->GetTuple(e.Source));
    superArc.first.second = (int)*(vertexInfo->GetTuple(e.Target));

    superArc.second.resize(vertexList->GetNumberOfTuples());
    vertexNumber += vertexList->GetNumberOfTuples();
    for (unsigned int i = 0; i < superArc.second.size(); i++)
      superArc.second[i] = vertexList->GetVariantValue(i).ToInt();

    before.push_back(superArc);
  } while (eIt->HasNext());

  segmentedVertices.resize(vertexNumber);
  for (unsigned int i = 0; i < segmentedVertices.size(); i++)
    segmentedVertices[i] = false;

  svtkIdType prevArcId = -1, arcId = 0;
  while (arcId != prevArcId)
  {
    prevArcId = arcId;
    arcId = GetPreviousArcId();
  }
  prevArcId = -1;

  while (prevArcId != arcId)
  {
    if (this->GetArc(arcId))
    {
      int down, up;
      down = this->GetNode((this->GetArc(arcId))->NodeId0)->VertexId;
      up = this->GetNode((this->GetArc(arcId))->NodeId1)->VertexId;

      std::pair<std::pair<int, int>, std::vector<int> > superArc;

      superArc.first.first = down;
      superArc.first.second = up;

      after.push_back(superArc);
    }
    prevArcId = arcId;
    arcId = GetNextArcId();
  }

  std::pair<int, int> destinationArc;
  std::map<int, bool> processedOutputArcs;

  // now map the unsimplified arcs onto the simplified ones
  for (unsigned int i = 0; i < before.size(); i++)
  {
    std::vector<int> simplifiedCriticalNodes;
    destinationArc = before[i].first;
    for (unsigned int j = 0; j < this->cancellationHistory.size(); j++)
    {
      for (unsigned int k = 0; k < this->cancellationHistory[j].removedArcs.size(); k++)
      {
        if ((destinationArc.first == this->cancellationHistory[j].removedArcs[k].first) &&
          (destinationArc.second == this->cancellationHistory[j].removedArcs[k].second))
        {
          // the arc has been involved in a cancellation
          destinationArc = this->cancellationHistory[j].insertedArcs[0];

          if (this->cancellationHistory[j].removedArcs.size() > 1)
          {
            if (((this->cancellationHistory[j].removedArcs[0].first == destinationArc.first) &&
                  (this->cancellationHistory[j].removedArcs[1].second == destinationArc.second)) ||
              ((this->cancellationHistory[j].removedArcs[1].first == destinationArc.first) &&
                (this->cancellationHistory[j].removedArcs[0].second == destinationArc.second)))
            {
              for (unsigned int l = 0; l < this->cancellationHistory[j].removedArcs.size(); l++)
              {
                if ((this->cancellationHistory[j].removedArcs[l].first != destinationArc.first) &&
                  (this->cancellationHistory[j].removedArcs[l].first != destinationArc.second))
                {
                  // this critical node will become a degree two node, let's
                  // remember it
                  simplifiedCriticalNodes.push_back(
                    this->cancellationHistory[j].removedArcs[l].first);
                }
                if ((this->cancellationHistory[j].removedArcs[l].second != destinationArc.first) &&
                  (this->cancellationHistory[j].removedArcs[l].second != destinationArc.second))
                {
                  // same thing as above
                  simplifiedCriticalNodes.push_back(
                    this->cancellationHistory[j].removedArcs[l].second);
                }
              }
            }
          }
        }
      }
    }

    // at this point the deg2-nodes are in before[i].second

    // now find the projection in the simplified graph
    for (unsigned int j = 0; j < after.size(); j++)
    {
      if (destinationArc == after[j].first)
      {
        std::map<int, bool>::iterator aIt;
        aIt = processedOutputArcs.find(j);
        if (aIt == processedOutputArcs.end())
        {
          if (before[i].first == destinationArc)
          {
            // non-simplified arc
            processedOutputArcs[j] = true;
            after[j].second = before[i].second;
          }
          if (before[i].first != destinationArc)
          {
            // adding content of before[i].second to after[j].second
            for (unsigned int k = 0; k < before[i].second.size(); k++)
            {
              if (!segmentedVertices[before[i].second[k]])
              {
                after[j].second.push_back(before[i].second[k]);
                segmentedVertices[before[i].second[k]] = true;
              }
            }
          }
          for (unsigned int k = 0; k < simplifiedCriticalNodes.size(); k++)
          {
            if (!segmentedVertices[simplifiedCriticalNodes[k]])
            {
              after[j].second.push_back(simplifiedCriticalNodes[k]);
              segmentedVertices[simplifiedCriticalNodes[k]] = true;
            }
          }
          break;
        }
      }
    }
  }
  // ensure the sorting on the arcs
  for (unsigned int i = 0; i < after.size(); i++)
  {
    std::vector<std::pair<int, double> > scalarValues;
    for (unsigned int j = 0; j < after[i].second.size(); j++)
    {
      std::pair<int, double> scalarVertex;
      scalarVertex.first = after[i].second[j];
      std::map<int, double>::iterator sIt;
      sIt = ScalarField.find(scalarVertex.first);
      if (sIt != ScalarField.end())
      {
        scalarVertex.second = sIt->second;
        scalarValues.push_back(scalarVertex);
      }
    }
    std::sort(scalarValues.begin(), scalarValues.end(), svtkReebGraphVertexSoS);
    for (unsigned int j = 0; j < after[i].second.size(); j++)
      after[i].second[j] = scalarValues[j].first;
  }

  // now construct the svtkMutableDirectedGraph
  // first, clean up the current graph
  while (this->Parent->GetNumberOfEdges())
    this->Parent->RemoveEdge(0);
  while (this->Parent->GetNumberOfVertices())
    this->Parent->RemoveVertex(0);

  this->Parent->GetVertexData()->RemoveArray("Vertex Ids");
  this->Parent->GetEdgeData()->RemoveArray("Vertex Ids");

  svtkIdType prevNodeId = -1, nodeId = 0;
  while (prevNodeId != nodeId)
  {
    prevNodeId = nodeId;
    nodeId = GetPreviousNodeId();
  }
  prevNodeId = -1;

  svtkVariantArray* vertexProperties = svtkVariantArray::New();
  vertexProperties->SetNumberOfValues(1);

  svtkIdTypeArray* vertexIds = svtkIdTypeArray::New();
  vertexIds->SetName("Vertex Ids");
  this->Parent->GetVertexData()->AddArray(vertexIds);

  std::map<int, int> vMap;
  int vIt = 0;
  while (prevNodeId != nodeId)
  {
    svtkIdType nodeVertexId = GetNodeVertexId(nodeId);
    vMap[nodeVertexId] = vIt;
    vertexProperties->SetValue(0, nodeVertexId);
    this->Parent->AddVertex(vertexProperties);

    prevNodeId = nodeId;
    nodeId = GetNextNodeId();
    vIt++;
  }
  vertexIds->Delete();
  vertexProperties->Delete();

  svtkVariantArray* deg2NodeIds = svtkVariantArray::New();
  deg2NodeIds->SetName("Vertex Ids");
  this->Parent->GetEdgeData()->AddArray(deg2NodeIds);

  for (unsigned int i = 0; i < after.size(); i++)
  {
    std::map<int, int>::iterator downIt, upIt;
    downIt = vMap.find(after[i].first.first);
    upIt = vMap.find(after[i].first.second);

    if ((downIt != vMap.end()) && (upIt != vMap.end()))
    {
      svtkVariantArray* edgeProperties = svtkVariantArray::New();
      svtkIdTypeArray* vertexList = svtkIdTypeArray::New();
      vertexList->SetNumberOfValues(static_cast<svtkIdType>(after[i].second.size()));
      for (unsigned int j = 0; j < after[i].second.size(); j++)
        vertexList->SetValue(j, after[i].second[j]);
      edgeProperties->SetNumberOfValues(1);
      edgeProperties->SetValue(0, vertexList);
      this->Parent->AddEdge(downIt->second, upIt->second, edgeProperties);
      vertexList->Delete();
      edgeProperties->Delete();
    }
  }
  deg2NodeIds->Delete();

  this->cancellationHistory.clear();

  return 0;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Simplify(
  double simplificationThreshold, svtkReebGraphSimplificationMetric* simplificationMetric)
{
  int deletionNumber = 0;

  this->Storage->cancellationHistory.clear();
  this->Storage->historyOn = true;

  this->Storage->ArcNumber = 0;
  this->Storage->NodeNumber = 0;

  deletionNumber = this->Storage->SimplifyBranches(simplificationThreshold, simplificationMetric) +
    this->Storage->SimplifyLoops(simplificationThreshold, simplificationMetric) +
    this->Storage->SimplifyBranches(simplificationThreshold, simplificationMetric);

  this->Storage->historyOn = false;

  return deletionNumber;
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::FlushLabels()
{
  for (int A = 1; A < this->MainArcTable.Size; A++)
  {
    if (!(this->GetArc(A)->LabelId1 == ((int)-2)))
      this->GetArc(A)->LabelId0 = this->GetArc(A)->LabelId1 = 0;
  }

  if (this->MainLabelTable.Buffer)
  {
    free(this->MainLabelTable.Buffer);
  }

  this->MainLabelTable.Buffer = (svtkReebLabel*)malloc(sizeof(svtkReebLabel) * 2);
  this->MainLabelTable.Size = 2;
  this->MainLabelTable.Number = 1;
  this->MainLabelTable.FreeZone = 1;
  this->GetLabel(1)->HNext = ((int)-2);
  this->GetLabel(1)->ArcId = 0;
}

//----------------------------------------------------------------------------
void svtkReebGraph::DeepCopy(svtkDataObject* src)
{

  svtkReebGraph* srcG = svtkReebGraph::SafeDownCast(src);

  if (srcG)
  {
    this->Storage->DeepCopy(srcG->Storage);
  }

  svtkMutableDirectedGraph::DeepCopy(srcG);
}

//----------------------------------------------------------------------------
void svtkReebGraph::Set(svtkMutableDirectedGraph* g)
{
  svtkMutableDirectedGraph::DeepCopy(g);
}

//----------------------------------------------------------------------------
void svtkReebGraph::CloseStream()
{

  svtkIdType prevArcId = -1, arcId = 0;
  while (arcId != prevArcId)
  {
    prevArcId = arcId;
    arcId = this->Storage->GetPreviousArcId();
  }
  prevArcId = -1;

  // loop over the arcs and build the local adjacency map

  // vertex -> (down vertices, up vertices)
  std::map<int, std::pair<std::vector<int>, std::vector<int> > > localAdjacency;
  while (prevArcId != arcId)
  {
    svtkIdType downVertexId, upVertexId;
    downVertexId = this->Storage->GetNode((this->Storage->GetArc(arcId))->NodeId0)->VertexId;
    upVertexId = this->Storage->GetNode((this->Storage->GetArc(arcId))->NodeId1)->VertexId;

    std::map<int, std::pair<std::vector<int>, std::vector<int> > >::iterator aIt;

    // lookUp for the down vertex
    aIt = localAdjacency.find(downVertexId);
    if (aIt == localAdjacency.end())
    {
      std::pair<std::vector<int>, std::vector<int> > adjacencyItem;
      adjacencyItem.second.push_back(upVertexId);
      localAdjacency[downVertexId] = adjacencyItem;
    }
    else
    {
      aIt->second.second.push_back(upVertexId);
    }

    // same thing for the up vertex
    aIt = localAdjacency.find(upVertexId);
    if (aIt == localAdjacency.end())
    {
      std::pair<std::vector<int>, std::vector<int> > adjacencyItem;
      adjacencyItem.first.push_back(downVertexId);
      localAdjacency[upVertexId] = adjacencyItem;
    }
    else
    {
      aIt->second.first.push_back(downVertexId);
    }

    prevArcId = arcId;
    arcId = this->Storage->GetNextArcId();
  }

  // now build the super-arcs with deg-2 nodes

  // <vertex,vertex>,<vertex list> (arc, deg2 node list)
  std::vector<std::pair<std::pair<int, int>, std::vector<int> > > globalAdjacency;

  std::map<int, std::pair<std::vector<int>, std::vector<int> > >::iterator aIt;
  aIt = localAdjacency.begin();
  do
  {
    if (!((aIt->second.first.size() == 1) && (aIt->second.second.size() == 1)))
    {
      // not a deg-2 node
      if (!aIt->second.second.empty())
      {
        // start the sweep up
        for (unsigned int i = 0; i < aIt->second.second.size(); i++)
        {
          std::vector<int> deg2List;
          std::map<int, std::pair<std::vector<int>, std::vector<int> > >::iterator nextIt;

          nextIt = localAdjacency.find(aIt->second.second[i]);
          while ((nextIt->second.first.size() == 1) && (nextIt->second.second.size() == 1))
          {
            deg2List.push_back(nextIt->first);
            nextIt = localAdjacency.find(nextIt->second.second[0]);
          }
          globalAdjacency.push_back(std::pair<std::pair<int, int>, std::vector<int> >(
            std::pair<int, int>(aIt->first, nextIt->first), deg2List));
        }
      }
    }
    ++aIt;
  } while (aIt != localAdjacency.end());

  // now cleanup the internal representation
  //int nmyend = 0;
  for (svtkIdType N = 1; N < this->Storage->MainNodeTable.Size; N++)
  {
    // clear the node
    if (this->Storage->GetNode(N)->ArcUpId == ((int)-2))
      continue;

    svtkReebGraph::Implementation::svtkReebNode* n = this->Storage->GetNode(N);

    if (!n->IsFinalized)
    {
      //nmyend++;
      this->Storage->EndVertex(N);
    }
  }

  this->Storage->FlushLabels();

  // now construct the actual graph
  svtkIdType prevNodeId = -1, nodeId = 0;
  while (prevNodeId != nodeId)
  {
    prevNodeId = nodeId;
    nodeId = this->Storage->GetPreviousNodeId();
  }
  prevNodeId = -1;

  svtkVariantArray* vertexProperties = svtkVariantArray::New();
  vertexProperties->SetNumberOfValues(1);

  svtkIdTypeArray* vertexIds = svtkIdTypeArray::New();
  vertexIds->SetName("Vertex Ids");
  GetVertexData()->AddArray(vertexIds);

  std::map<int, int> vMap;
  int vIt = 0;

  while (prevNodeId != nodeId)
  {
    svtkIdType nodeVertexId = this->Storage->GetNodeVertexId(nodeId);
    vMap[nodeVertexId] = vIt;
    vertexProperties->SetValue(0, nodeVertexId);
    AddVertex(vertexProperties);

    prevNodeId = nodeId;
    nodeId = this->Storage->GetNextNodeId();
    vIt++;
  }
  vertexIds->Delete();
  vertexProperties->Delete();

  svtkVariantArray* deg2NodeIds = svtkVariantArray::New();
  deg2NodeIds->SetName("Vertex Ids");
  GetEdgeData()->AddArray(deg2NodeIds);

  for (unsigned int i = 0; i < globalAdjacency.size(); i++)
  {
    std::map<int, int>::iterator downIt, upIt;
    downIt = vMap.find(globalAdjacency[i].first.first);
    upIt = vMap.find(globalAdjacency[i].first.second);

    if ((downIt != vMap.end()) && (upIt != vMap.end()))
    {
      svtkVariantArray* edgeProperties = svtkVariantArray::New();
      svtkIdTypeArray* vertexList = svtkIdTypeArray::New();
      vertexList->SetNumberOfValues(static_cast<svtkIdType>(globalAdjacency[i].second.size()));
      for (unsigned int j = 0; j < globalAdjacency[i].second.size(); j++)
        vertexList->SetValue(j, globalAdjacency[i].second[j]);
      edgeProperties->SetNumberOfValues(1);
      edgeProperties->SetValue(0, vertexList);
      AddEdge(downIt->second, upIt->second, edgeProperties);
      vertexList->Delete();
      edgeProperties->Delete();
    }
  }
  deg2NodeIds->Delete();
}

//----------------------------------------------------------------------------
svtkReebGraph::svtkReebGraph()
{
  this->Storage = new svtkReebGraph::Implementation;
  this->Storage->Parent = this;
}

//----------------------------------------------------------------------------
svtkReebGraph::~svtkReebGraph()
{
  delete this->Storage;
}

//----------------------------------------------------------------------------
void svtkReebGraph::PrintSelf(ostream& os, svtkIndent indent)
{
  svtkObject::PrintSelf(os, indent);
  os << indent << "Reeb graph general statistics:" << endl;
  os << indent << indent << "Number Of Node(s): " << this->Storage->GetNumberOfNodes() << endl;
  os << indent << indent << "Number Of Arc(s): " << this->Storage->GetNumberOfArcs() << endl;
  os << indent << indent
     << "Number Of Connected Component(s): " << this->Storage->GetNumberOfConnectedComponents()
     << endl;
  os << indent << indent << "Number Of Loop(s): " << this->Storage->GetNumberOfLoops() << endl;
}

void svtkReebGraph::PrintNodeData(ostream& os, svtkIndent indent)
{
  svtkIdType arcId = 0, nodeId = 0;
  os << indent << "Node Data:" << endl;
  svtkIdType prevNodeId = -1;

  // roll back to the beginning of the list
  while (prevNodeId != nodeId)
  {
    prevNodeId = nodeId;
    nodeId = this->Storage->GetPreviousNodeId();
  }
  prevNodeId = -1;

  while (prevNodeId != nodeId)
  {
    prevNodeId = nodeId;
    svtkIdList* downArcIdList = svtkIdList::New();
    svtkIdList* upArcIdList = svtkIdList::New();

    this->Storage->GetNodeDownArcIds(nodeId, downArcIdList);
    this->Storage->GetNodeUpArcIds(nodeId, upArcIdList);

    cout << indent << indent << "Node " << nodeId << ":" << endl;
    cout << indent << indent << indent;
    cout << "Vert: " << this->Storage->GetNodeVertexId(nodeId);
    cout << ", Val: " << this->Storage->GetNodeScalarValue(nodeId);
    cout << ", DwA:";
    for (svtkIdType i = 0; i < downArcIdList->GetNumberOfIds(); i++)
      cout << " " << this->Storage->GetArcDownNodeId(downArcIdList->GetId(i));
    cout << ", UpA:";
    for (svtkIdType i = 0; i < upArcIdList->GetNumberOfIds(); i++)
      cout << " " << this->Storage->GetArcUpNodeId(upArcIdList->GetId(i));
    cout << endl;

    downArcIdList->Delete();
    upArcIdList->Delete();
    nodeId = this->Storage->GetNextNodeId();
  }

  os << indent << "Arc Data:" << endl;
  svtkIdType prevArcId = -1;
  arcId = 0;

  // roll back to the beginning of the list
  while (prevArcId != arcId)
  {
    prevArcId = arcId;
    arcId = this->Storage->GetPreviousArcId();
  }
  prevArcId = -1;

  while (prevArcId != arcId)
  {
    prevArcId = arcId;
    cout << indent << indent << "Arc " << arcId << ":" << endl;
    cout << indent << indent << indent;
    cout << "Down: " << this->Storage->GetArcDownNodeId(arcId);
    cout << ", Up: " << this->Storage->GetArcUpNodeId(arcId);
    cout << ", Persistence: "
         << this->Storage->GetNodeScalarValue(this->Storage->GetArcUpNodeId(arcId)) -
        this->Storage->GetNodeScalarValue(this->Storage->GetArcDownNodeId(arcId));
    cout << endl;
    arcId = this->Storage->GetNextArcId();
  }
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::GetNodeDownArcIds(svtkIdType nodeId, svtkIdList* arcIdList)
{
  svtkIdType i = 0;

  if (!arcIdList)
    return;

  arcIdList->Reset();

  for (svtkIdType arcId = this->GetNode(nodeId)->ArcDownId; arcId;
       arcId = this->GetArc(arcId)->ArcDwId1)
  {
    arcIdList->InsertId(i, arcId);
    i++;
  }
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::GetNodeUpArcIds(svtkIdType nodeId, svtkIdList* arcIdList)
{
  svtkIdType i = 0;

  if (!arcIdList)
    return;

  for (svtkIdType arcId = this->GetNode(nodeId)->ArcUpId; arcId;
       arcId = this->GetArc(arcId)->ArcDwId0)
  {
    arcIdList->InsertId(i, arcId);
    i++;
  }
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::FindLoops()
{

  if (this->ArcLoopTable)
  {
    free(this->ArcLoopTable);
    this->ArcLoopTable = nullptr;
    this->LoopNumber = 0;
  }

  this->ConnectedComponentNumber = 0;

  int nstack = 0, mstack = 0;
  int* stack = nullptr;

  char* Ntouch = (char*)malloc(sizeof(char) * this->MainNodeTable.Size);
  char* Atouch = (char*)malloc(sizeof(char) * this->MainArcTable.Size);

  memset(Ntouch, 0, sizeof(char) * this->MainNodeTable.Size);

  for (int Node = 1; Node < this->MainNodeTable.Size; Node++)
  {
    // check that the node is clear
    if (this->GetNode(Node)->ArcUpId == ((int)-2))
      continue;

    if (!Ntouch[Node])
    {
      ++(this->ConnectedComponentNumber);

      memset(Atouch, 0, sizeof(bool) * this->MainArcTable.Size);

      Ntouch[Node] = 1;
      nstack = 0;
      svtkReebGraphStackPush(Node);

      while (svtkReebGraphStackSize())
      {

        int N = svtkReebGraphStackTop();
        svtkReebGraphStackPop();

        for (int dir = 0; dir <= 1; dir++)
        {
          for (int A = (!dir) ? (this->GetNode(N)->ArcDownId) : (this->GetNode(N)->ArcUpId); A;
               A = (!dir) ? (this->GetArc(A)->ArcDwId1) : (this->GetArc(A)->ArcDwId0))
          {
            int M = (!dir) ? (this->GetArc(A)->NodeId0) : (this->GetArc(A)->NodeId1);

            if (Atouch[A])
              continue;

            if (!Ntouch[M])
            {
              svtkReebGraphStackPush(M);
            }
            else
            {
              this->LoopNumber++;
              this->ArcLoopTable =
                (svtkIdType*)realloc(this->ArcLoopTable, sizeof(svtkIdType) * this->LoopNumber);
              this->ArcLoopTable[this->LoopNumber - 1] = A;
            }

            Atouch[A] = 1;
            Ntouch[M] = 1;
          }
        }
      }
    }
  }

  free(stack);
  free(Ntouch);
  free(Atouch);
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::AddMeshVertex(svtkIdType vertexId, double scalar)
{
  static bool firstVertex = true;

  ScalarField[vertexId] = scalar;

  svtkIdType N0;
  ResizeMainNodeTable(1);

  // create a new node in the graph...
  N0 = this->MainNodeTable.FreeZone;
  this->MainNodeTable.FreeZone = this->GetNode(N0)->ArcDownId;
  ++(this->MainNodeTable.Number);
  memset(this->GetNode(N0), 0, sizeof(svtkReebNode));

  svtkReebNode* node = this->GetNode(N0);
  node->VertexId = vertexId;
  node->Value = scalar;
  node->ArcDownId = 0;
  node->ArcUpId = 0;
  node->IsFinalized = false;

  if (firstVertex)
  {
    this->MinimumScalarValue = node->Value;
    this->MaximumScalarValue = node->Value;
  }
  else
  {
    if (node->Value > this->MaximumScalarValue)
      this->MaximumScalarValue = node->Value;
    if (node->Value < this->MinimumScalarValue)
      this->MinimumScalarValue = node->Value;
  }
  firstVertex = false;

  return N0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindDwLabel(svtkIdType nodeId, svtkReebLabelTag label)
{
  for (svtkIdType arcId = this->GetNode(nodeId)->ArcDownId; arcId;
       arcId = this->GetArc(arcId)->ArcDwId1)
  {
    for (svtkIdType labelId = this->GetArc(arcId)->LabelId0; labelId;
         labelId = this->GetLabel(labelId)->HNext)
    {
      if (this->GetLabel(labelId)->label == label)
        return labelId;
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::FindUpLabel(svtkIdType nodeId, svtkReebLabelTag label)
{
  for (svtkIdType arcId = this->GetNode(nodeId)->ArcUpId; arcId;
       arcId = this->GetArc(arcId)->ArcDwId0)
  {
    for (svtkIdType labelId = this->GetArc(arcId)->LabelId0; labelId;
         labelId = this->GetLabel(labelId)->HNext)
    {
      if (this->GetLabel(labelId)->label == label)
        return labelId;
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::ResizeMainArcTable(int newSize)
{
  int oldsize, i;
  if ((this->MainArcTable.Size - this->MainArcTable.Number) < newSize)
  {
    oldsize = this->MainArcTable.Size;
    if (!this->MainArcTable.Size)
      this->MainArcTable.Size = newSize;
    while ((this->MainArcTable.Size - this->MainArcTable.Number) < newSize)
      this->MainArcTable.Size <<= 1;

    this->MainArcTable.Buffer =
      (svtkReebArc*)realloc(this->MainArcTable.Buffer, sizeof(svtkReebArc) * this->MainArcTable.Size);
    for (i = oldsize; i < this->MainArcTable.Size - 1; i++)
    {
      this->GetArc(i)->LabelId0 = i + 1;
      // clear arc
      this->GetArc(i)->LabelId1 = ((int)-2);
    }

    this->GetArc(i)->LabelId0 = this->MainArcTable.FreeZone;
    // clear arc
    this->GetArc(i)->LabelId1 = ((int)-2);
    this->MainArcTable.FreeZone = oldsize;
  }
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::ResizeMainLabelTable(int newSize)
{
  int oldsize, i;
  if ((this->MainLabelTable.Size - this->MainLabelTable.Number) < newSize)
  {
    oldsize = this->MainLabelTable.Size;
    if (!this->MainLabelTable.Size)
      this->MainLabelTable.Size = newSize;
    while ((this->MainLabelTable.Size - this->MainLabelTable.Number) < newSize)
      this->MainLabelTable.Size <<= 1;

    this->MainLabelTable.Buffer = (svtkReebLabel*)realloc(
      this->MainLabelTable.Buffer, sizeof(svtkReebLabel) * this->MainLabelTable.Size);

    for (i = oldsize; i < this->MainLabelTable.Size - 1; i++)
    {
      this->GetLabel(i)->ArcId = i + 1;
      this->GetLabel(i)->HNext = ((int)-2);
    }

    this->GetLabel(i)->ArcId = this->MainLabelTable.FreeZone;
    this->GetLabel(i)->HNext = ((int)-2);
    this->MainLabelTable.FreeZone = oldsize;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::AddPath(
  int nodeNumber, svtkIdType* nodeOffset, svtkReebLabelTag label)
{
  svtkIdType i, Lprev, Ret = 0;

  this->ResizeMainArcTable(nodeNumber - 1);

  if (label)
    ResizeMainLabelTable(nodeNumber - 1);

  Lprev = 0;
  for (i = 0; i < (nodeNumber - 1); i++)
  {
    svtkIdType N0 = nodeOffset[i];
    svtkIdType N1 = nodeOffset[i + 1];

    // create a new arc in the graph
    int A = this->MainArcTable.FreeZone;
    this->MainArcTable.FreeZone = this->GetArc(A)->LabelId0;
    ++(this->MainArcTable.Number);
    memset(this->GetArc(A), 0, sizeof(svtkReebArc));
    svtkReebArc* arc = this->GetArc(A);

    int L = 0;

    if (!Ret)
      Ret = A;

    if (label)
    {
      svtkReebLabel* temporaryLabel;

      // create a new label in the graph
      L = this->MainLabelTable.FreeZone;
      this->MainLabelTable.FreeZone = this->GetLabel(L)->ArcId;
      ++(this->MainLabelTable.Number);
      memset(this->GetLabel(L), 0, sizeof(svtkReebLabel));
      temporaryLabel = this->GetLabel(L);

      temporaryLabel->ArcId = A;
      temporaryLabel->label = label;
      temporaryLabel->VPrev = Lprev;
    }

    arc->NodeId0 = N0;
    arc->NodeId1 = N1;
    arc->LabelId0 = arc->LabelId1 = L;

    svtkReebGraphAddUpArc(this, N0, A);
    svtkReebGraphAddDownArc(this, N1, A);

    if (label)
    {
      if (Lprev)
        this->GetLabel(Lprev)->VNext = L;
      Lprev = L;
    }
  }

  return Ret;
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::Collapse(svtkIdType startingNode, svtkIdType endingNode,
  svtkReebLabelTag startingLabel, svtkReebLabelTag endingLabel)
{

  int L0, L0n, L1, L1n;
  int cont[3] = { 0, 0, 0 }, Case;

  if (startingNode == endingNode)
    return;

  svtkReebNode* nstart = this->GetNode(startingNode);
  svtkReebNode* nend = this->GetNode(endingNode);

  if (!svtkReebGraphIsSmaller(this, startingNode, endingNode, nstart, nend))
  {
    svtkReebGraphSwapVars(int, startingNode, endingNode);
    svtkReebGraphSwapVars(svtkReebNode*, nstart, nend);
  }

  L0 = FindUpLabel(startingNode, startingLabel);
  L1 = FindUpLabel(startingNode, endingLabel);

  while (1)
  {
    int A0 = this->GetLabel(L0)->ArcId;
    svtkReebArc* a0 = this->GetArc(A0);
    int A1 = this->GetLabel(L1)->ArcId;
    svtkReebArc* a1 = this->GetArc(A1);

    svtkReebNode* down0 = this->GetNode(a0->NodeId0);
    svtkReebNode* up0 = this->GetNode(a0->NodeId1);
    svtkReebNode* up1 = this->GetNode(a1->NodeId1);

    /* it is the same arc, no semplification is done */
    if (A0 == A1)
    {
      Case = 0;
      L0n = this->GetLabel(L0)->VNext;
      L1n = this->GetLabel(L1)->VNext;
    }
    /* there are two arcs connecting the same start-end node */
    else if (A0 != A1 && a0->NodeId1 == a1->NodeId1)
    {
      Case = 1;
      svtkReebGraphRemoveUpArc(this, a0->NodeId0, A1);
      svtkReebGraphRemoveDownArc(this, a0->NodeId1, A1);

      for (int Lcur = this->GetArc(A1)->LabelId0; Lcur; Lcur = this->GetLabel(Lcur)->HNext)
      {
        this->GetLabel(Lcur)->ArcId = A0;
      }

      this->GetLabel(this->GetArc(A1)->LabelId0)->HPrev = this->GetArc(A0)->LabelId1;
      this->GetLabel(this->GetArc(A0)->LabelId1)->HNext = this->GetArc(A1)->LabelId0;
      this->GetArc(A0)->LabelId1 = this->GetArc(A1)->LabelId1;

      this->GetArc(A1)->LabelId0 = 0;
      this->GetArc(A1)->LabelId1 = 0;

      // delete the arc from the graph...
      this->GetArc(A1)->LabelId1 = ((int)-2);
      this->GetArc(A1)->LabelId0 = this->MainArcTable.FreeZone;
      this->MainArcTable.FreeZone = (A1);
      --(this->MainArcTable.Number);

      L0n = this->GetLabel(L0)->VNext;
      L1n = this->GetLabel(L1)->VNext;
    }
    else
    {
      if (historyOn)
      {
        svtkReebCancellation c;
        int downVertex, middleVertex, upVertex;
        downVertex = down0->VertexId;
        middleVertex = up0->VertexId;
        upVertex = up1->VertexId;
        c.removedArcs.push_back(std::pair<int, int>(downVertex, upVertex));
        c.insertedArcs.push_back(std::pair<int, int>(downVertex, middleVertex));
        c.insertedArcs.push_back(std::pair<int, int>(middleVertex, upVertex));
        this->cancellationHistory.push_back(c);
      }
      // a more complicate situation, collapse reaching the less ending point of
      // the arcs.
      Case = 2;
      {
        svtkReebNode* a0n1 = this->GetNode(a0->NodeId1);
        svtkReebNode* a1n1 = this->GetNode(a1->NodeId1);
        if (!svtkReebGraphIsSmaller(this, a0->NodeId1, a1->NodeId1, a0n1, a1n1))
        {
          svtkReebGraphSwapVars(int, A0, A1);
          svtkReebGraphSwapVars(int, L0, L1);
          svtkReebGraphSwapVars(svtkReebArc*, a0, a1);
        }
      }

      svtkReebGraphRemoveUpArc(this, a0->NodeId0, A1);
      a1->NodeId0 = a0->NodeId1;
      svtkReebGraphAddUpArc(this, a0->NodeId1, A1);

      //"replicate" labels from A1 to A0
      for (int Lcur = this->GetArc(A1)->LabelId0; Lcur; Lcur = this->GetLabel(Lcur)->HNext)
      {
        int Lnew;
        ResizeMainLabelTable(1);

        // create a new label in the graph
        Lnew = this->MainLabelTable.FreeZone;
        this->MainLabelTable.FreeZone = this->GetLabel(Lnew)->ArcId;
        ++(this->MainLabelTable.Number);
        memset(this->GetLabel(Lnew), 0, sizeof(svtkReebLabel));

        svtkReebLabel* lnew = this->GetLabel(Lnew);
        svtkReebLabel* lcur = this->GetLabel(Lcur);
        lnew->ArcId = A0;
        lnew->VPrev = lcur->VPrev;

        if (lcur->VPrev)
          this->GetLabel(lcur->VPrev)->VNext = Lnew;

        lcur->VPrev = Lnew;
        lnew->VNext = Lcur;
        lnew->label = lcur->label;

        lnew->HNext = 0;
        lnew->HPrev = this->GetArc(A0)->LabelId1;
        this->GetLabel(this->GetArc(A0)->LabelId1)->HNext = Lnew;

        this->GetArc(A0)->LabelId1 = Lnew;
      }

      L0n = this->GetLabel(L0)->VNext;
      L1n = L1;
    }

    ++cont[Case];

    int N0 = a0->NodeId0;
    svtkReebNode* n0 = this->GetNode(N0);

    if (n0->IsFinalized && svtkReebGraphIsRegular(this, n0))
    {
      if (historyOn)
      {
        svtkReebArc *up = this->GetArc(n0->ArcUpId), *down = this->GetArc(n0->ArcDownId);

        svtkReebCancellation c;
        int v0, v1, v2, v3;

        v0 = this->GetNode(up->NodeId0)->VertexId;
        v1 = this->GetNode(up->NodeId1)->VertexId;
        v2 = this->GetNode(down->NodeId0)->VertexId;
        v3 = this->GetNode(down->NodeId1)->VertexId;

        c.removedArcs.push_back(std::pair<int, int>(v0, v1));
        c.removedArcs.push_back(std::pair<int, int>(v2, v3));
        c.insertedArcs.push_back(std::pair<int, int>(v2, v1));
        this->cancellationHistory.push_back(c);
      }
      this->CollapseVertex(N0, n0);
    }

    /* end condition */
    if (a0->NodeId1 == endingNode)
    {
      svtkReebNode* nendNode = this->GetNode(endingNode);

      if (nendNode->IsFinalized && svtkReebGraphIsRegular(this, nendNode))
      {
        if (historyOn)
        {
          svtkReebArc *up = this->GetArc(this->GetNode(endingNode)->ArcUpId),
                     *down = this->GetArc(this->GetNode(endingNode)->ArcDownId);

          svtkReebCancellation c;

          int v0, v1, v2, v3;
          v0 = this->GetNode(up->NodeId0)->VertexId;
          v1 = this->GetNode(up->NodeId1)->VertexId;
          v2 = this->GetNode(down->NodeId0)->VertexId;
          v3 = this->GetNode(down->NodeId1)->VertexId;

          c.removedArcs.push_back(std::pair<int, int>(v0, v1));
          c.removedArcs.push_back(std::pair<int, int>(v2, v3));
          c.insertedArcs.push_back(std::pair<int, int>(v2, v1));
          this->cancellationHistory.push_back(c);
        }
        this->CollapseVertex(endingNode, nendNode);
      }

      return;
    }

    L0 = L0n;
    L1 = L1n;
  }
}

void svtkReebGraph::Implementation::SimplifyLabels(
  const svtkIdType nodeId, svtkReebLabelTag onlyLabel, bool goDown, bool goUp)
{
  int A, L, Lnext;
  svtkReebLabel* l;
  svtkReebNode* n = this->GetNode(nodeId);

  // I remove all Labels (paths) which start from me
  if (goDown)
  {
    int Anext;
    for (A = n->ArcDownId; A; A = Anext)
    {
      Anext = this->GetArc(A)->ArcDwId1;
      for (L = this->GetArc(A)->LabelId0; L; L = Lnext)
      {
        Lnext = this->GetLabel(L)->HNext;

        if (!(l = this->GetLabel(L))->VNext) //...starts from me!
        {
          if (!onlyLabel || onlyLabel == this->GetLabel(L)->label)
          {
            int Lprev;
            for (int Lcur = L; Lcur; Lcur = Lprev)
            {
              svtkReebLabel* lcur = this->GetLabel(Lcur);
              Lprev = lcur->VPrev;
              int CurA = lcur->ArcId;
              if (lcur->HPrev)
                this->GetLabel(lcur->HPrev)->HNext = lcur->HNext;
              else
                this->GetArc(CurA)->LabelId0 = lcur->HNext;
              if (lcur->HNext)
                this->GetLabel(lcur->HNext)->HPrev = lcur->HPrev;
              else
                this->GetArc(CurA)->LabelId1 = lcur->HPrev;

              // delete the label
              this->GetLabel(Lcur)->HNext = ((int)-2);
              this->GetLabel(Lcur)->ArcId = this->MainLabelTable.FreeZone;
              this->MainLabelTable.FreeZone = (Lcur);
              --(this->MainLabelTable.Number);
            }
          }
        }
      }
    }
  }

  // Remove all Labels (paths) which start from here

  if (goUp && !(this->GetNode(nodeId)->ArcUpId == ((int)-2)))
  {
    int Anext;
    for (A = n->ArcUpId; A; A = Anext)
    {
      Anext = this->GetArc(A)->ArcDwId0;
      for (L = this->GetArc(A)->LabelId0; L; L = Lnext)
      {
        Lnext = this->GetLabel(L)->HNext;

        if (!(l = this->GetLabel(L))->VPrev) //...starts from me!
        {
          if (!onlyLabel || onlyLabel == this->GetLabel(L)->label)
          {
            int myLnext;
            for (int Lcur = L; Lcur; Lcur = myLnext)
            {
              svtkReebLabel* lcur = this->GetLabel(Lcur);
              myLnext = lcur->VNext;
              int CurA = lcur->ArcId;
              svtkReebArc* cura = this->GetArc(CurA);
              if (lcur->HPrev)
              {
                this->GetLabel(lcur->HPrev)->HNext = lcur->HNext;
              }
              else
              {
                cura->LabelId0 = lcur->HNext;
              }
              if (lcur->HNext)
              {
                this->GetLabel(lcur->HNext)->HPrev = lcur->HPrev;
              }
              else
              {
                cura->LabelId1 = lcur->HPrev;
              }

              // delete the label...
              this->GetLabel(Lcur)->HNext = ((int)-2);
              this->GetLabel(Lcur)->ArcId = this->MainLabelTable.FreeZone;
              this->MainLabelTable.FreeZone = (Lcur);
              --(this->MainLabelTable.Number);
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkReebGraph::Implementation::EndVertex(const svtkIdType N)
{
  svtkReebNode* n = this->GetNode(N);

  n->IsFinalized = true;

  if (!(this->GetNode(N)->ArcUpId == ((int)-2)))
  {
    this->SimplifyLabels(N);

    if (!(this->GetNode(N)->ArcUpId == ((int)-2)))
    {
      // special case for regular point. A node is regular if it has one
      // arc down and one arc up. In this case it can disappear

      if (svtkReebGraphIsRegular(this, n))
      {
        this->CollapseVertex(N, n);
      }
    }
  }
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::AddMeshTetrahedron(svtkIdType vertex0Id, double f0,
  svtkIdType vertex1Id, double f1, svtkIdType vertex2Id, double f2, svtkIdType vertex3Id, double f3)
{
  svtkIdType vertex0, vertex1, vertex2, vertex3;

  vertex0 = this->VertexStream[vertex0Id];
  vertex1 = this->VertexStream[vertex1Id];
  vertex2 = this->VertexStream[vertex2Id];
  vertex3 = this->VertexStream[vertex3Id];

  int N0 = this->VertexMap[vertex0];
  int N1 = this->VertexMap[vertex1];
  int N2 = this->VertexMap[vertex2];
  int N3 = this->VertexMap[vertex3];

  // Consistency less check
  if (f3 < f2 || (f3 == f2 && vertex3 < vertex2))
  {
    svtkReebGraphSwapVars(int, vertex2, vertex3);
    svtkReebGraphSwapVars(int, N2, N3);
    svtkReebGraphSwapVars(double, f2, f3);
  }
  if (f2 < f1 || (f2 == f1 && vertex2 < vertex1))
  {
    svtkReebGraphSwapVars(int, vertex1, vertex2);
    svtkReebGraphSwapVars(int, N1, N2);
    svtkReebGraphSwapVars(double, f1, f2);
  }
  if (f1 < f0 || (f1 == f0 && vertex1 < vertex0))
  {
    svtkReebGraphSwapVars(int, vertex0, vertex1);
    svtkReebGraphSwapVars(int, N0, N1);
    svtkReebGraphSwapVars(double, f0, f1);
  }
  if (f3 < f2 || (f3 == f2 && vertex3 < vertex2))
  {
    svtkReebGraphSwapVars(int, vertex2, vertex3);
    svtkReebGraphSwapVars(int, N2, N3);
    svtkReebGraphSwapVars(double, f2, f3);
  }
  if (f2 < f1 || (f2 == f1 && vertex2 < vertex1))
  {
    svtkReebGraphSwapVars(int, vertex1, vertex2);
    svtkReebGraphSwapVars(int, N1, N2);
    svtkReebGraphSwapVars(double, f1, f2);
  }
  if (f3 < f2 || (f3 == f2 && vertex3 < vertex2))
  {
    svtkReebGraphSwapVars(int, vertex2, vertex3);
    svtkReebGraphSwapVars(int, N2, N3);
    svtkReebGraphSwapVars(double, f2, f3);
  }

  svtkIdType t0[] = { vertex0, vertex1, vertex2 }, t1[] = { vertex0, vertex1, vertex3 },
            t2[] = { vertex0, vertex2, vertex3 }, t3[] = { vertex1, vertex2, vertex3 };
  svtkIdType* cellIds[4];
  cellIds[0] = t0;
  cellIds[1] = t1;
  cellIds[2] = t2;
  cellIds[3] = t3;

  for (int i = 0; i < 3; i++)
  {
    int n0 = this->VertexMap[cellIds[i][0]], n1 = this->VertexMap[cellIds[i][1]],
        n2 = this->VertexMap[cellIds[i][2]];

    svtkReebLabelTag Label01 =
      ((svtkReebLabelTag)cellIds[i][0]) | (((svtkReebLabelTag)cellIds[i][1]) << 32);
    svtkReebLabelTag Label12 =
      ((svtkReebLabelTag)cellIds[i][1]) | (((svtkReebLabelTag)cellIds[i][2]) << 32);
    svtkReebLabelTag Label02 =
      ((svtkReebLabelTag)cellIds[i][0]) | (((svtkReebLabelTag)cellIds[i][2]) << 32);

    if (!this->FindUpLabel(n0, Label01))
    {
      svtkIdType N01[] = { n0, n1 };
      this->AddPath(2, N01, Label01);
    }
    if (!this->FindUpLabel(n1, Label12))
    {
      svtkIdType N12[] = { n1, n2 };
      this->AddPath(2, N12, Label12);
    }
    if (!this->FindUpLabel(n0, Label02))
    {
      svtkIdType N02[] = { n0, n2 };
      this->AddPath(2, N02, Label02);
    }

    this->Collapse(n0, n1, Label01, Label02);
    this->Collapse(n1, n2, Label12, Label02);
  }

  if (!(--(this->TriangleVertexMap[vertex0])))
    this->EndVertex(N0);
  if (!(--(this->TriangleVertexMap[vertex1])))
    this->EndVertex(N1);
  if (!(--(this->TriangleVertexMap[vertex2])))
    this->EndVertex(N2);
  if (!(--(this->TriangleVertexMap[vertex3])))
    this->EndVertex(N3);

  return 1;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::AddMeshTriangle(
  svtkIdType vertex0Id, double f0, svtkIdType vertex1Id, double f1, svtkIdType vertex2Id, double f2)
{

  int vertex0 = this->VertexStream[vertex0Id], vertex1 = this->VertexStream[vertex1Id],
      vertex2 = this->VertexStream[vertex2Id];

  int N0 = this->VertexMap[vertex0];
  int N1 = this->VertexMap[vertex1];
  int N2 = this->VertexMap[vertex2];

  // Consistency less check
  if (f2 < f1 || (f2 == f1 && vertex2 < vertex1))
  {
    svtkReebGraphSwapVars(int, vertex1, vertex2);
    svtkReebGraphSwapVars(int, N1, N2);
    svtkReebGraphSwapVars(double, f1, f2);
  }
  if (f1 < f0 || (f1 == f0 && vertex1 < vertex0))
  {
    svtkReebGraphSwapVars(int, vertex0, vertex1);
    svtkReebGraphSwapVars(int, N0, N1);
    // there is a useless assignment to f0 below
    // keeping for clarity and consistency
    svtkReebGraphSwapVars(double, f0, f1);
  }
  if (f2 < f1 || (f2 == f1 && vertex2 < vertex1))
  {
    svtkReebGraphSwapVars(int, vertex1, vertex2);
    svtkReebGraphSwapVars(int, N1, N2);
    // there is a useless assignment below
    // keeping for clarity and consistency
    svtkReebGraphSwapVars(double, f1, f2);
  }

  svtkReebLabelTag Label01 = ((svtkReebLabelTag)vertex0) | (((svtkReebLabelTag)vertex1) << 32);
  svtkReebLabelTag Label12 = ((svtkReebLabelTag)vertex1) | (((svtkReebLabelTag)vertex2) << 32);
  svtkReebLabelTag Label02 = ((svtkReebLabelTag)vertex0) | (((svtkReebLabelTag)vertex2) << 32);

  if (!this->FindUpLabel(N0, Label01))
  {
    svtkIdType N01[] = { N0, N1 };
    this->AddPath(2, N01, Label01);
  }
  if (!this->FindUpLabel(N1, Label12))
  {
    svtkIdType N12[] = { N1, N2 };
    this->AddPath(2, N12, Label12);
  }
  if (!this->FindUpLabel(N0, Label02))
  {
    svtkIdType N02[] = { N0, N2 };
    this->AddPath(2, N02, Label02);
  }

  this->Collapse(N0, N1, Label01, Label02);
  this->Collapse(N1, N2, Label12, Label02);

  if (!(--(this->TriangleVertexMap[vertex0])))
    this->EndVertex(N0);
  if (!(--(this->TriangleVertexMap[vertex1])))
    this->EndVertex(N1);
  if (!(--(this->TriangleVertexMap[vertex2])))
    this->EndVertex(N2);

  return 1;
}

//----------------------------------------------------------------------------
int svtkReebGraph::StreamTetrahedron(svtkIdType vertex0Id, double scalar0, svtkIdType vertex1Id,
  double scalar1, svtkIdType vertex2Id, double scalar2, svtkIdType vertex3Id, double scalar3)
{
  return this->Storage->StreamTetrahedron(
    vertex0Id, scalar0, vertex1Id, scalar1, vertex2Id, scalar2, vertex3Id, scalar3);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::StreamTetrahedron(svtkIdType vertex0Id, double scalar0,
  svtkIdType vertex1Id, double scalar1, svtkIdType vertex2Id, double scalar2, svtkIdType vertex3Id,
  double scalar3)
{
  if (!this->VertexMapAllocatedSize)
  {
    // first allocate an arbitrary size
    this->VertexMapAllocatedSize = svtkReebGraphInitialStreamSize;
    this->VertexMap = (svtkIdType*)malloc(sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    memset(this->VertexMap, 0, sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    this->VertexStream.clear();
  }
  else if (this->VertexMapSize >= this->VertexMapAllocatedSize - 4)
  {
    int oldSize = this->VertexMapAllocatedSize;
    this->VertexMapAllocatedSize <<= 1;
    this->VertexMap =
      (svtkIdType*)realloc(this->VertexMap, sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    for (int i = oldSize; i < this->VertexMapAllocatedSize - 1; i++)
      this->VertexMap[i] = 0;
  }

  // same thing with the triangle map
  if (!this->TriangleVertexMapAllocatedSize)
  {
    // first allocate an arbitrary size
    this->TriangleVertexMapAllocatedSize = svtkReebGraphInitialStreamSize;
    this->TriangleVertexMap = (int*)malloc(sizeof(int) * this->TriangleVertexMapAllocatedSize);
    memset(this->TriangleVertexMap, 0, sizeof(int) * this->TriangleVertexMapAllocatedSize);
  }
  else if (this->TriangleVertexMapSize >= this->TriangleVertexMapAllocatedSize - 4)
  {
    int oldSize = this->TriangleVertexMapAllocatedSize;
    this->TriangleVertexMapAllocatedSize <<= 1;
    this->TriangleVertexMap =
      (int*)realloc(this->TriangleVertexMap, sizeof(int) * this->TriangleVertexMapAllocatedSize);
    for (int i = oldSize; i < this->TriangleVertexMapAllocatedSize - 1; i++)
      this->TriangleVertexMap[i] = 0;
  }

  // Add the vertices to the stream
  std::map<int, int>::iterator sIter;

  // vertex0
  sIter = this->VertexStream.find(vertex0Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex0Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex0Id, scalar0);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  // vertex1
  sIter = this->VertexStream.find(vertex1Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex1Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex1Id, scalar1);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  // vertex2
  sIter = this->VertexStream.find(vertex2Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex2Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex2Id, scalar2);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  // vertex3
  sIter = this->VertexStream.find(vertex3Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex3Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex3Id, scalar3);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  this->AddMeshTetrahedron(
    vertex0Id, scalar0, vertex1Id, scalar1, vertex2Id, scalar2, vertex3Id, scalar3);

  return 0;
}

//----------------------------------------------------------------------------
int svtkReebGraph::StreamTriangle(svtkIdType vertex0Id, double scalar0, svtkIdType vertex1Id,
  double scalar1, svtkIdType vertex2Id, double scalar2)
{
  return this->Storage->StreamTriangle(vertex0Id, scalar0, vertex1Id, scalar1, vertex2Id, scalar2);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::StreamTriangle(svtkIdType vertex0Id, double scalar0,
  svtkIdType vertex1Id, double scalar1, svtkIdType vertex2Id, double scalar2)
{
  if (!this->VertexMapAllocatedSize)
  {
    // first allocate an arbitrary size
    this->VertexMapAllocatedSize = svtkReebGraphInitialStreamSize;
    this->VertexMap = (svtkIdType*)malloc(sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    memset(this->VertexMap, 0, sizeof(svtkIdType) * this->VertexMapAllocatedSize);
  }
  else if (this->VertexMapSize >= this->VertexMapAllocatedSize - 3)
  {
    int oldSize = this->VertexMapAllocatedSize;
    this->VertexMapAllocatedSize <<= 1;
    this->VertexMap =
      (svtkIdType*)realloc(this->VertexMap, sizeof(svtkIdType) * this->VertexMapAllocatedSize);
    for (int i = oldSize; i < this->VertexMapAllocatedSize - 1; i++)
      this->VertexMap[i] = 0;
  }

  // same thing with the triangle map
  if (!this->TriangleVertexMapAllocatedSize)
  {
    // first allocate an arbitrary size
    this->TriangleVertexMapAllocatedSize = svtkReebGraphInitialStreamSize;
    this->TriangleVertexMap = (int*)malloc(sizeof(int) * this->TriangleVertexMapAllocatedSize);
    memset(this->TriangleVertexMap, 0, sizeof(int) * this->TriangleVertexMapAllocatedSize);
  }
  else if (this->TriangleVertexMapSize >= this->TriangleVertexMapAllocatedSize - 3)
  {
    int oldSize = this->TriangleVertexMapAllocatedSize;
    this->TriangleVertexMapAllocatedSize <<= 1;
    this->TriangleVertexMap =
      (int*)realloc(this->TriangleVertexMap, sizeof(int) * this->TriangleVertexMapAllocatedSize);
    for (int i = oldSize; i < this->TriangleVertexMapAllocatedSize - 1; i++)
      this->TriangleVertexMap[i] = 0;
  }

  // Add the vertices to the stream
  std::map<int, int>::iterator sIter;

  // vertex0
  sIter = this->VertexStream.find(vertex0Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex0Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex0Id, scalar0);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  // vertex1
  sIter = this->VertexStream.find(vertex1Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex1Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex1Id, scalar1);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  // vertex2
  sIter = this->VertexStream.find(vertex2Id);
  if (sIter == this->VertexStream.end())
  {
    // this vertex hasn't been streamed yet, let's add it
    this->VertexStream[vertex2Id] = this->VertexMapSize;
    this->VertexMap[this->VertexMapSize] = this->AddMeshVertex(vertex2Id, scalar2);
    this->VertexMapSize++;
    this->TriangleVertexMapSize++;
  }

  this->AddMeshTriangle(vertex0Id, scalar0, vertex1Id, scalar1, vertex2Id, scalar2);

  return 0;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkPolyData* mesh, svtkDataArray* scalarField)
{
  for (svtkIdType i = 0; i < mesh->GetNumberOfCells(); i++)
  {
    svtkCell* triangle = mesh->GetCell(i);
    svtkIdList* trianglePointList = triangle->GetPointIds();
    if (trianglePointList->GetNumberOfIds() != 3)
      return svtkReebGraph::ERR_NOT_A_SIMPLICIAL_MESH;
    StreamTriangle(trianglePointList->GetId(0),
      scalarField->GetComponent(trianglePointList->GetId(0), 0), trianglePointList->GetId(1),
      scalarField->GetComponent(trianglePointList->GetId(1), 0), trianglePointList->GetId(2),
      scalarField->GetComponent(trianglePointList->GetId(2), 0));
  }

  this->Storage->inputMesh = mesh;
  this->Storage->inputScalarField = scalarField;

  this->CloseStream();

  return 0;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkUnstructuredGrid* mesh, svtkDataArray* scalarField)
{
  for (svtkIdType i = 0; i < mesh->GetNumberOfCells(); i++)
  {
    svtkCell* tet = mesh->GetCell(i);
    svtkIdList* tetPointList = tet->GetPointIds();
    if (tetPointList->GetNumberOfIds() != 4)
      return svtkReebGraph::ERR_NOT_A_SIMPLICIAL_MESH;
    StreamTetrahedron(tetPointList->GetId(0), scalarField->GetComponent(tetPointList->GetId(0), 0),
      tetPointList->GetId(1), scalarField->GetComponent(tetPointList->GetId(1), 0),
      tetPointList->GetId(2), scalarField->GetComponent(tetPointList->GetId(2), 0),
      tetPointList->GetId(3), scalarField->GetComponent(tetPointList->GetId(3), 0));
  }

  this->Storage->inputMesh = mesh;
  this->Storage->inputScalarField = scalarField;

  this->CloseStream();

  return 0;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::GetNumberOfArcs()
{
  if (!this->ArcNumber)
    for (svtkIdType arcId = 1; arcId < this->MainArcTable.Size; arcId++)
    {
      // check if arc is cleared
      if (!(this->GetArc(arcId)->LabelId1 == ((int)-2)))
        this->ArcNumber++;
    }

  return this->ArcNumber;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::GetNumberOfConnectedComponents()
{
  if (!this->ArcLoopTable)
    this->FindLoops();
  return this->ConnectedComponentNumber;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::GetNumberOfNodes()
{
  if (!this->NodeNumber)
    for (svtkIdType nodeId = 1; nodeId < this->MainNodeTable.Size; nodeId++)
    {
      // check if node is cleared
      if (!(this->GetNode(nodeId)->ArcUpId == ((int)-2)))
        this->NodeNumber++;
    }

  return this->NodeNumber;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetNextNodeId()
{
  for (svtkIdType nodeId = this->currentNodeId + 1; nodeId < this->MainNodeTable.Size; nodeId++)
  {
    // check if node is cleared
    if (!(this->GetNode(nodeId)->ArcUpId == ((int)-2)))
    {
      this->currentNodeId = nodeId;
      return this->currentNodeId;
    }
  }

  return this->currentNodeId;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetPreviousNodeId()
{
  if (!this->currentNodeId)
  {
    return this->GetNextNodeId();
  }

  for (svtkIdType nodeId = this->currentNodeId - 1; nodeId > 0; nodeId--)
  {
    // check if node is cleared
    if (!(this->GetNode(nodeId)->ArcUpId == ((int)-2)))
    {
      this->currentNodeId = nodeId;
      return this->currentNodeId;
    }
  }

  return this->currentNodeId;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetNextArcId()
{
  for (svtkIdType arcId = this->currentArcId + 1; arcId < this->MainArcTable.Size; arcId++)
  {
    // check if arc is cleared
    if (!(this->GetArc(arcId)->LabelId1 == ((int)-2)))
    {
      this->currentArcId = arcId;
      return this->currentArcId;
    }
  }

  return this->currentArcId;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetPreviousArcId()
{
  if (!this->currentArcId)
  {
    return this->GetNextArcId();
  }

  for (svtkIdType arcId = this->currentArcId - 1; arcId > 0; arcId--)
  {
    // check if arc is cleared
    if (!(this->GetArc(arcId)->LabelId1 == ((int)-2)))
    {
      this->currentArcId = arcId;
      return this->currentArcId;
    }
  }

  return this->currentArcId;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetArcDownNodeId(svtkIdType arcId)
{
  return (this->GetArc(arcId))->NodeId0;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetArcUpNodeId(svtkIdType arcId)
{
  return (this->GetArc(arcId))->NodeId1;
}

//----------------------------------------------------------------------------
double svtkReebGraph::Implementation::GetNodeScalarValue(svtkIdType nodeId)
{
  return (this->GetNode(nodeId))->Value;
}

//----------------------------------------------------------------------------
svtkIdType svtkReebGraph::Implementation::GetNodeVertexId(svtkIdType nodeId)
{
  return (this->GetNode(nodeId))->VertexId;
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkPolyData* mesh, svtkIdType scalarFieldId)
{
  svtkPointData* pointData = mesh->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(scalarFieldId);

  if (!scalarField)
    return svtkReebGraph::ERR_NO_SUCH_FIELD;

  return this->Build(mesh, scalarField);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkUnstructuredGrid* mesh, svtkIdType scalarFieldId)
{
  svtkPointData* pointData = mesh->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(scalarFieldId);

  if (!scalarField)
    return svtkReebGraph::ERR_NO_SUCH_FIELD;

  return this->Build(mesh, scalarField);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkPolyData* mesh, const char* scalarFieldName)
{
  int scalarFieldId = 0;

  svtkPointData* pointData = mesh->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(scalarFieldName, scalarFieldId);

  if (!scalarField)
    return svtkReebGraph::ERR_NO_SUCH_FIELD;

  return this->Build(mesh, scalarField);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Build(svtkUnstructuredGrid* mesh, const char* scalarFieldName)
{
  int scalarFieldId = 0;

  svtkPointData* pointData = mesh->GetPointData();
  svtkDataArray* scalarField = pointData->GetArray(scalarFieldName, scalarFieldId);

  if (!scalarField)
    return svtkReebGraph::ERR_NO_SUCH_FIELD;

  return this->Build(mesh, scalarField);
}

//----------------------------------------------------------------------------
int svtkReebGraph::Implementation::GetNumberOfLoops()
{
  if (!this->ArcLoopTable)
    this->FindLoops();
  return this->LoopNumber - this->RemovedLoopNumber;
}

//----------------------------------------------------------------------------
inline svtkIdType svtkReebGraph::Implementation::AddArc(svtkIdType nodeId0, svtkIdType nodeId1)
{
  if (!svtkReebGraphIsSmaller(
        this, nodeId0, nodeId1, this->GetNode(nodeId0), this->GetNode(nodeId1)))
    svtkReebGraphSwapVars(svtkIdType, nodeId0, nodeId1);
  svtkIdType nodesvtkReebArcble[] = { nodeId0, nodeId1 };
  return AddPath(2, nodesvtkReebArcble, 0);
}
