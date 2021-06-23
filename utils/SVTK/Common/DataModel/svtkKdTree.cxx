/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkKdTree.cxx

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

#include "svtkKdTree.h"

#include "svtkBSPCuts.h"
#include "svtkBSPIntersections.h"
#include "svtkCallbackCommand.h"
#include "svtkCell.h"
#include "svtkCellArray.h"
#include "svtkDataSet.h"
#include "svtkDataSetCollection.h"
#include "svtkFloatArray.h"
#include "svtkGarbageCollector.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkImageData.h"
#include "svtkIntArray.h"
#include "svtkKdNode.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointSet.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkRectilinearGrid.h"
#include "svtkTimerLog.h"
#include "svtkUniformGrid.h"
#include "svtkUnsignedCharArray.h"

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <set>

namespace
{
class TimeLog // Similar to svtkTimerLogScope, but can be disabled at runtime.
{
  const std::string Event;
  int Timing;

public:
  TimeLog(const char* event, int timing)
    : Event(event ? event : "")
    , Timing(timing)
  {
    if (this->Timing)
    {
      svtkTimerLog::MarkStartEvent(this->Event.c_str());
    }
  }

  ~TimeLog()
  {
    if (this->Timing)
    {
      svtkTimerLog::MarkEndEvent(this->Event.c_str());
    }
  }

  static void StartEvent(const char* event, int timing)
  {
    if (timing)
    {
      svtkTimerLog::MarkStartEvent(event);
    }
  }

  static void EndEvent(const char* event, int timing)
  {
    if (timing)
    {
      svtkTimerLog::MarkEndEvent(event);
    }
  }

private:
  // Explicit disable copy/assignment to prevent MSVC from complaining (C4512)
  TimeLog(const TimeLog&) = delete;
  TimeLog& operator=(const TimeLog&) = delete;
};
}

#define SCOPETIMER(msg)                                                                            \
  TimeLog _timer("KdTree: " msg, this->Timing);                                                    \
  (void)_timer
#define TIMER(msg) TimeLog::StartEvent("KdTree: " msg, this->Timing)
#define TIMERDONE(msg) TimeLog::EndEvent("KdTree: " msg, this->Timing)

//-----------------------------------------------------------------------------
static void LastInputDeletedCallback(svtkObject* svtkNotUsed(caller), unsigned long svtkNotUsed(eid),
  void* _self, void* svtkNotUsed(calldata))
{
  svtkKdTree* self = reinterpret_cast<svtkKdTree*>(_self);

  self->InvalidateGeometry();
}

// helper class for ordering the points in svtkKdTree::FindClosestNPoints()
namespace
{
class OrderPoints
{
public:
  OrderPoints(int N)
  {
    this->NumDesiredPoints = N;
    this->NumPoints = 0;
    this->LargestDist2 = SVTK_FLOAT_MAX;
  }

  void InsertPoint(float dist2, svtkIdType id)
  {
    if (dist2 <= this->LargestDist2 || this->NumPoints < this->NumDesiredPoints)
    {
      std::map<float, std::list<svtkIdType> >::iterator it = this->dist2ToIds.find(dist2);
      this->NumPoints++;
      if (it == this->dist2ToIds.end())
      {
        std::list<svtkIdType> idset;
        idset.push_back(id);
        this->dist2ToIds[dist2] = idset;
      }
      else
      {
        it->second.push_back(id);
      }
      if (this->NumPoints > this->NumDesiredPoints)
      {
        it = this->dist2ToIds.end();
        --it;
        if ((this->NumPoints - it->second.size()) > this->NumDesiredPoints)
        {
          this->NumPoints -= it->second.size();
          std::map<float, std::list<svtkIdType> >::iterator it2 = it;
          --it2;
          this->LargestDist2 = it2->first;
          this->dist2ToIds.erase(it);
        }
      }
    }
  }
  void GetSortedIds(svtkIdList* ids)
  {
    ids->Reset();
    svtkIdType numIds = static_cast<svtkIdType>(
      (this->NumDesiredPoints < this->NumPoints) ? this->NumDesiredPoints : this->NumPoints);
    ids->SetNumberOfIds(numIds);
    svtkIdType counter = 0;
    std::map<float, std::list<svtkIdType> >::iterator it = this->dist2ToIds.begin();
    while (counter < numIds && it != this->dist2ToIds.end())
    {
      std::list<svtkIdType>::iterator lit = it->second.begin();
      while (counter < numIds && lit != it->second.end())
      {
        ids->InsertId(counter, *lit);
        counter++;
        ++lit;
      }
      ++it;
    }
  }

  float GetLargestDist2() { return this->LargestDist2; }

private:
  size_t NumDesiredPoints, NumPoints;
  float LargestDist2;
  std::map<float, std::list<svtkIdType> > dist2ToIds; // map from dist^2 to a list of ids
};
}

svtkStandardNewMacro(svtkKdTree);

//----------------------------------------------------------------------------
svtkKdTree::svtkKdTree()
{
  this->FudgeFactor = 0;
  this->MaxWidth = 0.0;
  this->MaxLevel = 20;
  this->Level = 0;

  this->NumberOfRegionsOrLess = 0;
  this->NumberOfRegionsOrMore = 0;

  this->ValidDirections = (1 << svtkKdTree::XDIM) | (1 << svtkKdTree::YDIM) | (1 << svtkKdTree::ZDIM);

  this->MinCells = 100;
  this->NumberOfRegions = 0;

  this->DataSets = svtkDataSetCollection::New();

  this->Top = nullptr;
  this->RegionList = nullptr;

  this->Timing = 0;
  this->TimerLog = nullptr;

  this->IncludeRegionBoundaryCells = 0;
  this->GenerateRepresentationUsingDataBounds = 0;

  this->InitializeCellLists();
  this->CellRegionList = nullptr;

  this->NumberOfLocatorPoints = 0;
  this->LocatorPoints = nullptr;
  this->LocatorIds = nullptr;
  this->LocatorRegionLocation = nullptr;

  this->LastDataCacheSize = 0;
  this->LastNumDataSets = 0;
  this->ClearLastBuildCache();

  this->BSPCalculator = nullptr;
  this->Cuts = nullptr;
  this->UserDefinedCuts = 0;

  this->Progress = 0;
  this->ProgressOffset = 0;
  this->ProgressScale = 1.0;
}

//----------------------------------------------------------------------------
void svtkKdTree::DeleteAllDescendants(svtkKdNode* nd)
{
  svtkKdNode* left = nd->GetLeft();
  svtkKdNode* right = nd->GetRight();

  if (left && left->GetLeft())
  {
    svtkKdTree::DeleteAllDescendants(left);
  }

  if (right && right->GetLeft())
  {
    svtkKdTree::DeleteAllDescendants(right);
  }

  if (left && right)
  {
    nd->DeleteChildNodes(); // undo AddChildNodes
    left->Delete();         // undo svtkKdNode::New()
    right->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::InitializeCellLists()
{
  this->CellList.dataSet = nullptr;
  this->CellList.regionIds = nullptr;
  this->CellList.nRegions = 0;
  this->CellList.cells = nullptr;
  this->CellList.boundaryCells = nullptr;
  this->CellList.emptyList = nullptr;
}

//----------------------------------------------------------------------------
void svtkKdTree::DeleteCellLists()
{
  int i;
  int num = this->CellList.nRegions;

  delete[] this->CellList.regionIds;

  if (this->CellList.cells)
  {
    for (i = 0; i < num; i++)
    {
      this->CellList.cells[i]->Delete();
    }

    delete[] this->CellList.cells;
  }

  if (this->CellList.boundaryCells)
  {
    for (i = 0; i < num; i++)
    {
      this->CellList.boundaryCells[i]->Delete();
    }
    delete[] this->CellList.boundaryCells;
  }

  if (this->CellList.emptyList)
  {
    this->CellList.emptyList->Delete();
  }

  this->InitializeCellLists();
}

//----------------------------------------------------------------------------
svtkKdTree::~svtkKdTree()
{
  if (this->DataSets)
  {
    this->DataSets->Delete();
    this->DataSets = nullptr;
  }

  this->FreeSearchStructure();

  this->DeleteCellLists();

  delete[] this->CellRegionList;
  this->CellRegionList = nullptr;

  if (this->TimerLog)
  {
    this->TimerLog->Delete();
  }

  this->ClearLastBuildCache();

  this->SetCalculator(nullptr);
  this->SetCuts(nullptr);
}
//----------------------------------------------------------------------------
void svtkKdTree::SetCalculator(svtkKdNode* kd)
{
  if (this->BSPCalculator)
  {
    this->BSPCalculator->Delete();
    this->BSPCalculator = nullptr;
  }

  if (!this->UserDefinedCuts)
  {
    this->SetCuts(nullptr, 0);
  }

  if (kd == nullptr)
  {
    return;
  }

  if (!this->UserDefinedCuts)
  {
    svtkBSPCuts* cuts = svtkBSPCuts::New();
    cuts->CreateCuts(kd);
    this->SetCuts(cuts, 0);
  }

  this->BSPCalculator = svtkBSPIntersections::New();
  this->BSPCalculator->SetCuts(this->Cuts);
}
//----------------------------------------------------------------------------
void svtkKdTree::SetCuts(svtkBSPCuts* cuts)
{
  this->SetCuts(cuts, 1);
}
//----------------------------------------------------------------------------
void svtkKdTree::SetCuts(svtkBSPCuts* cuts, int userDefined)
{
  if (userDefined != 0)
  {
    userDefined = 1;
  }

  if ((cuts == this->Cuts) && (userDefined == this->UserDefinedCuts))
  {
    return;
  }

  if (!this->Cuts || !this->Cuts->Equals(cuts))
  {
    this->Modified();
  }

  if (this->Cuts)
  {
    if (this->UserDefinedCuts)
    {
      this->Cuts->UnRegister(this);
    }
    else
    {
      this->Cuts->Delete();
    }

    this->Cuts = nullptr;
    this->UserDefinedCuts = 0;
  }

  if (cuts == nullptr)
  {
    return;
  }

  this->Cuts = cuts;
  this->UserDefinedCuts = userDefined;

  if (this->UserDefinedCuts)
  {
    this->Cuts->Register(this);
  }
}
//----------------------------------------------------------------------------
// Add and remove data sets.  We don't update this->Modify() here, because
// changing the data sets doesn't necessarily require rebuilding the
// k-d tree.  We only need to build a new k-d tree in BuildLocator if
// the geometry has changed, and we check for that with NewGeometry in
// BuildLocator.  We Modify() for changes that definitely require a
// rebuild of the tree, like changing the depth of the k-d tree.

void svtkKdTree::SetDataSet(svtkDataSet* set)
{
  this->DataSets->RemoveAllItems();
  this->AddDataSet(set);
}

void svtkKdTree::AddDataSet(svtkDataSet* set)
{
  if (set == nullptr)
  {
    return;
  }

  if (this->DataSets->IsItemPresent(set))
  {
    return;
  }

  this->DataSets->AddItem(set);
}

void svtkKdTree::RemoveDataSet(svtkDataSet* set)
{
  this->DataSets->RemoveItem(set);
}

void svtkKdTree::RemoveDataSet(int index)
{
  this->DataSets->RemoveItem(index);
}

void svtkKdTree::RemoveAllDataSets()
{
  this->DataSets->RemoveAllItems();
}

//-----------------------------------------------------------------------------

int svtkKdTree::GetNumberOfDataSets()
{
  return this->DataSets->GetNumberOfItems();
}

int svtkKdTree::GetDataSetIndex(svtkDataSet* set)
{
  // This is weird, but IsItemPresent returns the index + 1 (so that 0
  // corresponds to item not present).
  return this->DataSets->IsItemPresent(set) - 1;
}

svtkDataSet* svtkKdTree::GetDataSet(int index)
{
  return this->DataSets->GetItem(index);
}

int svtkKdTree::GetDataSetsNumberOfCells(int from, int to)
{
  int numCells = 0;

  for (int i = from; i <= to; i++)
  {
    svtkDataSet* data = this->GetDataSet(i);
    if (data)
    {
      numCells += data->GetNumberOfCells();
    }
  }

  return numCells;
}

//----------------------------------------------------------------------------
int svtkKdTree::GetNumberOfCells()
{
  return this->GetDataSetsNumberOfCells(0, this->GetNumberOfDataSets() - 1);
}

//----------------------------------------------------------------------------
void svtkKdTree::GetBounds(double* bounds)
{
  if (this->Top)
  {
    this->Top->GetBounds(bounds);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::GetRegionBounds(int regionID, double bounds[6])
{
  if ((regionID < 0) || (regionID >= this->NumberOfRegions))
  {
    svtkErrorMacro(<< "svtkKdTree::GetRegionBounds invalid region");
    return;
  }

  svtkKdNode* node = this->RegionList[regionID];

  node->GetBounds(bounds);
}

//----------------------------------------------------------------------------
void svtkKdTree::GetRegionDataBounds(int regionID, double bounds[6])
{
  if ((regionID < 0) || (regionID >= this->NumberOfRegions))
  {
    svtkErrorMacro(<< "svtkKdTree::GetRegionDataBounds invalid region");
    return;
  }

  svtkKdNode* node = this->RegionList[regionID];

  node->GetDataBounds(bounds);
}

//----------------------------------------------------------------------------
svtkKdNode** svtkKdTree::_GetRegionsAtLevel(int level, svtkKdNode** nodes, svtkKdNode* kd)
{
  if (level > 0)
  {
    svtkKdNode** nodes0 = _GetRegionsAtLevel(level - 1, nodes, kd->GetLeft());
    svtkKdNode** nodes1 = _GetRegionsAtLevel(level - 1, nodes0, kd->GetRight());

    return nodes1;
  }
  else
  {
    nodes[0] = kd;
    return nodes + 1;
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::GetRegionsAtLevel(int level, svtkKdNode** nodes)
{
  if ((level < 0) || (level > this->Level))
  {
    return;
  }

  svtkKdTree::_GetRegionsAtLevel(level, nodes, this->Top);
}

//----------------------------------------------------------------------------
void svtkKdTree::GetLeafNodeIds(svtkKdNode* node, svtkIntArray* ids)
{
  int id = node->GetID();

  if (id < 0)
  {
    svtkKdTree::GetLeafNodeIds(node->GetLeft(), ids);
    svtkKdTree::GetLeafNodeIds(node->GetRight(), ids);
  }
  else
  {
    ids->InsertNextValue(id);
  }
}

//----------------------------------------------------------------------------
float* svtkKdTree::ComputeCellCenters()
{
  svtkDataSet* allSets = nullptr;
  return this->ComputeCellCenters(allSets);
}

//----------------------------------------------------------------------------
float* svtkKdTree::ComputeCellCenters(int set)
{
  svtkDataSet* data = this->GetDataSet(set);
  if (!data)
  {
    svtkErrorMacro(<< "svtkKdTree::ComputeCellCenters no such data set");
    return nullptr;
  }
  return this->ComputeCellCenters(data);
}

//----------------------------------------------------------------------------
float* svtkKdTree::ComputeCellCenters(svtkDataSet* set)
{
  SCOPETIMER("ComputeCellCenters");
  this->UpdateSubOperationProgress(0);
  int totalCells;

  if (set)
  {
    totalCells = set->GetNumberOfCells();
  }
  else
  {
    totalCells = this->GetNumberOfCells(); // all data sets
  }

  if (totalCells == 0)
  {
    return nullptr;
  }

  float* center = new float[3 * totalCells];

  if (!center)
  {
    return nullptr;
  }

  int maxCellSize = 0;

  if (set)
  {
    maxCellSize = set->GetMaxCellSize();
  }
  else
  {
    svtkCollectionSimpleIterator cookie;
    this->DataSets->InitTraversal(cookie);
    for (svtkDataSet* iset = this->DataSets->GetNextDataSet(cookie); iset != nullptr;
         iset = this->DataSets->GetNextDataSet(cookie))
    {
      int cellSize = iset->GetMaxCellSize();
      maxCellSize = (cellSize > maxCellSize) ? cellSize : maxCellSize;
    }
  }

  double* weights = new double[maxCellSize];

  float* cptr = center;
  double dcenter[3];

  if (set)
  {
    for (int j = 0; j < totalCells; j++)
    {
      this->ComputeCellCenter(set->GetCell(j), dcenter, weights);
      cptr[0] = static_cast<float>(dcenter[0]);
      cptr[1] = static_cast<float>(dcenter[1]);
      cptr[2] = static_cast<float>(dcenter[2]);
      cptr += 3;
      if (j % 1000 == 0)
      {
        this->UpdateSubOperationProgress(static_cast<double>(j) / totalCells);
      }
    }
  }
  else
  {
    svtkCollectionSimpleIterator cookie;
    this->DataSets->InitTraversal(cookie);
    for (svtkDataSet* iset = this->DataSets->GetNextDataSet(cookie); iset != nullptr;
         iset = this->DataSets->GetNextDataSet(cookie))
    {
      int nCells = iset->GetNumberOfCells();

      for (int j = 0; j < nCells; j++)
      {
        this->ComputeCellCenter(iset->GetCell(j), dcenter, weights);
        cptr[0] = static_cast<float>(dcenter[0]);
        cptr[1] = static_cast<float>(dcenter[1]);
        cptr[2] = static_cast<float>(dcenter[2]);
        cptr += 3;
        if (j % 1000 == 0)
        {
          this->UpdateSubOperationProgress(static_cast<double>(j) / totalCells);
        }
      }
    }
  }

  delete[] weights;

  this->UpdateSubOperationProgress(1.0);
  return center;
}

//----------------------------------------------------------------------------
void svtkKdTree::ComputeCellCenter(svtkDataSet* set, int cellId, float* center)
{
  double dcenter[3];

  this->ComputeCellCenter(set, cellId, dcenter);

  center[0] = static_cast<float>(dcenter[0]);
  center[1] = static_cast<float>(dcenter[1]);
  center[2] = static_cast<float>(dcenter[2]);
}

//----------------------------------------------------------------------------
void svtkKdTree::ComputeCellCenter(svtkDataSet* set, int cellId, double* center)
{
  int setNum;

  if (set)
  {
    setNum = this->GetDataSetIndex(set);

    if (setNum < 0)
    {
      svtkErrorMacro(<< "svtkKdTree::ComputeCellCenter invalid data set");
      return;
    }
  }
  else
  {
    set = this->GetDataSet();
  }

  if ((cellId < 0) || (cellId >= set->GetNumberOfCells()))
  {
    svtkErrorMacro(<< "svtkKdTree::ComputeCellCenter invalid cell ID");
    return;
  }

  double* weights = new double[set->GetMaxCellSize()];

  this->ComputeCellCenter(set->GetCell(cellId), center, weights);

  delete[] weights;
}

//----------------------------------------------------------------------------
void svtkKdTree::ComputeCellCenter(svtkCell* cell, double* center, double* weights)
{
  double pcoords[3];

  int subId = cell->GetParametricCenter(pcoords);

  cell->EvaluateLocation(subId, pcoords, center, weights);
}

//----------------------------------------------------------------------------
// Build the kdtree structure based on location of cell centroids.
//
void svtkKdTree::BuildLocator()
{
  SCOPETIMER("BuildLocator");

  this->UpdateProgress(0);
  int nCells = 0;
  int i;

  if ((this->Top != nullptr) && (this->BuildTime > this->GetMTime()) && (this->NewGeometry() == 0))
  {
    return;
  }

  nCells = this->GetNumberOfCells();

  if (nCells == 0)
  {
    svtkErrorMacro(<< "svtkKdTree::BuildLocator - No cells to subdivide");
    return;
  }

  svtkDebugMacro(<< "Creating Kdtree");
  this->InvokeEvent(svtkCommand::StartEvent);

  if ((this->Timing) && (this->TimerLog == nullptr))
  {
    this->TimerLog = svtkTimerLog::New();
  }

  TIMER("Set up to build k-d tree");

  this->FreeSearchStructure();

  // volume bounds - push out a little if flat
  svtkCollectionSimpleIterator cookie;
  this->DataSets->InitTraversal(cookie);
  svtkDataSet* iset = this->DataSets->GetNextDataSet(cookie);
  double volBounds[6];
  iset->GetBounds(volBounds);

  while ((iset = this->DataSets->GetNextDataSet(cookie)))
  {
    double setBounds[6];
    iset->GetBounds(setBounds);
    if (setBounds[0] < volBounds[0])
    {
      volBounds[0] = setBounds[0];
    }
    if (setBounds[2] < volBounds[2])
    {
      volBounds[2] = setBounds[2];
    }
    if (setBounds[4] < volBounds[4])
    {
      volBounds[4] = setBounds[4];
    }
    if (setBounds[1] > volBounds[1])
    {
      volBounds[1] = setBounds[1];
    }
    if (setBounds[3] > volBounds[3])
    {
      volBounds[3] = setBounds[3];
    }
    if (setBounds[5] > volBounds[5])
    {
      volBounds[5] = setBounds[5];
    }
  }

  double diff[3], aLittle = 0.0;
  this->MaxWidth = 0.0;

  for (i = 0; i < 3; i++)
  {
    diff[i] = volBounds[2 * i + 1] - volBounds[2 * i];
    this->MaxWidth = static_cast<float>((diff[i] > this->MaxWidth) ? diff[i] : this->MaxWidth);
  }

  this->FudgeFactor = this->MaxWidth * 10e-6;

  aLittle = this->MaxWidth / 100.0;

  for (i = 0; i < 3; i++)
  {
    if (diff[i] <= 0)
    {
      volBounds[2 * i] -= aLittle;
      volBounds[2 * i + 1] += aLittle;
    }
    else // need lower bound to be strictly less than any point in decomposition
    {
      volBounds[2 * i] -= this->FudgeFactor;
      volBounds[2 * i + 1] += this->FudgeFactor;
    }
  }
  TIMERDONE("Set up to build k-d tree");

  if (this->UserDefinedCuts)
  {
    // Actually, we will not compute the k-d tree.  We will use a
    // k-d tree provided to us.

    int fail = this->ProcessUserDefinedCuts(volBounds);

    if (fail)
    {
      return;
    }
  }
  else
  {
    // cell centers - basis of spatial decomposition

    TIMER("Create centroid list");
    this->ProgressOffset = 0;
    this->ProgressScale = 0.3;
    float* ptarray = this->ComputeCellCenters();

    TIMERDONE("Create centroid list");

    if (!ptarray)
    {
      svtkErrorMacro(<< "svtkKdTree::BuildLocator - insufficient memory");
      return;
    }

    // create kd tree structure that balances cell centers

    svtkKdNode* kd = this->Top = svtkKdNode::New();

    kd->SetBounds(
      volBounds[0], volBounds[1], volBounds[2], volBounds[3], volBounds[4], volBounds[5]);

    kd->SetNumberOfPoints(nCells);

    kd->SetDataBounds(
      volBounds[0], volBounds[1], volBounds[2], volBounds[3], volBounds[4], volBounds[5]);

    TIMER("Build tree");

    this->ProgressOffset += this->ProgressScale;
    this->ProgressScale = 0.7;
    this->DivideRegion(kd, ptarray, nullptr, 0);

    TIMERDONE("Build tree");

    // In the process of building the k-d tree regions,
    //   the cell centers became reordered, so no point
    //   in saving them, for example to build cell lists.

    delete[] ptarray;
  }

  this->SetActualLevel();
  this->BuildRegionList();

  this->InvokeEvent(svtkCommand::EndEvent);

  this->UpdateBuildTime();

  this->SetCalculator(this->Top);

  this->UpdateProgress(1.0);
}

int svtkKdTree::ProcessUserDefinedCuts(double* minBounds)
{
  SCOPETIMER("ProcessUserDefinedCuts");

  if (!this->Cuts)
  {
    svtkErrorMacro(<< "svtkKdTree::ProcessUserDefinedCuts - no cuts");
    return 1;
  }
  // Fix the bounds for the entire partitioning.  They must be at
  // least as large of the bounds of all the data sets.

  svtkKdNode* kd = this->Cuts->GetKdNodeTree();
  double bounds[6];
  kd->GetBounds(bounds);
  int fixBounds = 0;

  for (int j = 0; j < 3; j++)
  {
    int min = 2 * j;
    int max = min + 1;

    if (minBounds[min] < bounds[min])
    {
      bounds[min] = minBounds[min];
      fixBounds = 1;
    }
    if (minBounds[max] > bounds[max])
    {
      bounds[max] = minBounds[max];
      fixBounds = 1;
    }
  }

  this->Top = svtkKdTree::CopyTree(kd);

  if (fixBounds)
  {
    this->SetNewBounds(bounds);
  }

  // We don't really know the data bounds, so we'll just set them
  // to the spatial bounds.

  svtkKdTree::SetDataBoundsToSpatialBounds(this->Top);

  // And, we don't know how many points are in each region.  The number
  // in the provided svtkBSPCuts object was for some other dataset.  So
  // we zero out those fields.

  svtkKdTree::ZeroNumberOfPoints(this->Top);

  return 0;
}
//----------------------------------------------------------------------------
void svtkKdTree::ZeroNumberOfPoints(svtkKdNode* kd)
{
  kd->SetNumberOfPoints(0);

  if (kd->GetLeft())
  {
    svtkKdTree::ZeroNumberOfPoints(kd->GetLeft());
    svtkKdTree::ZeroNumberOfPoints(kd->GetRight());
  }
}
//----------------------------------------------------------------------------
void svtkKdTree::SetNewBounds(double* bounds)
{
  svtkKdNode* kd = this->Top;

  if (!kd)
  {
    return;
  }

  int fixDimLeft[6], fixDimRight[6];
  int go = 0;

  double kdb[6];
  kd->GetBounds(kdb);

  for (int i = 0; i < 3; i++)
  {
    int min = 2 * i;
    int max = 2 * i + 1;

    fixDimLeft[min] = fixDimRight[min] = 0;
    fixDimLeft[max] = fixDimRight[max] = 0;

    if (kdb[min] > bounds[min])
    {
      kdb[min] = bounds[min];
      go = fixDimLeft[min] = fixDimRight[min] = 1;
    }
    if (kdb[max] < bounds[max])
    {
      kdb[max] = bounds[max];
      go = fixDimLeft[max] = fixDimRight[max] = 1;
    }
  }

  if (go)
  {
    kd->SetBounds(kdb[0], kdb[1], kdb[2], kdb[3], kdb[4], kdb[5]);

    if (kd->GetLeft())
    {
      int cutDim = kd->GetDim() * 2;

      fixDimLeft[cutDim + 1] = 0;
      svtkKdTree::_SetNewBounds(kd->GetLeft(), bounds, fixDimLeft);

      fixDimRight[cutDim] = 0;
      svtkKdTree::_SetNewBounds(kd->GetRight(), bounds, fixDimRight);
    }
  }
}
//----------------------------------------------------------------------------
void svtkKdTree::_SetNewBounds(svtkKdNode* kd, double* b, int* fixDim)
{
  int go = 0;
  int fixDimLeft[6], fixDimRight[6];

  double kdb[6];
  kd->GetBounds(kdb);

  for (int i = 0; i < 6; i++)
  {
    if (fixDim[i])
    {
      kdb[i] = b[i];
      go = 1;
    }
    fixDimLeft[i] = fixDim[i];
    fixDimRight[i] = fixDim[i];
  }

  if (go)
  {
    kd->SetBounds(kdb[0], kdb[1], kdb[2], kdb[3], kdb[4], kdb[5]);

    if (kd->GetLeft())
    {
      int cutDim = kd->GetDim() * 2;

      fixDimLeft[cutDim + 1] = 0;
      svtkKdTree::_SetNewBounds(kd->GetLeft(), b, fixDimLeft);

      fixDimRight[cutDim] = 0;
      svtkKdTree::_SetNewBounds(kd->GetRight(), b, fixDimRight);
    }
  }
}
//----------------------------------------------------------------------------
svtkKdNode* svtkKdTree::CopyTree(svtkKdNode* kd)
{
  svtkKdNode* top = svtkKdNode::New();
  svtkKdTree::CopyKdNode(top, kd);
  svtkKdTree::CopyChildNodes(top, kd);

  return top;
}
//----------------------------------------------------------------------------
void svtkKdTree::CopyChildNodes(svtkKdNode* to, svtkKdNode* from)
{
  if (from->GetLeft())
  {
    svtkKdNode* left = svtkKdNode::New();
    svtkKdNode* right = svtkKdNode::New();

    svtkKdTree::CopyKdNode(left, from->GetLeft());
    svtkKdTree::CopyKdNode(right, from->GetRight());

    to->AddChildNodes(left, right);

    svtkKdTree::CopyChildNodes(to->GetLeft(), from->GetLeft());
    svtkKdTree::CopyChildNodes(to->GetRight(), from->GetRight());
  }
}
//----------------------------------------------------------------------------
void svtkKdTree::CopyKdNode(svtkKdNode* to, svtkKdNode* from)
{
  to->SetMinBounds(from->GetMinBounds());
  to->SetMaxBounds(from->GetMaxBounds());
  to->SetMinDataBounds(from->GetMinDataBounds());
  to->SetMaxDataBounds(from->GetMaxDataBounds());
  to->SetID(from->GetID());
  to->SetMinID(from->GetMinID());
  to->SetMaxID(from->GetMaxID());
  to->SetNumberOfPoints(from->GetNumberOfPoints());
  to->SetDim(from->GetDim());
}

//----------------------------------------------------------------------------
int svtkKdTree::ComputeLevel(svtkKdNode* kd)
{
  if (!kd)
  {
    return 0;
  }

  int iam = 1;

  if (kd->GetLeft() != nullptr)
  {
    int depth1 = svtkKdTree::ComputeLevel(kd->GetLeft());
    int depth2 = svtkKdTree::ComputeLevel(kd->GetRight());

    if (depth1 > depth2)
    {
      iam += depth1;
    }
    else
    {
      iam += depth2;
    }
  }
  return iam;
}
//----------------------------------------------------------------------------
void svtkKdTree::SetDataBoundsToSpatialBounds(svtkKdNode* kd)
{
  kd->SetMinDataBounds(kd->GetMinBounds());
  kd->SetMaxDataBounds(kd->GetMaxBounds());

  if (kd->GetLeft())
  {
    svtkKdTree::SetDataBoundsToSpatialBounds(kd->GetLeft());
    svtkKdTree::SetDataBoundsToSpatialBounds(kd->GetRight());
  }
}
//----------------------------------------------------------------------------
int svtkKdTree::SelectCutDirection(svtkKdNode* kd)
{
  int dim = 0, i;

  int xdir = 1 << svtkKdTree::XDIM;
  int ydir = 1 << svtkKdTree::YDIM;
  int zdir = 1 << svtkKdTree::ZDIM;

  // determine direction in which to divide this region

  if (this->ValidDirections == xdir)
  {
    dim = svtkKdTree::XDIM;
  }
  else if (this->ValidDirections == ydir)
  {
    dim = svtkKdTree::YDIM;
  }
  else if (this->ValidDirections == zdir)
  {
    dim = svtkKdTree::ZDIM;
  }
  else
  {
    // divide in the longest direction, for more compact regions

    double diff[3], dataBounds[6], maxdiff;
    kd->GetDataBounds(dataBounds);

    for (i = 0; i < 3; i++)
    {
      diff[i] = dataBounds[i * 2 + 1] - dataBounds[i * 2];
    }

    maxdiff = -1.0;

    if ((this->ValidDirections & xdir) && (diff[svtkKdTree::XDIM] > maxdiff))
    {
      dim = svtkKdTree::XDIM;
      maxdiff = diff[svtkKdTree::XDIM];
    }

    if ((this->ValidDirections & ydir) && (diff[svtkKdTree::YDIM] > maxdiff))
    {
      dim = svtkKdTree::YDIM;
      maxdiff = diff[svtkKdTree::YDIM];
    }

    if ((this->ValidDirections & zdir) && (diff[svtkKdTree::ZDIM] > maxdiff))
    {
      dim = svtkKdTree::ZDIM;
    }
  }
  return dim;
}
//----------------------------------------------------------------------------
int svtkKdTree::DivideTest(int size, int level)
{
  if (level >= this->MaxLevel)
    return 0;

  int minCells = this->GetMinCells();

  if (minCells && (minCells > (size / 2)))
    return 0;

  int nRegionsNow = 1 << level;
  int nRegionsNext = nRegionsNow << 1;

  if (this->NumberOfRegionsOrLess && (nRegionsNext > this->NumberOfRegionsOrLess))
    return 0;
  if (this->NumberOfRegionsOrMore && (nRegionsNow >= this->NumberOfRegionsOrMore))
    return 0;

  return 1;
}
//----------------------------------------------------------------------------
int svtkKdTree::DivideRegion(svtkKdNode* kd, float* c1, int* ids, int level)
{
  int ok = this->DivideTest(kd->GetNumberOfPoints(), level);

  if (!ok)
  {
    return 0;
  }

  int maxdim = this->SelectCutDirection(kd);

  kd->SetDim(maxdim);

  int dim1 = maxdim; // best cut direction
  int dim2 = -1;     // other valid cut directions
  int dim3 = -1;

  int otherDirections = this->ValidDirections ^ (1 << maxdim);

  if (otherDirections)
  {
    int x = otherDirections & (1 << svtkKdTree::XDIM);
    int y = otherDirections & (1 << svtkKdTree::YDIM);
    int z = otherDirections & (1 << svtkKdTree::ZDIM);

    if (x)
    {
      dim2 = svtkKdTree::XDIM;

      if (y)
      {
        dim3 = svtkKdTree::YDIM;
      }
      else if (z)
      {
        dim3 = svtkKdTree::ZDIM;
      }
    }
    else if (y)
    {
      dim2 = svtkKdTree::YDIM;

      if (z)
      {
        dim3 = svtkKdTree::ZDIM;
      }
    }
    else if (z)
    {
      dim2 = svtkKdTree::ZDIM;
    }
  }

  this->DoMedianFind(kd, c1, ids, dim1, dim2, dim3);

  if (kd->GetLeft() == nullptr)
  {
    return 0; // unable to divide region further
  }

  int nleft = kd->GetLeft()->GetNumberOfPoints();

  int* leftIds = ids;
  int* rightIds = ids ? ids + nleft : nullptr;

  this->DivideRegion(kd->GetLeft(), c1, leftIds, level + 1);

  this->DivideRegion(kd->GetRight(), c1 + nleft * 3, rightIds, level + 1);

  return 0;
}

//----------------------------------------------------------------------------
// Rearrange the point array.  Try dim1 first.  If there's a problem
// go to dim2, then dim3.
//
void svtkKdTree::DoMedianFind(svtkKdNode* kd, float* c1, int* ids, int dim1, int dim2, int dim3)
{
  double coord;
  int dim;
  int midpt;

  int npoints = kd->GetNumberOfPoints();

  int dims[3] = { dim1, dim2, dim3 };

  for (dim = 0; dim < 3; dim++)
  {
    if (dims[dim] < 0)
    {
      break;
    }

    midpt = svtkKdTree::Select(dims[dim], c1, ids, npoints, coord);

    if (midpt == 0)
    {
      continue; // fatal
    }

    kd->SetDim(dims[dim]);

    svtkKdTree::AddNewRegions(kd, c1, midpt, dims[dim], coord);

    break; // division is fine
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::AddNewRegions(svtkKdNode* kd, float* c1, int midpt, int dim, double coord)
{
  svtkKdNode* left = svtkKdNode::New();
  svtkKdNode* right = svtkKdNode::New();

  int npoints = kd->GetNumberOfPoints();

  int nleft = midpt;
  int nright = npoints - midpt;

  kd->AddChildNodes(left, right);

  double bounds[6];
  kd->GetBounds(bounds);

  left->SetBounds(bounds[0], ((dim == svtkKdTree::XDIM) ? coord : bounds[1]), bounds[2],
    ((dim == svtkKdTree::YDIM) ? coord : bounds[3]), bounds[4],
    ((dim == svtkKdTree::ZDIM) ? coord : bounds[5]));

  left->SetNumberOfPoints(nleft);

  right->SetBounds(((dim == svtkKdTree::XDIM) ? coord : bounds[0]), bounds[1],
    ((dim == svtkKdTree::YDIM) ? coord : bounds[2]), bounds[3],
    ((dim == svtkKdTree::ZDIM) ? coord : bounds[4]), bounds[5]);

  right->SetNumberOfPoints(nright);

  left->SetDataBounds(c1);
  right->SetDataBounds(c1 + nleft * 3);
}
// Use Floyd & Rivest (1975) to find the median:
// Given an array X with element indices ranging from L to R, and
// a K such that L <= K <= R, rearrange the elements such that
// X[K] contains the ith sorted element,  where i = K - L + 1, and
// all the elements X[j], j < k satisfy X[j] < X[K], and all the
// elements X[j], j > k satisfy X[j] >= X[K].

#define Exchange(array, ids, x, y)                                                                 \
  {                                                                                                \
    float temp[3];                                                                                 \
    temp[0] = array[3 * x];                                                                        \
    temp[1] = array[3 * x + 1];                                                                    \
    temp[2] = array[3 * x + 2];                                                                    \
    array[3 * x] = array[3 * y];                                                                   \
    array[3 * x + 1] = array[3 * y + 1];                                                           \
    array[3 * x + 2] = array[3 * y + 2];                                                           \
    array[3 * y] = temp[0];                                                                        \
    array[3 * y + 1] = temp[1];                                                                    \
    array[3 * y + 2] = temp[2];                                                                    \
    if (ids)                                                                                       \
    {                                                                                              \
      svtkIdType tempid = ids[x];                                                                   \
      ids[x] = ids[y];                                                                             \
      ids[y] = tempid;                                                                             \
    }                                                                                              \
  }

#define sign(x) (((x) < 0) ? (-1) : (1))

//----------------------------------------------------------------------------
int svtkKdTree::Select(int dim, float* c1, int* ids, int nvals, double& coord)
{
  int left = 0;
  int mid = nvals / 2;
  int right = nvals - 1;

  svtkKdTree::_Select(dim, c1, ids, left, right, mid);

  // We need to be careful in the case where the "mid"
  // value is repeated several times in the array.  We
  // want to roll the dividing index (mid) back to the
  // first occurrence in the array, so that there is no
  // ambiguity about which spatial region a given point
  // belongs in.
  //
  // The array has been rearranged (in _Select) like this:
  //
  // All values c1[n], left <= n < mid, satisfy c1[n] <= c1[mid]
  // All values c1[n], mid < n <= right, satisfy c1[n] >= c1[mid]
  //
  // In addition, by careful construction, there is a J <= mid
  // such that
  //
  // All values c1[n], n < J, satisfy c1[n] < c1[mid] STRICTLY
  // All values c1[n], J <= n <= mid, satisfy c1[n] = c1[mid]
  // All values c1[n], mid < n <= right , satisfy c1[n] >= c1[mid]
  //
  // We need to roll back the "mid" value to the "J".  This
  // means our spatial regions are less balanced, but there
  // is no ambiguity regarding which region a point belongs in.

  int midValIndex = mid * 3 + dim;

  while ((mid > left) && (c1[midValIndex - 3] == c1[midValIndex]))
  {
    mid--;
    midValIndex -= 3;
  }

  if (mid == left)
  {
    return mid; // failed to divide region
  }

  float leftMax = svtkKdTree::FindMaxLeftHalf(dim, c1, mid);

  coord = (static_cast<double>(c1[midValIndex]) + static_cast<double>(leftMax)) / 2.0;

  return mid;
}

//----------------------------------------------------------------------------
float svtkKdTree::FindMaxLeftHalf(int dim, float* c1, int K)
{
  int i;

  float* Xcomponent = c1 + dim;
  float max = Xcomponent[0];

  for (i = 3; i < K * 3; i += 3)
  {
    if (Xcomponent[i] > max)
    {
      max = Xcomponent[i];
    }
  }
  return max;
}

//----------------------------------------------------------------------------
// Note: The indices (L, R, X) into the point array should be svtkIdType rather
// than ints, but this change causes the k-d tree build time to double.
// _Select is the heart of this build, called for every sub-interval that
// is to be reordered.  We will leave these as ints now.

void svtkKdTree::_Select(int dim, float* X, int* ids, int L, int R, int K)
{
  int N, I, J, S, SD, LL, RR;
  float Z, T;
  int manyTValues = 0;

  while (R > L)
  {
    if (R - L > 600)
    {
      // "Recurse on a sample of size S to get an estimate for the
      // (K-L+1)-th smallest element into X[K], biased slightly so
      // that the (K-L+1)-th element is expected to lie in the
      // smaller set after partitioning"

      N = R - L + 1;
      I = K - L + 1;
      Z = static_cast<float>(log(static_cast<float>(N)));
      S = static_cast<int>(.5 * exp(2 * Z / 3));
      SD = static_cast<int>(.5 * sqrt(Z * S * static_cast<float>(N - S) / N) * sign(I - N / 2));
      LL = svtkMath::Max(L, K - static_cast<int>(I * static_cast<float>(S) / N) + SD);
      RR = svtkMath::Min(R, K + static_cast<int>((N - I) * static_cast<float>(S) / N) + SD);
      _Select(dim, X, ids, LL, RR, K);
    }

    float* Xcomponent = X + dim; // x, y or z component

    T = Xcomponent[K * 3];

    // "the following code partitions X[L:R] about T."

    I = L;
    J = R;

    Exchange(X, ids, L, K);

    if (Xcomponent[R * 3] >= T)
    {
      if (Xcomponent[R * 3] == T)
        manyTValues++;
      Exchange(X, ids, R, L);
    }
    while (I < J)
    {
      Exchange(X, ids, I, J);

      while (Xcomponent[(++I) * 3] < T)
      {
        ;
      }

      while ((J > L) && (Xcomponent[(--J) * 3] >= T))
      {
        if (!manyTValues && (J > L) && (Xcomponent[J * 3] == T))
        {
          manyTValues = 1;
        }
      }
    }

    if (Xcomponent[L * 3] == T)
    {
      Exchange(X, ids, L, J);
    }
    else
    {
      J++;
      Exchange(X, ids, J, R);
    }

    if ((J < K) && manyTValues)
    {
      // Select has a worst case - when many of the same values
      // are repeated in an array.  It is bad enough that it is
      // worth detecting and optimizing for.  Here we're taking the
      // interval of values that are >= T, and rearranging it into
      // an interval of values = T followed by those > T.

      I = J;
      J = R + 1;

      while (I < J)
      {
        while ((++I < J) && (Xcomponent[I * 3] == T))
        {
          ;
        }
        if (I == J)
          break;

        while ((--J > I) && (Xcomponent[J * 3] > T))
        {
          ;
        }
        if (J == I)
          break;

        Exchange(X, ids, I, J);
      }

      // I and J are at the first array element that is > T
      // If K is within the sequence of T's, we're done, else
      // move the new [L,R] interval to the sequence of values
      // that are strictly greater than T.

      if (K < J)
      {
        J = K;
      }
      else
      {
        J = J - 1;
      }
    }

    // "now adjust L,R so they surround the subset containing
    // the (K-L+1)-th smallest element"

    if (J <= K)
    {
      L = J + 1;
    }
    if (K <= J)
    {
      R = J - 1;
    }
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::SelfRegister(svtkKdNode* kd)
{
  if (kd->GetLeft() == nullptr)
  {
    this->RegionList[kd->GetID()] = kd;
  }
  else
  {
    this->SelfRegister(kd->GetLeft());
    this->SelfRegister(kd->GetRight());
  }
}

//----------------------------------------------------------------------------
int svtkKdTree::SelfOrder(int startId, svtkKdNode* kd)
{
  int nextId;

  if (kd->GetLeft() == nullptr)
  {
    kd->SetID(startId);
    kd->SetMaxID(startId);
    kd->SetMinID(startId);

    nextId = startId + 1;
  }
  else
  {
    kd->SetID(-1);
    nextId = svtkKdTree::SelfOrder(startId, kd->GetLeft());
    nextId = svtkKdTree::SelfOrder(nextId, kd->GetRight());

    kd->SetMinID(startId);
    kd->SetMaxID(nextId - 1);
  }

  return nextId;
}

void svtkKdTree::BuildRegionList()
{
  SCOPETIMER("BuildRegionList");

  if (this->Top == nullptr)
  {
    return;
  }

  this->NumberOfRegions = svtkKdTree::SelfOrder(0, this->Top);

  this->RegionList = new svtkKdNode*[this->NumberOfRegions];

  this->SelfRegister(this->Top);
}

//----------------------------------------------------------------------------
// K-d tree from points, for finding duplicate and near-by points
//
void svtkKdTree::BuildLocatorFromPoints(svtkPointSet* pointset)
{
  this->BuildLocatorFromPoints(pointset->GetPoints());
}

void svtkKdTree::BuildLocatorFromPoints(svtkPoints* ptArray)
{
  this->BuildLocatorFromPoints(&ptArray, 1);
}

//----------------------------------------------------------------------------
void svtkKdTree::BuildLocatorFromPoints(svtkPoints** ptArrays, int numPtArrays)
{
  int ptId;
  int i;

  int totalNumPoints = 0;

  for (i = 0; i < numPtArrays; i++)
  {
    totalNumPoints += ptArrays[i]->GetNumberOfPoints();
  }

  if (totalNumPoints < 1)
  {
    svtkErrorMacro(<< "svtkKdTree::BuildLocatorFromPoints - no points");
    return;
  }

  if (totalNumPoints >= SVTK_INT_MAX)
  {
    // The heart of the k-d tree build is the recursive median find in
    // _Select.  It rearranges point IDs along with points themselves.
    // When point IDs are stored in an "int" instead of a svtkIdType,
    // performance doubles.  So we store point IDs in an "int" during
    // the calculation.  This will need to be rewritten if true 64 bit
    // IDs are required.

    svtkErrorMacro(<< "BuildLocatorFromPoints - intentional 64 bit error - time to rewrite code");
    return;
  }

  svtkDebugMacro(<< "Creating Kdtree");

  if ((this->Timing) && (this->TimerLog == nullptr))
  {
    this->TimerLog = svtkTimerLog::New();
  }

  TIMER("Set up to build k-d tree");

  this->FreeSearchStructure();
  this->ClearLastBuildCache();

  // Fix bounds - (1) push out a little if flat
  // (2) pull back the x, y and z lower bounds a little bit so that
  // points are clearly "inside" the spatial region.  Point p is
  // "inside" region r = [r1, r2] if r1 < p <= r2.

  double bounds[6], diff[3];

  ptArrays[0]->GetBounds(bounds);

  for (i = 1; i < numPtArrays; i++)
  {
    double tmpbounds[6];
    ptArrays[i]->GetBounds(tmpbounds);

    if (tmpbounds[0] < bounds[0])
    {
      bounds[0] = tmpbounds[0];
    }
    if (tmpbounds[2] < bounds[2])
    {
      bounds[2] = tmpbounds[2];
    }
    if (tmpbounds[4] < bounds[4])
    {
      bounds[4] = tmpbounds[4];
    }
    if (tmpbounds[1] > bounds[1])
    {
      bounds[1] = tmpbounds[1];
    }
    if (tmpbounds[3] > bounds[3])
    {
      bounds[3] = tmpbounds[3];
    }
    if (tmpbounds[5] > bounds[5])
    {
      bounds[5] = tmpbounds[5];
    }
  }

  this->MaxWidth = 0.0;

  for (i = 0; i < 3; i++)
  {
    diff[i] = bounds[2 * i + 1] - bounds[2 * i];
    this->MaxWidth = static_cast<float>((diff[i] > this->MaxWidth) ? diff[i] : this->MaxWidth);
  }

  this->FudgeFactor = this->MaxWidth * 10e-6;

  double aLittle = this->MaxWidth * 10e-2;

  for (i = 0; i < 3; i++)
  {
    if (diff[i] < aLittle) // case (1) above
    {
      double temp = bounds[2 * i];
      bounds[2 * i] = bounds[2 * i + 1] - aLittle;
      bounds[2 * i + 1] = temp + aLittle;
    }
    else // case (2) above
    {
      bounds[2 * i] -= this->FudgeFactor;
      bounds[2 * i + 1] += this->FudgeFactor;
    }
  }

  // root node of k-d tree - it's the whole space

  svtkKdNode* kd = this->Top = svtkKdNode::New();

  kd->SetBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

  kd->SetNumberOfPoints(totalNumPoints);

  kd->SetDataBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

  this->LocatorIds = new int[totalNumPoints];
  this->LocatorPoints = new float[3 * totalNumPoints];

  if (!this->LocatorPoints || !this->LocatorIds)
  {
    this->FreeSearchStructure();
    svtkErrorMacro(<< "svtkKdTree::BuildLocatorFromPoints - memory allocation");
    return;
  }

  int* ptIds = this->LocatorIds;
  float* points = this->LocatorPoints;

  for (i = 0, ptId = 0; i < numPtArrays; i++)
  {
    int npoints = ptArrays[i]->GetNumberOfPoints();
    int nvals = npoints * 3;

    int pointArrayType = ptArrays[i]->GetDataType();

    if (pointArrayType == SVTK_FLOAT)
    {
      svtkDataArray* da = ptArrays[i]->GetData();
      svtkFloatArray* fa = svtkArrayDownCast<svtkFloatArray>(da);
      memcpy(points + ptId, fa->GetPointer(0), sizeof(float) * nvals);
      ptId += nvals;
    }
    else
    {
      // Hopefully point arrays are usually floats.  This conversion will
      // really slow things down.

      for (svtkIdType ii = 0; ii < npoints; ii++)
      {
        double* pt = ptArrays[i]->GetPoint(ii);

        points[ptId++] = static_cast<float>(pt[0]);
        points[ptId++] = static_cast<float>(pt[1]);
        points[ptId++] = static_cast<float>(pt[2]);
      }
    }
  }

  for (ptId = 0; ptId < totalNumPoints; ptId++)
  {
    // _Select dominates DivideRegion algorithm, operating on
    // ints is much fast than operating on long longs

    ptIds[ptId] = ptId;
  }

  TIMERDONE("Set up to build k-d tree");

  TIMER("Build tree");

  this->DivideRegion(kd, points, ptIds, 0);

  this->SetActualLevel();
  this->BuildRegionList();

  // Create a location array for the points of each region

  this->LocatorRegionLocation = new int[this->NumberOfRegions];

  int idx = 0;

  for (int reg = 0; reg < this->NumberOfRegions; reg++)
  {
    this->LocatorRegionLocation[reg] = idx;

    idx += this->RegionList[reg]->GetNumberOfPoints();
  }

  this->NumberOfLocatorPoints = idx;

  this->SetCalculator(this->Top);

  TIMERDONE("Build tree");
}

//----------------------------------------------------------------------------
// Query functions subsequent to BuildLocatorFromPoints,
// relating to duplicate and nearby points
//
svtkIdTypeArray* svtkKdTree::BuildMapForDuplicatePoints(float tolerance = 0.0)
{
  int i;

  if ((tolerance < 0.0) || (tolerance >= this->MaxWidth))
  {
    svtkWarningMacro(<< "svtkKdTree::BuildMapForDuplicatePoints - invalid tolerance");
    tolerance = this->MaxWidth;
  }

  TIMER("Find duplicate points");

  int* idCount = new int[this->NumberOfRegions];
  int** uniqueFound = new int*[this->NumberOfRegions];

  if (!idCount || !uniqueFound)
  {
    delete[] idCount;
    delete[] uniqueFound;

    svtkErrorMacro(<< "svtkKdTree::BuildMapForDuplicatePoints memory allocation");
    return nullptr;
  }

  memset(idCount, 0, sizeof(int) * this->NumberOfRegions);

  for (i = 0; i < this->NumberOfRegions; i++)
  {
    uniqueFound[i] = new int[this->RegionList[i]->GetNumberOfPoints()];

    if (!uniqueFound[i])
    {
      delete[] idCount;
      for (int j = 0; j < i; j++)
        delete[] uniqueFound[j];
      delete[] uniqueFound;
      svtkErrorMacro(<< "svtkKdTree::BuildMapForDuplicatePoints memory allocation");
      return nullptr;
    }
  }

  float tolerance2 = tolerance * tolerance;

  svtkIdTypeArray* uniqueIds = svtkIdTypeArray::New();
  uniqueIds->SetNumberOfValues(this->NumberOfLocatorPoints);

  int idx = 0;
  int nextRegionId = 0;
  float* point = this->LocatorPoints;

  while (idx < this->NumberOfLocatorPoints)
  {
    // first point we have in this region

    int currentId = this->LocatorIds[idx];

    int regionId = this->GetRegionContainingPoint(point[0], point[1], point[2]);

    if ((regionId == -1) || (regionId != nextRegionId))
    {
      delete[] idCount;
      for (i = 0; i < this->NumberOfRegions; i++)
        delete[] uniqueFound[i];
      delete[] uniqueFound;
      uniqueIds->Delete();
      svtkErrorMacro(<< "svtkKdTree::BuildMapForDuplicatePoints corrupt k-d tree");
      return nullptr;
    }

    int duplicateFound = -1;

    if ((tolerance > 0.0) && (regionId > 0))
    {
      duplicateFound = this->SearchNeighborsForDuplicate(
        regionId, point, uniqueFound, idCount, tolerance, tolerance2);
    }

    if (duplicateFound >= 0)
    {
      uniqueIds->SetValue(currentId, this->LocatorIds[duplicateFound]);
    }
    else
    {
      uniqueFound[regionId][idCount[regionId]++] = idx;
      uniqueIds->SetValue(currentId, currentId);
    }

    // test the other points in this region

    int numRegionPoints = this->RegionList[regionId]->GetNumberOfPoints();

    int secondIdx = idx + 1;
    int nextFirstIdx = idx + numRegionPoints;

    for (int idx2 = secondIdx; idx2 < nextFirstIdx; idx2++)
    {
      point += 3;
      currentId = this->LocatorIds[idx2];

      duplicateFound =
        this->SearchRegionForDuplicate(point, uniqueFound[regionId], idCount[regionId], tolerance2);

      if ((tolerance > 0.0) && (duplicateFound < 0) && (regionId > 0))
      {
        duplicateFound = this->SearchNeighborsForDuplicate(
          regionId, point, uniqueFound, idCount, tolerance, tolerance2);
      }

      if (duplicateFound >= 0)
      {
        uniqueIds->SetValue(currentId, this->LocatorIds[duplicateFound]);
      }
      else
      {
        uniqueFound[regionId][idCount[regionId]++] = idx2;
        uniqueIds->SetValue(currentId, currentId);
      }
    }

    idx = nextFirstIdx;
    point += 3;
    nextRegionId++;
  }

  for (i = 0; i < this->NumberOfRegions; i++)
  {
    delete[] uniqueFound[i];
  }
  delete[] uniqueFound;
  delete[] idCount;

  TIMERDONE("Find duplicate points");

  return uniqueIds;
}

//----------------------------------------------------------------------------
int svtkKdTree::SearchRegionForDuplicate(float* point, int* pointsSoFar, int len, float tolerance2)
{
  int duplicateFound = -1;
  int id;

  for (id = 0; id < len; id++)
  {
    int otherId = pointsSoFar[id];

    float* otherPoint = this->LocatorPoints + (otherId * 3);

    float distance2 = svtkMath::Distance2BetweenPoints(point, otherPoint);

    if (distance2 <= tolerance2)
    {
      duplicateFound = otherId;
      break;
    }
  }
  return duplicateFound;
}

//----------------------------------------------------------------------------
int svtkKdTree::SearchNeighborsForDuplicate(
  int regionId, float* point, int** pointsSoFar, int* len, float tolerance, float tolerance2)
{
  int duplicateFound = -1;

  float dist2 =
    this->RegionList[regionId]->GetDistance2ToInnerBoundary(point[0], point[1], point[2]);

  if (dist2 >= tolerance2)
  {
    // There are no other regions with data that are within the
    // tolerance distance of this point.

    return duplicateFound;
  }

  // Find all regions that are within the tolerance distance of the point

  int* regionIds = new int[this->NumberOfRegions];

  this->BSPCalculator->ComputeIntersectionsUsingDataBoundsOn();

#ifdef USE_SPHERE

  // Find all regions which intersect a sphere around the point
  // with radius equal to tolerance.  Compute the intersection using
  // the bounds of data within regions, not the bounds of the region.

  int nRegions = this->BSPCalculator->IntersectsSphere2(
    regionIds, this->NumberOfRegions, point[0], point[1], point[2], tolerance2);
#else

  // Technically, we want all regions that intersect a sphere around the
  // point. But this takes much longer to compute than a box.  We'll compute
  // all regions that intersect a box.  We may occasionally get a region
  // we don't need, but that's OK.

  double box[6];
  box[0] = point[0] - tolerance;
  box[1] = point[0] + tolerance;
  box[2] = point[1] - tolerance;
  box[3] = point[1] + tolerance;
  box[4] = point[2] - tolerance;
  box[5] = point[2] + tolerance;

  int nRegions = this->BSPCalculator->IntersectsBox(regionIds, this->NumberOfRegions, box);

#endif

  this->BSPCalculator->ComputeIntersectionsUsingDataBoundsOff();

  for (int reg = 0; reg < nRegions; reg++)
  {
    if ((regionIds[reg] == regionId) || (len[reg] == 0))
    {
      continue;
    }

    duplicateFound = this->SearchRegionForDuplicate(point, pointsSoFar[reg], len[reg], tolerance2);

    if (duplicateFound)
    {
      break;
    }
  }

  delete[] regionIds;

  return duplicateFound;
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindPoint(double x[3])
{
  return this->FindPoint(x[0], x[1], x[2]);
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindPoint(double x, double y, double z)
{
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindPoint - must build locator first");
    return -1;
  }

  int regionId = this->GetRegionContainingPoint(x, y, z);

  if (regionId == -1)
  {
    return -1;
  }

  int idx = this->LocatorRegionLocation[regionId];

  svtkIdType ptId = -1;

  float* point = this->LocatorPoints + (idx * 3);

  float fx = static_cast<float>(x);
  float fy = static_cast<float>(y);
  float fz = static_cast<float>(z);

  for (int i = 0; i < this->RegionList[regionId]->GetNumberOfPoints(); i++)
  {
    if ((point[0] == fx) && (point[1] == fy) && (point[2] == fz))
    {
      ptId = static_cast<svtkIdType>(this->LocatorIds[idx + i]);
      break;
    }

    point += 3;
  }

  return ptId;
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindClosestPoint(double x[3], double& dist2)
{
  svtkIdType id = this->FindClosestPoint(x[0], x[1], x[2], dist2);

  return id;
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindClosestPoint(double x, double y, double z, double& dist2)
{
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindClosestPoint: must build locator first");
    return -1;
  }

  double minDistance2 = 0.0;

  int closeId = -1, newCloseId = -1;
  double newDistance2 = 4 * this->MaxWidth * this->MaxWidth;

  int regionId = this->GetRegionContainingPoint(x, y, z);

  if (regionId < 0)
  {
    // This point is not inside the space divided by the k-d tree.
    // Find the point on the boundary that is closest to it.

    double pt[3];
    this->Top->GetDistance2ToBoundary(x, y, z, pt, 1);

    double* min = this->Top->GetMinBounds();
    double* max = this->Top->GetMaxBounds();

    // GetDistance2ToBoundary will sometimes return a point *just*
    // *barely* outside the bounds of the region.  Move that point to
    // just barely *inside* instead.

    if (pt[0] <= min[0])
    {
      pt[0] = min[0] + this->FudgeFactor;
    }
    if (pt[1] <= min[1])
    {
      pt[1] = min[1] + this->FudgeFactor;
    }
    if (pt[2] <= min[2])
    {
      pt[2] = min[2] + this->FudgeFactor;
    }
    if (pt[0] >= max[0])
    {
      pt[0] = max[0] - this->FudgeFactor;
    }
    if (pt[1] >= max[1])
    {
      pt[1] = max[1] - this->FudgeFactor;
    }
    if (pt[2] >= max[2])
    {
      pt[2] = max[2] - this->FudgeFactor;
    }

    regionId = this->GetRegionContainingPoint(pt[0], pt[1], pt[2]);

    closeId = this->_FindClosestPointInRegion(regionId, x, y, z, minDistance2);

    // Check to see if neighboring regions have a closer point

    newCloseId = this->FindClosestPointInSphere(x, y, z,
      sqrt(minDistance2), // radius
      regionId,           // skip this region
      newDistance2);      // distance to closest point
  }
  else // Point is inside a k-d tree region
  {
    closeId = this->_FindClosestPointInRegion(regionId, x, y, z, minDistance2);

    if (minDistance2 > 0.0)
    {
      float dist2ToBoundary = this->RegionList[regionId]->GetDistance2ToInnerBoundary(x, y, z);

      if (dist2ToBoundary < minDistance2)
      {
        // The closest point may be in a neighboring region

        newCloseId = this->FindClosestPointInSphere(x, y, z,
          sqrt(minDistance2), // radius
          regionId,           // skip this region
          newDistance2);
      }
    }
  }

  if (newDistance2 < minDistance2 && newCloseId != -1)
  {
    closeId = newCloseId;
    minDistance2 = newDistance2;
  }

  svtkIdType closePointId = static_cast<svtkIdType>(this->LocatorIds[closeId]);

  dist2 = minDistance2;

  return closePointId;
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindClosestPointWithinRadius(double radius, const double x[3], double& dist2)
{
  int localCloseId = this->FindClosestPointInSphere(x[0], x[1], x[2], radius, -1, dist2);
  if (localCloseId >= 0)
  {
    return static_cast<svtkIdType>(this->LocatorIds[localCloseId]);
  }
  return -1;
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindClosestPointInRegion(int regionId, double* x, double& dist2)
{
  return this->FindClosestPointInRegion(regionId, x[0], x[1], x[2], dist2);
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::FindClosestPointInRegion(
  int regionId, double x, double y, double z, double& dist2)
{
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindClosestPointInRegion - must build locator first");
    return -1;
  }
  int localId = this->_FindClosestPointInRegion(regionId, x, y, z, dist2);

  svtkIdType originalId = -1;

  if (localId >= 0)
  {
    originalId = static_cast<svtkIdType>(this->LocatorIds[localId]);
  }

  return originalId;
}

//----------------------------------------------------------------------------
int svtkKdTree::_FindClosestPointInRegion(int regionId, double x, double y, double z, double& dist2)
{
  int minId = 0;

  double minDistance2 = 4 * this->MaxWidth * this->MaxWidth;

  int idx = this->LocatorRegionLocation[regionId];

  float* candidate = this->LocatorPoints + (idx * 3);

  int numPoints = this->RegionList[regionId]->GetNumberOfPoints();
  for (int i = 0; i < numPoints; i++)
  {
    double dx = (x - candidate[0]) * (x - candidate[0]);

    if (dx < minDistance2)
    {
      double dxy = dx + ((y - candidate[1]) * (y - candidate[1]));

      if (dxy < minDistance2)
      {
        double dxyz = dxy + ((z - candidate[2]) * (z - candidate[2]));

        if (dxyz < minDistance2)
        {
          minId = idx + i;
          minDistance2 = dxyz;

          if (dxyz == 0.0)
          {
            break;
          }
        }
      }
    }

    candidate += 3;
  }

  dist2 = minDistance2;

  return minId;
}

//----------------------------------------------------------------------------
int svtkKdTree::FindClosestPointInSphere(
  double x, double y, double z, double radius, int skipRegion, double& dist2)
{
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindClosestPointInSphere - must build locator first");
    return -1;
  }
  int* regionIds = new int[this->NumberOfRegions];

  this->BSPCalculator->ComputeIntersectionsUsingDataBoundsOn();

  int nRegions = this->BSPCalculator->IntersectsSphere2(
    regionIds, this->NumberOfRegions, x, y, z, radius * radius);

  this->BSPCalculator->ComputeIntersectionsUsingDataBoundsOff();

  double minDistance2 = 4 * this->MaxWidth * this->MaxWidth;
  int localCloseId = -1;

  bool recheck = 0; // used to flag that we should recheck the distance
  for (int reg = 0; reg < nRegions; reg++)
  {
    if (regionIds[reg] == skipRegion)
    {
      continue;
    }

    int neighbor = regionIds[reg];

    // recheck that the bin is closer than the current minimum distance
    if (!recheck || this->RegionList[neighbor]->GetDistance2ToBoundary(x, y, z, 1) < minDistance2)
    {
      double newDistance2;
      int newLocalCloseId = this->_FindClosestPointInRegion(neighbor, x, y, z, newDistance2);

      if (newDistance2 < minDistance2 && newDistance2 <= radius * radius)
      {
        minDistance2 = newDistance2;
        localCloseId = newLocalCloseId;
        recheck = 1; // changed the minimum distance so mark to check subsequent bins
      }
    }
  }

  delete[] regionIds;

  dist2 = minDistance2;
  return localCloseId;
}

//----------------------------------------------------------------------------
void svtkKdTree::FindPointsWithinRadius(double R, const double x[3], svtkIdList* result)
{
  result->Reset();
  // don't forget to square the radius
  this->FindPointsWithinRadius(this->Top, R * R, x, result);
}

//----------------------------------------------------------------------------
void svtkKdTree::FindPointsWithinRadius(
  svtkKdNode* node, double R2, const double x[3], svtkIdList* result)

{
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindPointsWithinRadius - must build locator first");
    return;
  }

  double b[6];
  node->GetBounds(b);

  double mindist2 = 0; // distance to closest vertex of BB
  double maxdist2 = 0; // distance to furthest vertex of BB
  // x-dir
  if (x[0] < b[0])
  {
    mindist2 = (b[0] - x[0]) * (b[0] - x[0]);
    maxdist2 = (b[1] - x[0]) * (b[1] - x[0]);
  }
  else if (x[0] > b[1])
  {
    mindist2 = (b[1] - x[0]) * (b[1] - x[0]);
    maxdist2 = (b[0] - x[0]) * (b[0] - x[0]);
  }
  else if ((b[1] - x[0]) > (x[0] - b[0]))
  {
    maxdist2 = (b[1] - x[0]) * (b[1] - x[0]);
  }
  else
  {
    maxdist2 = (b[0] - x[0]) * (b[0] - x[0]);
  }
  // y-dir
  if (x[1] < b[2])
  {
    mindist2 += (b[2] - x[1]) * (b[2] - x[1]);
    maxdist2 += (b[3] - x[1]) * (b[3] - x[1]);
  }
  else if (x[1] > b[3])
  {
    mindist2 += (b[3] - x[1]) * (b[3] - x[1]);
    maxdist2 += (b[2] - x[1]) * (b[2] - x[1]);
  }
  else if ((b[3] - x[1]) > (x[1] - b[2]))
  {
    maxdist2 += (b[3] - x[1]) * (b[3] - x[1]);
  }
  else
  {
    maxdist2 += (b[2] - x[1]) * (b[2] - x[1]);
  }
  // z-dir
  if (x[2] < b[4])
  {
    mindist2 += (b[4] - x[2]) * (b[4] - x[2]);
    maxdist2 += (b[5] - x[2]) * (b[5] - x[2]);
  }
  else if (x[2] > b[5])
  {
    mindist2 += (b[5] - x[2]) * (b[5] - x[2]);
    maxdist2 += (b[4] - x[2]) * (b[4] - x[2]);
  }
  else if ((b[5] - x[2]) > (x[2] - b[4]))
  {
    maxdist2 += (b[5] - x[2]) * (b[5] - x[2]);
  }
  else
  {
    maxdist2 += (x[2] - b[4]) * (x[2] - b[4]);
  }

  if (mindist2 > R2)
  {
    // non-intersecting
    return;
  }

  if (maxdist2 <= R2)
  {
    // sphere contains BB
    this->AddAllPointsInRegion(node, result);
    return;
  }

  // partial intersection of sphere & BB
  if (node->GetLeft() == nullptr)
  {
    int regionID = node->GetID();
    int regionLoc = this->LocatorRegionLocation[regionID];
    float* pt = this->LocatorPoints + (regionLoc * 3);
    svtkIdType numPoints = this->RegionList[regionID]->GetNumberOfPoints();
    for (svtkIdType i = 0; i < numPoints; i++)
    {
      double dist2 = (pt[0] - x[0]) * (pt[0] - x[0]) + (pt[1] - x[1]) * (pt[1] - x[1]) +
        (pt[2] - x[2]) * (pt[2] - x[2]);
      if (dist2 <= R2)
      {
        svtkIdType ptId = static_cast<svtkIdType>(this->LocatorIds[regionLoc + i]);
        result->InsertNextId(ptId);
      }
      pt += 3;
    }
  }
  else
  {
    this->FindPointsWithinRadius(node->GetLeft(), R2, x, result);
    this->FindPointsWithinRadius(node->GetRight(), R2, x, result);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::FindClosestNPoints(int N, const double x[3], svtkIdList* result)
{
  result->Reset();
  if (N <= 0)
  {
    return;
  }
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindClosestNPoints - must build locator first");
    return;
  }

  int numTotalPoints = this->Top->GetNumberOfPoints();
  if (numTotalPoints < N)
  {
    svtkWarningMacro("Number of requested points is greater than total number of points in KdTree");
    N = numTotalPoints;
  }
  result->SetNumberOfIds(N);

  // now we want to go about finding a region that contains at least N points
  // but not many more -- hopefully the region contains X as well but we
  // can't depend on that
  svtkKdNode* node = this->Top;
  svtkKdNode* startingNode = nullptr;
  if (!node->ContainsPoint(x[0], x[1], x[2], 0))
  {
    // point is not in the region
    int numPoints = node->GetNumberOfPoints();
    svtkKdNode* prevNode = node;
    while (node->GetLeft() && numPoints > N)
    {
      prevNode = node;
      double leftDist2 = node->GetLeft()->GetDistance2ToBoundary(x[0], x[1], x[2], 1);
      double rightDist2 = node->GetRight()->GetDistance2ToBoundary(x[0], x[1], x[2], 1);
      if (leftDist2 < rightDist2)
      {
        node = node->GetLeft();
      }
      else
      {
        node = node->GetRight();
      }
      numPoints = node->GetNumberOfPoints();
    }
    if (numPoints < N)
    {
      startingNode = prevNode;
    }
    else
    {
      startingNode = node;
    }
  }
  else
  {
    int numPoints = node->GetNumberOfPoints();
    svtkKdNode* prevNode = node;
    while (node->GetLeft() && numPoints > N)
    {
      prevNode = node;
      if (node->GetLeft()->ContainsPoint(x[0], x[1], x[2], 0))
      {
        node = node->GetLeft();
      }
      else
      {
        node = node->GetRight();
      }
      numPoints = node->GetNumberOfPoints();
    }
    if (numPoints < N)
    {
      startingNode = prevNode;
    }
    else
    {
      startingNode = node;
    }
  }

  // now that we have a starting region, go through its points
  // and order them
  int regionId = startingNode->GetID();
  int numPoints = startingNode->GetNumberOfPoints();
  int where;
  if (regionId >= 0)
  {
    where = this->LocatorRegionLocation[regionId];
  }
  else
  {
    svtkKdNode* left = startingNode->GetLeft();
    svtkKdNode* next = left->GetLeft();
    while (next)
    {
      left = next;
      next = next->GetLeft();
    }
    int leftRegionId = left->GetID();
    where = this->LocatorRegionLocation[leftRegionId];
  }
  int* ids = this->LocatorIds + where;
  float* pt = this->LocatorPoints + (where * 3);
  float xfloat[3] = { static_cast<float>(x[0]), static_cast<float>(x[1]),
    static_cast<float>(x[2]) };
  OrderPoints orderedPoints(N);
  for (int i = 0; i < numPoints; i++)
  {
    float dist2 = svtkMath::Distance2BetweenPoints(xfloat, pt);
    orderedPoints.InsertPoint(dist2, ids[i]);
    pt += 3;
  }

  // to finish up we have to check other regions for
  // closer points
  float LargestDist2 = orderedPoints.GetLargestDist2();
  double delta[3] = { 0, 0, 0 };
  double bounds[6];
  node = this->Top;
  std::queue<svtkKdNode*> nodesToBeSearched;
  nodesToBeSearched.push(node);
  while (!nodesToBeSearched.empty())
  {
    node = nodesToBeSearched.front();
    nodesToBeSearched.pop();
    if (node == startingNode)
    {
      continue;
    }
    svtkKdNode* left = node->GetLeft();
    if (left)
    {
      left->GetDataBounds(bounds);
      if (svtkMath::PointIsWithinBounds(const_cast<double*>(x), bounds, delta) == 1 ||
        left->GetDistance2ToBoundary(x[0], x[1], x[2], 1) < LargestDist2)
      {
        nodesToBeSearched.push(left);
      }
      node->GetRight()->GetDataBounds(bounds);
      if (svtkMath::PointIsWithinBounds(const_cast<double*>(x), bounds, delta) == 1 ||
        node->GetRight()->GetDistance2ToBoundary(x[0], x[1], x[2], 1) < LargestDist2)
      {
        nodesToBeSearched.push(node->GetRight());
      }
    }
    else if (node->GetDistance2ToBoundary(x[0], x[1], x[2], 1) < LargestDist2)
    {
      regionId = node->GetID();
      numPoints = node->GetNumberOfPoints();
      where = this->LocatorRegionLocation[regionId];
      ids = this->LocatorIds + where;
      pt = this->LocatorPoints + (where * 3);
      for (int i = 0; i < numPoints; i++)
      {
        float dist2 = svtkMath::Distance2BetweenPoints(xfloat, pt);
        orderedPoints.InsertPoint(dist2, ids[i]);
        pt += 3;
      }
      LargestDist2 = orderedPoints.GetLargestDist2();
    }
  }
  orderedPoints.GetSortedIds(result);
}

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkKdTree::GetPointsInRegion(int regionId)
{
  if ((regionId < 0) || (regionId >= this->NumberOfRegions))
  {
    svtkErrorMacro(<< "svtkKdTree::GetPointsInRegion invalid region ID");
    return nullptr;
  }

  if (!this->LocatorIds)
  {
    svtkErrorMacro(<< "svtkKdTree::GetPointsInRegion build locator first");
    return nullptr;
  }

  int numPoints = this->RegionList[regionId]->GetNumberOfPoints();
  int where = this->LocatorRegionLocation[regionId];

  svtkIdTypeArray* ptIds = svtkIdTypeArray::New();
  ptIds->SetNumberOfValues(numPoints);

  int* ids = this->LocatorIds + where;

  for (int i = 0; i < numPoints; i++)
  {
    ptIds->SetValue(i, ids[i]);
  }

  return ptIds;
}

//----------------------------------------------------------------------------
// Code to save state/time of last k-d tree build, and to
// determine if a data set's geometry has changed since the
// last build.
//
void svtkKdTree::InvalidateGeometry()
{
  // Remove observers to data sets.
  for (int i = 0; i < this->LastNumDataSets; i++)
  {
    this->LastInputDataSets[i]->RemoveObserver(this->LastDataSetObserverTags[i]);
  }

  this->LastNumDataSets = 0;
}

//-----------------------------------------------------------------------------
void svtkKdTree::ClearLastBuildCache()
{
  this->InvalidateGeometry();

  if (this->LastDataCacheSize > 0)
  {
    delete[] this->LastInputDataSets;
    delete[] this->LastDataSetObserverTags;
    delete[] this->LastDataSetType;
    delete[] this->LastInputDataInfo;
    delete[] this->LastBounds;
    delete[] this->LastNumCells;
    delete[] this->LastNumPoints;
    this->LastDataCacheSize = 0;
  }
  this->LastNumDataSets = 0;
  this->LastInputDataSets = nullptr;
  this->LastDataSetObserverTags = nullptr;
  this->LastDataSetType = nullptr;
  this->LastInputDataInfo = nullptr;
  this->LastBounds = nullptr;
  this->LastNumPoints = nullptr;
  this->LastNumCells = nullptr;
}

//----------------------------------------------------------------------------
void svtkKdTree::UpdateBuildTime()
{
  this->BuildTime.Modified();

  // Save enough information so that next time we execute,
  // we can determine whether input geometry has changed.

  this->InvalidateGeometry();

  int numDataSets = this->GetNumberOfDataSets();
  if (numDataSets > this->LastDataCacheSize)
  {
    this->ClearLastBuildCache();

    this->LastInputDataSets = new svtkDataSet*[numDataSets];
    this->LastDataSetObserverTags = new unsigned long[numDataSets];
    this->LastDataSetType = new int[numDataSets];
    this->LastInputDataInfo = new double[9 * numDataSets];
    this->LastBounds = new double[6 * numDataSets];
    this->LastNumPoints = new svtkIdType[numDataSets];
    this->LastNumCells = new svtkIdType[numDataSets];
    this->LastDataCacheSize = numDataSets;
  }

  this->LastNumDataSets = numDataSets;

  int nextds = 0;

  svtkCollectionSimpleIterator cookie;
  this->DataSets->InitTraversal(cookie);
  for (svtkDataSet* in = this->DataSets->GetNextDataSet(cookie); in != nullptr;
       in = this->DataSets->GetNextDataSet(cookie))
  {
    if (nextds >= numDataSets)
    {
      svtkErrorMacro(<< "svtkKdTree::UpdateBuildTime corrupt counts");
      return;
    }

    svtkCallbackCommand* cbc = svtkCallbackCommand::New();
    cbc->SetCallback(LastInputDeletedCallback);
    cbc->SetClientData(this);
    this->LastDataSetObserverTags[nextds] = in->AddObserver(svtkCommand::DeleteEvent, cbc);
    cbc->Delete();

    this->LastInputDataSets[nextds] = in;

    this->LastNumPoints[nextds] = in->GetNumberOfPoints();
    this->LastNumCells[nextds] = in->GetNumberOfCells();

    in->GetBounds(this->LastBounds + 6 * nextds);

    int type = this->LastDataSetType[nextds] = in->GetDataObjectType();

    if ((type == SVTK_IMAGE_DATA) || (type == SVTK_UNIFORM_GRID))
    {
      double origin[3], spacing[3];
      int dims[3];

      if (type == SVTK_IMAGE_DATA)
      {
        svtkImageData* id = svtkImageData::SafeDownCast(in);
        id->GetDimensions(dims);
        id->GetOrigin(origin);
        id->GetSpacing(spacing);
      }
      else
      {
        svtkUniformGrid* ug = svtkUniformGrid::SafeDownCast(in);
        ug->GetDimensions(dims);
        ug->GetOrigin(origin);
        ug->GetSpacing(spacing);
      }

      this->SetInputDataInfo(nextds, dims, origin, spacing);
    }

    nextds++;
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::SetInputDataInfo(int i, int dims[3], double origin[3], double spacing[3])
{
  int idx = 9 * i;
  this->LastInputDataInfo[idx++] = static_cast<double>(dims[0]);
  this->LastInputDataInfo[idx++] = static_cast<double>(dims[1]);
  this->LastInputDataInfo[idx++] = static_cast<double>(dims[2]);
  this->LastInputDataInfo[idx++] = origin[0];
  this->LastInputDataInfo[idx++] = origin[1];
  this->LastInputDataInfo[idx++] = origin[2];
  this->LastInputDataInfo[idx++] = spacing[0];
  this->LastInputDataInfo[idx++] = spacing[1];
  this->LastInputDataInfo[idx++] = spacing[2];
}

//----------------------------------------------------------------------------
int svtkKdTree::CheckInputDataInfo(int i, int dims[3], double origin[3], double spacing[3])
{
  int sameValues = 1;
  int idx = 9 * i;

  if ((dims[0] != static_cast<int>(this->LastInputDataInfo[idx])) ||
    (dims[1] != static_cast<int>(this->LastInputDataInfo[idx + 1])) ||
    (dims[2] != static_cast<int>(this->LastInputDataInfo[idx + 2])) ||
    (origin[0] != this->LastInputDataInfo[idx + 3]) ||
    (origin[1] != this->LastInputDataInfo[idx + 4]) ||
    (origin[2] != this->LastInputDataInfo[idx + 5]) ||
    (spacing[0] != this->LastInputDataInfo[idx + 6]) ||
    (spacing[1] != this->LastInputDataInfo[idx + 7]) ||
    (spacing[2] != this->LastInputDataInfo[idx + 8]))
  {
    sameValues = 0;
  }

  return sameValues;
}

//----------------------------------------------------------------------------
int svtkKdTree::NewGeometry()
{
  if (this->GetNumberOfDataSets() != this->LastNumDataSets)
  {
    return 1;
  }

  svtkDataSet** tmp = new svtkDataSet*[this->GetNumberOfDataSets()];
  for (int i = 0; i < this->GetNumberOfDataSets(); i++)
  {
    tmp[i] = this->GetDataSet(i);
  }

  int itsNew = this->NewGeometry(tmp, this->GetNumberOfDataSets());

  delete[] tmp;

  return itsNew;
}
int svtkKdTree::NewGeometry(svtkDataSet** sets, int numSets)
{
  int newGeometry = 0;
#if 0
  svtkPointSet *ps = nullptr;
#endif
  svtkRectilinearGrid* rg = nullptr;
  svtkImageData* id = nullptr;
  svtkUniformGrid* ug = nullptr;
  int same = 0;
  int dims[3];
  double origin[3], spacing[3];

  if (numSets != this->LastNumDataSets)
  {
    return 1;
  }

  for (int i = 0; i < numSets; i++)
  {
    svtkDataSet* in = this->LastInputDataSets[i];
    int type = in->GetDataObjectType();

    if (type != this->LastDataSetType[i])
    {
      newGeometry = 1;
      break;
    }

    switch (type)
    {
      case SVTK_POLY_DATA:
      case SVTK_UNSTRUCTURED_GRID:
      case SVTK_STRUCTURED_GRID:
#if 0
        // For now, svtkPExodusReader creates a whole new
        // svtkUnstructuredGrid, even when just changing
        // field arrays.  So we'll just check bounds.
        ps = svtkPointSet::SafeDownCast(in);
        if (ps->GetPoints()->GetMTime() > this->BuildTime)
        {
          newGeometry = 1;
        }
#else
        if ((sets[i]->GetNumberOfPoints() != this->LastNumPoints[i]) ||
          (sets[i]->GetNumberOfCells() != this->LastNumCells[i]))
        {
          newGeometry = 1;
        }
        else
        {
          double b[6];
          sets[i]->GetBounds(b);
          double* lastb = this->LastBounds + 6 * i;

          for (int dim = 0; dim < 6; dim++)
          {
            if (*lastb++ != b[dim])
            {
              newGeometry = 1;
              break;
            }
          }
        }
#endif
        break;

      case SVTK_RECTILINEAR_GRID:

        rg = svtkRectilinearGrid::SafeDownCast(in);
        if ((rg->GetXCoordinates()->GetMTime() > this->BuildTime) ||
          (rg->GetYCoordinates()->GetMTime() > this->BuildTime) ||
          (rg->GetZCoordinates()->GetMTime() > this->BuildTime))
        {
          newGeometry = 1;
        }
        break;

      case SVTK_IMAGE_DATA:
      case SVTK_STRUCTURED_POINTS:

        id = svtkImageData::SafeDownCast(in);

        id->GetDimensions(dims);
        id->GetOrigin(origin);
        id->GetSpacing(spacing);

        same = this->CheckInputDataInfo(i, dims, origin, spacing);

        if (!same)
        {
          newGeometry = 1;
        }

        break;

      case SVTK_UNIFORM_GRID:

        ug = svtkUniformGrid::SafeDownCast(in);

        ug->GetDimensions(dims);
        ug->GetOrigin(origin);
        ug->GetSpacing(spacing);

        same = this->CheckInputDataInfo(i, dims, origin, spacing);

        if (!same)
        {
          newGeometry = 1;
        }
        else if (ug->GetPointGhostArray() && ug->GetPointGhostArray()->GetMTime() > this->BuildTime)
        {
          newGeometry = 1;
        }
        else if (ug->GetCellGhostArray() && ug->GetCellGhostArray()->GetMTime() > this->BuildTime)
        {
          newGeometry = 1;
        }
        break;

      default:
        svtkWarningMacro(<< "svtkKdTree::NewGeometry: unanticipated type");

        newGeometry = 1;
    }
    if (newGeometry)
    {
      break;
    }
  }

  return newGeometry;
}

//----------------------------------------------------------------------------
void svtkKdTree::__printTree(svtkKdNode* kd, int depth, int v)
{
  if (v)
  {
    kd->PrintVerboseNode(depth);
  }
  else
  {
    kd->PrintNode(depth);
  }

  if (kd->GetLeft())
  {
    svtkKdTree::__printTree(kd->GetLeft(), depth + 1, v);
  }
  if (kd->GetRight())
  {
    svtkKdTree::__printTree(kd->GetRight(), depth + 1, v);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::_printTree(int v)
{
  svtkKdTree::__printTree(this->Top, 0, v);
}

//----------------------------------------------------------------------------
void svtkKdTree::PrintRegion(int id)
{
  this->RegionList[id]->PrintNode(0);
}

//----------------------------------------------------------------------------
void svtkKdTree::PrintTree()
{
  _printTree(0);
}

//----------------------------------------------------------------------------
void svtkKdTree::PrintVerboseTree()
{
  _printTree(1);
}

//----------------------------------------------------------------------------
void svtkKdTree::FreeSearchStructure()
{
  SCOPETIMER("FreeSearchStructure");

  if (this->Top)
  {
    svtkKdTree::DeleteAllDescendants(this->Top);
    this->Top->Delete();
    this->Top = nullptr;
  }

  delete[] this->RegionList;
  this->RegionList = nullptr;

  this->NumberOfRegions = 0;
  this->SetActualLevel();

  this->DeleteCellLists();

  delete[] this->CellRegionList;
  this->CellRegionList = nullptr;

  delete[] this->LocatorPoints;
  this->LocatorPoints = nullptr;

  delete[] this->LocatorIds;
  this->LocatorIds = nullptr;

  delete[] this->LocatorRegionLocation;
  this->LocatorRegionLocation = nullptr;
}

//----------------------------------------------------------------------------
// build PolyData representation of all spacial regions------------
//
void svtkKdTree::GenerateRepresentation(int level, svtkPolyData* pd)
{
  if (this->GenerateRepresentationUsingDataBounds)
  {
    this->GenerateRepresentationDataBounds(level, pd);
  }
  else
  {
    this->GenerateRepresentationWholeSpace(level, pd);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::GenerateRepresentationWholeSpace(int level, svtkPolyData* pd)
{
  int i;

  svtkPoints* pts;
  svtkCellArray* polys;

  if (this->Top == nullptr)
  {
    svtkErrorMacro(<< "svtkKdTree::GenerateRepresentation empty tree");
    return;
  }

  if ((level < 0) || (level > this->Level))
  {
    level = this->Level;
  }

  // points and quads for level 0 bounding box
  svtkIdType npoints = 8;
  svtkIdType npolys = 6;

  for (i = 1; i < level; i++)
  {
    int levelPolys = 1 << (i - 1);
    npoints += (4 * levelPolys);
    npolys += levelPolys;
  }

  pts = svtkPoints::New();
  pts->Allocate(npoints);
  polys = svtkCellArray::New();
  polys->AllocateEstimate(npolys, 4);

  // level 0 bounding box

  svtkIdType ids[8];
  svtkIdType idList[4];
  double x[3];
  svtkKdNode* kd = this->Top;

  double* min = kd->GetMinBounds();
  double* max = kd->GetMaxBounds();

  x[0] = min[0];
  x[1] = max[1];
  x[2] = min[2];
  ids[0] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = max[1];
  x[2] = min[2];
  ids[1] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = max[1];
  x[2] = max[2];
  ids[2] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = max[1];
  x[2] = max[2];
  ids[3] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = min[1];
  x[2] = min[2];
  ids[4] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = min[1];
  x[2] = min[2];
  ids[5] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = min[1];
  x[2] = max[2];
  ids[6] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = min[1];
  x[2] = max[2];
  ids[7] = pts->InsertNextPoint(x);

  idList[0] = ids[0];
  idList[1] = ids[1];
  idList[2] = ids[2];
  idList[3] = ids[3];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[1];
  idList[1] = ids[5];
  idList[2] = ids[6];
  idList[3] = ids[2];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[5];
  idList[1] = ids[4];
  idList[2] = ids[7];
  idList[3] = ids[6];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[4];
  idList[1] = ids[0];
  idList[2] = ids[3];
  idList[3] = ids[7];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[3];
  idList[1] = ids[2];
  idList[2] = ids[6];
  idList[3] = ids[7];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[1];
  idList[1] = ids[0];
  idList[2] = ids[4];
  idList[3] = ids[5];
  polys->InsertNextCell(4, idList);

  if (kd->GetLeft() && (level > 0))
  {
    this->_generateRepresentationWholeSpace(kd, pts, polys, level - 1);
  }

  pd->SetPoints(pts);
  pts->Delete();
  pd->SetPolys(polys);
  polys->Delete();
  pd->Squeeze();
}

//----------------------------------------------------------------------------
void svtkKdTree::_generateRepresentationWholeSpace(
  svtkKdNode* kd, svtkPoints* pts, svtkCellArray* polys, int level)
{
  int i;
  double p[4][3];
  svtkIdType ids[4];

  if ((level < 0) || (kd->GetLeft() == nullptr))
  {
    return;
  }

  double* min = kd->GetMinBounds();
  double* max = kd->GetMaxBounds();
  double* leftmax = kd->GetLeft()->GetMaxBounds();

  // splitting plane

  switch (kd->GetDim())
  {
    case XDIM:

      p[0][0] = leftmax[0];
      p[0][1] = max[1];
      p[0][2] = max[2];
      p[1][0] = leftmax[0];
      p[1][1] = max[1];
      p[1][2] = min[2];
      p[2][0] = leftmax[0];
      p[2][1] = min[1];
      p[2][2] = min[2];
      p[3][0] = leftmax[0];
      p[3][1] = min[1];
      p[3][2] = max[2];

      break;

    case YDIM:

      p[0][0] = min[0];
      p[0][1] = leftmax[1];
      p[0][2] = max[2];
      p[1][0] = min[0];
      p[1][1] = leftmax[1];
      p[1][2] = min[2];
      p[2][0] = max[0];
      p[2][1] = leftmax[1];
      p[2][2] = min[2];
      p[3][0] = max[0];
      p[3][1] = leftmax[1];
      p[3][2] = max[2];

      break;

    case ZDIM:

      p[0][0] = min[0];
      p[0][1] = min[1];
      p[0][2] = leftmax[2];
      p[1][0] = min[0];
      p[1][1] = max[1];
      p[1][2] = leftmax[2];
      p[2][0] = max[0];
      p[2][1] = max[1];
      p[2][2] = leftmax[2];
      p[3][0] = max[0];
      p[3][1] = min[1];
      p[3][2] = leftmax[2];

      break;
  }

  for (i = 0; i < 4; i++)
    ids[i] = pts->InsertNextPoint(p[i]);

  polys->InsertNextCell(4, ids);

  this->_generateRepresentationWholeSpace(kd->GetLeft(), pts, polys, level - 1);
  this->_generateRepresentationWholeSpace(kd->GetRight(), pts, polys, level - 1);
}

//----------------------------------------------------------------------------
void svtkKdTree::GenerateRepresentationDataBounds(int level, svtkPolyData* pd)
{
  int i;
  svtkPoints* pts;
  svtkCellArray* polys;

  if (this->Top == nullptr)
  {
    svtkErrorMacro(<< "svtkKdTree::GenerateRepresentation no tree");
    return;
  }

  if ((level < 0) || (level > this->Level))
  {
    level = this->Level;
  }

  svtkIdType npoints = 0;
  svtkIdType npolys = 0;

  for (i = 0; i < level; i++)
  {
    int levelBoxes = 1 << i;
    npoints += (8 * levelBoxes);
    npolys += (6 * levelBoxes);
  }

  pts = svtkPoints::New();
  pts->Allocate(npoints);
  polys = svtkCellArray::New();
  polys->AllocateEstimate(npolys, 4);

  _generateRepresentationDataBounds(this->Top, pts, polys, level);

  pd->SetPoints(pts);
  pts->Delete();
  pd->SetPolys(polys);
  polys->Delete();
  pd->Squeeze();
}

//----------------------------------------------------------------------------
void svtkKdTree::_generateRepresentationDataBounds(
  svtkKdNode* kd, svtkPoints* pts, svtkCellArray* polys, int level)
{
  if (level > 0)
  {
    if (kd->GetLeft())
    {
      _generateRepresentationDataBounds(kd->GetLeft(), pts, polys, level - 1);
      _generateRepresentationDataBounds(kd->GetRight(), pts, polys, level - 1);
    }

    return;
  }
  svtkKdTree::AddPolys(kd, pts, polys);
}

//----------------------------------------------------------------------------
// PolyData rep. of all spacial regions, shrunk to data bounds-------
//
void svtkKdTree::AddPolys(svtkKdNode* kd, svtkPoints* pts, svtkCellArray* polys)
{
  svtkIdType ids[8];
  svtkIdType idList[4];
  double x[3];

  double* min;
  double* max;

  if (this->GenerateRepresentationUsingDataBounds)
  {
    min = kd->GetMinDataBounds();
    max = kd->GetMaxDataBounds();
  }
  else
  {
    min = kd->GetMinBounds();
    max = kd->GetMaxBounds();
  }

  x[0] = min[0];
  x[1] = max[1];
  x[2] = min[2];
  ids[0] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = max[1];
  x[2] = min[2];
  ids[1] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = max[1];
  x[2] = max[2];
  ids[2] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = max[1];
  x[2] = max[2];
  ids[3] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = min[1];
  x[2] = min[2];
  ids[4] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = min[1];
  x[2] = min[2];
  ids[5] = pts->InsertNextPoint(x);

  x[0] = max[0];
  x[1] = min[1];
  x[2] = max[2];
  ids[6] = pts->InsertNextPoint(x);

  x[0] = min[0];
  x[1] = min[1];
  x[2] = max[2];
  ids[7] = pts->InsertNextPoint(x);

  idList[0] = ids[0];
  idList[1] = ids[1];
  idList[2] = ids[2];
  idList[3] = ids[3];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[1];
  idList[1] = ids[5];
  idList[2] = ids[6];
  idList[3] = ids[2];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[5];
  idList[1] = ids[4];
  idList[2] = ids[7];
  idList[3] = ids[6];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[4];
  idList[1] = ids[0];
  idList[2] = ids[3];
  idList[3] = ids[7];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[3];
  idList[1] = ids[2];
  idList[2] = ids[6];
  idList[3] = ids[7];
  polys->InsertNextCell(4, idList);

  idList[0] = ids[1];
  idList[1] = ids[0];
  idList[2] = ids[4];
  idList[3] = ids[5];
  polys->InsertNextCell(4, idList);
}

//----------------------------------------------------------------------------
// PolyData representation of a list of spacial regions------------
//
void svtkKdTree::GenerateRepresentation(int* regions, int len, svtkPolyData* pd)
{
  int i;
  svtkPoints* pts;
  svtkCellArray* polys;

  if (this->Top == nullptr)
  {
    svtkErrorMacro(<< "svtkKdTree::GenerateRepresentation no tree");
    return;
  }

  const svtkIdType npoints = 8 * len;
  const svtkIdType npolys = 6 * len;

  pts = svtkPoints::New();
  pts->Allocate(npoints);
  polys = svtkCellArray::New();
  polys->AllocateEstimate(npolys, 4);

  for (i = 0; i < len; i++)
  {
    if ((regions[i] < 0) || (regions[i] >= this->NumberOfRegions))
    {
      break;
    }

    svtkKdTree::AddPolys(this->RegionList[regions[i]], pts, polys);
  }

  pd->SetPoints(pts);
  pts->Delete();
  pd->SetPolys(polys);
  polys->Delete();
  pd->Squeeze();
}

//----------------------------------------------------------------------------
//  Cell ID lists
//
#define SORTLIST(l, lsize) std::sort(l, (l) + (lsize))

#define REMOVEDUPLICATES(l, lsize, newsize)                                                        \
  {                                                                                                \
    int ii, jj;                                                                                    \
    for (ii = 0, jj = 0; ii < (lsize); ii++)                                                       \
    {                                                                                              \
      if ((ii > 0) && (l[ii] == l[jj - 1]))                                                        \
      {                                                                                            \
        continue;                                                                                  \
      }                                                                                            \
      if (jj != ii)                                                                                \
      {                                                                                            \
        l[jj] = l[ii];                                                                             \
      }                                                                                            \
      jj++;                                                                                        \
    }                                                                                              \
    newsize = jj;                                                                                  \
  }

//----------------------------------------------------------------------------
int svtkKdTree::FoundId(svtkIntArray* idArray, int id)
{
  // This is a simple linear search, because I think it is rare that
  // an id array would be provided, and if one is it should be small.

  int found = 0;
  int len = idArray->GetNumberOfTuples();
  int* ids = idArray->GetPointer(0);

  for (int i = 0; i < len; i++)
  {
    if (ids[i] == id)
    {
      found = 1;
    }
  }

  return found;
}

//----------------------------------------------------------------------------
int svtkKdTree::findRegion(svtkKdNode* node, float x, float y, float z)
{
  return svtkKdTree::findRegion(
    node, static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}

//----------------------------------------------------------------------------
int svtkKdTree::findRegion(svtkKdNode* node, double x, double y, double z)
{
  int regionId;

  if (!node->ContainsPoint(x, y, z, 0))
  {
    return -1;
  }

  if (node->GetLeft() == nullptr)
  {
    regionId = node->GetID();
  }
  else
  {
    regionId = svtkKdTree::findRegion(node->GetLeft(), x, y, z);

    if (regionId < 0)
    {
      regionId = svtkKdTree::findRegion(node->GetRight(), x, y, z);
    }
  }

  return regionId;
}

//----------------------------------------------------------------------------
void svtkKdTree::CreateCellLists()
{
  this->CreateCellLists(static_cast<int*>(nullptr), 0);
}

//----------------------------------------------------------------------------
void svtkKdTree::CreateCellLists(int* regionList, int listSize)
{
  this->CreateCellLists(this->GetDataSet(), regionList, listSize);
}

//----------------------------------------------------------------------------
void svtkKdTree::CreateCellLists(int dataSetIndex, int* regionList, int listSize)
{
  svtkDataSet* dataSet = this->GetDataSet(dataSetIndex);
  if (!dataSet)
  {
    svtkErrorMacro(<< "svtkKdTree::CreateCellLists invalid data set");
    return;
  }

  this->CreateCellLists(dataSet, regionList, listSize);
}

//----------------------------------------------------------------------------
void svtkKdTree::CreateCellLists(svtkDataSet* set, int* regionList, int listSize)
{
  int i, AllRegions;

  if (this->GetDataSetIndex(set) < 0)
  {
    svtkErrorMacro(<< "svtkKdTree::CreateCellLists invalid data set");
    return;
  }

  svtkKdTree::_cellList* list = &this->CellList;

  if (list->nRegions > 0)
  {
    this->DeleteCellLists();
  }

  list->emptyList = svtkIdList::New();

  list->dataSet = set;

  if ((regionList == nullptr) || (listSize == 0))
  {
    list->nRegions = this->NumberOfRegions; // all regions
  }
  else
  {
    list->regionIds = new int[listSize];

    if (!list->regionIds)
    {
      svtkErrorMacro(<< "svtkKdTree::CreateCellLists memory allocation");
      return;
    }

    memcpy(list->regionIds, regionList, sizeof(int) * listSize);
    SORTLIST(list->regionIds, listSize);
    REMOVEDUPLICATES(list->regionIds, listSize, list->nRegions);

    if (list->nRegions == this->NumberOfRegions)
    {
      delete[] list->regionIds;
      list->regionIds = nullptr;
    }
  }

  if (list->nRegions == this->NumberOfRegions)
  {
    AllRegions = 1;
  }
  else
  {
    AllRegions = 0;
  }

  int* idlist = nullptr;
  int idListLen = 0;

  if (this->IncludeRegionBoundaryCells)
  {
    list->boundaryCells = new svtkIdList*[list->nRegions];

    if (!list->boundaryCells)
    {
      svtkErrorMacro(<< "svtkKdTree::CreateCellLists memory allocation");
      return;
    }

    for (i = 0; i < list->nRegions; i++)
    {
      list->boundaryCells[i] = svtkIdList::New();
    }
    idListLen = this->NumberOfRegions;

    idlist = new int[idListLen];
  }

  int* listptr = nullptr;

  if (!AllRegions)
  {
    listptr = new int[this->NumberOfRegions];

    if (!listptr)
    {
      svtkErrorMacro(<< "svtkKdTree::CreateCellLists memory allocation");
      delete[] idlist;
      return;
    }

    for (i = 0; i < this->NumberOfRegions; i++)
    {
      listptr[i] = -1;
    }
  }

  list->cells = new svtkIdList*[list->nRegions];

  if (!list->cells)
  {
    svtkErrorMacro(<< "svtkKdTree::CreateCellLists memory allocation");
    delete[] idlist;
    delete[] listptr;
    return;
  }

  for (i = 0; i < list->nRegions; i++)
  {
    list->cells[i] = svtkIdList::New();

    if (listptr)
    {
      listptr[list->regionIds[i]] = i;
    }
  }

  // acquire a list in cell Id order of the region Id each
  // cell centroid falls in

  int* regList = this->CellRegionList;

  if (regList == nullptr)
  {
    regList = this->AllGetRegionContainingCell();
  }

  int setNum = this->GetDataSetIndex(set);

  if (setNum > 0)
  {
    int ncells = this->GetDataSetsNumberOfCells(0, setNum - 1);
    regList += ncells;
  }

  int nCells = set->GetNumberOfCells();

  for (int cellId = 0; cellId < nCells; cellId++)
  {
    if (this->IncludeRegionBoundaryCells)
    {
      // Find all regions the cell intersects, including
      // the region the cell centroid lies in.
      // This can be an expensive calculation, intersections
      // of a convex region with axis aligned boxes.

      int nRegions = this->BSPCalculator->IntersectsCell(
        idlist, idListLen, set->GetCell(cellId), regList[cellId]);

      if (nRegions == 1)
      {
        int idx = (listptr) ? listptr[idlist[0]] : idlist[0];

        if (idx >= 0)
        {
          list->cells[idx]->InsertNextId(cellId);
        }
      }
      else
      {
        for (int r = 0; r < nRegions; r++)
        {
          int regionId = idlist[r];

          int idx = (listptr) ? listptr[regionId] : regionId;

          if (idx < 0)
          {
            continue;
          }

          if (regionId == regList[cellId])
          {
            list->cells[idx]->InsertNextId(cellId);
          }
          else
          {
            list->boundaryCells[idx]->InsertNextId(cellId);
          }
        }
      }
    }
    else
    {
      // just find the region the cell centroid lies in - easy

      int regionId = regList[cellId];

      int idx = (listptr) ? listptr[regionId] : regionId;

      if (idx >= 0)
      {
        list->cells[idx]->InsertNextId(cellId);
      }
    }
  }

  delete[] listptr;
  delete[] idlist;
}

//----------------------------------------------------------------------------
svtkIdList* svtkKdTree::GetList(int regionId, svtkIdList** which)
{
  int i;
  struct _cellList* list = &this->CellList;
  svtkIdList* cellIds = nullptr;

  if (which && (list->nRegions == this->NumberOfRegions))
  {
    cellIds = which[regionId];
  }
  else if (which)
  {
    for (i = 0; i < list->nRegions; i++)
    {
      if (list->regionIds[i] == regionId)
      {
        cellIds = which[i];
        break;
      }
    }
  }
  else
  {
    cellIds = list->emptyList;
  }

  return cellIds;
}

//----------------------------------------------------------------------------
svtkIdList* svtkKdTree::GetCellList(int regionID)
{
  return this->GetList(regionID, this->CellList.cells);
}

//----------------------------------------------------------------------------
svtkIdList* svtkKdTree::GetBoundaryCellList(int regionID)
{
  return this->GetList(regionID, this->CellList.boundaryCells);
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::GetCellLists(
  svtkIntArray* regions, int setIndex, svtkIdList* inRegionCells, svtkIdList* onBoundaryCells)
{
  svtkDataSet* set = this->GetDataSet(setIndex);
  if (!set)
  {
    svtkErrorMacro(<< "svtkKdTree::GetCellLists no such data set");
    return 0;
  }
  return this->GetCellLists(regions, set, inRegionCells, onBoundaryCells);
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::GetCellLists(
  svtkIntArray* regions, svtkIdList* inRegionCells, svtkIdList* onBoundaryCells)
{
  return this->GetCellLists(regions, this->GetDataSet(), inRegionCells, onBoundaryCells);
}

//----------------------------------------------------------------------------
svtkIdType svtkKdTree::GetCellLists(
  svtkIntArray* regions, svtkDataSet* set, svtkIdList* inRegionCells, svtkIdList* onBoundaryCells)
{
  int reg, regionId;
  svtkIdType cell, cellId, numCells;
  svtkIdList* cellIds;

  svtkIdType totalCells = 0;

  if ((inRegionCells == nullptr) && (onBoundaryCells == nullptr))
  {
    return totalCells;
  }

  int nregions = regions->GetNumberOfTuples();

  if (nregions == 0)
  {
    return totalCells;
  }

  // Do I have cell lists for all these regions?  If not, build cell
  // lists for all regions for this data set.

  int rebuild = 0;

  if (this->CellList.dataSet != set)
  {
    rebuild = 1;
  }
  else if (nregions > this->CellList.nRegions)
  {
    rebuild = 1;
  }
  else if ((onBoundaryCells != nullptr) && (this->CellList.boundaryCells == nullptr))
  {
    rebuild = 1;
  }
  else if (this->CellList.nRegions < this->NumberOfRegions)
  {
    // these two lists should generally be "short"

    int* haveIds = this->CellList.regionIds;

    for (int wantReg = 0; wantReg < nregions; wantReg++)
    {
      int wantRegion = regions->GetValue(wantReg);
      int gotId = 0;

      for (int haveReg = 0; haveReg < this->CellList.nRegions; haveReg++)
      {
        if (haveIds[haveReg] == wantRegion)
        {
          gotId = 1;
          break;
        }
      }
      if (!gotId)
      {
        rebuild = 1;
        break;
      }
    }
  }

  if (rebuild)
  {
    if (onBoundaryCells != nullptr)
    {
      this->IncludeRegionBoundaryCellsOn();
    }
    this->CreateCellLists(set, nullptr, 0); // for all regions
  }

  // OK, we have cell lists for these regions.  Make lists of region
  // cells and boundary cells.

  int CheckSet = (onBoundaryCells && (nregions > 1));

  std::set<svtkIdType> ids;
  std::pair<std::set<svtkIdType>::iterator, bool> idRec;

  svtkIdType totalRegionCells = 0;
  svtkIdType totalBoundaryCells = 0;

  svtkIdList** inRegionList = new svtkIdList*[nregions];

  // First the cell IDs with centroid in the regions

  for (reg = 0; reg < nregions; reg++)
  {
    regionId = regions->GetValue(reg);

    inRegionList[reg] = this->GetCellList(regionId);

    totalRegionCells += inRegionList[reg]->GetNumberOfIds();
  }

  if (inRegionCells)
  {
    inRegionCells->Initialize();
    inRegionCells->SetNumberOfIds(totalRegionCells);
  }

  int nextCell = 0;

  for (reg = 0; reg < nregions; reg++)
  {
    cellIds = inRegionList[reg];

    numCells = cellIds->GetNumberOfIds();

    for (cell = 0; cell < numCells; cell++)
    {
      if (inRegionCells)
      {
        inRegionCells->SetId(nextCell++, cellIds->GetId(cell));
      }

      if (CheckSet)
      {
        // We have to save the ids, so we don't include
        // them as boundary cells.  A cell in one region
        // may be a boundary cell of another region.

        ids.insert(cellIds->GetId(cell));
      }
    }
  }

  delete[] inRegionList;

  if (onBoundaryCells == nullptr)
  {
    return totalRegionCells;
  }

  // Now the list of all cells on the boundary of the regions,
  // which do not have their centroid in one of the regions

  onBoundaryCells->Initialize();

  for (reg = 0; reg < nregions; reg++)
  {
    regionId = regions->GetValue(reg);

    cellIds = this->GetBoundaryCellList(regionId);

    numCells = cellIds->GetNumberOfIds();

    for (cell = 0; cell < numCells; cell++)
    {
      cellId = cellIds->GetId(cell);

      if (CheckSet)
      {
        // if we already included this cell because it is within
        // one of the regions, or on the boundary of another, skip it

        idRec = ids.insert(cellId);

        if (idRec.second == 0)
        {
          continue;
        }
      }

      onBoundaryCells->InsertNextId(cellId);
      totalBoundaryCells++;
    }

    totalCells += totalBoundaryCells;
  }

  return totalCells;
}

//----------------------------------------------------------------------------
int svtkKdTree::GetRegionContainingCell(svtkIdType cellID)
{
  return this->GetRegionContainingCell(this->GetDataSet(), cellID);
}

//----------------------------------------------------------------------------
int svtkKdTree::GetRegionContainingCell(int setIndex, svtkIdType cellID)
{
  svtkDataSet* set = this->GetDataSet(setIndex);
  if (!set)
  {
    svtkErrorMacro(<< "svtkKdTree::GetRegionContainingCell no such data set");
    return -1;
  }
  return this->GetRegionContainingCell(set, cellID);
}

//----------------------------------------------------------------------------
int svtkKdTree::GetRegionContainingCell(svtkDataSet* set, svtkIdType cellID)
{
  int regionID = -1;

  if (this->GetDataSetIndex(set) < 0)
  {
    svtkErrorMacro(<< "svtkKdTree::GetRegionContainingCell no such data set");
    return -1;
  }
  if ((cellID < 0) || (cellID >= set->GetNumberOfCells()))
  {
    svtkErrorMacro(<< "svtkKdTree::GetRegionContainingCell bad cell ID");
    return -1;
  }

  if (this->CellRegionList)
  {
    if (set == this->GetDataSet()) // 99.99999% of the time
    {
      return this->CellRegionList[cellID];
    }

    int setNum = this->GetDataSetIndex(set);

    int offset = this->GetDataSetsNumberOfCells(0, setNum - 1);

    return this->CellRegionList[offset + cellID];
  }

  float center[3];

  this->ComputeCellCenter(set, cellID, center);

  regionID = this->GetRegionContainingPoint(center[0], center[1], center[2]);

  return regionID;
}

//----------------------------------------------------------------------------
int* svtkKdTree::AllGetRegionContainingCell()
{
  if (this->CellRegionList)
  {
    return this->CellRegionList;
  }
  this->CellRegionList = new int[this->GetNumberOfCells()];

  if (!this->CellRegionList)
  {
    svtkErrorMacro(<< "svtkKdTree::AllGetRegionContainingCell memory allocation");
    return nullptr;
  }

  int* listPtr = this->CellRegionList;

  svtkCollectionSimpleIterator cookie;
  this->DataSets->InitTraversal(cookie);
  for (svtkDataSet* iset = this->DataSets->GetNextDataSet(cookie); iset != nullptr;
       iset = this->DataSets->GetNextDataSet(cookie))
  {
    int setCells = iset->GetNumberOfCells();

    float* centers = this->ComputeCellCenters(iset);

    float* pt = centers;

    for (int cellId = 0; cellId < setCells; cellId++)
    {
      listPtr[cellId] = this->GetRegionContainingPoint(pt[0], pt[1], pt[2]);

      pt += 3;
    }

    listPtr += setCells;

    delete[] centers;
  }

  return this->CellRegionList;
}

//----------------------------------------------------------------------------
int svtkKdTree::GetRegionContainingPoint(double x, double y, double z)
{
  return svtkKdTree::findRegion(this->Top, x, y, z);
}
//----------------------------------------------------------------------------
int svtkKdTree::MinimalNumberOfConvexSubRegions(svtkIntArray* regionIdList, double** convexSubRegions)
{
  int nids = 0;

  if ((regionIdList == nullptr) || ((nids = regionIdList->GetNumberOfTuples()) == 0))
  {
    svtkErrorMacro(<< "svtkKdTree::MinimalNumberOfConvexSubRegions no regions specified");
    return 0;
  }

  int i;
  int* ids = regionIdList->GetPointer(0);

  if (nids == 1)
  {
    if ((ids[0] < 0) || (ids[0] >= this->NumberOfRegions))
    {
      svtkErrorMacro(<< "svtkKdTree::MinimalNumberOfConvexSubRegions bad region ID");
      return 0;
    }

    double* bounds = new double[6];

    this->RegionList[ids[0]]->GetBounds(bounds);

    *convexSubRegions = bounds;

    return 1;
  }

  // create a sorted list of unique region Ids

  std::set<int> idSet;
  std::set<int>::iterator it;

  for (i = 0; i < nids; i++)
  {
    idSet.insert(ids[i]);
  }

  int nUniqueIds = static_cast<int>(idSet.size());

  int* idList = new int[nUniqueIds];

  for (i = 0, it = idSet.begin(); it != idSet.end(); ++it, ++i)
  {
    idList[i] = *it;
  }

  svtkKdNode** regions = new svtkKdNode*[nUniqueIds];

  int nregions = svtkKdTree::__ConvexSubRegions(idList, nUniqueIds, this->Top, regions);

  double* bounds = new double[nregions * 6];

  for (i = 0; i < nregions; i++)
  {
    regions[i]->GetBounds(bounds + (i * 6));
  }

  *convexSubRegions = bounds;

  delete[] idList;
  delete[] regions;

  return nregions;
}
//----------------------------------------------------------------------------
int svtkKdTree::__ConvexSubRegions(int* ids, int len, svtkKdNode* tree, svtkKdNode** nodes)
{
  int nregions = tree->GetMaxID() - tree->GetMinID() + 1;

  if (nregions == len)
  {
    *nodes = tree;
    return 1;
  }

  if (tree->GetLeft() == nullptr)
  {
    return 0;
  }

  int min = ids[0];
  int max = ids[len - 1];

  int leftMax = tree->GetLeft()->GetMaxID();
  int rightMin = tree->GetRight()->GetMinID();

  if (max <= leftMax)
  {
    return svtkKdTree::__ConvexSubRegions(ids, len, tree->GetLeft(), nodes);
  }
  else if (min >= rightMin)
  {
    return svtkKdTree::__ConvexSubRegions(ids, len, tree->GetRight(), nodes);
  }
  else
  {
    int leftIds = 1;

    for (int i = 1; i < len - 1; i++)
    {
      if (ids[i] <= leftMax)
      {
        leftIds++;
      }
      else
      {
        break;
      }
    }

    int numNodesLeft = svtkKdTree::__ConvexSubRegions(ids, leftIds, tree->GetLeft(), nodes);

    int numNodesRight = svtkKdTree::__ConvexSubRegions(
      ids + leftIds, len - leftIds, tree->GetRight(), nodes + numNodesLeft);

    return (numNodesLeft + numNodesRight);
  }
}

//----------------------------------------------------------------------------
int svtkKdTree::ViewOrderRegionsInDirection(
  svtkIntArray* regionIds, const double directionOfProjection[3], svtkIntArray* orderedList)
{
  int i;

  svtkIntArray* IdsOfInterest = nullptr;

  if (regionIds && (regionIds->GetNumberOfTuples() > 0))
  {
    // Created sorted list of unique ids

    std::set<int> ids;
    std::set<int>::iterator it;
    int nids = regionIds->GetNumberOfTuples();

    for (i = 0; i < nids; i++)
    {
      ids.insert(regionIds->GetValue(i));
    }

    if (ids.size() < static_cast<unsigned int>(this->NumberOfRegions))
    {
      IdsOfInterest = svtkIntArray::New();
      IdsOfInterest->SetNumberOfValues(static_cast<svtkIdType>(ids.size()));

      for (it = ids.begin(), i = 0; it != ids.end(); ++it, ++i)
      {
        IdsOfInterest->SetValue(i, *it);
      }
    }
  }

  int size = this->_ViewOrderRegionsInDirection(IdsOfInterest, directionOfProjection, orderedList);

  if (IdsOfInterest)
  {
    IdsOfInterest->Delete();
  }

  return size;
}

//----------------------------------------------------------------------------
int svtkKdTree::ViewOrderAllRegionsInDirection(
  const double directionOfProjection[3], svtkIntArray* orderedList)
{
  return this->_ViewOrderRegionsInDirection(nullptr, directionOfProjection, orderedList);
}

//----------------------------------------------------------------------------
int svtkKdTree::ViewOrderRegionsFromPosition(
  svtkIntArray* regionIds, const double cameraPosition[3], svtkIntArray* orderedList)
{
  int i;

  svtkIntArray* IdsOfInterest = nullptr;

  if (regionIds && (regionIds->GetNumberOfTuples() > 0))
  {
    // Created sorted list of unique ids

    std::set<int> ids;
    std::set<int>::iterator it;
    int nids = regionIds->GetNumberOfTuples();

    for (i = 0; i < nids; i++)
    {
      ids.insert(regionIds->GetValue(i));
    }

    if (ids.size() < static_cast<unsigned int>(this->NumberOfRegions))
    {
      IdsOfInterest = svtkIntArray::New();
      IdsOfInterest->SetNumberOfValues(static_cast<svtkIdType>(ids.size()));

      for (it = ids.begin(), i = 0; it != ids.end(); ++it, ++i)
      {
        IdsOfInterest->SetValue(i, *it);
      }
    }
  }

  int size = this->_ViewOrderRegionsFromPosition(IdsOfInterest, cameraPosition, orderedList);

  if (IdsOfInterest)
  {
    IdsOfInterest->Delete();
  }

  return size;
}

//----------------------------------------------------------------------------
int svtkKdTree::ViewOrderAllRegionsFromPosition(
  const double cameraPosition[3], svtkIntArray* orderedList)
{
  return this->_ViewOrderRegionsFromPosition(nullptr, cameraPosition, orderedList);
}

//----------------------------------------------------------------------------
int svtkKdTree::_ViewOrderRegionsInDirection(
  svtkIntArray* IdsOfInterest, const double dir[3], svtkIntArray* orderedList)
{
  int nextId = 0;

  int numValues = (IdsOfInterest ? IdsOfInterest->GetNumberOfTuples() : this->NumberOfRegions);

  orderedList->Initialize();
  orderedList->SetNumberOfValues(numValues);

  int size =
    svtkKdTree::__ViewOrderRegionsInDirection(this->Top, orderedList, IdsOfInterest, dir, nextId);
  if (size < 0)
  {
    svtkErrorMacro(<< "svtkKdTree::DepthOrderRegions k-d tree structure is corrupt");
    orderedList->Initialize();
    return 0;
  }

  return size;
}
//----------------------------------------------------------------------------
int svtkKdTree::__ViewOrderRegionsInDirection(
  svtkKdNode* node, svtkIntArray* list, svtkIntArray* IdsOfInterest, const double dir[3], int nextId)
{
  if (node->GetLeft() == nullptr)
  {
    if (!IdsOfInterest || svtkKdTree::FoundId(IdsOfInterest, node->GetID()))
    {
      list->SetValue(nextId, node->GetID());
      nextId = nextId + 1;
    }

    return nextId;
  }

  int cutPlane = node->GetDim();

  if ((cutPlane < 0) || (cutPlane > 2))
  {
    return -1;
  }

  double closest = dir[cutPlane] * -1;

  svtkKdNode* closeNode = (closest < 0) ? node->GetLeft() : node->GetRight();
  svtkKdNode* farNode = (closest >= 0) ? node->GetLeft() : node->GetRight();

  int nextNextId =
    svtkKdTree::__ViewOrderRegionsInDirection(closeNode, list, IdsOfInterest, dir, nextId);

  if (nextNextId == -1)
  {
    return -1;
  }

  nextNextId =
    svtkKdTree::__ViewOrderRegionsInDirection(farNode, list, IdsOfInterest, dir, nextNextId);

  return nextNextId;
}

//----------------------------------------------------------------------------
int svtkKdTree::_ViewOrderRegionsFromPosition(
  svtkIntArray* IdsOfInterest, const double pos[3], svtkIntArray* orderedList)
{
  int nextId = 0;

  int numValues = (IdsOfInterest ? IdsOfInterest->GetNumberOfTuples() : this->NumberOfRegions);

  orderedList->Initialize();
  orderedList->SetNumberOfValues(numValues);

  int size =
    svtkKdTree::__ViewOrderRegionsFromPosition(this->Top, orderedList, IdsOfInterest, pos, nextId);
  if (size < 0)
  {
    svtkErrorMacro(<< "svtkKdTree::DepthOrderRegions k-d tree structure is corrupt");
    orderedList->Initialize();
    return 0;
  }

  return size;
}
//----------------------------------------------------------------------------
int svtkKdTree::__ViewOrderRegionsFromPosition(
  svtkKdNode* node, svtkIntArray* list, svtkIntArray* IdsOfInterest, const double pos[3], int nextId)
{
  if (node->GetLeft() == nullptr)
  {
    if (!IdsOfInterest || svtkKdTree::FoundId(IdsOfInterest, node->GetID()))
    {
      list->SetValue(nextId, node->GetID());
      nextId = nextId + 1;
    }

    return nextId;
  }

  int cutPlane = node->GetDim();

  if ((cutPlane < 0) || (cutPlane > 2))
  {
    return -1;
  }

  double closest = pos[cutPlane] - node->GetDivisionPosition();

  svtkKdNode* closeNode = (closest < 0) ? node->GetLeft() : node->GetRight();
  svtkKdNode* farNode = (closest >= 0) ? node->GetLeft() : node->GetRight();

  int nextNextId =
    svtkKdTree::__ViewOrderRegionsFromPosition(closeNode, list, IdsOfInterest, pos, nextId);

  if (nextNextId == -1)
  {
    return -1;
  }

  nextNextId =
    svtkKdTree::__ViewOrderRegionsFromPosition(farNode, list, IdsOfInterest, pos, nextNextId);

  return nextNextId;
}

//----------------------------------------------------------------------------
// These requests change the boundaries of the k-d tree, so must
// update the MTime.
//
void svtkKdTree::NewPartitioningRequest(int req)
{
  if (req != this->ValidDirections)
  {
    this->Modified();
    this->ValidDirections = req;
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitXPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::YDIM) | (1 << svtkKdTree::ZDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitYPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::ZDIM) | (1 << svtkKdTree::XDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitZPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::XDIM) | (1 << svtkKdTree::YDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitXYPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::ZDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitYZPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::XDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitZXPartitioning()
{
  this->NewPartitioningRequest((1 << svtkKdTree::YDIM));
}

//----------------------------------------------------------------------------
void svtkKdTree::OmitNoPartitioning()
{
  int req = ((1 << svtkKdTree::XDIM) | (1 << svtkKdTree::YDIM) | (1 << svtkKdTree::ZDIM));
  this->NewPartitioningRequest(req);
}

//---------------------------------------------------------------------------
void svtkKdTree::PrintTiming(ostream& os, svtkIndent)
{
  svtkTimerLog::DumpLogWithIndents(&os, 0.0f);
}

//----------------------------------------------------------------------------
void svtkKdTree::UpdateProgress(double amt)
{
  this->Progress = amt;
  this->InvokeEvent(svtkCommand::ProgressEvent, static_cast<void*>(&amt));
}

//----------------------------------------------------------------------------
void svtkKdTree::UpdateSubOperationProgress(double amt)
{
  this->UpdateProgress(this->ProgressOffset + this->ProgressScale * amt);
}

//----------------------------------------------------------------------------
void svtkKdTree::FindPointsInArea(double* area, svtkIdTypeArray* ids, bool clearArray)
{
  if (clearArray)
  {
    ids->Reset();
  }
  if (!this->LocatorPoints)
  {
    svtkErrorMacro(<< "svtkKdTree::FindPointsInArea - must build locator first");
    return;
  }
  this->FindPointsInArea(this->Top, area, ids);
}

//----------------------------------------------------------------------------
void svtkKdTree::FindPointsInArea(svtkKdNode* node, double* area, svtkIdTypeArray* ids)
{
  double b[6];
  node->GetBounds(b);

  if (b[0] > area[1] || b[1] < area[0] || b[2] > area[3] || b[3] < area[2] || b[4] > area[5] ||
    b[5] < area[4])
  {
    return;
  }

  bool contains = false;
  if (area[0] <= b[0] && b[1] <= area[1] && area[2] <= b[2] && b[3] <= area[3] && area[4] <= b[4] &&
    b[5] <= area[5])
  {
    contains = true;
  }

  if (contains)
  {
    this->AddAllPointsInRegion(node, ids);
  }
  else // intersects
  {
    if (node->GetLeft() == nullptr)
    {
      int regionID = node->GetID();
      int regionLoc = this->LocatorRegionLocation[regionID];
      float* pt = this->LocatorPoints + (regionLoc * 3);
      svtkIdType numPoints = this->RegionList[regionID]->GetNumberOfPoints();
      for (svtkIdType i = 0; i < numPoints; i++)
      {
        if (area[0] <= pt[0] && pt[0] <= area[1] && area[2] <= pt[1] && pt[1] <= area[3] &&
          area[4] <= pt[2] && pt[2] <= area[5])
        {
          svtkIdType ptId = static_cast<svtkIdType>(this->LocatorIds[regionLoc + i]);
          ids->InsertNextValue(ptId);
        }
        pt += 3;
      }
    }
    else
    {
      this->FindPointsInArea(node->GetLeft(), area, ids);
      this->FindPointsInArea(node->GetRight(), area, ids);
    }
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::AddAllPointsInRegion(svtkKdNode* node, svtkIdTypeArray* ids)
{
  if (node->GetLeft() == nullptr)
  {
    int regionID = node->GetID();
    int regionLoc = this->LocatorRegionLocation[regionID];
    svtkIdType numPoints = this->RegionList[regionID]->GetNumberOfPoints();
    for (svtkIdType i = 0; i < numPoints; i++)
    {
      svtkIdType ptId = static_cast<svtkIdType>(this->LocatorIds[regionLoc + i]);
      ids->InsertNextValue(ptId);
    }
  }
  else
  {
    this->AddAllPointsInRegion(node->GetLeft(), ids);
    this->AddAllPointsInRegion(node->GetRight(), ids);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::AddAllPointsInRegion(svtkKdNode* node, svtkIdList* ids)
{
  if (node->GetLeft() == nullptr)
  {
    int regionID = node->GetID();
    int regionLoc = this->LocatorRegionLocation[regionID];
    svtkIdType numPoints = this->RegionList[regionID]->GetNumberOfPoints();
    for (svtkIdType i = 0; i < numPoints; i++)
    {
      svtkIdType ptId = static_cast<svtkIdType>(this->LocatorIds[regionLoc + i]);
      ids->InsertNextId(ptId);
    }
  }
  else
  {
    this->AddAllPointsInRegion(node->GetLeft(), ids);
    this->AddAllPointsInRegion(node->GetRight(), ids);
  }
}

//----------------------------------------------------------------------------
void svtkKdTree::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "ValidDirections: " << this->ValidDirections << endl;
  os << indent << "MinCells: " << this->MinCells << endl;
  os << indent << "NumberOfRegionsOrLess: " << this->NumberOfRegionsOrLess << endl;
  os << indent << "NumberOfRegionsOrMore: " << this->NumberOfRegionsOrMore << endl;

  os << indent << "NumberOfRegions: " << this->NumberOfRegions << endl;

  os << indent << "DataSets: " << this->DataSets << endl;

  os << indent << "Top: " << this->Top << endl;
  os << indent << "RegionList: " << this->RegionList << endl;

  os << indent << "Timing: " << this->Timing << endl;
  os << indent << "TimerLog: " << this->TimerLog << endl;

  os << indent << "IncludeRegionBoundaryCells: ";
  os << this->IncludeRegionBoundaryCells << endl;
  os << indent << "GenerateRepresentationUsingDataBounds: ";
  os << this->GenerateRepresentationUsingDataBounds << endl;

  if (this->CellList.nRegions > 0)
  {
    os << indent << "CellList.dataSet " << this->CellList.dataSet << endl;
    os << indent << "CellList.regionIds " << this->CellList.regionIds << endl;
    os << indent << "CellList.nRegions " << this->CellList.nRegions << endl;
    os << indent << "CellList.cells " << this->CellList.cells << endl;
    os << indent << "CellList.boundaryCells " << this->CellList.boundaryCells << endl;
  }
  os << indent << "CellRegionList: " << this->CellRegionList << endl;

  os << indent << "LocatorPoints: " << this->LocatorPoints << endl;
  os << indent << "NumberOfLocatorPoints: " << this->NumberOfLocatorPoints << endl;
  os << indent << "LocatorIds: " << this->LocatorIds << endl;
  os << indent << "LocatorRegionLocation: " << this->LocatorRegionLocation << endl;

  os << indent << "FudgeFactor: " << this->FudgeFactor << endl;
  os << indent << "MaxWidth: " << this->MaxWidth << endl;

  os << indent << "Cuts: ";
  if (this->Cuts)
  {
    this->Cuts->PrintSelf(os << endl, indent.GetNextIndent());
  }
  else
  {
    os << "(none)" << endl;
  }
  os << indent << "Progress: " << this->Progress << endl;
}
