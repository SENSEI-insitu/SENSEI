/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBSPIntersections.cxx

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

#include "svtkBSPIntersections.h"
#include "svtkBSPCuts.h"
#include "svtkCell.h"
#include "svtkKdNode.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"

#include <set>

svtkStandardNewMacro(svtkBSPIntersections);

#define REGIONCHECK(err)                                                                           \
  if (this->BuildRegionList())                                                                     \
  {                                                                                                \
    return err;                                                                                    \
  }

#define REGIONIDCHECK_RETURNERR(id, err)                                                           \
  if (this->BuildRegionList())                                                                     \
  {                                                                                                \
    return err;                                                                                    \
  }                                                                                                \
  if (((id) < 0) || ((id) >= this->NumberOfRegions))                                               \
  {                                                                                                \
    svtkErrorMacro(<< "Invalid region ID");                                                         \
    return (err);                                                                                  \
  }

//----------------------------------------------------------------------------

svtkCxxSetObjectMacro(svtkBSPIntersections, Cuts, svtkBSPCuts);

//----------------------------------------------------------------------------

// Don't use svtkSetMacro or svtkBooleanMacro on these.  They will
// update the Mtime, which is incorrect.

void svtkBSPIntersections::SetComputeIntersectionsUsingDataBounds(int c)
{
  this->ComputeIntersectionsUsingDataBounds = (c != 0);
}

void svtkBSPIntersections::ComputeIntersectionsUsingDataBoundsOn()
{
  this->ComputeIntersectionsUsingDataBounds = 1;
}

void svtkBSPIntersections::ComputeIntersectionsUsingDataBoundsOff()
{
  this->ComputeIntersectionsUsingDataBounds = 0;
}

//----------------------------------------------------------------------------
svtkBSPIntersections::svtkBSPIntersections()
{
  this->Cuts = nullptr;
  this->NumberOfRegions = 0;
  this->RegionList = nullptr;
  this->ComputeIntersectionsUsingDataBounds = 0;
  svtkMath::UninitializeBounds(this->CellBoundsCache);
}

//----------------------------------------------------------------------------
svtkBSPIntersections::~svtkBSPIntersections()
{
  this->SetCuts(nullptr);
  delete[] this->RegionList;
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::BuildRegionList()
{
  if ((this->RegionList != nullptr) && (this->RegionListBuildTime > this->GetMTime()))
  {
    return 0;
  }

  delete[] this->RegionList;
  this->RegionList = nullptr;

  svtkKdNode* top = nullptr;
  if (this->Cuts)
  {
    top = this->Cuts->GetKdNodeTree();
  }

  if (!top)
  {
    return 1;
  }

  this->NumberOfRegions = svtkBSPIntersections::NumberOfLeafNodes(top);

  if (this->NumberOfRegions < 1)
  {
    svtkErrorMacro(<< "svtkBSPIntersections::BuildRegionList no cuts in svtkBSPCut object");
    return 1;
  }

  this->RegionList = new svtkKdNode*[this->NumberOfRegions];

  if (!this->RegionList)
  {
    svtkErrorMacro(<< "svtkBSPIntersections::BuildRegionList memory allocation");
    return 1;
  }

  int fail = this->SelfRegister(top);

  if (fail)
  {
    svtkErrorMacro(<< "svtkBSPIntersections::BuildRegionList bad ids in svtkBSPCut object");
    return 1;
  }

  int min = 0;
  int max = 0;

  svtkBSPIntersections::SetIDRanges(top, min, max);

  this->RegionListBuildTime.Modified();

  return 0;
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::SelfRegister(svtkKdNode* kd)
{
  int fail = 0;

  if (kd->GetLeft() == nullptr)
  {
    int id = kd->GetID();

    if ((id < 0) || (id >= this->NumberOfRegions))
    {
      return 1;
    }
    this->RegionList[id] = kd;
  }
  else
  {
    fail = this->SelfRegister(kd->GetLeft());

    if (!fail)
    {
      fail = this->SelfRegister(kd->GetRight());
    }
  }

  return fail;
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::NumberOfLeafNodes(svtkKdNode* kd)
{
  int nLeafNodes = 1;

  if (kd->GetLeft() != nullptr)
  {
    int numLeft = svtkBSPIntersections::NumberOfLeafNodes(kd->GetLeft());
    int numRight = svtkBSPIntersections::NumberOfLeafNodes(kd->GetRight());

    nLeafNodes = numLeft + numRight;
  }

  return nLeafNodes;
}
//----------------------------------------------------------------------------
void svtkBSPIntersections::SetIDRanges(svtkKdNode* kd, int& min, int& max)
{
  int tempMin = 0;
  int tempMax = 0;

  if (kd->GetLeft() == nullptr)
  {
    min = kd->GetID();
    max = kd->GetID();
  }
  else
  {
    svtkBSPIntersections::SetIDRanges(kd->GetLeft(), min, max);
    svtkBSPIntersections::SetIDRanges(kd->GetRight(), tempMin, tempMax);
    max = (tempMax > max) ? tempMax : max;
    min = (tempMin < min) ? tempMin : min;
  }

  kd->SetMinID(min);
  kd->SetMaxID(max);
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::GetBounds(double* bounds)
{
  REGIONCHECK(1);

  this->Cuts->GetKdNodeTree()->GetBounds(bounds);

  return 0;
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::GetNumberOfRegions()
{
  REGIONCHECK(0)

  return this->NumberOfRegions;
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::GetRegionBounds(int regionID, double bounds[6])
{
  REGIONIDCHECK_RETURNERR(regionID, 1)

  svtkKdNode* node = this->RegionList[regionID];

  node->GetBounds(bounds);

  return 0;
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::GetRegionDataBounds(int regionID, double bounds[6])
{
  REGIONIDCHECK_RETURNERR(regionID, 1)

  svtkKdNode* node = this->RegionList[regionID];

  node->GetDataBounds(bounds);

  return 0;
}

//----------------------------------------------------------------------------
//  Query functions ----------------------------------------------------
//    K-d Trees are particularly efficient with region intersection
//    queries, like finding all regions that intersect a view frustum
//
// Intersection with axis-aligned box----------------------------------
//

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsBox(int regionId, double* x)
{
  return this->IntersectsBox(regionId, x[0], x[1], x[2], x[3], x[4], x[5]);
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsBox(
  int regionId, double x0, double x1, double y0, double y1, double z0, double z1)
{
  REGIONIDCHECK_RETURNERR(regionId, 0);

  svtkKdNode* node = this->RegionList[regionId];

  return node->IntersectsBox(x0, x1, y0, y1, z0, z1, this->ComputeIntersectionsUsingDataBounds);
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsBox(int* ids, int len, double* x)
{
  return this->IntersectsBox(ids, len, x[0], x[1], x[2], x[3], x[4], x[5]);
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsBox(
  int* ids, int len, double x0, double x1, double y0, double y1, double z0, double z1)
{
  REGIONCHECK(0);

  int nnodes = 0;

  if (len > 0)
  {
    nnodes = this->_IntersectsBox(this->Cuts->GetKdNodeTree(), ids, len, x0, x1, y0, y1, z0, z1);
  }
  return nnodes;
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::_IntersectsBox(svtkKdNode* node, int* ids, int len, double x0, double x1,
  double y0, double y1, double z0, double z1)
{
  int result, nnodes1, nnodes2, listlen;
  int* idlist;

  result = node->IntersectsBox(x0, x1, y0, y1, z0, z1, this->ComputeIntersectionsUsingDataBounds);

  if (!result)
  {
    return 0;
  }

  if (node->GetLeft() == nullptr)
  {
    ids[0] = node->GetID();
    return 1;
  }

  nnodes1 = _IntersectsBox(node->GetLeft(), ids, len, x0, x1, y0, y1, z0, z1);

  idlist = ids + nnodes1;
  listlen = len - nnodes1;

  if (listlen > 0)
  {
    nnodes2 = _IntersectsBox(node->GetRight(), idlist, listlen, x0, x1, y0, y1, z0, z1);
  }
  else
  {
    nnodes2 = 0;
  }

  return (nnodes1 + nnodes2);
}

//----------------------------------------------------------------------------
// Intersection with a sphere---------------------------------------
//
int svtkBSPIntersections::IntersectsSphere2(
  int regionId, double x, double y, double z, double rSquared)
{
  REGIONIDCHECK_RETURNERR(regionId, 0);

  svtkKdNode* node = this->RegionList[regionId];

  return node->IntersectsSphere2(x, y, z, rSquared, this->ComputeIntersectionsUsingDataBounds);
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsSphere2(
  int* ids, int len, double x, double y, double z, double rSquared)
{
  REGIONCHECK(0)

  int nnodes = 0;

  if (len > 0)
  {
    nnodes = this->_IntersectsSphere2(this->Cuts->GetKdNodeTree(), ids, len, x, y, z, rSquared);
  }
  return nnodes;
}

//----------------------------------------------------------------------------
int svtkBSPIntersections::_IntersectsSphere2(
  svtkKdNode* node, int* ids, int len, double x, double y, double z, double rSquared)
{
  int result, nnodes1, nnodes2, listlen;
  int* idlist;

  result = node->IntersectsSphere2(x, y, z, rSquared, this->ComputeIntersectionsUsingDataBounds);

  if (!result)
  {
    return 0;
  }

  if (node->GetLeft() == nullptr)
  {
    ids[0] = node->GetID();
    return 1;
  }

  nnodes1 = _IntersectsSphere2(node->GetLeft(), ids, len, x, y, z, rSquared);

  idlist = ids + nnodes1;
  listlen = len - nnodes1;

  if (listlen > 0)
  {
    nnodes2 = _IntersectsSphere2(node->GetRight(), idlist, listlen, x, y, z, rSquared);
  }
  else
  {
    nnodes2 = 0;
  }

  return (nnodes1 + nnodes2);
}

//----------------------------------------------------------------------------
// Intersection with arbitrary svtkCell -----------------------------
//

//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsCell(int regionId, svtkCell* cell, int cellRegion)
{
  REGIONIDCHECK_RETURNERR(regionId, 0);

  svtkKdNode* node = this->RegionList[regionId];

  return node->IntersectsCell(cell, this->ComputeIntersectionsUsingDataBounds, cellRegion);
}
//----------------------------------------------------------------------------
void svtkBSPIntersections::SetCellBounds(svtkCell* cell, double* bounds)
{
  svtkPoints* pts = cell->GetPoints();
  pts->Modified(); // SVTK bug - so bounds will be re-calculated
  pts->GetBounds(bounds);
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::IntersectsCell(int* ids, int len, svtkCell* cell, int cellRegion)
{
  REGIONCHECK(0)

  svtkBSPIntersections::SetCellBounds(cell, this->CellBoundsCache);

  return this->_IntersectsCell(this->Cuts->GetKdNodeTree(), ids, len, cell, cellRegion);
}
//----------------------------------------------------------------------------
int svtkBSPIntersections::_IntersectsCell(
  svtkKdNode* node, int* ids, int len, svtkCell* cell, int cellRegion)
{
  int result, nnodes1, nnodes2, listlen, intersects;
  int* idlist;

  intersects = node->IntersectsCell(
    cell, this->ComputeIntersectionsUsingDataBounds, cellRegion, this->CellBoundsCache);

  if (intersects)
  {
    if (node->GetLeft())
    {
      nnodes1 = this->_IntersectsCell(node->GetLeft(), ids, len, cell, cellRegion);

      idlist = ids + nnodes1;
      listlen = len - nnodes1;

      if (listlen > 0)
      {
        nnodes2 = this->_IntersectsCell(node->GetRight(), idlist, listlen, cell, cellRegion);
      }
      else
      {
        nnodes2 = 0;
      }

      result = nnodes1 + nnodes2;
    }
    else
    {
      ids[0] = node->GetID(); // leaf node (spatial region)

      result = 1;
    }
  }
  else
  {
    result = 0;
  }

  return result;
}

//----------------------------------------------------------------------------
void svtkBSPIntersections::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Cuts: ";
  if (this->Cuts)
  {
    this->Cuts->PrintSelf(os << endl, indent.GetNextIndent());
  }
  else
  {
    os << "(none)" << endl;
  }
  os << indent << "NumberOfRegions: " << this->NumberOfRegions << endl;
  os << indent << "RegionList: " << this->RegionList << endl;
  os << indent << "RegionListBuildTime: " << this->RegionListBuildTime << endl;
  os << indent
     << "ComputeIntersectionsUsingDataBounds: " << this->ComputeIntersectionsUsingDataBounds
     << endl;
  double* d = this->CellBoundsCache;
  os << indent << "CellBoundsCache " << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << " "
     << d[4] << " " << d[5] << " " << endl;
}
