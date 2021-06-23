/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkClosestNPointsStrategy.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkClosestNPointsStrategy.h"

#include "svtkCell.h"
#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkPointLocator.h"
#include "svtkPointSet.h"
#include "svtkStaticPointLocator.h"

#include <set>

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkClosestNPointsStrategy);

//----------------------------------------------------------------------------
svtkClosestNPointsStrategy::svtkClosestNPointsStrategy()
{
  this->ClosestNPoints = 9;
}

//----------------------------------------------------------------------------
svtkClosestNPointsStrategy::~svtkClosestNPointsStrategy() {}

//-----------------------------------------------------------------------------
svtkIdType svtkClosestNPointsStrategy::FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell,
  svtkIdType cellId, double tol2, int& subId, double pcoords[3], double* weights)
{
  // First try standard strategy which is reasonably fast
  svtkIdType foundCell =
    this->Superclass::FindCell(x, cell, gencell, cellId, tol2, subId, pcoords, weights);
  if (foundCell >= 0)
  {
    return foundCell;
  }

  // Couldn't find anything so try more time consuming strategy.  It is
  // possible that the closest point is not part of a cell containing the
  // query point (i.e., a hanging node situation). In this case, look for the
  // N closest points (beyond any coincident points identified
  // previously). Typically N=9 (somewhat arbitrary, empirical, based on 2:1
  // subdivision of hexahedral cells). Using large N affects performance but
  // produces better results.
  svtkIdType numPts = this->NearPointIds->GetNumberOfIds();
  this->PointLocator->FindClosestNPoints(numPts + this->ClosestNPoints, x, this->NearPointIds);
  numPts = this->NearPointIds->GetNumberOfIds();

  svtkIdType i, j, ptId, numCells;
  int ret;
  double closest[3], dist2;
  for (i = 0; i < numPts; ++i)
  {
    ptId = this->NearPointIds->GetId(i);
    this->PointSet->GetPointCells(ptId, this->CellIds);
    numCells = this->CellIds->GetNumberOfIds();
    for (j = 0; j < numCells; j++)
    {
      cellId = this->CellIds->GetId(j);
      if (this->VisitedCells.find(cellId) == this->VisitedCells.end())
      {
        cell = this->SelectCell(this->PointSet, cellId, nullptr, gencell);
        ret = cell->EvaluatePosition(x, closest, subId, pcoords, dist2, weights);
        if (ret != -1 && dist2 <= tol2)
        {
          return cellId;
        }
        this->VisitedCells.insert(cellId);
      }
    }
  }

  return -1;
}

//----------------------------------------------------------------------------
void svtkClosestNPointsStrategy::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
