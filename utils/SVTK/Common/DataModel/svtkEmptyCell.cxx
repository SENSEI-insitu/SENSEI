/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEmptyCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkEmptyCell.h"

#include "svtkCellArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkEmptyCell);

//----------------------------------------------------------------------------
int svtkEmptyCell::EvaluatePosition(const double svtkNotUsed(x)[3], double closestPoint[3],
  int& subId, double pcoords[3], double& dist2, double svtkNotUsed(weights)[])
{
  pcoords[0] = pcoords[1] = pcoords[2] = -1.0;
  subId = 0;
  if (closestPoint != nullptr)
  {
    closestPoint[0] = closestPoint[1] = closestPoint[2] = 0.0;
    dist2 = -1.0;
  }
  return 0;
}

//----------------------------------------------------------------------------
void svtkEmptyCell::EvaluateLocation(int& svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3],
  double x[3], double* svtkNotUsed(weights))
{
  x[0] = x[1] = x[2] = 0.0;
}

//----------------------------------------------------------------------------
int svtkEmptyCell::CellBoundary(
  int svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3], svtkIdList* pts)
{
  pts->Reset();
  return 0;
}

//----------------------------------------------------------------------------
void svtkEmptyCell::Contour(double svtkNotUsed(value), svtkDataArray* svtkNotUsed(cellScalars),
  svtkIncrementalPointLocator* svtkNotUsed(locator), svtkCellArray* svtkNotUsed(verts),
  svtkCellArray* svtkNotUsed(lines), svtkCellArray* svtkNotUsed(polys), svtkPointData* svtkNotUsed(inPd),
  svtkPointData* svtkNotUsed(outPd), svtkCellData* svtkNotUsed(inCd), svtkIdType svtkNotUsed(cellId),
  svtkCellData* svtkNotUsed(outCd))
{
}

//----------------------------------------------------------------------------
// Project point on line. If it lies between 0<=t<=1 and distance off line
// is less than tolerance, intersection detected.
int svtkEmptyCell::IntersectWithLine(const double svtkNotUsed(p1)[3], const double svtkNotUsed(p2)[3],
  double svtkNotUsed(tol), double& svtkNotUsed(t), double svtkNotUsed(x)[3],
  double svtkNotUsed(pcoords)[3], int& svtkNotUsed(subId))
{
  return 0;
}

//----------------------------------------------------------------------------
int svtkEmptyCell::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  pts->Reset();
  ptIds->Reset();
  return 1;
}

//----------------------------------------------------------------------------
void svtkEmptyCell::Derivatives(int svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3],
  const double* svtkNotUsed(values), int svtkNotUsed(dim), double* svtkNotUsed(derivs))
{
}

//----------------------------------------------------------------------------
void svtkEmptyCell::Clip(double svtkNotUsed(value), svtkDataArray* svtkNotUsed(cellScalars),
  svtkIncrementalPointLocator* svtkNotUsed(locator), svtkCellArray* svtkNotUsed(verts),
  svtkPointData* svtkNotUsed(inPD), svtkPointData* svtkNotUsed(outPD), svtkCellData* svtkNotUsed(inCD),
  svtkIdType svtkNotUsed(cellId), svtkCellData* svtkNotUsed(outCD), int svtkNotUsed(insideOut))
{
}

//----------------------------------------------------------------------------
void svtkEmptyCell::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
