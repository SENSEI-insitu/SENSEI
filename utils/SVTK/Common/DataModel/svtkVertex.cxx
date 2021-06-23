/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVertex.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkVertex.h"

#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkIncrementalPointLocator.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkVertex);

//----------------------------------------------------------------------------
// Construct the vertex with a single point.
svtkVertex::svtkVertex()
{
  this->Points->SetNumberOfPoints(1);
  this->PointIds->SetNumberOfIds(1);
  for (int i = 0; i < 1; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, 0);
  }
}

//----------------------------------------------------------------------------
// Make a new svtkVertex object with the same information as this object.
int svtkVertex::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& dist2, double weights[])
{
  double X[3];

  subId = 0;
  pcoords[1] = pcoords[2] = 0.0;

  this->Points->GetPoint(0, X);
  if (closestPoint)
  {
    closestPoint[0] = X[0];
    closestPoint[1] = X[1];
    closestPoint[2] = X[2];
  }

  dist2 = svtkMath::Distance2BetweenPoints(X, x);
  weights[0] = 1.0;

  if (dist2 == 0.0)
  {
    pcoords[0] = 0.0;
    return 1;
  }
  else
  {
    pcoords[0] = -1.0;
    return 0;
  }
}

//----------------------------------------------------------------------------
void svtkVertex::EvaluateLocation(
  int& svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3], double x[3], double* weights)
{
  this->Points->GetPoint(0, x);

  weights[0] = 1.0;
}

//----------------------------------------------------------------------------
// Given parametric coordinates of a point, return the closest cell boundary,
// and whether the point is inside or outside of the cell. The cell boundary
// is defined by a list of points (pts) that specify a vertex (1D cell).
// If the return value of the method is != 0, then the point is inside the cell.
int svtkVertex::CellBoundary(int svtkNotUsed(subId), const double pcoords[3], svtkIdList* pts)
{

  pts->SetNumberOfIds(1);
  pts->SetId(0, this->PointIds->GetId(0));

  if (pcoords[0] != 0.0)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

//----------------------------------------------------------------------------
// Generate contouring primitives. The scalar list cellScalars are
// scalar values at each cell point. The point locator is essentially a
// points list that merges points as they are inserted (i.e., prevents
// duplicates).
void svtkVertex::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* svtkNotUsed(lines),
  svtkCellArray* svtkNotUsed(polys), svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
  svtkIdType cellId, svtkCellData* outCd)
{
  if (value == cellScalars->GetComponent(0, 0))
  {
    int newCellId;
    svtkIdType pts[1];
    pts[0] = locator->InsertNextPoint(this->Points->GetPoint(0));
    if (outPd)
    {
      outPd->CopyData(inPd, this->PointIds->GetId(0), pts[0]);
    }
    newCellId = verts->InsertNextCell(1, pts);
    if (outCd)
    {
      outCd->CopyData(inCd, cellId, newCellId);
    }
  }
}

//----------------------------------------------------------------------------
// Intersect with a ray. Return parametric coordinates (both line and cell)
// and global intersection coordinates, given ray definition and tolerance.
// The method returns non-zero value if intersection occurs.
int svtkVertex::IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
  double x[3], double pcoords[3], int& subId)
{
  int i;
  double X[3], ray[3], rayFactor, projXYZ[3];

  subId = 0;
  pcoords[1] = pcoords[2] = 0.0;

  this->Points->GetPoint(0, X);

  for (i = 0; i < 3; i++)
  {
    ray[i] = p2[i] - p1[i];
  }
  if ((rayFactor = svtkMath::Dot(ray, ray)) == 0.0)
  {
    return 0;
  }
  //
  //  Project each point onto ray. Determine whether point is within tolerance.
  //
  t = (ray[0] * (X[0] - p1[0]) + ray[1] * (X[1] - p1[1]) + ray[2] * (X[2] - p1[2])) / rayFactor;

  if (t >= 0.0 && t <= 1.0)
  {
    for (i = 0; i < 3; i++)
    {
      projXYZ[i] = p1[i] + t * ray[i];
      if (fabs(X[i] - projXYZ[i]) > tol)
      {
        break;
      }
    }

    if (i > 2) // within tolerance
    {
      pcoords[0] = 0.0;
      x[0] = X[0];
      x[1] = X[1];
      x[2] = X[2];
      return 1;
    }
  }

  pcoords[0] = -1.0;
  return 0;
}

//----------------------------------------------------------------------------
// Triangulate the vertex. This method fills pts and ptIds with information
// from the only point in the vertex.
int svtkVertex::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  pts->Reset();
  ptIds->Reset();
  pts->InsertPoint(0, this->Points->GetPoint(0));
  ptIds->InsertId(0, this->PointIds->GetId(0));

  return 1;
}

//----------------------------------------------------------------------------
// Get the derivative of the vertex. Returns (0.0, 0.0, 0.0) for all
// dimensions.
void svtkVertex::Derivatives(int svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3],
  const double* svtkNotUsed(values), int dim, double* derivs)
{
  int i, idx;

  for (i = 0; i < dim; i++)
  {
    idx = i * dim;
    derivs[idx] = 0.0;
    derivs[idx + 1] = 0.0;
    derivs[idx + 2] = 0.0;
  }
}

//----------------------------------------------------------------------------
void svtkVertex::Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
  svtkCellArray* verts, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd, int insideOut)
{
  double s, x[3];
  int newCellId;
  svtkIdType pts[1];

  s = cellScalars->GetComponent(0, 0);

  if ((!insideOut && s > value) || (insideOut && s <= value))
  {
    this->Points->GetPoint(0, x);
    if (locator->InsertUniquePoint(x, pts[0]))
    {
      outPd->CopyData(inPd, this->PointIds->GetId(0), pts[0]);
    }
    newCellId = verts->InsertNextCell(1, pts);
    outCd->CopyData(inCd, cellId, newCellId);
  }
}

//----------------------------------------------------------------------------
// Compute interpolation functions
void svtkVertex::InterpolationFunctions(const double[3], double weights[1])
{
  weights[0] = 1.0;
}

//----------------------------------------------------------------------------
void svtkVertex::InterpolationDerivs(const double[3], double derivs[3])
{
  derivs[0] = 0.0;
  derivs[1] = 0.0;
  derivs[2] = 0.0;
}

//----------------------------------------------------------------------------
static double svtkVertexCellPCoords[3] = { 0.0, 0.0, 0.0 };
double* svtkVertex::GetParametricCoords()
{
  return svtkVertexCellPCoords;
}

//----------------------------------------------------------------------------
void svtkVertex::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
