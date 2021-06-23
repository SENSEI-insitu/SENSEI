/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticLinearQuad.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// Thanks to Soeren Gebbert  who developed this class and
// integrated it into SVTK 5.0.

#include "svtkQuadraticLinearQuad.h"

#include "svtkDoubleArray.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkQuad.h"
#include "svtkQuadraticEdge.h"

svtkStandardNewMacro(svtkQuadraticLinearQuad);

//----------------------------------------------------------------------------
// Construct the quadratic linear quad with six points.
svtkQuadraticLinearQuad::svtkQuadraticLinearQuad()
{
  this->Edge = svtkQuadraticEdge::New();
  this->LinEdge = svtkLine::New();
  this->Quad = svtkQuad::New();
  this->Scalars = svtkDoubleArray::New();
  this->Scalars->SetNumberOfTuples(4); // vertices of a linear quad
  this->Points->SetNumberOfPoints(6);
  this->PointIds->SetNumberOfIds(6);
  for (int i = 0; i < 6; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, 0);
  }
}

//----------------------------------------------------------------------------
svtkQuadraticLinearQuad::~svtkQuadraticLinearQuad()
{
  this->Edge->Delete();
  this->LinEdge->Delete();
  this->Quad->Delete();
  this->Scalars->Delete();
}

//----------------------------------------------------------------------------
static int LinearQuads[2][4] = {
  { 0, 4, 5, 3 },
  { 4, 1, 2, 5 },
};

static int LinearQuadEdges[4][3] = {
  { 0, 1, 4 },
  { 1, 2, -1 },
  { 2, 3, 5 },
  { 3, 0, -1 },
};

//----------------------------------------------------------------------------
int* svtkQuadraticLinearQuad::GetEdgeArray(svtkIdType edgeId)
{
  return LinearQuadEdges[edgeId];
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticLinearQuad::GetEdge(int edgeId)
{
  edgeId = (edgeId < 0 ? 0 : (edgeId > 3 ? 3 : edgeId));

  // We have 2 linear edges
  if (edgeId == 1 || edgeId == 3)
  {
    this->LinEdge->PointIds->SetId(0, this->PointIds->GetId(LinearQuadEdges[edgeId][0]));
    this->LinEdge->PointIds->SetId(1, this->PointIds->GetId(LinearQuadEdges[edgeId][1]));

    this->LinEdge->Points->SetPoint(0, this->Points->GetPoint(LinearQuadEdges[edgeId][0]));
    this->LinEdge->Points->SetPoint(1, this->Points->GetPoint(LinearQuadEdges[edgeId][1]));

    return this->LinEdge;
  }
  // and two quadratic edges
  else // (edgeId == 0 || edgeId == 2)
  {
    this->Edge->PointIds->SetId(0, this->PointIds->GetId(LinearQuadEdges[edgeId][0]));
    this->Edge->PointIds->SetId(1, this->PointIds->GetId(LinearQuadEdges[edgeId][1]));
    this->Edge->PointIds->SetId(2, this->PointIds->GetId(LinearQuadEdges[edgeId][2]));

    this->Edge->Points->SetPoint(0, this->Points->GetPoint(LinearQuadEdges[edgeId][0]));
    this->Edge->Points->SetPoint(1, this->Points->GetPoint(LinearQuadEdges[edgeId][1]));
    this->Edge->Points->SetPoint(2, this->Points->GetPoint(LinearQuadEdges[edgeId][2]));

    return this->Edge;
  }
}

//----------------------------------------------------------------------------
int svtkQuadraticLinearQuad::EvaluatePosition(const double x[3], double* closestPoint, int& subId,
  double pcoords[3], double& minDist2, double* weights)
{
  double pc[3], dist2;
  int ignoreId, i, returnStatus = 0, status;
  double tempWeights[4];
  double closest[3];

  // two linear quads are used
  for (minDist2 = SVTK_DOUBLE_MAX, i = 0; i < 2; i++)
  {
    this->Quad->Points->SetPoint(0, this->Points->GetPoint(LinearQuads[i][0]));
    this->Quad->Points->SetPoint(1, this->Points->GetPoint(LinearQuads[i][1]));
    this->Quad->Points->SetPoint(2, this->Points->GetPoint(LinearQuads[i][2]));
    this->Quad->Points->SetPoint(3, this->Points->GetPoint(LinearQuads[i][3]));

    status = this->Quad->EvaluatePosition(x, closest, ignoreId, pc, dist2, tempWeights);
    if (status != -1 && dist2 < minDist2)
    {
      returnStatus = status;
      minDist2 = dist2;
      subId = i;
      pcoords[0] = pc[0];
      pcoords[1] = pc[1];
    }
  }

  // adjust parametric coordinates
  if (returnStatus != -1)
  {
    if (subId == 0)
    {
      pcoords[0] /= 2.0;
    }
    else if (subId == 1)
    {
      pcoords[0] = 0.5 + (pcoords[0] / 2.0);
    }
    pcoords[2] = 0.0;
    if (closestPoint != nullptr)
    {
      // Compute both closestPoint and weights
      this->EvaluateLocation(subId, pcoords, closestPoint, weights);
    }
    else
    {
      // Compute weights only
      this->InterpolationFunctions(pcoords, weights);
    }
  }

  return returnStatus;
}

//----------------------------------------------------------------------------
void svtkQuadraticLinearQuad::EvaluateLocation(
  int& svtkNotUsed(subId), const double pcoords[3], double x[3], double* weights)
{
  int i, j;
  double pt[3];

  this->InterpolationFunctions(pcoords, weights);

  x[0] = x[1] = x[2] = 0.0;
  for (i = 0; i < 6; i++)
  {
    this->Points->GetPoint(i, pt);
    for (j = 0; j < 3; j++)
    {
      x[j] += pt[j] * weights[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkQuadraticLinearQuad::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  return this->Quad->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
void svtkQuadraticLinearQuad::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      this->Quad->Points->SetPoint(j, this->Points->GetPoint(LinearQuads[i][j]));
      this->Quad->PointIds->SetId(j, this->PointIds->GetId(LinearQuads[i][j]));
      this->Scalars->SetValue(j, cellScalars->GetTuple1(LinearQuads[i][j]));
    }
    this->Quad->Contour(
      value, this->Scalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  }
}

//----------------------------------------------------------------------------
// Clip this quadratic quad using scalar value provided. Like contouring,
// except that it cuts the quad to produce other quads and triangles.
void svtkQuadraticLinearQuad::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 4; j++) // for each of the four vertices of the linear quad
    {
      this->Quad->Points->SetPoint(j, this->Points->GetPoint(LinearQuads[i][j]));
      this->Quad->PointIds->SetId(j, this->PointIds->GetId(LinearQuads[i][j]));
      this->Scalars->SetTuple(j, cellScalars->GetTuple(LinearQuads[i][j]));
    }
    this->Quad->Clip(
      value, this->Scalars, locator, polys, inPd, outPd, inCd, cellId, outCd, insideOut);
  }
}

//----------------------------------------------------------------------------
// Line-line intersection. Intersection has to occur within [0,1] parametric
// coordinates and with specified tolerance.
int svtkQuadraticLinearQuad::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  int subTest, i;
  subId = 0;

  // intersect the two linear quads
  for (i = 0; i < 2; i++)
  {
    this->Quad->Points->SetPoint(0, this->Points->GetPoint(LinearQuads[i][0]));
    this->Quad->Points->SetPoint(1, this->Points->GetPoint(LinearQuads[i][1]));
    this->Quad->Points->SetPoint(2, this->Points->GetPoint(LinearQuads[i][2]));
    this->Quad->Points->SetPoint(3, this->Points->GetPoint(LinearQuads[i][3]));

    if (this->Quad->IntersectWithLine(p1, p2, tol, t, x, pcoords, subTest))
    {
      return 1;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
int svtkQuadraticLinearQuad::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  pts->Reset();
  ptIds->Reset();

  // Create four linear triangles:
  // Choose the triangulation that minimizes the edge length
  // across the cell.
  double x0[3], x1[3], x2[3], x3[3], x4[3], x5[3];
  this->Points->GetPoint(0, x0);
  this->Points->GetPoint(1, x1);
  this->Points->GetPoint(2, x2);
  this->Points->GetPoint(3, x3);
  this->Points->GetPoint(4, x4);
  this->Points->GetPoint(5, x5);

  if (svtkMath::Distance2BetweenPoints(x0, x5) <= svtkMath::Distance2BetweenPoints(x3, x4))
  {
    ptIds->InsertId(0, this->PointIds->GetId(0));
    ptIds->InsertId(1, this->PointIds->GetId(4));
    ptIds->InsertId(2, this->PointIds->GetId(5));
    pts->InsertPoint(0, this->Points->GetPoint(0));
    pts->InsertPoint(1, this->Points->GetPoint(4));
    pts->InsertPoint(2, this->Points->GetPoint(5));

    ptIds->InsertId(3, this->PointIds->GetId(0));
    ptIds->InsertId(4, this->PointIds->GetId(5));
    ptIds->InsertId(5, this->PointIds->GetId(3));
    pts->InsertPoint(3, this->Points->GetPoint(0));
    pts->InsertPoint(4, this->Points->GetPoint(5));
    pts->InsertPoint(5, this->Points->GetPoint(3));
  }
  else
  {
    ptIds->InsertId(0, this->PointIds->GetId(0));
    ptIds->InsertId(1, this->PointIds->GetId(4));
    ptIds->InsertId(2, this->PointIds->GetId(3));
    pts->InsertPoint(0, this->Points->GetPoint(0));
    pts->InsertPoint(1, this->Points->GetPoint(4));
    pts->InsertPoint(2, this->Points->GetPoint(3));

    ptIds->InsertId(3, this->PointIds->GetId(4));
    ptIds->InsertId(4, this->PointIds->GetId(5));
    ptIds->InsertId(5, this->PointIds->GetId(3));
    pts->InsertPoint(3, this->Points->GetPoint(4));
    pts->InsertPoint(4, this->Points->GetPoint(5));
    pts->InsertPoint(5, this->Points->GetPoint(3));
  }

  if (svtkMath::Distance2BetweenPoints(x4, x2) <= svtkMath::Distance2BetweenPoints(x5, x1))
  {
    ptIds->InsertId(6, this->PointIds->GetId(4));
    ptIds->InsertId(7, this->PointIds->GetId(1));
    ptIds->InsertId(8, this->PointIds->GetId(2));
    pts->InsertPoint(6, this->Points->GetPoint(4));
    pts->InsertPoint(7, this->Points->GetPoint(1));
    pts->InsertPoint(8, this->Points->GetPoint(2));

    ptIds->InsertId(9, this->PointIds->GetId(4));
    ptIds->InsertId(10, this->PointIds->GetId(2));
    ptIds->InsertId(11, this->PointIds->GetId(5));
    pts->InsertPoint(9, this->Points->GetPoint(4));
    pts->InsertPoint(10, this->Points->GetPoint(2));
    pts->InsertPoint(11, this->Points->GetPoint(5));
  }
  else
  {
    ptIds->InsertId(6, this->PointIds->GetId(4));
    ptIds->InsertId(7, this->PointIds->GetId(1));
    ptIds->InsertId(8, this->PointIds->GetId(5));
    pts->InsertPoint(6, this->Points->GetPoint(4));
    pts->InsertPoint(7, this->Points->GetPoint(1));
    pts->InsertPoint(8, this->Points->GetPoint(5));

    ptIds->InsertId(9, this->PointIds->GetId(1));
    ptIds->InsertId(10, this->PointIds->GetId(2));
    ptIds->InsertId(11, this->PointIds->GetId(5));
    pts->InsertPoint(9, this->Points->GetPoint(1));
    pts->InsertPoint(10, this->Points->GetPoint(2));
    pts->InsertPoint(11, this->Points->GetPoint(5));
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkQuadraticLinearQuad::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  double x0[3], x1[3], x2[3], deltaX[3], weights[6];
  int i, j;
  double functionDerivs[12];

  this->Points->GetPoint(0, x0);
  this->Points->GetPoint(1, x1);
  this->Points->GetPoint(2, x2);

  this->InterpolationFunctions(pcoords, weights);
  this->InterpolationDerivs(pcoords, functionDerivs);

  for (i = 0; i < 3; i++)
  {
    deltaX[i] = x1[i] - x0[i] - x2[i];
  }
  for (i = 0; i < dim; i++)
  {
    for (j = 0; j < 3; j++)
    {
      if (deltaX[j] != 0)
      {
        derivs[3 * i + j] = (values[2 * i + 1] - values[2 * i]) / deltaX[j];
      }
      else
      {
        derivs[3 * i + j] = 0;
      }
    }
  }
}

//----------------------------------------------------------------------------
// Compute interpolation functions. The first four nodes are the corner
// vertices; the others are mid-edge nodes.
void svtkQuadraticLinearQuad::InterpolationFunctions(const double pcoords[3], double weights[6])
{
  double x = pcoords[0];
  double y = pcoords[1];

  // corners
  weights[0] = -1.0 * (2.0 * x - 1.0) * (x - 1.0) * (y - 1.0);
  weights[1] = -1.0 * (2.0 * x - 1.0) * (x) * (y - 1.0);
  weights[2] = (2.0 * x - 1.0) * (x) * (y);
  weights[3] = (2.0 * x - 1.0) * (x - 1.0) * (y);

  // Edge middle nodes
  weights[4] = 4.0 * (x) * (1.0 - x) * (1.0 - y);
  weights[5] = 4.0 * (x) * (1.0 - x) * (y);
}

//----------------------------------------------------------------------------
// Derivatives in parametric space.
void svtkQuadraticLinearQuad::InterpolationDerivs(const double pcoords[3], double derivs[12])
{
  double x = pcoords[0];
  double y = pcoords[1];

  // Derivatives in the x-direction
  // corners
  derivs[0] = -1.0 * (4.0 * x - 3.0) * (y - 1.0);
  derivs[1] = -1.0 * (4.0 * x - 1.0) * (y - 1.0);
  derivs[2] = (4.0 * x - 1.0) * (y);
  derivs[3] = (4.0 * x - 3.0) * (y);
  // Edge middle nodes
  derivs[4] = 4.0 * (1.0 - 2.0 * x) * (1.0 - y);
  derivs[5] = 4.0 * (1.0 - 2.0 * x) * (y);

  // Derivatives in the y-direction
  // corners
  derivs[6] = -1.0 * (2.0 * x - 1.0) * (x - 1.0);
  derivs[7] = -1.0 * (2.0 * x - 1.0) * (x);
  derivs[8] = (2.0 * x - 1.0) * (x);
  derivs[9] = (2.0 * x - 1.0) * (x - 1.0);
  // Edge middle nodes
  derivs[10] = -4.0 * (x) * (1.0 - x);
  derivs[11] = 4.0 * (x) * (1.0 - x);
}

//----------------------------------------------------------------------------
static double svtkQLinQuadCellPCoords[18] = {
  0.0, 0.0, 0.0, //
  1.0, 0.0, 0.0, //
  1.0, 1.0, 0.0, //
  0.0, 1.0, 0.0, //
  0.5, 0.0, 0.0, //
  0.5, 1.0, 0.0  //
};

double* svtkQuadraticLinearQuad::GetParametricCoords()
{
  return svtkQLinQuadCellPCoords;
}

//----------------------------------------------------------------------------
void svtkQuadraticLinearQuad::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Edge:\n";
  this->Edge->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Quad:\n";
  this->Quad->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Scalars:\n";
  this->Scalars->PrintSelf(os, indent.GetNextIndent());
}
