/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticTetra.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkQuadraticTetra.h"

#include "svtkDoubleArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkQuadraticEdge.h"
#include "svtkQuadraticTriangle.h"
#include "svtkTetra.h"

svtkStandardNewMacro(svtkQuadraticTetra);

//----------------------------------------------------------------------------
// Construct the tetra with ten points.
svtkQuadraticTetra::svtkQuadraticTetra()
{
  this->Edge = svtkQuadraticEdge::New();
  this->Face = svtkQuadraticTriangle::New();
  this->Tetra = svtkTetra::New();
  this->Scalars = svtkDoubleArray::New();
  this->Scalars->SetNumberOfTuples(4);

  this->Points->SetNumberOfPoints(10);
  this->PointIds->SetNumberOfIds(10);
  for (int i = 0; i < 10; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, 0);
  }
}

//----------------------------------------------------------------------------
svtkQuadraticTetra::~svtkQuadraticTetra()
{
  this->Edge->Delete();
  this->Face->Delete();
  this->Tetra->Delete();
  this->Scalars->Delete();
}

//----------------------------------------------------------------------------
// clip each of the four vertices; the remaining octahedron is
// divided into four tetrahedron.
static int LinearTetras[3][8][4] = {
  { { 0, 4, 6, 7 }, { 4, 1, 5, 8 }, { 6, 5, 2, 9 }, { 7, 8, 9, 3 }, { 6, 4, 5, 8 }, { 6, 5, 9, 8 },
    { 6, 9, 7, 8 }, { 6, 7, 4, 8 } },
  { { 0, 4, 6, 7 }, { 4, 1, 5, 8 }, { 6, 5, 2, 9 }, { 7, 8, 9, 3 }, { 4, 8, 5, 9 }, { 4, 5, 6, 9 },
    { 4, 6, 7, 9 }, { 4, 7, 8, 9 } },
  { { 0, 4, 6, 7 }, { 4, 1, 5, 8 }, { 6, 5, 2, 9 }, { 7, 8, 9, 3 }, { 5, 9, 6, 7 }, { 5, 6, 4, 7 },
    { 5, 4, 8, 7 }, { 5, 8, 9, 7 } },
};

static constexpr svtkIdType TetraFaces[4][6] = {
  { 0, 1, 3, 4, 8, 7 },
  { 1, 2, 3, 5, 9, 8 },
  { 2, 0, 3, 6, 7, 9 },
  { 0, 2, 1, 6, 5, 4 },
};

static constexpr svtkIdType TetraEdges[6][3] = {
  { 0, 1, 4 },
  { 1, 2, 5 },
  { 2, 0, 6 },
  { 0, 3, 7 },
  { 1, 3, 8 },
  { 2, 3, 9 },
};

//------------------------Tuple1----------------------------------------------------
const svtkIdType* svtkQuadraticTetra::GetEdgeArray(svtkIdType edgeId)
{
  return TetraEdges[edgeId];
}
//----------------------------------------------------------------------------
const svtkIdType* svtkQuadraticTetra::GetFaceArray(svtkIdType faceId)
{
  return TetraFaces[faceId];
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticTetra::GetEdge(int edgeId)
{
  edgeId = (edgeId < 0 ? 0 : (edgeId > 5 ? 5 : edgeId));

  // load point id's
  this->Edge->PointIds->SetId(0, this->PointIds->GetId(TetraEdges[edgeId][0]));
  this->Edge->PointIds->SetId(1, this->PointIds->GetId(TetraEdges[edgeId][1]));
  this->Edge->PointIds->SetId(2, this->PointIds->GetId(TetraEdges[edgeId][2]));

  // load coordinates
  this->Edge->Points->SetPoint(0, this->Points->GetPoint(TetraEdges[edgeId][0]));
  this->Edge->Points->SetPoint(1, this->Points->GetPoint(TetraEdges[edgeId][1]));
  this->Edge->Points->SetPoint(2, this->Points->GetPoint(TetraEdges[edgeId][2]));

  return this->Edge;
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticTetra::GetFace(int faceId)
{
  faceId = (faceId < 0 ? 0 : (faceId > 3 ? 3 : faceId));

  // load point id's and coordinates
  for (int i = 0; i < 6; i++)
  {
    this->Face->PointIds->SetId(i, this->PointIds->GetId(TetraFaces[faceId][i]));
    this->Face->Points->SetPoint(i, this->Points->GetPoint(TetraFaces[faceId][i]));
  }

  return this->Face;
}

//----------------------------------------------------------------------------
namespace
{
static const double SVTK_DIVERGED = 1.e6;
static const int SVTK_TETRA_MAX_ITERATION = 20;
static const double SVTK_TETRA_CONVERGED = 1.e-05;
}

int svtkQuadraticTetra::EvaluatePosition(const double* x, double closestPoint[3], int& subId,
  double pcoords[3], double& dist2, double weights[])
{
  double params[3] = { .25, .25, .25 };
  double derivs[30];

  // compute a bound on the volume to get a scale for an acceptable determinant
  double longestEdge = 0;
  for (int i = 0; i < 6; i++)
  {
    double pt0[3], pt1[3];
    this->Points->GetPoint(TetraEdges[i][0], pt0);
    this->Points->GetPoint(TetraEdges[i][1], pt1);
    double d2 = svtkMath::Distance2BetweenPoints(pt0, pt1);
    if (longestEdge < d2)
    {
      longestEdge = d2;
    }
  }
  // longestEdge value is already squared
  double volumeBound = pow(longestEdge, 1.5);
  double determinantTolerance = 1e-20 < .00001 * volumeBound ? 1e-20 : .00001 * volumeBound;

  //  set initial position for Newton's method
  subId = 0;
  pcoords[0] = pcoords[1] = pcoords[2] = 0.25;
  //  enter iteration loop
  int converged = 0;
  for (int iteration = 0; !converged && (iteration < SVTK_TETRA_MAX_ITERATION); iteration++)
  {
    //  calculate element interpolation functions and derivatives
    this->InterpolationFunctions(pcoords, weights);
    this->InterpolationDerivs(pcoords, derivs);

    //  calculate newton functions
    double fcol[3] = { 0, 0, 0 }, rcol[3] = { 0, 0, 0 }, scol[3] = { 0, 0, 0 },
           tcol[3] = { 0, 0, 0 };
    for (int i = 0; i < 10; i++)
    {
      double pt[3];
      this->Points->GetPoint(i, pt);
      for (int j = 0; j < 3; j++)
      {
        fcol[j] += pt[j] * weights[i];
        rcol[j] += pt[j] * derivs[i];
        scol[j] += pt[j] * derivs[i + 10];
        tcol[j] += pt[j] * derivs[i + 20];
      }
    }

    for (int i = 0; i < 3; i++)
    {
      fcol[i] -= x[i];
    }

    //  compute determinants and generate improvements
    double d = svtkMath::Determinant3x3(rcol, scol, tcol);
    if (fabs(d) < determinantTolerance)
    {
      return -1;
    }

    pcoords[0] = params[0] - 0.5 * svtkMath::Determinant3x3(fcol, scol, tcol) / d;
    pcoords[1] = params[1] - 0.5 * svtkMath::Determinant3x3(rcol, fcol, tcol) / d;
    pcoords[2] = params[2] - 0.5 * svtkMath::Determinant3x3(rcol, scol, fcol) / d;

    //  check for convergence
    if (((fabs(pcoords[0] - params[0])) < SVTK_TETRA_CONVERGED) &&
      ((fabs(pcoords[1] - params[1])) < SVTK_TETRA_CONVERGED) &&
      ((fabs(pcoords[2] - params[2])) < SVTK_TETRA_CONVERGED))
    {
      converged = 1;
    }

    // Test for bad divergence (S.Hirschberg 11.12.2001)
    else if ((fabs(pcoords[0]) > SVTK_DIVERGED) || (fabs(pcoords[1]) > SVTK_DIVERGED) ||
      (fabs(pcoords[2]) > SVTK_DIVERGED))
    {
      return -1;
    }

    //  if not converged, repeat
    else
    {
      params[0] = pcoords[0];
      params[1] = pcoords[1];
      params[2] = pcoords[2];
    }
  }

  //  if not converged, set the parametric coordinates to arbitrary values
  //  outside of element
  if (!converged)
  {
    return -1;
  }

  this->InterpolationFunctions(pcoords, weights);

  if (pcoords[0] >= -0.001 && pcoords[0] <= 1.001 && pcoords[1] >= -0.001 && pcoords[1] <= 1.001 &&
    pcoords[2] >= -0.001 && pcoords[2] <= 1.001 && pcoords[0] + pcoords[1] + pcoords[2] <= 1.001)
  {
    if (closestPoint)
    {
      closestPoint[0] = x[0];
      closestPoint[1] = x[1];
      closestPoint[2] = x[2];
      dist2 = 0.0; // inside tetra
    }
    return 1;
  }
  else
  {
    double pc[3], w[10];
    if (closestPoint)
    {
      for (int i = 0; i < 3; i++) // only approximate, not really true for warped tetra
      {
        if (pcoords[i] < 0.0)
        {
          pc[i] = 0.0;
        }
        else if (pcoords[i] > 1.0)
        {
          pc[i] = 1.0;
        }
        else
        {
          pc[i] = pcoords[i];
        }
      }
      this->EvaluateLocation(subId, pc, closestPoint, static_cast<double*>(w));
      dist2 = svtkMath::Distance2BetweenPoints(closestPoint, x);
    }
    return 0;
  }
}

//----------------------------------------------------------------------------
void svtkQuadraticTetra::EvaluateLocation(
  int& svtkNotUsed(subId), const double pcoords[3], double x[3], double* weights)
{
  int i, j;
  double pt[3];

  this->InterpolationFunctions(pcoords, weights);

  x[0] = x[1] = x[2] = 0.0;
  for (i = 0; i < 10; i++)
  {
    this->Points->GetPoint(i, pt);
    for (j = 0; j < 3; j++)
    {
      x[j] += pt[j] * weights[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkQuadraticTetra::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  for (int i = 0; i < 4; ++i) // For each of the four vertices of the tet
  {
    this->Tetra->PointIds->SetId(i, this->PointIds->GetId(i));
  }

  return this->Tetra->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
void svtkQuadraticTetra::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  // Determine how to tessellate. This will depend on the scalars (to try and minimize
  // artifacts).
  double sDiff0 = fabs(cellScalars->GetTuple1(8) - cellScalars->GetTuple1(6));
  double sDiff1 = fabs(cellScalars->GetTuple1(9) - cellScalars->GetTuple1(4));
  double sDiff2 = fabs(cellScalars->GetTuple1(7) - cellScalars->GetTuple1(5));
  int dir = ((sDiff0 < sDiff1 ? (sDiff0 < sDiff2 ? 0 : 2) : (sDiff1 < sDiff2 ? 1 : 2)));

  for (int i = 0; i < 8; i++) // for each subdivided tetra
  {
    for (int j = 0; j < 4; j++) // for each of the four vertices of the tetra
    {
      this->Tetra->Points->SetPoint(j, this->Points->GetPoint(LinearTetras[dir][i][j]));
      this->Tetra->PointIds->SetId(j, this->PointIds->GetId(LinearTetras[dir][i][j]));
      this->Scalars->SetValue(j, cellScalars->GetTuple1(LinearTetras[dir][i][j]));
    }
    this->Tetra->Contour(
      value, this->Scalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  }
}

//----------------------------------------------------------------------------
// Line-line intersection. Intersection has to occur within [0,1] parametric
// coordinates and with specified tolerance.
int svtkQuadraticTetra::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  int intersection = 0;
  double tTemp;
  double pc[3], xTemp[3];
  int faceNum;

  t = SVTK_DOUBLE_MAX;
  for (faceNum = 0; faceNum < 4; faceNum++)
  {
    for (int i = 0; i < 6; i++)
    {
      this->Face->Points->SetPoint(i, this->Points->GetPoint(TetraFaces[faceNum][i]));
    }

    if (this->Face->IntersectWithLine(p1, p2, tol, tTemp, xTemp, pc, subId))
    {
      intersection = 1;
      if (tTemp < t)
      {
        t = tTemp;
        x[0] = xTemp[0];
        x[1] = xTemp[1];
        x[2] = xTemp[2];
        switch (faceNum)
        {
          case 0:
            pcoords[0] = pc[0];
            pcoords[1] = pc[1];
            pcoords[2] = 0.0;
            break;

          case 1:
            pcoords[0] = 0.0;
            pcoords[1] = pc[1];
            pcoords[2] = 0.0;
            break;

          case 2:
            pcoords[0] = pc[0];
            pcoords[1] = 0.0;
            pcoords[2] = 0.0;
            break;

          case 3:
            pcoords[0] = pc[0];
            pcoords[1] = pc[1];
            pcoords[2] = pc[2];
            break;
        }
      }
    }
  }
  return intersection;
}

//----------------------------------------------------------------------------
int svtkQuadraticTetra::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  pts->Reset();
  ptIds->Reset();

  for (int i = 0; i < 8; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      ptIds->InsertId(4 * i + j, this->PointIds->GetId(LinearTetras[0][i][j]));
      pts->InsertPoint(4 * i + j, this->Points->GetPoint(LinearTetras[0][i][j]));
    }
  }

  return 1;
}

//----------------------------------------------------------------------------
// Given parametric coordinates compute inverse Jacobian transformation
// matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
// function derivatives.
void svtkQuadraticTetra::JacobianInverse(
  const double pcoords[3], double** inverse, double derivs[60])
{
  int i, j;
  double *m[3], m0[3], m1[3], m2[3];
  double x[3];

  // compute interpolation function derivatives
  this->InterpolationDerivs(pcoords, derivs);

  // create Jacobian matrix
  m[0] = m0;
  m[1] = m1;
  m[2] = m2;
  for (i = 0; i < 3; i++) // initialize matrix
  {
    m0[i] = m1[i] = m2[i] = 0.0;
  }

  for (j = 0; j < 10; j++)
  {
    this->Points->GetPoint(j, x);
    for (i = 0; i < 3; i++)
    {
      m0[i] += x[i] * derivs[j];
      m1[i] += x[i] * derivs[10 + j];
      m2[i] += x[i] * derivs[20 + j];
    }
  }

  // now find the inverse
  if (svtkMath::InvertMatrix(m, inverse, 3) == 0)
  {
    svtkErrorMacro(<< "Jacobian inverse not found");
    return;
  }
}

//----------------------------------------------------------------------------
void svtkQuadraticTetra::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  double *jI[3], j0[3], j1[3], j2[3];
  double functionDerivs[30], sum[3];
  int i, j, k;

  // compute inverse Jacobian and interpolation function derivatives
  jI[0] = j0;
  jI[1] = j1;
  jI[2] = j2;
  this->JacobianInverse(pcoords, jI, functionDerivs);

  // now compute derivates of values provided
  for (k = 0; k < dim; k++) // loop over values per vertex
  {
    sum[0] = sum[1] = sum[2] = 0.0;
    for (i = 0; i < 10; i++) // loop over interp. function derivatives
    {
      sum[0] += functionDerivs[i] * values[dim * i + k];
      sum[1] += functionDerivs[10 + i] * values[dim * i + k];
      sum[2] += functionDerivs[20 + i] * values[dim * i + k];
    }
    for (j = 0; j < 3; j++) // loop over derivative directions
    {
      derivs[3 * k + j] = sum[0] * jI[j][0] + sum[1] * jI[j][1] + sum[2] * jI[j][2];
    }
  }
}

//----------------------------------------------------------------------------
// Clip this quadratic tetra using the scalar value provided. Like contouring,
// except that it cuts the tetra to produce other tetra.
void svtkQuadraticTetra::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* tetras, svtkPointData* inPd,
  svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  // Determine how to tessellate. This will depend on the scalars (to try and minimize
  // artifacts).
  double sDiff0 = fabs(cellScalars->GetTuple1(8) - cellScalars->GetTuple1(6));
  double sDiff1 = fabs(cellScalars->GetTuple1(9) - cellScalars->GetTuple1(4));
  double sDiff2 = fabs(cellScalars->GetTuple1(7) - cellScalars->GetTuple1(5));
  int dir = ((sDiff0 < sDiff1 ? (sDiff0 < sDiff2 ? 0 : 2) : (sDiff1 < sDiff2 ? 1 : 2)));

  for (int i = 0; i < 8; i++) // for each subdivided tetra
  {
    for (int j = 0; j < 4; j++) // for each of the four vertices of the tetra
    {
      this->Tetra->Points->SetPoint(j, this->Points->GetPoint(LinearTetras[dir][i][j]));
      this->Tetra->PointIds->SetId(j, this->PointIds->GetId(LinearTetras[dir][i][j]));
      this->Scalars->SetValue(j, cellScalars->GetTuple1(LinearTetras[dir][i][j]));
    }
    this->Tetra->Clip(
      value, this->Scalars, locator, tetras, inPd, outPd, inCd, cellId, outCd, insideOut);
  }
}

//----------------------------------------------------------------------------
int svtkQuadraticTetra::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.25;
  return 0;
}

//----------------------------------------------------------------------------
// Compute interpolation functions. First four nodes are the
// tetrahedron corner vertices; the others are mid-edge nodes.
void svtkQuadraticTetra::InterpolationFunctions(const double pcoords[3], double weights[10])
{
  double r = pcoords[0];
  double s = pcoords[1];
  double t = pcoords[2];
  double u = 1.0 - r - s - t;

  // corners
  weights[0] = u * (2.0 * u - 1.0);
  weights[1] = r * (2.0 * r - 1.0);
  weights[2] = s * (2.0 * s - 1.0);
  weights[3] = t * (2.0 * t - 1.0);

  // midedge
  weights[4] = 4.0 * u * r;
  weights[5] = 4.0 * r * s;
  weights[6] = 4.0 * s * u;
  weights[7] = 4.0 * u * t;
  weights[8] = 4.0 * r * t;
  weights[9] = 4.0 * s * t;
}

//----------------------------------------------------------------------------
// Derivatives in parametric space.
void svtkQuadraticTetra::InterpolationDerivs(const double pcoords[3], double derivs[30])
{
  double r = pcoords[0];
  double s = pcoords[1];
  double t = pcoords[2];

  // r-derivatives: dW0/dr to dW9/dr
  derivs[0] = 4.0 * (r + s + t) - 3.0;
  derivs[1] = 4.0 * r - 1.0;
  derivs[2] = 0.0;
  derivs[3] = 0.0;
  derivs[4] = 4.0 - 8.0 * r - 4.0 * s - 4.0 * t;
  derivs[5] = 4.0 * s;
  derivs[6] = -4.0 * s;
  derivs[7] = -4.0 * t;
  derivs[8] = 4.0 * t;
  derivs[9] = 0.0;

  // s-derivatives: dW0/ds to dW9/ds
  derivs[10] = 4.0 * (r + s + t) - 3.0;
  derivs[11] = 0.0;
  derivs[12] = 4.0 * s - 1.0;
  derivs[13] = 0.0;
  derivs[14] = -4.0 * r;
  derivs[15] = 4.0 * r;
  derivs[16] = 4.0 - 4.0 * r - 8.0 * s - 4.0 * t;
  derivs[17] = -4.0 * t;
  derivs[18] = 0.0;
  derivs[19] = 4.0 * t;

  // t-derivatives: dW0/dt to dW9/dt
  derivs[20] = 4.0 * (r + s + t) - 3.0;
  derivs[21] = 0.0;
  derivs[22] = 0.0;
  derivs[23] = 4.0 * t - 1.0;
  derivs[24] = -4.0 * r;
  derivs[25] = 0.0;
  derivs[26] = -4.0 * s;
  derivs[27] = 4.0 - 4.0 * r - 4.0 * s - 8.0 * t;
  derivs[28] = 4.0 * r;
  derivs[29] = 4.0 * s;
}

//----------------------------------------------------------------------------
double svtkQuadraticTetra::GetParametricDistance(const double pcoords[3])
{
  int i;
  double pDist, pDistMax = 0.0;
  double pc[4];

  pc[0] = pcoords[0];
  pc[1] = pcoords[1];
  pc[2] = pcoords[2];
  pc[3] = 1.0 - pcoords[0] - pcoords[1] - pcoords[2];

  for (i = 0; i < 4; i++)
  {
    if (pc[i] < 0.0)
    {
      pDist = -pc[i];
    }
    else if (pc[i] > 1.0)
    {
      pDist = pc[i] - 1.0;
    }
    else // inside the cell in the parametric direction
    {
      pDist = 0.0;
    }
    if (pDist > pDistMax)
    {
      pDistMax = pDist;
    }
  }

  return pDistMax;
}

//----------------------------------------------------------------------------
static double svtkQTetraCellPCoords[30] = {
  0.0, 0.0, 0.0, //
  1.0, 0.0, 0.0, //
  0.0, 1.0, 0.0, //
  0.0, 0.0, 1.0, //
  0.5, 0.0, 0.0, //
  0.5, 0.5, 0.0, //
  0.0, 0.5, 0.0, //
  0.0, 0.0, 0.5, //
  0.5, 0.0, 0.5, //
  0.0, 0.5, 0.5  //
};

double* svtkQuadraticTetra::GetParametricCoords()
{
  return svtkQTetraCellPCoords;
}

//----------------------------------------------------------------------------
void svtkQuadraticTetra::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Edge:\n";
  this->Edge->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Face:\n";
  this->Face->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Tetra:\n";
  this->Tetra->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Scalars:\n";
  this->Scalars->PrintSelf(os, indent.GetNextIndent());
}
