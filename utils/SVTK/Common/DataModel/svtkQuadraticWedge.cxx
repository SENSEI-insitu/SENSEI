/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticWedge.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkQuadraticWedge.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkQuadraticEdge.h"
#include "svtkQuadraticQuad.h"
#include "svtkQuadraticTriangle.h"
#include "svtkWedge.h"

svtkStandardNewMacro(svtkQuadraticWedge);

//----------------------------------------------------------------------------
// Construct the wedge with 15 points + 3 extra points for internal
// computation.
svtkQuadraticWedge::svtkQuadraticWedge()
{
  // At times the cell looks like it has 18 points (during interpolation)
  // We initially allocate for 18.
  this->Points->SetNumberOfPoints(18);
  this->PointIds->SetNumberOfIds(18);
  for (int i = 0; i < 18; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, 0);
  }
  this->Points->SetNumberOfPoints(15);
  this->PointIds->SetNumberOfIds(15);

  this->Edge = svtkQuadraticEdge::New();
  this->Face = svtkQuadraticQuad::New();
  this->TriangleFace = svtkQuadraticTriangle::New();
  this->Wedge = svtkWedge::New();

  this->PointData = svtkPointData::New();
  this->CellData = svtkCellData::New();
  this->CellScalars = svtkDoubleArray::New();
  this->CellScalars->SetNumberOfTuples(18);
  this->Scalars = svtkDoubleArray::New();
  this->Scalars->SetNumberOfTuples(6); // num of vertices
}

//----------------------------------------------------------------------------
svtkQuadraticWedge::~svtkQuadraticWedge()
{
  this->Edge->Delete();
  this->Face->Delete();
  this->TriangleFace->Delete();
  this->Wedge->Delete();

  this->PointData->Delete();
  this->CellData->Delete();
  this->CellScalars->Delete();
  this->Scalars->Delete();
}

//----------------------------------------------------------------------------
// instead of using an hexahedron we could use two prims/wedge...
static int LinearWedges[8][6] = {
  { 0, 8, 6, 12, 17, 15 },
  { 6, 8, 7, 15, 17, 16 },
  { 6, 7, 1, 15, 16, 13 },
  { 8, 2, 7, 17, 14, 16 },
  { 12, 17, 15, 3, 11, 9 },
  { 15, 17, 16, 9, 11, 10 },
  { 15, 16, 13, 9, 10, 4 },
  { 17, 14, 16, 11, 5, 10 },
};

static constexpr svtkIdType WedgeFaces[5][8] = {
  { 0, 1, 2, 6, 7, 8, 0, 0 },
  { 3, 5, 4, 11, 10, 9, 0, 0 },
  { 0, 3, 4, 1, 12, 9, 13, 6 },
  { 1, 4, 5, 2, 13, 10, 14, 7 },
  { 2, 5, 3, 0, 14, 11, 12, 8 },
};

static constexpr svtkIdType WedgeEdges[9][3] = {
  { 0, 1, 6 },
  { 1, 2, 7 },
  { 2, 0, 8 },
  { 3, 4, 9 },
  { 4, 5, 10 },
  { 5, 3, 11 },
  { 0, 3, 12 },
  { 1, 4, 13 },
  { 2, 5, 14 },
};

static double MidPoints[3][3] = {
  { 0.5, 0.0, 0.5 },
  { 0.5, 0.5, 0.5 },
  { 0.0, 0.5, 0.5 },
};
//----------------------------------------------------------------------------
const svtkIdType* svtkQuadraticWedge::GetEdgeArray(svtkIdType edgeId)
{
  return WedgeEdges[edgeId];
}
//----------------------------------------------------------------------------
const svtkIdType* svtkQuadraticWedge::GetFaceArray(svtkIdType faceId)
{
  return WedgeFaces[faceId];
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticWedge::GetEdge(int edgeId)
{
  edgeId = (edgeId < 0 ? 0 : (edgeId > 8 ? 8 : edgeId));

  for (int i = 0; i < 3; i++)
  {
    this->Edge->PointIds->SetId(i, this->PointIds->GetId(WedgeEdges[edgeId][i]));
    this->Edge->Points->SetPoint(i, this->Points->GetPoint(WedgeEdges[edgeId][i]));
  }

  return this->Edge;
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticWedge::GetFace(int faceId)
{
  faceId = (faceId < 0 ? 0 : (faceId > 4 ? 4 : faceId));

  // load point id's and coordinates
  // be careful with the last two:
  if (faceId < 2)
  {
    for (int i = 0; i < 6; i++)
    {
      this->TriangleFace->PointIds->SetId(i, this->PointIds->GetId(WedgeFaces[faceId][i]));
      this->TriangleFace->Points->SetPoint(i, this->Points->GetPoint(WedgeFaces[faceId][i]));
    }
    return this->TriangleFace;
  }
  else
  {
    for (int i = 0; i < 8; i++)
    {
      this->Face->PointIds->SetId(i, this->PointIds->GetId(WedgeFaces[faceId][i]));
      this->Face->Points->SetPoint(i, this->Points->GetPoint(WedgeFaces[faceId][i]));
    }
    return this->Face;
  }
}

//----------------------------------------------------------------------------
static const double SVTK_DIVERGED = 1.e6;
static const int SVTK_WEDGE_MAX_ITERATION = 10;
static const double SVTK_WEDGE_CONVERGED = 1.e-03;

int svtkQuadraticWedge::EvaluatePosition(const double* x, double closestPoint[3], int& subId,
  double pcoords[3], double& dist2, double weights[])
{
  double params[3] = { 0.5, 0.5, 0.5 };
  double derivs[3 * 15];

  // compute a bound on the volume to get a scale for an acceptable determinant
  double longestEdge = 0;
  for (int i = 0; i < 9; i++)
  {
    double pt0[3], pt1[3];
    this->Points->GetPoint(WedgeEdges[i][0], pt0);
    this->Points->GetPoint(WedgeEdges[i][1], pt1);
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
  pcoords[0] = pcoords[1] = pcoords[2] = .5;

  //  enter iteration loop
  int converged = 0;
  for (int iteration = 0; !converged && (iteration < SVTK_WEDGE_MAX_ITERATION); iteration++)
  {
    //  calculate element interpolation functions and derivatives
    this->InterpolationFunctions(pcoords, weights);
    this->InterpolationDerivs(pcoords, derivs);

    //  calculate newton functions
    double fcol[3] = { 0, 0, 0 }, rcol[3] = { 0, 0, 0 }, scol[3] = { 0, 0, 0 },
           tcol[3] = { 0, 0, 0 };
    for (int i = 0; i < 15; i++)
    {
      double pt[3];
      this->Points->GetPoint(i, pt);
      for (int j = 0; j < 3; j++)
      {
        fcol[j] += pt[j] * weights[i];
        rcol[j] += pt[j] * derivs[i];
        scol[j] += pt[j] * derivs[i + 15];
        tcol[j] += pt[j] * derivs[i + 30];
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
      svtkDebugMacro(<< "Determinant incorrect, iteration " << iteration);
      return -1;
    }

    pcoords[0] = params[0] - 0.5 * svtkMath::Determinant3x3(fcol, scol, tcol) / d;
    pcoords[1] = params[1] - 0.5 * svtkMath::Determinant3x3(rcol, fcol, tcol) / d;
    pcoords[2] = params[2] - 0.5 * svtkMath::Determinant3x3(rcol, scol, fcol) / d;

    //  check for convergence
    if (((fabs(pcoords[0] - params[0])) < SVTK_WEDGE_CONVERGED) &&
      ((fabs(pcoords[1] - params[1])) < SVTK_WEDGE_CONVERGED) &&
      ((fabs(pcoords[2] - params[2])) < SVTK_WEDGE_CONVERGED))
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
    pcoords[2] >= -0.001 && pcoords[2] <= 1.001 && pcoords[0] + pcoords[1] <= 1.001)
  {
    if (closestPoint)
    {
      closestPoint[0] = x[0];
      closestPoint[1] = x[1];
      closestPoint[2] = x[2];
      dist2 = 0.0; // inside wedge
    }
    return 1;
  }
  else
  {
    double pc[3], w[15];
    if (closestPoint)
    {
      for (int i = 0; i < 3; i++) // only approximate, not really true for warped hexa
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
void svtkQuadraticWedge::EvaluateLocation(
  int& svtkNotUsed(subId), const double pcoords[3], double x[3], double* weights)
{
  double pt[3];

  this->InterpolationFunctions(pcoords, weights);

  x[0] = x[1] = x[2] = 0.0;
  for (int i = 0; i < 15; i++)
  {
    this->Points->GetPoint(i, pt);
    for (int j = 0; j < 3; j++)
    {
      x[j] += pt[j] * weights[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkQuadraticWedge::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  return this->Wedge->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
void svtkQuadraticWedge::Subdivide(
  svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars)
{
  int numMidPts, i, j;
  double weights[15];
  double x[3];
  double s;

  // Copy point and cell attribute data, first make sure it's empty:
  this->PointData->Initialize();
  this->CellData->Initialize();
  // Make sure to copy ALL arrays. These field data have to be
  // identical to the input field data. Otherwise, CopyData
  // that occurs later may not work because the output field
  // data was initialized (CopyAllocate) with the input field
  // data.
  this->PointData->CopyAllOn();
  this->CellData->CopyAllOn();
  this->PointData->CopyAllocate(inPd, 18);
  this->CellData->CopyAllocate(inCd, 8);
  for (i = 0; i < 15; i++)
  {
    this->PointData->CopyData(inPd, this->PointIds->GetId(i), i);
    this->CellScalars->SetValue(i, cellScalars->GetTuple1(i));
  }
  for (i = 0; i < 8; i++)
  {
    this->CellData->CopyData(inCd, cellId, i);
  }

  // Interpolate new values
  double p[3];
  this->Points->Resize(18);
  this->CellScalars->Resize(18);
  for (numMidPts = 0; numMidPts < 3; numMidPts++)
  {
    this->InterpolationFunctions(MidPoints[numMidPts], weights);

    x[0] = x[1] = x[2] = 0.0;
    s = 0.0;
    for (i = 0; i < 15; i++)
    {
      this->Points->GetPoint(i, p);
      for (j = 0; j < 3; j++)
      {
        x[j] += p[j] * weights[i];
      }
      s += cellScalars->GetTuple1(i) * weights[i];
    }
    this->Points->SetPoint(15 + numMidPts, x);
    this->CellScalars->SetValue(15 + numMidPts, s);
    this->PointData->InterpolatePoint(inPd, 15 + numMidPts, this->PointIds, weights);
  }
}

//----------------------------------------------------------------------------
void svtkQuadraticWedge::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  // subdivide into 8 linear wedges
  this->Subdivide(inPd, inCd, cellId, cellScalars);
  // contour each linear wedge separately
  for (int i = 0; i < 8; i++) // for each wedge
  {
    for (int j = 0; j < 6; j++) // for each point of wedge
    {
      this->Wedge->Points->SetPoint(j, this->Points->GetPoint(LinearWedges[i][j]));
      this->Wedge->PointIds->SetId(j, LinearWedges[i][j]);
      this->Scalars->SetValue(j, this->CellScalars->GetValue(LinearWedges[i][j]));
    }
    this->Wedge->Contour(value, this->Scalars, locator, verts, lines, polys, this->PointData, outPd,
      this->CellData, i, outCd);
  }
}

//----------------------------------------------------------------------------
// Line-hex intersection. Intersection has to occur within [0,1] parametric
// coordinates and with specified tolerance.
int svtkQuadraticWedge::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  int intersection = 0;
  double tTemp;
  double pc[3], xTemp[3];
  int faceNum;
  int inter;

  t = SVTK_DOUBLE_MAX;
  for (faceNum = 0; faceNum < 5; faceNum++)
  {
    // We have 8 nodes on rect face
    // and 6 on triangle faces
    if (faceNum < 2)
    {
      for (int i = 0; i < 6; i++)
      {
        this->TriangleFace->Points->SetPoint(i, this->Points->GetPoint(WedgeFaces[faceNum][i]));
      }
      inter = this->TriangleFace->IntersectWithLine(p1, p2, tol, tTemp, xTemp, pc, subId);
    }
    else
    {
      for (int i = 0; i < 8; i++)
      {
        this->Face->Points->SetPoint(i, this->Points->GetPoint(WedgeFaces[faceNum][i]));
      }
      inter = this->Face->IntersectWithLine(p1, p2, tol, tTemp, xTemp, pc, subId);
    }
    if (inter)
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
            pcoords[0] = 0.0;
            pcoords[1] = pc[1];
            pcoords[2] = pc[0];
            break;

          case 1:
            pcoords[0] = 1.0;
            pcoords[1] = pc[0];
            pcoords[2] = pc[1];
            break;

          case 2:
            pcoords[0] = pc[0];
            pcoords[1] = 0.0;
            pcoords[2] = pc[1];
            break;

          case 3:
            pcoords[0] = pc[1];
            pcoords[1] = 1.0;
            pcoords[2] = pc[0];
            break;

          case 4:
            pcoords[0] = pc[1];
            pcoords[1] = pc[0];
            pcoords[2] = 0.0;
            break;

          case 5:
            pcoords[0] = pc[0];
            pcoords[1] = pc[1];
            pcoords[2] = 1.0;
            break;
        }
      }
    }
  }
  return intersection;
}

//----------------------------------------------------------------------------
int svtkQuadraticWedge::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  // divide up into 16 tets
  pts->SetNumberOfPoints(16 * 4);
  ptIds->SetNumberOfIds(16 * 4);

  svtkIdType ids[16][4] = {
    { 0, 7, 6, 12 },
    { 6, 7, 1, 13 },
    { 9, 6, 7, 12 },
    { 0, 8, 7, 12 },
    { 8, 2, 7, 14 },
    { 10, 11, 3, 12 },
    { 11, 10, 8, 12 },
    { 10, 7, 8, 12 },
    { 9, 10, 3, 12 },
    { 10, 9, 7, 12 },
    { 9, 7, 6, 13 },
    { 9, 10, 7, 13 },
    { 10, 9, 4, 13 },
    { 10, 8, 7, 14 },
    { 5, 11, 10, 14 },
    { 11, 8, 10, 14 },
  };
  svtkIdType counter = 0;
  for (int i = 0; i < 16; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      ptIds->SetId(counter, this->PointIds->GetId(ids[i][j]));
      pts->SetPoint(counter, this->Points->GetPoint(ids[i][j]));
      counter++;
    }
  }

  return 1;
}

//----------------------------------------------------------------------------
// Given parametric coordinates compute inverse Jacobian transformation
// matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
// function derivatives.
void svtkQuadraticWedge::JacobianInverse(
  const double pcoords[3], double** inverse, double derivs[45])
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

  for (j = 0; j < 15; j++)
  {
    this->Points->GetPoint(j, x);
    for (i = 0; i < 3; i++)
    {
      m0[i] += x[i] * derivs[j];
      m1[i] += x[i] * derivs[15 + j];
      m2[i] += x[i] * derivs[30 + j];
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
void svtkQuadraticWedge::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  double *jI[3], j0[3], j1[3], j2[3];
  double functionDerivs[3 * 15], sum[3];
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
    for (i = 0; i < 15; i++) // loop over interp. function derivatives
    {
      sum[0] += functionDerivs[i] * values[dim * i + k];
      sum[1] += functionDerivs[15 + i] * values[dim * i + k];
      sum[2] += functionDerivs[30 + i] * values[dim * i + k];
    }
    for (j = 0; j < 3; j++) // loop over derivative directions
    {
      derivs[3 * k + j] = sum[0] * jI[j][0] + sum[1] * jI[j][1] + sum[2] * jI[j][2];
    }
  }
}

//----------------------------------------------------------------------------
// Clip this quadratic wedge using scalar value provided. Like contouring,
// except that it cuts the wedge to produce tetrahedra.
void svtkQuadraticWedge::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* tets, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  // create eight linear hexes
  this->Subdivide(inPd, inCd, cellId, cellScalars);

  // contour each linear hex separately
  for (int i = 0; i < 8; i++) // for each subdivided wedge
  {
    for (int j = 0; j < 6; j++) // for each of the six vertices of the wedge
    {
      this->Wedge->Points->SetPoint(j, this->Points->GetPoint(LinearWedges[i][j]));
      this->Wedge->PointIds->SetId(j, LinearWedges[i][j]);
      this->Scalars->SetValue(j, this->CellScalars->GetValue(LinearWedges[i][j]));
    }
    this->Wedge->Clip(value, this->Scalars, locator, tets, this->PointData, outPd, this->CellData,
      i, outCd, insideOut);
  }
}

//----------------------------------------------------------------------------
// Compute interpolation functions for the fifteen nodes.
void svtkQuadraticWedge::InterpolationFunctions(const double pcoords[3], double weights[15])
{
  // SVTK needs parametric coordinates to be between (0,1). Isoparametric
  // shape functions are formulated between (-1,1). Here we do a
  // coordinate system conversion from (0,1) to (-1,1).
  double r = pcoords[0];
  double s = pcoords[1];
  double t = pcoords[2];
  // corners
  weights[0] = 2 * (1 - r - s) * (1 - t) * (.5 - r - s - t);
  weights[1] = 2 * r * (1 - t) * (r - t - 0.5);
  weights[2] = 2 * s * (1 - t) * (s - t - 0.5);
  weights[3] = 2 * (1 - r - s) * t * (t - r - s - 0.5);
  weights[4] = 2 * r * t * (r + t - 1.5);
  weights[5] = 2 * s * t * (s + t - 1.5);

  // midsides of triangles
  weights[6] = 4 * r * (1 - r - s) * (1 - t);
  weights[7] = 4 * r * s * (1 - t);
  weights[8] = 4 * (1 - r - s) * s * (1 - t);
  weights[9] = 4 * r * (1 - r - s) * t;
  weights[10] = 4 * r * s * t;
  weights[11] = 4 * (1 - r - s) * s * t;

  // midsides of rectangles
  weights[12] = 4 * t * (1 - r - s) * (1 - t);
  weights[13] = 4 * t * r * (1 - t);
  weights[14] = 4 * t * s * (1 - t);
}

//----------------------------------------------------------------------------
// Derivatives in parametric space.
void svtkQuadraticWedge::InterpolationDerivs(const double pcoords[3], double derivs[45])
{
  // SVTK needs parametric coordinates to be between (0,1). Isoparametric
  // shape functions are formulated between (-1,1). Here we do a
  // coordinate system conversion from (0,1) to (-1,1).
  double r = pcoords[0];
  double s = pcoords[1];
  double t = pcoords[2];
  // r-derivatives
  // corners
  derivs[0] = 2 * (1 - t) * (-1.5 + 2 * r + 2 * s + t);
  derivs[1] = 2 * (1 - t) * (-0.5 + 2 * r - t);
  derivs[2] = 0;
  derivs[3] = 2 * t * (-0.5 + 2 * r + 2 * s - t);
  derivs[4] = 2 * t * (-1.5 + 2 * r + t);
  derivs[5] = 0;
  // midsides of triangles
  derivs[6] = 4 * (1 - t) * (1 - 2 * r - s);
  derivs[7] = 4 * (1 - t) * s;
  derivs[8] = -derivs[7];
  derivs[9] = 4 * t * (1 - 2 * r - s);
  derivs[10] = 4 * s * t;
  derivs[11] = -derivs[10];
  // midsides of rectangles
  derivs[12] = -4 * t * (1 - t);
  derivs[13] = -derivs[12];
  derivs[14] = 0;

  // s-derivatives
  // corners
  derivs[15] = derivs[0];
  derivs[16] = 0;
  derivs[17] = 2 * (1 - t) * (-0.5 + 2 * s - t);
  derivs[18] = derivs[3];
  derivs[19] = 0;
  derivs[20] = 2 * t * (-1.5 + 2 * s + t);
  // midsides of triangles
  derivs[21] = -4 * (1 - t) * r;
  derivs[22] = -derivs[21];
  derivs[23] = 4 * (1 - t) * (1 - r - 2 * s);
  derivs[24] = -4 * r * t;
  derivs[25] = -derivs[24];
  derivs[26] = 4 * t * (1 - r - 2 * s);
  // midsides of rectangles
  derivs[27] = derivs[12];
  derivs[28] = 0;
  derivs[29] = -derivs[27];

  // t-derivatives
  // corners
  derivs[30] = 2 * (1 - r - s) * (-1.5 + r + s + 2 * t);
  derivs[31] = 2 * r * (-0.5 - r + 2 * t);
  derivs[32] = 2 * s * (-0.5 - s + 2 * t);
  derivs[33] = 2 * (1 - r - s) * (-0.5 - r - s + 2 * t);
  derivs[34] = 2 * r * (-1.5 + r + 2 * t);
  derivs[35] = 2 * s * (-1.5 + s + 2 * t);
  // midsides of triangles
  derivs[36] = -4 * r * (1 - r - s);
  derivs[37] = -4 * r * s;
  derivs[38] = -4 * s * (1 - r - s);
  derivs[39] = -derivs[36];
  derivs[40] = -derivs[37];
  derivs[41] = -derivs[38];
  // midsides of rectangles
  derivs[42] = 4 * (1 - 2 * t) * (1 - r - s);
  derivs[43] = 4 * (1 - 2 * t) * r;
  derivs[44] = 4 * (1 - 2 * t) * s;
}

//----------------------------------------------------------------------------
static double svtkQWedgeCellPCoords[45] = {
  0.0, 0.0, 0.0, //
  1.0, 0.0, 0.0, //
  0.0, 1.0, 0.0, //
  0.0, 0.0, 1.0, //
  1.0, 0.0, 1.0, //
  0.0, 1.0, 1.0, //
  0.5, 0.0, 0.0, //
  0.5, 0.5, 0.0, //
  0.0, 0.5, 0.0, //
  0.5, 0.0, 1.0, //
  0.5, 0.5, 1.0, //
  0.0, 0.5, 1.0, //
  0.0, 0.0, 0.5, //
  1.0, 0.0, 0.5, //
  0.0, 1.0, 0.5  //
};
double* svtkQuadraticWedge::GetParametricCoords()
{
  return svtkQWedgeCellPCoords;
}

//----------------------------------------------------------------------------
void svtkQuadraticWedge::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Edge:\n";
  this->Edge->PrintSelf(os, indent.GetNextIndent());
  os << indent << "TriangleFace:\n";
  this->TriangleFace->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Face:\n";
  this->Face->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Wedge:\n";
  this->Wedge->PrintSelf(os, indent.GetNextIndent());
  os << indent << "PointData:\n";
  this->PointData->PrintSelf(os, indent.GetNextIndent());
  os << indent << "CellData:\n";
  this->CellData->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Scalars:\n";
  this->Scalars->PrintSelf(os, indent.GetNextIndent());
}
