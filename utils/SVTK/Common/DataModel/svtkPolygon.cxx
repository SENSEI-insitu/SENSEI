/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolygon.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPolygon.h"

#include "svtkBox.h"
#include "svtkCellArray.h"
#include "svtkDataSet.h"
#include "svtkDoubleArray.h"
#include "svtkIncrementalPointLocator.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkMathUtilities.h"
#include "svtkMergePoints.h"
#include "svtkObjectFactory.h"
#include "svtkPlane.h"
#include "svtkPoints.h"
#include "svtkPriorityQueue.h"
#include "svtkQuad.h"
#include "svtkSmartPointer.h"
#include "svtkTriangle.h"

#include <vector>

svtkStandardNewMacro(svtkPolygon);

//----------------------------------------------------------------------------
// Instantiate polygon.
svtkPolygon::svtkPolygon()
{
  this->Tris = svtkIdList::New();
  this->Tris->Allocate(SVTK_CELL_SIZE);
  this->Triangle = svtkTriangle::New();
  this->Quad = svtkQuad::New();
  this->TriScalars = svtkDoubleArray::New();
  this->TriScalars->Allocate(3);
  this->Line = svtkLine::New();
  this->Tolerance = 0.0;
  this->SuccessfulTriangulation = 0;
  this->Normal[0] = this->Normal[1] = this->Normal[2] = 0.0;
  this->UseMVCInterpolation = false;
}

//----------------------------------------------------------------------------
svtkPolygon::~svtkPolygon()
{
  this->Tris->Delete();
  this->Triangle->Delete();
  this->Quad->Delete();
  this->TriScalars->Delete();
  this->Line->Delete();
}

//----------------------------------------------------------------------------
double svtkPolygon::ComputeArea()
{
  double normal[3]; // not used, but required for the
                    // following ComputeArea call
  return svtkPolygon::ComputeArea(
    this->GetPoints(), this->GetNumberOfPoints(), this->GetPointIds()->GetPointer(0), normal);
}

//----------------------------------------------------------------------------
bool svtkPolygon::IsConvex()
{
  return svtkPolygon::IsConvex(
    this->GetPoints(), this->GetNumberOfPoints(), this->GetPointIds()->GetPointer(0));
}

#define SVTK_POLYGON_FAILURE (-1)
#define SVTK_POLYGON_OUTSIDE 0
#define SVTK_POLYGON_INSIDE 1
#define SVTK_POLYGON_INTERSECTION 2
#define SVTK_POLYGON_ON_LINE 3

//----------------------------------------------------------------------------
//
// In many of the functions that follow, the Points and PointIds members
// of the Cell are assumed initialized.  This is usually done indirectly
// through the GetCell(id) method in the DataSet objects.
//

// Compute the polygon normal from a points list, and a list of point ids
// that index into the points list. Parameter pts can be nullptr, indicating that
// the polygon indexing is {0, 1, ..., numPts-1}. This version will handle
// non-convex polygons.
void svtkPolygon::ComputeNormal(svtkPoints* p, int numPts, const svtkIdType* pts, double* n)
{
  int i;
  double v[3][3], *v0 = v[0], *v1 = v[1], *v2 = v[2], *tmp;
  double ax, ay, az, bx, by, bz;
  //
  // Check for special triangle case. Saves extra work.
  //
  n[0] = n[1] = n[2] = 0.0;
  if (numPts < 3)
  {
    return;
  }

  if (numPts == 3)
  {
    if (pts)
    {
      p->GetPoint(pts[0], v0);
      p->GetPoint(pts[1], v1);
      p->GetPoint(pts[2], v2);
    }
    else
    {
      p->GetPoint(0, v0);
      p->GetPoint(1, v1);
      p->GetPoint(2, v2);
    }
    svtkTriangle::ComputeNormal(v0, v1, v2, n);
    return;
  }

  //  Because polygon may be concave, need to accumulate cross products to
  //  determine true normal.
  //

  // set things up for loop
  if (pts)
  {
    p->GetPoint(pts[0], v1);
    p->GetPoint(pts[1], v2);
  }
  else
  {
    p->GetPoint(0, v1);
    p->GetPoint(1, v2);
  }

  for (i = 0; i < numPts; i++)
  {
    tmp = v0;
    v0 = v1;
    v1 = v2;
    v2 = tmp;

    if (pts)
    {
      p->GetPoint(pts[(i + 2) % numPts], v2);
    }
    else
    {
      p->GetPoint((i + 2) % numPts, v2);
    }

    // order is important!!! to maintain consistency with polygon vertex order
    ax = v2[0] - v1[0];
    ay = v2[1] - v1[1];
    az = v2[2] - v1[2];
    bx = v0[0] - v1[0];
    by = v0[1] - v1[1];
    bz = v0[2] - v1[2];

    n[0] += (ay * bz - az * by);
    n[1] += (az * bx - ax * bz);
    n[2] += (ax * by - ay * bx);
  }

  svtkMath::Normalize(n);
}

//----------------------------------------------------------------------------
// Compute the polygon normal from a points list, and a list of point ids
// that index into the points list. This version will handle non-convex
// polygons.
void svtkPolygon::ComputeNormal(svtkIdTypeArray* ids, svtkPoints* p, double n[3])
{
  return svtkPolygon::ComputeNormal(p, ids->GetNumberOfTuples(), ids->GetPointer(0), n);
}

//----------------------------------------------------------------------------
// Compute the polygon normal from a list of doubleing points. This version
// will handle non-convex polygons.
void svtkPolygon::ComputeNormal(svtkPoints* p, double* n)
{
  return svtkPolygon::ComputeNormal(p, p->GetNumberOfPoints(), nullptr, n);
}

//----------------------------------------------------------------------------
// Compute the polygon normal from an array of points. This version assumes
// that the polygon is convex, and looks for the first valid normal.
void svtkPolygon::ComputeNormal(int numPts, double* pts, double n[3])
{
  int i;
  double *v1, *v2, *v3;
  double length;
  double ax, ay, az;
  double bx, by, bz;

  //  Because some polygon vertices are colinear, need to make sure
  //  first non-zero normal is found.
  //
  v1 = pts;
  v2 = pts + 3;
  v3 = pts + 6;

  for (i = 0; i < numPts - 2; i++)
  {
    ax = v2[0] - v1[0];
    ay = v2[1] - v1[1];
    az = v2[2] - v1[2];
    bx = v3[0] - v1[0];
    by = v3[1] - v1[1];
    bz = v3[2] - v1[2];

    n[0] = (ay * bz - az * by);
    n[1] = (az * bx - ax * bz);
    n[2] = (ax * by - ay * bx);

    length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (length != 0.0)
    {
      n[0] /= length;
      n[1] /= length;
      n[2] /= length;
      return;
    }
    else
    {
      v1 = v2;
      v2 = v3;
      v3 += 3;
    }
  } // over all points
}

//----------------------------------------------------------------------------
// Determine whether or not a polygon is convex from a points list and a list
// of point ids that index into the points list. Parameter pts can be nullptr,
// indicating that the polygon indexing is {0, 1, ..., numPts-1}.
bool svtkPolygon::IsConvex(svtkPoints* p, int numPts, svtkIdType* pts)
{
  int i;
  double v[3][3], *v0 = v[0], *v1 = v[1], *v2 = v[2], *tmp, a[3], aMag, b[3], bMag;
  double n[3] = { 0., 0., 0. }, ni[3] = { 0., 0., 0. };
  bool nComputed = false;

  if (numPts < 3)
  {
    return false;
  }

  if (numPts == 3)
  {
    return true;
  }

  if (pts)
  {
    p->GetPoint(pts[0], v1);
    p->GetPoint(pts[1], v2);
  }
  else
  {
    p->GetPoint(0, v1);
    p->GetPoint(1, v2);
  }

  for (i = 0; i <= numPts; i++)
  {
    tmp = v0;
    v0 = v1;
    v1 = v2;
    v2 = tmp;

    if (pts)
    {
      p->GetPoint(pts[(i + 2) % numPts], v2);
    }
    else
    {
      p->GetPoint((i + 2) % numPts, v2);
    }

    // order is important!!! to maintain consistency with polygon vertex order
    a[0] = v2[0] - v1[0];
    a[1] = v2[1] - v1[1];
    a[2] = v2[2] - v1[2];
    b[0] = v0[0] - v1[0];
    b[1] = v0[1] - v1[1];
    b[2] = v0[2] - v1[2];

    if (!nComputed)
    {
      aMag = svtkMath::Norm(a);
      bMag = svtkMath::Norm(b);
      if (aMag > SVTK_DBL_EPSILON && bMag > SVTK_DBL_EPSILON)
      {
        svtkMath::Cross(a, b, n);
        nComputed = svtkMath::Norm(n) > SVTK_DBL_EPSILON * (aMag < bMag ? bMag : aMag);
      }
      continue;
    }

    svtkMath::Cross(a, b, ni);
    if (svtkMath::Norm(ni) > SVTK_DBL_EPSILON && svtkMath::Dot(n, ni) < 0.)
    {
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
bool svtkPolygon::IsConvex(svtkIdTypeArray* ids, svtkPoints* p)
{
  return svtkPolygon::IsConvex(p, ids->GetNumberOfTuples(), ids->GetPointer(0));
}

//----------------------------------------------------------------------------
bool svtkPolygon::IsConvex(svtkPoints* p)
{
  return svtkPolygon::IsConvex(p, p->GetNumberOfPoints(), nullptr);
}

//----------------------------------------------------------------------------
int svtkPolygon::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& minDist2, double weights[])
{
  int i;
  double p0[3], p10[3], l10, p20[3], l20, n[3], cp[3];
  double ray[3];

  subId = 0;
  this->ParameterizePolygon(p0, p10, l10, p20, l20, n);
  this->InterpolateFunctions(x, weights);
  svtkPlane::ProjectPoint(x, p0, n, cp);

  for (i = 0; i < 3; i++)
  {
    ray[i] = cp[i] - p0[i];
  }
  pcoords[0] = svtkMath::Dot(ray, p10) / (l10 * l10);
  pcoords[1] = svtkMath::Dot(ray, p20) / (l20 * l20);
  pcoords[2] = 0.0;

  if (pcoords[0] >= 0.0 && pcoords[0] <= 1.0 && pcoords[1] >= 0.0 && pcoords[1] <= 1.0 &&
    (this->PointInPolygon(cp, this->Points->GetNumberOfPoints(),
       static_cast<svtkDoubleArray*>(this->Points->GetData())->GetPointer(0), this->GetBounds(),
       n) == SVTK_POLYGON_INSIDE))
  {
    if (closestPoint)
    {
      closestPoint[0] = cp[0];
      closestPoint[1] = cp[1];
      closestPoint[2] = cp[2];
      minDist2 = svtkMath::Distance2BetweenPoints(x, closestPoint);
    }
    return 1;
  }

  // If here, point is outside of polygon, so need to find distance to boundary
  //
  else
  {
    double t, dist2;
    int numPts;
    double closest[3];
    double pt1[3], pt2[3];

    if (closestPoint)
    {
      numPts = this->Points->GetNumberOfPoints();
      for (minDist2 = SVTK_DOUBLE_MAX, i = 0; i < numPts; i++)
      {
        this->Points->GetPoint(i, pt1);
        this->Points->GetPoint((i + 1) % numPts, pt2);
        dist2 = svtkLine::DistanceToLine(x, pt1, pt2, t, closest);
        if (dist2 < minDist2)
        {
          closestPoint[0] = closest[0];
          closestPoint[1] = closest[1];
          closestPoint[2] = closest[2];
          minDist2 = dist2;
        }
      }
    }
    return 0;
  }
}

//----------------------------------------------------------------------------
void svtkPolygon::EvaluateLocation(
  int& svtkNotUsed(subId), const double pcoords[3], double x[3], double* weights)
{
  int i;
  double p0[3], p10[3], l10, p20[3], l20, n[3];

  this->ParameterizePolygon(p0, p10, l10, p20, l20, n);
  for (i = 0; i < 3; i++)
  {
    x[i] = p0[i] + pcoords[0] * p10[i] + pcoords[1] * p20[i];
  }

  this->InterpolateFunctions(x, weights);
}

//----------------------------------------------------------------------------
// Compute interpolation weights using 1/r**2 normalized sum or mean value
// coordinate.
void svtkPolygon::InterpolateFunctions(const double x[3], double* weights)
{
  // Compute interpolation weights using mean value coordinate.
  if (this->UseMVCInterpolation)
  {
    this->InterpolateFunctionsUsingMVC(x, weights);
    return;
  }

  // Compute interpolation weights using 1/r**2 normalized sum.
  int i;
  int numPts = this->Points->GetNumberOfPoints();
  double sum, pt[3];

  for (sum = 0.0, i = 0; i < numPts; i++)
  {
    this->Points->GetPoint(i, pt);
    weights[i] = svtkMath::Distance2BetweenPoints(x, pt);
    if (weights[i] == 0.0) // exact hit
    {
      for (int j = 0; j < numPts; j++)
      {
        weights[j] = 0.0;
      }
      weights[i] = 1.0;
      return;
    }
    else
    {
      weights[i] = 1.0 / weights[i];
      sum += weights[i];
    }
  }

  for (i = 0; i < numPts; i++)
  {
    weights[i] /= sum;
  }
}

//----------------------------------------------------------------------------
// Compute interpolation weights using mean value coordinate.
void svtkPolygon::InterpolateFunctionsUsingMVC(const double x[3], double* weights)
{
  int numPts = this->Points->GetNumberOfPoints();

  // Begin by initializing weights.
  for (int i = 0; i < numPts; i++)
  {
    weights[i] = static_cast<double>(0.0);
  }

  // create local array for storing point-to-vertex vectors and distances
  std::vector<double> dist(numPts);
  std::vector<double> uVec(3 * numPts);
  static const double eps = 0.00000001;
  for (int i = 0; i < numPts; i++)
  {
    double pt[3];
    this->Points->GetPoint(i, pt);

    // point-to-vertex vector
    uVec[3 * i] = pt[0] - x[0];
    uVec[3 * i + 1] = pt[1] - x[1];
    uVec[3 * i + 2] = pt[2] - x[2];

    // distance
    dist[i] = svtkMath::Norm(uVec.data() + 3 * i);

    // handle special case when the point is really close to a vertex
    if (dist[i] < eps)
    {
      weights[i] = 1.0;
      return;
    }

    uVec[3 * i] /= dist[i];
    uVec[3 * i + 1] /= dist[i];
    uVec[3 * i + 2] /= dist[i];
  }

  // Now loop over all vertices to compute weight
  // w_i = ( tan(theta_i/2) + tan(theta_(i+1)/2) ) / dist_i
  // To do consider the simplification of
  // tan(alpha/2) = (1-cos(alpha))/sin(alpha)
  //              = (d0*d1 - cross(u0, u1))/(2*dot(u0,u1))
  std::vector<double> tanHalfTheta(numPts);
  for (int i = 0; i < numPts; i++)
  {
    int i1 = i + 1;
    if (i1 == numPts)
    {
      i1 = 0;
    }

    double* u0 = uVec.data() + 3 * i;
    double* u1 = uVec.data() + 3 * i1;

    double l = sqrt(svtkMath::Distance2BetweenPoints(u0, u1));
    double theta = 2.0 * asin(l / 2.0);

    // special case where x lies on an edge
    if (svtkMath::Pi() - theta < 0.001)
    {
      weights[i] = dist[i1] / (dist[i] + dist[i1]);
      weights[i1] = 1 - weights[i];
      return;
    }

    tanHalfTheta[i] = tan(theta / 2.0);
  }

  // Normal case
  for (int i = 0; i < numPts; i++)
  {
    int i1 = i - 1;
    if (i1 == -1)
    {
      i1 = numPts - 1;
    }

    weights[i] = (tanHalfTheta[i] + tanHalfTheta[i1]) / dist[i];
  }

  // normalize weight
  double sum = 0.0;
  for (int i = 0; i < numPts; i++)
  {
    sum += weights[i];
  }

  if (fabs(sum) < eps)
  {
    return;
  }

  for (int i = 0; i < numPts; i++)
  {
    weights[i] /= sum;
  }
}

//----------------------------------------------------------------------------
// Create a local s-t coordinate system for a polygon. The point p0 is
// the origin of the local system, p10 is s-axis vector, and p20 is the
// t-axis vector. (These are expressed in the modelling coordinate system and
// are vectors of dimension [3].) The values l20 and l20 are the lengths of
// the vectors p10 and p20, and n is the polygon normal.
int svtkPolygon::ParameterizePolygon(
  double* p0, double* p10, double& l10, double* p20, double& l20, double* n)
{
  int i, j;
  double s, t, p[3], p1[3], p2[3], sbounds[2], tbounds[2];
  int numPts = this->Points->GetNumberOfPoints();
  double x1[3], x2[3];

  if (numPts < 3)
  {
    return 0;
  }

  //  This is a two pass process: first create a p' coordinate system
  //  that is then adjusted to insure that the polygon points are all in
  //  the range 0<=s,t<=1.  The p' system is defined by the polygon normal,
  //  first vertex and the first edge.
  //
  this->ComputeNormal(this->Points, n);
  this->Points->GetPoint(0, x1);
  this->Points->GetPoint(1, x2);
  for (i = 0; i < 3; i++)
  {
    p0[i] = x1[i];
    p10[i] = x2[i] - x1[i];
  }
  svtkMath::Cross(n, p10, p20);

  // Determine lengths of edges
  //
  if ((l10 = svtkMath::Dot(p10, p10)) == 0.0 || (l20 = svtkMath::Dot(p20, p20)) == 0.0)
  {
    return 0;
  }

  //  Now evaluate all polygon points to determine min/max parametric
  //  coordinate values.
  //
  // first vertex has (s,t) = (0,0)
  sbounds[0] = 0.0;
  sbounds[1] = 0.0;
  tbounds[0] = 0.0;
  tbounds[1] = 0.0;

  for (i = 1; i < numPts; i++)
  {
    this->Points->GetPoint(i, x1);
    for (j = 0; j < 3; j++)
    {
      p[j] = x1[j] - p0[j];
    }
    s = (p[0] * p10[0] + p[1] * p10[1] + p[2] * p10[2]) / l10;
    t = (p[0] * p20[0] + p[1] * p20[1] + p[2] * p20[2]) / l20;
    sbounds[0] = (s < sbounds[0] ? s : sbounds[0]);
    sbounds[1] = (s > sbounds[1] ? s : sbounds[1]);
    tbounds[0] = (t < tbounds[0] ? t : tbounds[0]);
    tbounds[1] = (t > tbounds[1] ? t : tbounds[1]);
  }

  //  Re-evaluate coordinate system
  //
  for (i = 0; i < 3; i++)
  {
    p1[i] = p0[i] + sbounds[1] * p10[i] + tbounds[0] * p20[i];
    p2[i] = p0[i] + sbounds[0] * p10[i] + tbounds[1] * p20[i];
    p0[i] = p0[i] + sbounds[0] * p10[i] + tbounds[0] * p20[i];
    p10[i] = p1[i] - p0[i];
    p20[i] = p2[i] - p0[i];
  }
  l10 = svtkMath::Norm(p10);
  l20 = svtkMath::Norm(p20);

  return 1;
}

#define SVTK_POLYGON_CERTAIN 1
#define SVTK_POLYGON_UNCERTAIN 0
#define SVTK_POLYGON_RAY_TOL 1.e-03 // Tolerance for ray firing
#define SVTK_POLYGON_MAX_ITER 10    // Maximum iterations for ray-firing
#define SVTK_POLYGON_VOTE_THRESHOLD 2

//----------------------------------------------------------------------------
// Determine whether point is inside polygon. Function uses ray-casting
// to determine if point is inside polygon. Works for arbitrary polygon shape
// (e.g., non-convex). Returns 0 if point is not in polygon; 1 if it is.
// Can also return -1 to indicate degenerate polygon. Note: a point in
// bounding box check is NOT performed prior to in/out check. You may want
// to do this to improve performance.
int svtkPolygon::PointInPolygon(double x[3], int numPts, double* pts, double bounds[6], double* n)
{
  double *x1, *x2, xray[3], u, v;
  double rayMag, mag = 1, ray[3];
  int testResult, status, numInts, i;
  int iterNumber;
  int maxComp, comps[2];
  int deltaVotes;
  // do a quick bounds check
  if (x[0] < bounds[0] || x[0] > bounds[1] || x[1] < bounds[2] || x[1] > bounds[3] ||
    x[2] < bounds[4] || x[2] > bounds[5])
  {
    return SVTK_POLYGON_OUTSIDE;
  }

  //
  //  Define a ray to fire.  The ray is a random ray normal to the
  //  normal of the face.  The length of the ray is a function of the
  //  size of the face bounding box.
  //
  for (i = 0; i < 3; i++)
  {
    ray[i] = (bounds[2 * i + 1] - bounds[2 * i]) * 1.1 +
      fabs((bounds[2 * i + 1] + bounds[2 * i]) / 2.0 - x[i]);
  }

  if ((rayMag = svtkMath::Norm(ray)) == 0.0)
  {
    return SVTK_POLYGON_OUTSIDE;
  }

  //  Get the maximum component of the normal.
  //
  if (fabs(n[0]) > fabs(n[1]))
  {
    if (fabs(n[0]) > fabs(n[2]))
    {
      maxComp = 0;
      comps[0] = 1;
      comps[1] = 2;
    }
    else
    {
      maxComp = 2;
      comps[0] = 0;
      comps[1] = 1;
    }
  }
  else
  {
    if (fabs(n[1]) > fabs(n[2]))
    {
      maxComp = 1;
      comps[0] = 0;
      comps[1] = 2;
    }
    else
    {
      maxComp = 2;
      comps[0] = 0;
      comps[1] = 1;
    }
  }

  //  Check that max component is non-zero
  //
  if (n[maxComp] == 0.0)
  {
    return SVTK_POLYGON_FAILURE;
  }

  //  Enough information has been acquired to determine the random ray.
  //  Random rays are generated until one is satisfactory (i.e.,
  //  produces a ray of non-zero magnitude).  Also, since more than one
  //  ray may need to be fired, the ray-firing occurs in a large loop.
  //
  //  The variable iterNumber counts the number of iterations and is
  //  limited by the defined variable SVTK_POLYGON_MAX_ITER.
  //
  //  The variable deltaVotes keeps track of the number of votes for
  //  "in" versus "out" of the face.  When delta_vote > 0, more votes
  //  have counted for "in" than "out".  When delta_vote < 0, more votes
  //  have counted for "out" than "in".  When the delta_vote exceeds or
  //  equals the defined variable SVTK_POLYGON_VOTE_THRESHOLD, than the
  //  appropriate "in" or "out" status is returned.
  //
  for (deltaVotes = 0, iterNumber = 1;
       (iterNumber < SVTK_POLYGON_MAX_ITER) && (abs(deltaVotes) < SVTK_POLYGON_VOTE_THRESHOLD);
       iterNumber++)
  {
    //
    //  Generate ray
    //
    bool rayOK;
    for (rayOK = false; rayOK == false;)
    {
      ray[comps[0]] = svtkMath::Random(-rayMag, rayMag);
      ray[comps[1]] = svtkMath::Random(-rayMag, rayMag);
      ray[maxComp] = -(n[comps[0]] * ray[comps[0]] + n[comps[1]] * ray[comps[1]]) / n[maxComp];
      if ((mag = svtkMath::Norm(ray)) > rayMag * SVTK_TOL)
      {
        rayOK = true;
      }
    }

    //  The ray must be appropriately sized.
    //
    for (i = 0; i < 3; i++)
    {
      xray[i] = x[i] + (rayMag / mag) * ray[i];
    }

    //  The ray may now be fired against all the edges
    //
    for (numInts = 0, testResult = SVTK_POLYGON_CERTAIN, i = 0; i < numPts; i++)
    {
      x1 = pts + 3 * i;
      x2 = pts + 3 * ((i + 1) % numPts);

      //   Fire the ray and compute the number of intersections.  Be careful
      //   of degenerate cases (e.g., ray intersects at vertex).
      //

      if ((status = svtkLine::Intersection(x, xray, x1, x2, u, v)) == SVTK_POLYGON_INTERSECTION)
      {
        // This test checks for vertex and edge intersections
        // For example
        //  Vertex intersection
        //    (u=0 v=0), (u=0 v=1), (u=1 v=0), (u=1 v=0)
        //  Edge intersection
        //    (u=0 v!=0 v!=1), (u=1 v!=0 v!=1)
        //    (u!=0 u!=1 v=0), (u!=0 u!=1 v=1)
        if ((SVTK_POLYGON_RAY_TOL < u) && (u < 1.0 - SVTK_POLYGON_RAY_TOL) &&
          (SVTK_POLYGON_RAY_TOL < v) && (v < 1.0 - SVTK_POLYGON_RAY_TOL))
        {
          numInts++;
        }
        else
        {
          testResult = SVTK_POLYGON_UNCERTAIN;
        }
      }
      else if (status == SVTK_POLYGON_ON_LINE)
      {
        testResult = SVTK_POLYGON_UNCERTAIN;
      }
    }
    if (testResult == SVTK_POLYGON_CERTAIN)
    {
      if (numInts % 2 == 0)
      {
        --deltaVotes;
      }
      else
      {
        ++deltaVotes;
      }
    }
  } // try another ray

  //   If the number of intersections is odd, the point is in the polygon.
  //
  if (deltaVotes < 0)
  {
    return SVTK_POLYGON_OUTSIDE;
  }
  else
  {
    return SVTK_POLYGON_INSIDE;
  }
}

#define SVTK_POLYGON_TOLERANCE 1.0e-06

//----------------------------------------------------------------------------
// Triangulate polygon.
//
int svtkPolygon::Triangulate(svtkIdList* outTris)
{
  const double* bounds = this->GetBounds();

  double d = sqrt((bounds[1] - bounds[0]) * (bounds[1] - bounds[0]) +
    (bounds[3] - bounds[2]) * (bounds[3] - bounds[2]) +
    (bounds[5] - bounds[4]) * (bounds[5] - bounds[4]));
  this->Tolerance = SVTK_POLYGON_TOLERANCE * d;
  this->SuccessfulTriangulation = 1;

  this->Tris->Reset();
  int success = this->EarCutTriangulation();

  if (!success) // degenerate triangle encountered
  {
    svtkDebugMacro(<< "Degenerate polygon encountered during triangulation");
  }

  outTris->DeepCopy(this->Tris);
  return success;
}

//----------------------------------------------------------------------------
// Split into non-degenerate polygons prior to triangulation
//
int svtkPolygon::NonDegenerateTriangulate(svtkIdList* outTris)
{
  double pt[3], bounds[6];
  svtkIdType ptId, numPts;

  // ComputeBounds does not give the correct bounds
  // So we do it manually
  bounds[0] = SVTK_DOUBLE_MAX;
  bounds[1] = -SVTK_DOUBLE_MAX;
  bounds[2] = SVTK_DOUBLE_MAX;
  bounds[3] = -SVTK_DOUBLE_MAX;
  bounds[4] = SVTK_DOUBLE_MAX;
  bounds[5] = -SVTK_DOUBLE_MAX;

  numPts = this->GetNumberOfPoints();

  for (int i = 0; i < numPts; i++)
  {
    this->Points->GetPoint(i, pt);

    if (pt[0] < bounds[0])
    {
      bounds[0] = pt[0];
    }
    if (pt[1] < bounds[2])
    {
      bounds[2] = pt[1];
    }
    if (pt[2] < bounds[4])
    {
      bounds[4] = pt[2];
    }
    if (pt[0] > bounds[1])
    {
      bounds[1] = pt[0];
    }
    if (pt[1] > bounds[3])
    {
      bounds[3] = pt[1];
    }
    if (pt[2] > bounds[5])
    {
      bounds[5] = pt[2];
    }
  }

  outTris->Reset();
  outTris->Allocate(3 * (2 * numPts - 4));

  svtkPoints* newPts = svtkPoints::New();
  newPts->Allocate(numPts);

  svtkMergePoints* mergePoints = svtkMergePoints::New();
  mergePoints->InitPointInsertion(newPts, bounds);
  mergePoints->SetDivisions(10, 10, 10);

  svtkIdTypeArray* matchingIds = svtkIdTypeArray::New();
  matchingIds->SetNumberOfTuples(numPts);

  int numDuplicatePts = 0;

  for (int i = 0; i < numPts; i++)
  {
    this->Points->GetPoint(i, pt);
    if (mergePoints->InsertUniquePoint(pt, ptId))
    {
      matchingIds->SetValue(i, ptId + numDuplicatePts);
    }
    else
    {
      matchingIds->SetValue(i, ptId + numDuplicatePts);
      numDuplicatePts++;
    }
  }

  mergePoints->Delete();
  newPts->Delete();

  int numPtsRemoved = 0;
  svtkIdType tri[3];

  while (numPtsRemoved < numPts)
  {
    svtkIdType start = 0;
    svtkIdType end = numPts - 1;

    for (; start < numPts; start++)
    {
      if (matchingIds->GetValue(start) >= 0)
      {
        break;
      }
    }

    if (start >= end)
    {
      svtkErrorMacro("ERROR: start >= end");
      break;
    }

    for (int i = start; i < numPts; i++)
    {
      if (matchingIds->GetValue(i) < 0)
      {
        continue;
      }

      if (matchingIds->GetValue(i) != i)
      {
        start = (matchingIds->GetValue(i) + 1) % numPts;
        end = i;

        while (matchingIds->GetValue(start) < 0)
        {
          start++;
        }

        break;
      }
    }

    svtkPolygon* polygon = svtkPolygon::New();
    polygon->Points->SetDataTypeToDouble();

    int numPolygonPts = start < end ? end - start + 1 : end - start + numPts + 1;

    for (int i = 0; i < numPolygonPts; i++)
    {
      ptId = (start + i) % numPts;

      if (matchingIds->GetValue(ptId) >= 0)
      {
        numPtsRemoved++;
        matchingIds->SetValue(ptId, -1);

        polygon->PointIds->InsertNextId(ptId);
        polygon->Points->InsertNextPoint(this->Points->GetPoint(ptId));
      }
    }

    svtkIdList* outTriangles = svtkIdList::New();
    outTriangles->Allocate(3 * (2 * polygon->GetNumberOfPoints() - 4));

    polygon->Triangulate(outTriangles);

    int outNumTris = outTriangles->GetNumberOfIds();

    for (int i = 0; i < outNumTris; i += 3)
    {
      tri[0] = outTriangles->GetId(i);
      tri[1] = outTriangles->GetId(i + 1);
      tri[2] = outTriangles->GetId(i + 2);

      tri[0] = polygon->PointIds->GetId(tri[0]);
      tri[1] = polygon->PointIds->GetId(tri[1]);
      tri[2] = polygon->PointIds->GetId(tri[2]);

      outTris->InsertNextId(tri[0]);
      outTris->InsertNextId(tri[1]);
      outTris->InsertNextId(tri[2]);
    }

    polygon->Delete();
    outTriangles->Delete();
  }

  matchingIds->Delete();
  return 1;
}

//----------------------------------------------------------------------------
// Triangulate polygon and enforce that the ratio of the smallest triangle area
// to the polygon area is greater than a user-defined tolerance.
int svtkPolygon::BoundedTriangulate(svtkIdList* outTris, double tolerance)
{
  int i, j, k, success = 0, numPts = this->PointIds->GetNumberOfIds();
  double totalArea, area_static[SVTK_CELL_SIZE], *area;
  double p[3][3];

  // For most polygons, there should be fewer than SVTK_CELL_SIZE points. In
  // the event that we have a huge polygon, dynamically allocate an
  // appropriately sized array.
  std::vector<double> area_dynamic;
  if (numPts - 2 <= SVTK_CELL_SIZE)
  {
    area = &area_static[0];
  }
  else
  {
    area_dynamic.resize(numPts - 2);
    area = area_dynamic.data();
  }

  for (i = 0; i < numPts; i++)
  {
    this->Tris->Reset();

    success = this->UnbiasedEarCutTriangulation(i);

    if (!success)
    {
      continue;
    }

    totalArea = 0.;
    for (j = 0; j < numPts - 2; j++)
    {
      for (k = 0; k < 3; k++)
      {
        this->Points->GetPoint(this->Tris->GetId(3 * j + k), p[k]);
      }
      area[j] = svtkTriangle::TriangleArea(p[0], p[1], p[2]);
      totalArea += area[j];
    }

    for (j = 0; j < numPts - 2; j++)
    {
      if (area[j] / totalArea < tolerance)
      {
        success = 0;
        break;
      }
    }

    if (success == 1)
    {
      break;
    }
  }

  outTris->DeepCopy(this->Tris);

  return success;
}

//----------------------------------------------------------------------------
// Special structures for building loops. This is a double-linked list.
typedef struct _svtkPolyVertex
{
  int id;
  double x[3];
  double measure;
  _svtkPolyVertex* next;
  _svtkPolyVertex* previous;
} svtkLocalPolyVertex;

class svtkPolyVertexList
{ // structure to support triangulation
public:
  svtkPolyVertexList(svtkIdList* ptIds, svtkPoints* pts, double tol2);
  ~svtkPolyVertexList();

  int ComputeNormal();
  double ComputeMeasure(svtkLocalPolyVertex* vtx);
  void RemoveVertex(svtkLocalPolyVertex* vtx, svtkIdList* ids, svtkPriorityQueue* queue = nullptr);
  void RemoveVertex(int i, svtkIdList* ids, svtkPriorityQueue* queue = nullptr);
  int CanRemoveVertex(svtkLocalPolyVertex* vtx, double tol);
  int CanRemoveVertex(int id, double tol);

  int NumberOfVerts;
  svtkLocalPolyVertex* Array;
  svtkLocalPolyVertex* Head;
  double Normal[3];
};

//----------------------------------------------------------------------------
// tolerance is squared
svtkPolyVertexList::svtkPolyVertexList(svtkIdList* ptIds, svtkPoints* pts, double tol2)
{
  int numVerts = ptIds->GetNumberOfIds();
  this->NumberOfVerts = numVerts;
  this->Array = new svtkLocalPolyVertex[numVerts];
  int i;

  // now load the data into the array
  double x[3];
  for (i = 0; i < numVerts; i++)
  {
    this->Array[i].id = i;
    pts->GetPoint(i, x);
    this->Array[i].x[0] = x[0];
    this->Array[i].x[1] = x[1];
    this->Array[i].x[2] = x[2];
    this->Array[i].next = this->Array + (i + 1) % numVerts;
    if (i == 0)
    {
      this->Array[i].previous = this->Array + numVerts - 1;
    }
    else
    {
      this->Array[i].previous = this->Array + i - 1;
    }
  }

  // Make sure that there are no coincident vertices.
  // Beware of multiple coincident vertices.
  svtkLocalPolyVertex *vtx, *next;
  this->Head = this->Array;

  for (vtx = this->Head, i = 0; i < numVerts; i++)
  {
    next = vtx->next;
    if (svtkMath::Distance2BetweenPoints(vtx->x, next->x) < tol2)
    {
      next->next->previous = vtx;
      vtx->next = next->next;
      if (next == this->Head)
      {
        this->Head = vtx;
      }
      this->NumberOfVerts--;
    }
    else // can move forward
    {
      vtx = next;
    }
  }
}

//----------------------------------------------------------------------------
svtkPolyVertexList::~svtkPolyVertexList()
{
  delete[] this->Array;
}

//----------------------------------------------------------------------------
// Remove the vertex from the polygon (forming a triangle with
// its previous and next neighbors, and reinsert the neighbors
// into the priority queue.
void svtkPolyVertexList::RemoveVertex(
  svtkLocalPolyVertex* vtx, svtkIdList* tris, svtkPriorityQueue* queue)
{
  // Create triangle
  tris->InsertNextId(vtx->id);
  tris->InsertNextId(vtx->next->id);
  tris->InsertNextId(vtx->previous->id);

  // remove vertex; special case if single triangle left
  if (--this->NumberOfVerts < 3)
  {
    return;
  }
  if (vtx == this->Head)
  {
    this->Head = vtx->next;
  }
  vtx->previous->next = vtx->next;
  vtx->next->previous = vtx->previous;

  // recompute measure, reinsert into queue
  // note that id may have been previously deleted (with Pop()) if we
  // are dealing with a concave polygon and vertex couldn't be split.
  if (queue)
  {
    queue->DeleteId(vtx->previous->id);
    queue->DeleteId(vtx->next->id);
    if (this->ComputeMeasure(vtx->previous) > 0.0)
    {
      queue->Insert(vtx->previous->measure, vtx->previous->id);
    }
    if (this->ComputeMeasure(vtx->next) > 0.0)
    {
      queue->Insert(vtx->next->measure, vtx->next->id);
    }
  }
}

//----------------------------------------------------------------------------
// Remove the vertex from the polygon (forming a triangle with
// its previous and next neighbors, and reinsert the neighbors
// into the priority queue.
void svtkPolyVertexList::RemoveVertex(int i, svtkIdList* tris, svtkPriorityQueue* queue)
{
  return this->RemoveVertex(this->Array + i, tris, queue);
}

//----------------------------------------------------------------------------
int svtkPolyVertexList::ComputeNormal()
{
  svtkLocalPolyVertex* vtx = this->Head;
  double v1[3], v2[3], n[3], *anchor = vtx->x;

  this->Normal[0] = this->Normal[1] = this->Normal[2] = 0.0;
  for (vtx = vtx->next; vtx->next != this->Head; vtx = vtx->next)
  {
    v1[0] = vtx->x[0] - anchor[0];
    v1[1] = vtx->x[1] - anchor[1];
    v1[2] = vtx->x[2] - anchor[2];
    v2[0] = vtx->next->x[0] - anchor[0];
    v2[1] = vtx->next->x[1] - anchor[1];
    v2[2] = vtx->next->x[2] - anchor[2];
    svtkMath::Cross(v1, v2, n);
    this->Normal[0] += n[0];
    this->Normal[1] += n[1];
    this->Normal[2] += n[2];
  }
  if (svtkMath::Normalize(this->Normal) == 0.0)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

//----------------------------------------------------------------------------
// The measure is the ratio of triangle perimeter^2 to area;
// the sign of the measure is determined by dotting the local
// vector with the normal (concave features return a negative
// measure).
double svtkPolyVertexList::ComputeMeasure(svtkLocalPolyVertex* vtx)
{
  double v1[3], v2[3], v3[3], v4[3], area, perimeter;

  for (int i = 0; i < 3; i++)
  {
    v1[i] = vtx->x[i] - vtx->previous->x[i];
    v2[i] = vtx->next->x[i] - vtx->x[i];
    v3[i] = vtx->previous->x[i] - vtx->next->x[i];
  }
  svtkMath::Cross(v1, v2, v4); //|v4| is twice the area
  if ((area = svtkMath::Dot(v4, this->Normal)) < 0.0)
  {
    return (vtx->measure = -1.0); // concave or bad triangle
  }
  else if (area == 0.0)
  {
    return (vtx->measure = -SVTK_DOUBLE_MAX); // concave or bad triangle
  }
  else
  {
    perimeter = svtkMath::Norm(v1) + svtkMath::Norm(v2) + svtkMath::Norm(v3);
    return (vtx->measure = perimeter * perimeter / area);
  }
}

//----------------------------------------------------------------------------
// returns != 0 if vertex can be removed. Uses half-space
// comparison to determine whether ear-cut is valid, and may
// resort to line-plane intersections to resolve possible
// instersections with ear-cut.
int svtkPolyVertexList::CanRemoveVertex(svtkLocalPolyVertex* currentVtx, double tolerance)
{
  int i, sign, currentSign;
  double v[3], sN[3], *sPt, val, s, t;
  svtkLocalPolyVertex *previous, *next, *vtx;

  // Check for simple case
  if (this->NumberOfVerts <= 3)
  {
    return 1;
  }

  // Compute split plane, the point to be cut off
  // is always on the positive side of the plane.
  previous = currentVtx->previous;
  next = currentVtx->next;

  sPt = previous->x; // point on plane
  for (i = 0; i < 3; i++)
  {
    v[i] = next->x[i] - previous->x[i]; // vector passing through point
  }

  svtkMath::Cross(v, this->Normal, sN);
  if ((svtkMath::Normalize(sN)) == 0.0)
  {
    return 0; // bad split, indeterminant
  }

  // Traverse the other points to see if a) they are all on the
  // other side of the plane; and if not b) whether they intersect
  // the split line.
  int oneNegative = 0;
  val = svtkPlane::Evaluate(sN, sPt, next->next->x);
  currentSign = (val > tolerance ? 1 : (val < -tolerance ? -1 : 0));
  oneNegative = (currentSign < 0 ? 1 : 0); // very important

  // Intersections are only computed when the split half-space is crossed
  for (vtx = next->next->next; vtx != previous; vtx = vtx->next)
  {
    val = svtkPlane::Evaluate(sN, sPt, vtx->x);
    sign = (val > tolerance ? 1 : (val < -tolerance ? -1 : 0));
    if (sign != currentSign)
    {
      if (!oneNegative)
      {
        oneNegative = (sign < 0 ? 1 : 0); // very important
      }
      if (svtkLine::Intersection(sPt, next->x, vtx->x, vtx->previous->x, s, t) != 0)
      {
        return 0;
      }
      else
      {
        currentSign = sign;
      }
    } // if crossing occurs
  }   // for the rest of the loop

  if (!oneNegative)
  {
    return 0; // entire loop is on this side of plane
  }
  else
  {
    return 1;
  }
}

//----------------------------------------------------------------------------
// returns != 0 if vertex can be removed. Uses half-space
// comparison to determine whether ear-cut is valid, and may
// resort to line-plane intersections to resolve possible
// instersections with ear-cut.
int svtkPolyVertexList::CanRemoveVertex(int id, double tolerance)
{
  return this->CanRemoveVertex(this->Array + id, tolerance);
}

//----------------------------------------------------------------------------
// Triangulation method based on ear-cutting. Triangles, or ears, are
// cut off from the polygon based on the angle of the vertex. Small
// angles (narrow triangles) are cut off first. This implementation uses
// a priority queue to cut off ears with smallest angles. Also, the
// algorithm works in 3D (the points don't have to be projected into
// 2D, and the ordering direction of the points is nor important as
// long as the polygon edges do not self intersect).
int svtkPolygon::EarCutTriangulation()
{
  svtkPolyVertexList poly(this->PointIds, this->Points, this->Tolerance * this->Tolerance);
  svtkLocalPolyVertex* vtx;
  int i, id;

  // First compute the polygon normal the correct way
  //
  if (!poly.ComputeNormal())
  {
    return (this->SuccessfulTriangulation = 0);
  }

  // Now compute the angles between edges incident to each
  // vertex. Place the structure into a priority queue (those
  // vertices with smallest angle are to be removed first).
  //
  svtkPriorityQueue* VertexQueue = svtkPriorityQueue::New();
  VertexQueue->Allocate(poly.NumberOfVerts);
  for (i = 0, vtx = poly.Head; i < poly.NumberOfVerts; i++, vtx = vtx->next)
  {
    // concave (negative measure) vertices are not eligible for removal
    if (poly.ComputeMeasure(vtx) > 0.0)
    {
      VertexQueue->Insert(vtx->measure, vtx->id);
    }
  }

  // For each vertex in priority queue, and as long as there
  // are three or more vertices, remove the vertex (if possible)
  // and create a new triangle. If the number of vertices in the
  // queue is equal to the number of vertices, then the polygon
  // is convex and triangle removal can proceed without intersection
  // checks.
  //
  int numInQueue;
  while (poly.NumberOfVerts > 2 && (numInQueue = VertexQueue->GetNumberOfItems()) > 0)
  {
    if (numInQueue == poly.NumberOfVerts) // convex, pop away
    {
      id = VertexQueue->Pop();
      poly.RemoveVertex(id, this->Tris, VertexQueue);
    } // convex
    else
    {
      id = VertexQueue->Pop(); // removes it, even if can't be split
      if (poly.CanRemoveVertex(id, this->Tolerance))
      {
        poly.RemoveVertex(id, this->Tris, VertexQueue);
      }
    } // concave
  }   // while

  // Clean up
  VertexQueue->Delete();

  if (poly.NumberOfVerts > 2) // couldn't triangulate
  {
    return (this->SuccessfulTriangulation = 0);
  }
  return (this->SuccessfulTriangulation = 1);
}

//----------------------------------------------------------------------------
// Triangulation method based on ear-cutting. Triangles, or ears, are
// cut off from the polygon. This implementation does not bias the
// selection of ears; it sequentially progresses through each vertex
// starting at a user-defined seed value.
int svtkPolygon::UnbiasedEarCutTriangulation(int seed)
{
  svtkPolyVertexList poly(this->PointIds, this->Points, this->Tolerance * this->Tolerance);

  // First compute the polygon normal the correct way
  //
  if (!poly.ComputeNormal())
  {
    return (this->SuccessfulTriangulation = 0);
  }

  seed = abs(seed) % poly.NumberOfVerts;
  svtkLocalPolyVertex* vtx = poly.Array + seed;

  int marker = -1;

  while (poly.NumberOfVerts > 2)
  {
    if (poly.CanRemoveVertex(vtx, this->Tolerance))
    {
      poly.RemoveVertex(vtx, this->Tris);
    }
    vtx = vtx->next;

    if (vtx == poly.Head)
    {
      if (poly.NumberOfVerts == marker)
      {
        break;
      }
      marker = poly.NumberOfVerts;
    }
  }

  if (poly.NumberOfVerts > 2) // couldn't triangulate
  {
    return (this->SuccessfulTriangulation = 0);
  }
  return (this->SuccessfulTriangulation = 1);
}

//----------------------------------------------------------------------------
int svtkPolygon::CellBoundary(int svtkNotUsed(subId), const double pcoords[3], svtkIdList* pts)
{
  int i, numPts = this->PointIds->GetNumberOfIds();
  double x[3];
  int closestPoint = 0, previousPoint, nextPoint;
  double largestWeight = 0.0;
  double p0[3], p10[3], l10, p20[3], l20, n[3];

  pts->Reset();
  std::vector<double> weights(numPts);

  // determine global coordinates given parametric coordinates
  this->ParameterizePolygon(p0, p10, l10, p20, l20, n);
  for (i = 0; i < 3; i++)
  {
    x[i] = p0[i] + pcoords[0] * p10[i] + pcoords[1] * p20[i];
  }

  // find edge with largest and next largest weight values. This will be
  // the closest edge.
  this->InterpolateFunctions(x, weights.data());
  for (i = 0; i < numPts; i++)
  {
    if (weights[i] > largestWeight)
    {
      closestPoint = i;
      largestWeight = weights[i];
    }
  }

  pts->InsertId(0, this->PointIds->GetId(closestPoint));

  previousPoint = closestPoint - 1;
  nextPoint = closestPoint + 1;
  if (previousPoint < 0)
  {
    previousPoint = numPts - 1;
  }
  if (nextPoint >= numPts)
  {
    nextPoint = 0;
  }

  if (weights[previousPoint] > weights[nextPoint])
  {
    pts->InsertId(1, this->PointIds->GetId(previousPoint));
  }
  else
  {
    pts->InsertId(1, this->PointIds->GetId(nextPoint));
  }

  // determine whether point is inside of polygon
  if (pcoords[0] >= 0.0 && pcoords[0] <= 1.0 && pcoords[1] >= 0.0 && pcoords[1] <= 1.0 &&
    (this->PointInPolygon(x, this->Points->GetNumberOfPoints(),
       static_cast<svtkDoubleArray*>(this->Points->GetData())->GetPointer(0), this->GetBounds(),
       n) == SVTK_POLYGON_INSIDE))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

//----------------------------------------------------------------------------
void svtkPolygon::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  int i, success;
  int p1, p2, p3;

  this->TriScalars->SetNumberOfTuples(3);

  const double* bounds = this->GetBounds();

  double d = sqrt((bounds[1] - bounds[0]) * (bounds[1] - bounds[0]) +
    (bounds[3] - bounds[2]) * (bounds[3] - bounds[2]) +
    (bounds[5] - bounds[4]) * (bounds[5] - bounds[4]));
  this->Tolerance = SVTK_POLYGON_TOLERANCE * d;
  this->SuccessfulTriangulation = 1;
  this->ComputeNormal(this->Points, this->Normal);

  this->Tris->Reset();

  success = this->EarCutTriangulation();

  if (!success) // Just skip for now.
  {
  }
  else // Contour triangle
  {
    for (i = 0; i < this->Tris->GetNumberOfIds(); i += 3)
    {
      p1 = this->Tris->GetId(i);
      p2 = this->Tris->GetId(i + 1);
      p3 = this->Tris->GetId(i + 2);

      this->Triangle->Points->SetPoint(0, this->Points->GetPoint(p1));
      this->Triangle->Points->SetPoint(1, this->Points->GetPoint(p2));
      this->Triangle->Points->SetPoint(2, this->Points->GetPoint(p3));

      if (outPd)
      {
        this->Triangle->PointIds->SetId(0, this->PointIds->GetId(p1));
        this->Triangle->PointIds->SetId(1, this->PointIds->GetId(p2));
        this->Triangle->PointIds->SetId(2, this->PointIds->GetId(p3));
      }

      this->TriScalars->SetTuple(0, cellScalars->GetTuple(p1));
      this->TriScalars->SetTuple(1, cellScalars->GetTuple(p2));
      this->TriScalars->SetTuple(2, cellScalars->GetTuple(p3));

      this->Triangle->Contour(
        value, this->TriScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
    }
  }
}

//----------------------------------------------------------------------------
svtkCell* svtkPolygon::GetEdge(int edgeId)
{
  int numPts = this->Points->GetNumberOfPoints();

  // load point id's
  this->Line->PointIds->SetId(0, this->PointIds->GetId(edgeId));
  this->Line->PointIds->SetId(1, this->PointIds->GetId((edgeId + 1) % numPts));

  // load coordinates
  this->Line->Points->SetPoint(0, this->Points->GetPoint(edgeId));
  this->Line->Points->SetPoint(1, this->Points->GetPoint((edgeId + 1) % numPts));

  return this->Line;
}

//----------------------------------------------------------------------------
//
// Intersect this plane with finite line defined by p1 & p2 with tolerance tol.
//
int svtkPolygon::IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
  double x[3], double pcoords[3], int& subId)
{
  double pt1[3], n[3];
  double tol2 = tol * tol;
  double closestPoint[3];
  double dist2;
  int npts = this->GetNumberOfPoints();

  subId = 0;
  pcoords[0] = pcoords[1] = pcoords[2] = 0.0;

  // Define a plane to intersect with
  //
  this->Points->GetPoint(1, pt1);
  this->ComputeNormal(this->Points, n);

  // Intersect plane of the polygon with line
  //
  if (!svtkPlane::IntersectWithLine(p1, p2, n, pt1, t, x))
  {
    return 0;
  }

  // Evaluate position
  //
  std::vector<double> weights(npts);
  if (this->EvaluatePosition(x, closestPoint, subId, pcoords, dist2, weights.data()) >= 0)
  {
    if (dist2 <= tol2)
    {
      return 1;
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
int svtkPolygon::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  int i, success;
  double *bounds, d;

  pts->Reset();
  ptIds->Reset();

  bounds = this->GetBounds();
  d = sqrt((bounds[1] - bounds[0]) * (bounds[1] - bounds[0]) +
    (bounds[3] - bounds[2]) * (bounds[3] - bounds[2]) +
    (bounds[5] - bounds[4]) * (bounds[5] - bounds[4]));
  this->Tolerance = SVTK_POLYGON_TOLERANCE * d;
  this->SuccessfulTriangulation = 1;
  this->ComputeNormal(this->Points, this->Normal);

  this->Tris->Reset();

  success = this->EarCutTriangulation();

  if (!success) // Indicate possible failure
  {
    svtkDebugMacro(<< "Possible triangulation failure");
  }
  for (i = 0; i < this->Tris->GetNumberOfIds(); i++)
  {
    ptIds->InsertId(i, this->PointIds->GetId(this->Tris->GetId(i)));
    pts->InsertPoint(i, this->Points->GetPoint(this->Tris->GetId(i)));
  }

  return this->SuccessfulTriangulation;
}

//----------------------------------------------------------------------------
// Samples at three points to compute derivatives in local r-s coordinate
// system and projects vectors into 3D model coordinate system.
// Note that the results are usually inaccurate because
// this method actually returns the derivative of the interpolation
// function which is obtained using 1/r**2 normalized sum.
#define SVTK_SAMPLE_DISTANCE 0.01
void svtkPolygon::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  int i, j, k, idx;

  if (this->Points->GetNumberOfPoints() == 4)
  {
    for (i = 0; i < 4; i++)
    {
      this->Quad->Points->SetPoint(i, this->Points->GetPoint(i));
    }
    this->Quad->Derivatives(0, pcoords, values, dim, derivs);
    return;
  }
  else if (this->Points->GetNumberOfPoints() == 3)
  {
    for (i = 0; i < 3; i++)
    {
      this->Triangle->Points->SetPoint(i, this->Points->GetPoint(i));
    }
    this->Triangle->Derivatives(0, pcoords, values, dim, derivs);
    return;
  }

  double p0[3], p10[3], l10, p20[3], l20, n[3];
  double x[3][3], l1, l2, v1[3], v2[3];

  // setup parametric system and check for degeneracy
  if (this->ParameterizePolygon(p0, p10, l10, p20, l20, n) == 0)
  {
    for (j = 0; j < dim; j++)
    {
      for (i = 0; i < 3; i++)
      {
        derivs[j * dim + i] = 0.0;
      }
    }
    return;
  }

  int numVerts = this->PointIds->GetNumberOfIds();
  std::vector<double> weights(numVerts);
  std::vector<double> sample(dim * 3);

  // compute positions of three sample points
  for (i = 0; i < 3; i++)
  {
    x[0][i] = p0[i] + pcoords[0] * p10[i] + pcoords[1] * p20[i];
    x[1][i] = p0[i] + (pcoords[0] + SVTK_SAMPLE_DISTANCE) * p10[i] + pcoords[1] * p20[i];
    x[2][i] = p0[i] + pcoords[0] * p10[i] + (pcoords[1] + SVTK_SAMPLE_DISTANCE) * p20[i];
  }

  // for each sample point, sample data values
  for (idx = 0, k = 0; k < 3; k++) // loop over three sample points
  {
    this->InterpolateFunctions(x[k], weights.data());
    for (j = 0; j < dim; j++, idx++) // over number of derivates requested
    {
      sample[idx] = 0.0;
      for (i = 0; i < numVerts; i++)
      {
        sample[idx] += weights[i] * values[j + i * dim];
      }
    }
  }

  // compute differences along the two axes
  for (i = 0; i < 3; i++)
  {
    v1[i] = x[1][i] - x[0][i];
    v2[i] = x[2][i] - x[0][i];
  }
  l1 = svtkMath::Normalize(v1);
  l2 = svtkMath::Normalize(v2);

  // compute derivatives along x-y-z axes
  double ddx, ddy;
  for (j = 0; j < dim; j++)
  {
    ddx = (sample[dim + j] - sample[j]) / l1;
    ddy = (sample[2 * dim + j] - sample[j]) / l2;

    // project onto global x-y-z axes
    derivs[3 * j] = ddx * v1[0] + ddy * v2[0];
    derivs[3 * j + 1] = ddx * v1[1] + ddy * v2[1];
    derivs[3 * j + 2] = ddx * v1[2] + ddy * v2[2];
  }
}

//----------------------------------------------------------------------------
void svtkPolygon::Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
  svtkCellArray* tris, svtkPointData* inPD, svtkPointData* outPD, svtkCellData* inCD, svtkIdType cellId,
  svtkCellData* outCD, int insideOut)
{
  int i, success;
  int p1, p2, p3;

  this->TriScalars->SetNumberOfTuples(3);

  const double* bounds = this->GetBounds();
  double d = sqrt((bounds[1] - bounds[0]) * (bounds[1] - bounds[0]) +
    (bounds[3] - bounds[2]) * (bounds[3] - bounds[2]) +
    (bounds[5] - bounds[4]) * (bounds[5] - bounds[4]));
  this->Tolerance = SVTK_POLYGON_TOLERANCE * d;

  this->SuccessfulTriangulation = 1;
  this->ComputeNormal(this->Points, this->Normal);

  this->Tris->Reset();

  success = this->EarCutTriangulation();

  if (success) // clip triangles
  {
    for (i = 0; i < this->Tris->GetNumberOfIds(); i += 3)
    {
      p1 = this->Tris->GetId(i);
      p2 = this->Tris->GetId(i + 1);
      p3 = this->Tris->GetId(i + 2);

      this->Triangle->Points->SetPoint(0, this->Points->GetPoint(p1));
      this->Triangle->Points->SetPoint(1, this->Points->GetPoint(p2));
      this->Triangle->Points->SetPoint(2, this->Points->GetPoint(p3));

      this->Triangle->PointIds->SetId(0, this->PointIds->GetId(p1));
      this->Triangle->PointIds->SetId(1, this->PointIds->GetId(p2));
      this->Triangle->PointIds->SetId(2, this->PointIds->GetId(p3));

      this->TriScalars->SetTuple(0, cellScalars->GetTuple(p1));
      this->TriScalars->SetTuple(1, cellScalars->GetTuple(p2));
      this->TriScalars->SetTuple(2, cellScalars->GetTuple(p3));

      this->Triangle->Clip(
        value, this->TriScalars, locator, tris, inPD, outPD, inCD, cellId, outCD, insideOut);
    }
  }
}

//----------------------------------------------------------------------------
// Method intersects two polygons. You must supply the number of points and
// point coordinates (npts, *pts) and the bounding box (bounds) of the two
// polygons. Also supply a tolerance squared for controlling
// error. The method returns 1 if there is an intersection, and 0 if
// not. A single point of intersection x[3] is also returned if there
// is an intersection.
int svtkPolygon::IntersectPolygonWithPolygon(int npts, double* pts, double bounds[6], int npts2,
  double* pts2, double bounds2[6], double tol2, double x[3])
{
  double n[3], coords[3];
  int i, j;
  double *p1, *p2, ray[3];
  double t;

  //  Intersect each edge of first polygon against second
  //
  svtkPolygon::ComputeNormal(npts2, pts2, n);

  for (i = 0; i < npts; i++)
  {
    p1 = pts + 3 * i;
    p2 = pts + 3 * ((i + 1) % npts);

    for (j = 0; j < 3; j++)
    {
      ray[j] = p2[j] - p1[j];
    }
    if (!svtkBox::IntersectBox(bounds2, p1, ray, coords, t))
    {
      continue;
    }

    if ((svtkPlane::IntersectWithLine(p1, p2, n, pts2, t, x)) == 1)
    {
      if ((npts2 == 3 && svtkTriangle::PointInTriangle(x, pts2, pts2 + 3, pts2 + 6, tol2)) ||
        (npts2 > 3 && svtkPolygon::PointInPolygon(x, npts2, pts2, bounds2, n) == SVTK_POLYGON_INSIDE))
      {
        return 1;
      }
    }
    else
    {
      return 0;
    }
  }

  //  Intersect each edge of second polygon against first
  //
  svtkPolygon::ComputeNormal(npts, pts, n);

  for (i = 0; i < npts2; i++)
  {
    p1 = pts2 + 3 * i;
    p2 = pts2 + 3 * ((i + 1) % npts2);

    for (j = 0; j < 3; j++)
    {
      ray[j] = p2[j] - p1[j];
    }

    if (!svtkBox::IntersectBox(bounds, p1, ray, coords, t))
    {
      continue;
    }

    if ((svtkPlane::IntersectWithLine(p1, p2, n, pts, t, x)) == 1)
    {
      if ((npts == 3 && svtkTriangle::PointInTriangle(x, pts, pts + 3, pts + 6, tol2)) ||
        (npts > 3 && svtkPolygon::PointInPolygon(x, npts, pts, bounds, n) == SVTK_POLYGON_INSIDE))
      {
        return 1;
      }
    }
    else
    {
      return 0;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
// Compute the area of the polygon (oriented in 3D space). It uses an
// efficient approach where the area is computed in 2D and then projected into
// 3D space.
double svtkPolygon::ComputeArea(svtkPoints* p, svtkIdType numPts, const svtkIdType* pts, double n[3])
{
  if (numPts < 3)
  {
    return 0.0;
  }
  else
  {
    double area = 0.0;
    double nx, ny, nz;
    int coord, i;

    svtkPolygon::ComputeNormal(p, numPts, pts, n);

    // Select the projection direction
    nx = (n[0] > 0.0 ? n[0] : -n[0]); // abs x-coord
    ny = (n[1] > 0.0 ? n[1] : -n[1]); // abs y-coord
    nz = (n[2] > 0.0 ? n[2] : -n[2]); // abs z-coord

    coord = (nx > ny ? (nx > nz ? 0 : 2) : (ny > nz ? 1 : 2));

    // compute area of the 2D projection
    double x0[3], x1[3], x2[3], *v0, *v1, *v2;
    v0 = x0;
    v1 = x1;
    v2 = x2;

    for (i = 0; i < numPts; i++)
    {
      if (pts)
      {
        p->GetPoint(pts[i], v0);
        p->GetPoint(pts[(i + 1) % numPts], v1);
        p->GetPoint(pts[(i + 2) % numPts], v2);
      }
      else
      {
        p->GetPoint(i, v0);
        p->GetPoint((i + 1) % numPts, v1);
        p->GetPoint((i + 2) % numPts, v2);
      }
      switch (coord)
      {
        case 0:
          area += v1[1] * (v2[2] - v0[2]);
          continue;
        case 1:
          area += v1[0] * (v2[2] - v0[2]);
          continue;
        case 2:
          area += v1[0] * (v2[1] - v0[1]);
          continue;
      }
    }

    // scale to get area before projection
    switch (coord)
    {
      case 0:
        area /= (2.0 * nx);
        break;
      case 1:
        area /= (2.0 * ny);
        break;
      case 2:
        area /= (2.0 * nz);
    }
    return fabs(area);
  } // general polygon
}

//----------------------------------------------------------------------------
void svtkPolygon::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Tolerance: " << this->Tolerance << "\n";
  os << indent << "SuccessfulTriangulation: " << this->SuccessfulTriangulation << "\n";
  os << indent << "UseMVCInterpolation: " << this->UseMVCInterpolation << "\n";
  os << indent << "Normal: (" << this->Normal[0] << ", " << this->Normal[1] << ", "
     << this->Normal[2] << ")\n";
  os << indent << "Tris:\n";
  this->Tris->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Triangle:\n";
  this->Triangle->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Quad:\n";
  this->Quad->PrintSelf(os, indent.GetNextIndent());
  os << indent << "TriScalars:\n";
  this->TriScalars->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Line:\n";
  this->Line->PrintSelf(os, indent.GetNextIndent());
}

//----------------------------------------------------------------------------
// Compute the polygon centroid from a points list, the number of points, and an
// array of point ids that index into the points list. Returns false if the
// computation is invalid.
bool svtkPolygon::ComputeCentroid(svtkPoints* p, int numPts, const svtkIdType* ids, double c[3])
{
  // Strategy:
  // - Compute centroid of projected polygon on (x,y) if polygon is projectible, (x,z) otherwise
  // - Accumulate signed projected area as well, which is needed in the centroid's formula
  // - Infer 3rd dimension using polygon's normal.
  svtkIdType i;
  double p0[6];
  double normal[3] = { 0.0 };
  if (numPts < 2)
  {
    return false;
  }
  // Handle to the doubled area of the projected polygon on (x,y) or (x,z) if the polygon
  // projected on (x,y) is degenerate.
  double a = 0.0;
  p->GetPoint(ids[0], p0);

  svtkPolygon::ComputeNormal(p, numPts, ids, normal);
  svtkIdType xOffset = 0, yOffset = 0;
  // Checking if the polygon is colinear with z axis.
  // If it is, the centroid computation axis shall be shifted
  // because the projected polygon on (x,y) is degenerate.

  {
    constexpr double z[3] = { 0.0, 0.0, 1.0 };
    svtkMath::Cross(normal, z, c);
    // If the normal is orthogonal with z axis, the projected polygon is then a line...
    if (std::fabs(c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - 1.0) <= SVTK_DBL_EPSILON)
    {
      yOffset = 1;
      constexpr double y[3] = { 0.0, 1.0, 0.0 };
      svtkMath::Cross(normal, y, c);
      // If the normal is orthogonal with y axis, the projected polygon is then a line...
      if (std::fabs(c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - 1.0) <= SVTK_DBL_EPSILON)
      {
        xOffset = 1;
      }
    }
  }

  c[0] = c[1] = c[2] = 0.0;

  for (i = 0; i < numPts; i++)
  {
    p->GetPoint(ids[(i + 1) % numPts], p0 + 3 * !(i % 2));
    double det = (p0[3 * (i % 2) + xOffset] * p0[3 * !(i % 2) + 1 + yOffset] -
      p0[3 * !(i % 2) + xOffset] * p0[3 * (i % 2) + 1 + yOffset]);
    c[xOffset] += (p0[xOffset] + p0[3 + xOffset]) * det;
    c[1 + yOffset] += (p0[1 + yOffset] + p0[4 + yOffset]) * det;
    a += det;
  }
  if (std::abs(a) < SVTK_DBL_MIN)
  {
    // Polygon is degenerate
    return false;
  }
  c[xOffset] /= 3.0 * a;
  c[1 + yOffset] /= 3.0 * a;
  c[2 - xOffset - yOffset] = 1.0 / normal[2 - xOffset - yOffset] *
    (-normal[xOffset] * c[xOffset] - normal[1 + yOffset] * c[1 + yOffset] +
      svtkMath::Dot(normal, p0));
  return true;
}

//----------------------------------------------------------------------------
// Compute the polygon centroid from a points list and a list of point ids
// that index into the points list. Returns false if the computation is invalid.
bool svtkPolygon::ComputeCentroid(svtkIdTypeArray* ids, svtkPoints* p, double c[3])
{
  return svtkPolygon::ComputeCentroid(p, ids->GetNumberOfTuples(), ids->GetPointer(0), c);
}

//----------------------------------------------------------------------------
double svtkPolygon::DistanceToPolygon(
  double x[3], int numPts, double* pts, double bounds[6], double closest[3])
{
  // First check to see if the point is inside the polygon
  // do a quick bounds check
  if (x[0] >= bounds[0] && x[0] <= bounds[1] && x[1] >= bounds[2] && x[1] <= bounds[3] &&
    x[2] >= bounds[4] && x[2] <= bounds[5])
  {
    double n[3];
    svtkPolygon::ComputeNormal(numPts, pts, n);
    if (svtkPolygon::PointInPolygon(x, numPts, pts, bounds, n))
    {
      closest[0] = x[0];
      closest[1] = x[1];
      closest[2] = x[2];
      return 0.0;
    }
  }

  // Not inside, compute the distance of the point to the edges.
  double minDist2 = SVTK_FLOAT_MAX;
  double *p0, *p1, dist2, t, c[3];
  for (int i = 0; i < numPts; i++)
  {
    p0 = pts + 3 * i;
    p1 = pts + 3 * ((i + 1) % numPts);
    dist2 = svtkLine::DistanceToLine(x, p0, p1, t, c);
    if (dist2 < minDist2)
    {
      minDist2 = dist2;
      closest[0] = c[0];
      closest[1] = c[1];
      closest[2] = c[2];
    }
  }

  return sqrt(minDist2);
}

//----------------------------------------------------------------------------
int svtkPolygon::IntersectConvex2DCells(
  svtkCell* cell1, svtkCell* cell2, double tol, double p0[3], double p1[3])
{
  // Intersect the six total edges of the two triangles against each other. Two points are
  // all that are required.
  double *x[2], pcoords[3], t, x0[3], x1[3];
  x[0] = p0;
  x[1] = p1;
  int subId, idx = 0;
  double t2 = tol * tol;

  // Loop over edges of second polygon and intersect against first polygon
  svtkIdType i, numPts = cell2->Points->GetNumberOfPoints();
  for (i = 0; i < numPts; i++)
  {
    cell2->Points->GetPoint(i, x0);
    cell2->Points->GetPoint((i + 1) % numPts, x1);

    if (cell1->IntersectWithLine(x0, x1, tol, t, x[idx], pcoords, subId))
    {
      if (idx == 0)
      {
        idx++;
      }
      else if (((x[1][0] - x[0][0]) * (x[1][0] - x[0][0]) +
                 (x[1][1] - x[0][1]) * (x[1][1] - x[0][1]) +
                 (x[1][2] - x[0][2]) * (x[1][2] - x[0][2])) > t2)
      {
        return 2;
      }
    } // if edge intersection
  }   // over all edges

  // Loop over edges of first polygon and intersect against second polygon
  numPts = cell1->Points->GetNumberOfPoints();
  for (i = 0; i < numPts; i++)
  {
    cell1->Points->GetPoint(i, x0);
    cell1->Points->GetPoint((i + 1) % numPts, x1);

    if (cell2->IntersectWithLine(x0, x1, tol, t, x[idx], pcoords, subId))
    {
      if (idx == 0)
      {
        idx++;
      }
      else if (((x[1][0] - x[0][0]) * (x[1][0] - x[0][0]) +
                 (x[1][1] - x[0][1]) * (x[1][1] - x[0][1]) +
                 (x[1][2] - x[0][2]) * (x[1][2] - x[0][2])) > t2)
      {
        return 2;
      }
    } // if edge intersection
  }   // over all edges

  // Evaluate what we got
  if (idx == 1)
  {
    return 1; // everything intersecting at single point
  }
  else
  {
    return 0;
  }
}
