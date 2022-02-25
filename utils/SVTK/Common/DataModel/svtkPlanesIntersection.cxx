/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPlanesIntersection.cxx

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

#include "svtkPlanesIntersection.h"
#include "svtkCell.h"
#include "svtkFloatArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointsProjectedHull.h"

svtkStandardNewMacro(svtkPlanesIntersection);

// Experiment shows that we get plane equation values on the
//  order of 10e-6 when the point is actually on the plane

#define SVTK_SMALL_DOUBLE (10e-5)

const int Inside = 0;
const int Outside = 1;
const int Straddle = 2;

const int Xdim = 0; // don't change these three values
const int Ydim = 1;
const int Zdim = 2;

svtkPlanesIntersection::svtkPlanesIntersection()
{
  this->Planes = nullptr;
  this->RegionPts = nullptr;
}
svtkPlanesIntersection::~svtkPlanesIntersection()
{
  if (this->RegionPts)
  {
    this->RegionPts->Delete();
    this->RegionPts = nullptr;
  }
  delete[] this->Planes;
  this->Planes = nullptr;
}
void svtkPlanesIntersection::SetRegionVertices(svtkPoints* v)
{
  int i;
  if (this->RegionPts)
  {
    this->RegionPts->Delete();
  }
  this->RegionPts = svtkPointsProjectedHull::New();

  if (v->GetDataType() == SVTK_DOUBLE)
  {
    this->RegionPts->DeepCopy(v);
  }
  else
  {
    this->RegionPts->SetDataTypeToDouble();

    int npts = v->GetNumberOfPoints();
    this->RegionPts->SetNumberOfPoints(npts);

    double* pt;
    for (i = 0; i < npts; i++)
    {
      pt = v->GetPoint(i);
      RegionPts->SetPoint(i, pt[0], pt[1], pt[2]);
    }
  }
}
void svtkPlanesIntersection::SetRegionVertices(double* v, int nvertices)
{
  int i;
  if (this->RegionPts)
  {
    this->RegionPts->Delete();
  }
  this->RegionPts = svtkPointsProjectedHull::New();

  this->RegionPts->SetDataTypeToDouble();
  this->RegionPts->SetNumberOfPoints(nvertices);

  for (i = 0; i < nvertices; i++)
  {
    this->RegionPts->SetPoint(i, v + (i * 3));
  }
}
int svtkPlanesIntersection::GetRegionVertices(double* v, int nvertices)
{
  int i;
  if (this->RegionPts == nullptr)
  {
    this->ComputeRegionVertices();
  }

  int npts = this->RegionPts->GetNumberOfPoints();

  if (npts > nvertices)
  {
    npts = nvertices;
  }

  for (i = 0; i < npts; i++)
  {
    this->RegionPts->GetPoint(i, v + i * 3);
  }

  return npts;
}
int svtkPlanesIntersection::GetNumberOfRegionVertices()
{
  if (this->RegionPts == nullptr)
  {
    this->ComputeRegionVertices();
  }
  return this->RegionPts->GetNumberOfPoints();
}

//---------------------------------------------------------------------
// Determine whether the axis aligned box provided intersects
// the convex region bounded by the planes.
//---------------------------------------------------------------------

int svtkPlanesIntersection::IntersectsRegion(svtkPoints* R)
{
  int plane;
  int allInside = 0;
  int nplanes = this->GetNumberOfPlanes();

  if (nplanes < 4)
  {
    svtkErrorMacro("invalid region - less than 4 planes");
    return 0;
  }

  if (this->RegionPts == nullptr)
  {
    this->ComputeRegionVertices();
    if (this->RegionPts->GetNumberOfPoints() < 4)
    {
      svtkErrorMacro("Invalid region: zero-volume intersection");
      return 0;
    }
  }

  if (R->GetNumberOfPoints() < 8)
  {
    svtkErrorMacro("invalid box");
    return 0;
  }

  int* where = new int[nplanes];

  int intersects = -1;

  //  Here's the algorithm from Graphics Gems IV, page 81,
  //
  //  R is an axis aligned box (could represent a region in a spatial
  //    spatial partitioning of a volume of data).
  //
  //  P is a set of planes defining a convex region in space (could be
  //    a view frustum).
  //
  //  The question is does P intersect R.  We expect to be doing the
  //    calculation for one P and many Rs.

  //    You may wonder why we don't do what svtkClipPolyData does, which
  //    computes the following on every point of it's PolyData input:
  //
  //      for each point in the input
  //        for each plane defining the convex region
  //          evaluate plane eq to determine if outside, inside or on plane
  //
  //     For each cell, if some points are inside and some outside, then
  //     svtkClipPolyData decides it straddles the region and clips it.  If
  //     every point is inside, it tosses it.
  //
  //     The reason is that the Graphics Gems algorithm is faster in some
  //     cases (we may only need to evaluate one vertex of the box).  And
  //     also because if the convex region passes through the box without
  //     including any vertices of the box, all box vertices will be
  //     "outside" and the algorithm will fail.  svtkClipPolyData assumes
  //     cells are very small relative to the clip region.  In general
  //     the axis-aligned box may be a large portion of world coordinate
  //     space, and the convex region a view frustum representing a
  //     small portion of the screen.

  //  1.  If R does not intersect P's bounding box, return 0.

  if (this->IntersectsBoundingBox(R) == 0)
  {
    intersects = 0;
  }

  //  2.  If P's bounding box is entirely inside R, return 1.

  else if (this->EnclosesBoundingBox(R) == 1)
  {
    intersects = 1;
  }

  //  3.  For each face plane F of P
  //
  //      Suppose the plane equation is negative inside P and
  //      positive outside P. Choose the vertex (n) of R which is
  //      most in the direction of the negative pointing normal of
  //      the plane.  The opposite vertex (p) is most in the
  //      direction of the positive pointing normal.  (This is
  //      a very quick calculation.)
  //
  //      If n is on the positive side of the plane, R is
  //      completely outside of P, so return 0.
  //
  //      If n and p are both on the negative side, then R is on
  //      the "inside" of F.  Keep track to see if all R is inside
  //      all planes defining the region.

  else
  {
    if (this->Planes == nullptr)
    {
      this->SetPlaneEquations();
    }
    allInside = 1;

    for (plane = 0; plane < nplanes; plane++)
    {
      where[plane] = this->EvaluateFacePlane(plane, R);

      if (allInside && (where[plane] != Inside))
      {
        allInside = 0;
      }

      if (where[plane] == Outside)
      {
        intersects = 0;

        break;
      }
    }
  }

  if (intersects == -1)
  {

    //  4.  If n and p were "inside" all faces, R is inside P
    //      so return 1.

    if (allInside)
    {
      intersects = 1;
    }
    //  5.  For each of three orthographic projections (X, Y and Z)
    //
    //      Compute the equations of the edge lines of P in those views.
    //
    //      If R's projection lies outside any of these lines (using 2D
    //      version of n & p tests), return 0.

    else if ((this->IntersectsProjection(R, Xdim) == 0) ||
      (this->IntersectsProjection(R, Ydim) == 0) || (this->IntersectsProjection(R, Zdim) == 0))
    {
    }
    else
    {
      //    6.  Return 1.

      intersects = 1;
    }
  }

  delete[] where;

  return (intersects == 1);
}

// a static convenience function - since we have all the machinery
//  in this class, we can compute whether an arbitrary polygon intersects
//  an axis aligned box
//
// it is assumed "pts" represents a planar polygon
//

int svtkPlanesIntersection::PolygonIntersectsBBox(double bounds[6], svtkPoints* pts)
{
  // a bogus svtkPlanesIntersection object containing only one plane

  svtkPlanesIntersection* pi = svtkPlanesIntersection::New();

  pi->SetRegionVertices(pts);

  svtkPoints* Box = svtkPoints::New();
  Box->SetNumberOfPoints(8);
  Box->SetPoint(0, bounds[0], bounds[2], bounds[4]);
  Box->SetPoint(1, bounds[1], bounds[2], bounds[4]);
  Box->SetPoint(2, bounds[1], bounds[3], bounds[4]);
  Box->SetPoint(3, bounds[0], bounds[3], bounds[4]);
  Box->SetPoint(4, bounds[0], bounds[2], bounds[5]);
  Box->SetPoint(5, bounds[1], bounds[2], bounds[5]);
  Box->SetPoint(6, bounds[1], bounds[3], bounds[5]);
  Box->SetPoint(7, bounds[0], bounds[3], bounds[5]);

  int intersects = -1;

  //  1.  Does Box intersect the polygon's bounding box?

  if (pi->IntersectsBoundingBox(Box) == 0)
  {
    intersects = 0;
  }

  //  2.  If so, does Box entirely contain the polygon's bounding box?

  else if (pi->EnclosesBoundingBox(Box) == 1)
  {
    intersects = 1;
  }

  if (intersects == -1)
  {

    //  3. If not, determine whether the Box intersects the plane of the polygon

    svtkPoints* origin = svtkPoints::New();
    origin->SetNumberOfPoints(1);
    origin->SetPoint(0, pts->GetPoint(0));

    svtkFloatArray* normal = svtkFloatArray::New();
    normal->SetNumberOfComponents(3);
    normal->SetNumberOfTuples(1);

    // find 3 points that are not co-linear and compute a normal

    double nvec[3], p0[3], p1[3], pp[3];

    int npts = pts->GetNumberOfPoints();

    pts->GetPoint(0, p0);
    pts->GetPoint(1, p1);

    for (int p = 2; p < npts; p++)
    {
      pts->GetPoint(p, pp);

      svtkPlanesIntersection::ComputeNormal(p0, p1, pp, nvec);

      if (svtkPlanesIntersection::GoodNormal(nvec))
      {
        break;
      }
    }

    normal->SetTuple(0, nvec);

    pi->SetPoints(origin);
    pi->SetNormals(normal);

    origin->Delete();
    normal->Delete();

    pi->SetPlaneEquations();

    int where = pi->EvaluateFacePlane(0, Box);

    if (where != Straddle)
    {
      intersects = 0;
    }
  }

  if (intersects == -1)
  {

    //  4.  The Box intersects the plane of the polygon.
    //
    //      For each of three orthographic projections (X, Y and Z),
    //      compute the equations of the edge lines of the polygon in those views.
    //
    //      If Box's projection lies outside any of these projections, they
    //      don't intersect in 3D.  Otherwise they do intersect in 3D.
    //
    //      KDM: I'm pretty sure the above statement is untrue.  I can think of a
    //      situation where all 3 projections intersect, but the 3D intersection
    //      does not.  However, if the two intersect in 3D, then they will
    //      intersect in the 3 2D projections.  Since I'm not worried about
    //      false positives, I'm not going to fix this right now.

    if ((pi->IntersectsProjection(Box, Xdim) == 0) || (pi->IntersectsProjection(Box, Ydim) == 0) ||
      (pi->IntersectsProjection(Box, Zdim) == 0))
    {
      intersects = 0;
    }
    else
    {
      intersects = 1;
    }
  }

  Box->Delete();
  pi->Delete();

  return intersects;
}

//---------------------------------------------------------------------
// Some convenience functions that build a svtkPlanesIntersection object
// out of a convex region.
//---------------------------------------------------------------------

// a static convenience function that converts a 3D cell into a
// svtkPlanesIntersection object

svtkPlanesIntersection* svtkPlanesIntersection::Convert3DCell(svtkCell* cell)
{
  int i;
  int nfaces = cell->GetNumberOfFaces();

  svtkPoints* origins = svtkPoints::New();
  origins->SetNumberOfPoints(nfaces);

  svtkFloatArray* normals = svtkFloatArray::New();
  normals->SetNumberOfComponents(3);
  normals->SetNumberOfTuples(nfaces);

  double inside[3] = { 0.0, 0.0, 0.0 };

  for (i = 0; i < nfaces; i++)
  {
    svtkCell* face = cell->GetFace(i);

    svtkPoints* facePts = face->GetPoints();
    int npts = facePts->GetNumberOfPoints();

    double p0[3], p1[3], pp[3], n[3];

    facePts->GetPoint(0, p0);
    facePts->GetPoint(1, p1);

    for (int p = 2; p < npts; p++)
    {
      facePts->GetPoint(p, pp);

      svtkPlanesIntersection::ComputeNormal(pp, p1, p0, n);

      if (svtkPlanesIntersection::GoodNormal(n))
      {
        break;
      }
    }

    origins->SetPoint(i, pp);
    normals->SetTuple(i, n);

    inside[0] += p1[0];
    inside[1] += p1[1];
    inside[2] += p1[2];
  }

  inside[0] /= static_cast<double>(nfaces);
  inside[1] /= static_cast<double>(nfaces);
  inside[2] /= static_cast<double>(nfaces);

  // ensure that all normals are outward pointing

  for (i = 0; i < nfaces; i++)
  {
    double ns[3], xs[3];
    double n[3], x[3], p[4];

    normals->GetTuple(i, ns);
    origins->GetPoint(i, xs);

    n[0] = ns[0];
    x[0] = xs[0];
    n[1] = ns[1];
    x[1] = xs[1];
    n[2] = ns[2];
    x[2] = xs[2];

    double outside[3];

    outside[0] = x[0] + n[0];
    outside[1] = x[1] + n[1];
    outside[2] = x[2] + n[2];

    svtkPlanesIntersection::PlaneEquation(n, x, p);

    double insideVal = svtkPlanesIntersection::EvaluatePlaneEquation(inside, p);

    double normalDirection = svtkPlanesIntersection::EvaluatePlaneEquation(outside, p);

    int sameSide =
      ((insideVal < 0) && (normalDirection < 0)) || ((insideVal > 0) && (normalDirection > 0));

    if (sameSide)
    {
      ns[0] = -ns[0];
      ns[1] = -ns[1];
      ns[2] = -ns[2];

      normals->SetTuple(i, ns);
    }
  }

  svtkPlanesIntersection* pi = svtkPlanesIntersection::New();

  pi->SetPoints(origins);
  pi->SetNormals(normals);

  origins->Delete();
  normals->Delete();

  pi->SetRegionVertices(cell->GetPoints());

  return pi;
}

//--------------------------------------------------------------------------

void svtkPlanesIntersection::ComputeNormal(double* p1, double* p2, double* p3, double normal[3])
{
  double v1[3], v2[3];

  v1[0] = p1[0] - p2[0];
  v1[1] = p1[1] - p2[1];
  v1[2] = p1[2] - p2[2];
  v2[0] = p3[0] - p2[0];
  v2[1] = p3[1] - p2[1];
  v2[2] = p3[2] - p2[2];

  svtkMath::Cross(v1, v2, normal);
}
int svtkPlanesIntersection::GoodNormal(double* n)
{
  if ((n[0] < SVTK_SMALL_DOUBLE) || (n[0] > SVTK_SMALL_DOUBLE) || (n[1] < SVTK_SMALL_DOUBLE) ||
    (n[1] > SVTK_SMALL_DOUBLE) || (n[2] < SVTK_SMALL_DOUBLE) || (n[2] > SVTK_SMALL_DOUBLE))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}
double svtkPlanesIntersection::EvaluatePlaneEquation(double* x, double* p)
{
  return (x[0] * p[0] + x[1] * p[1] + x[2] * p[2] + p[3]);
}
void svtkPlanesIntersection::PlaneEquation(double* n, double* x, double* p)
{
  p[0] = n[0];
  p[1] = n[1];
  p[2] = n[2];
  p[3] = -(n[0] * x[0] + n[1] * x[1] + n[2] * x[2]);
}

// The plane equations ***********************************************

void svtkPlanesIntersection::SetPlaneEquations()
{
  int i;
  int nplanes = this->GetNumberOfPlanes();

  // svtkPlanes stores normals & pts instead of
  //   plane equation coefficients

  delete[] this->Planes;

  this->Planes = new double[nplanes * 4];

  for (i = 0; i < nplanes; i++)
  {
    double n[3], x[3];

    this->Points->GetPoint(i, x);
    this->Normals->GetTuple(i, n);

    double nd[3], xd[3];

    nd[0] = n[0];
    xd[0] = x[0];
    nd[1] = n[1];
    xd[1] = x[1];
    nd[2] = n[2];
    xd[2] = x[2];

    double* p = this->Planes + (i * 4);

    svtkPlanesIntersection::PlaneEquation(nd, xd, p);
  }
}

// Compute region vertices if not set explicitly ********************

void svtkPlanesIntersection::ComputeRegionVertices()
{
  double M[3][3];
  double rhs[3];
  double testv[3];
  int i, j, k;
  int nplanes = this->GetNumberOfPlanes();

  if (this->RegionPts)
  {
    this->RegionPts->Delete();
  }

  this->RegionPts = svtkPointsProjectedHull::New();

  if (nplanes <= 3)
  {
    svtkErrorMacro(<< "svtkPlanesIntersection::ComputeRegionVertices invalid region");
    return;
  }

  if (this->Planes == nullptr)
  {
    this->SetPlaneEquations();
  }

  // This is an expensive process.  Better if vertices are
  // set in SetRegionVertices().  We're testing every triple of
  // planes to see if they intersect in a point that is
  // not "outside" any plane.

  int nvertices = 0;

  for (i = 0; i < nplanes; i++)
  {
    for (j = i + 1; j < nplanes; j++)
    {
      for (k = j + 1; k < nplanes; k++)
      {
        this->planesMatrix(i, j, k, M);

        int notInvertible = this->Invert3x3(M);

        if (notInvertible)
        {
          continue;
        }
        this->planesRHS(i, j, k, rhs);

        svtkMath::Multiply3x3(M, rhs, testv);

        if (duplicate(testv))
        {
          continue;
        }
        int outside = this->outsideRegion(testv);

        if (!outside)
        {
          this->RegionPts->InsertPoint(nvertices, testv);
          nvertices++;
        }
      }
    }
  }
}
int svtkPlanesIntersection::duplicate(double testv[3]) const
{
  int i;
  double pt[3];
  int npts = this->RegionPts->GetNumberOfPoints();

  for (i = 0; i < npts; i++)
  {
    this->RegionPts->GetPoint(i, pt);

    if ((pt[0] == testv[0]) && (pt[1] == testv[1]) && (pt[2] == testv[2]))
    {
      return 1;
    }
  }
  return 0;
}
void svtkPlanesIntersection::planesMatrix(int p1, int p2, int p3, double M[3][3]) const
{
  int i;
  for (i = 0; i < 3; i++)
  {
    M[0][i] = this->Planes[p1 * 4 + i];
    M[1][i] = this->Planes[p2 * 4 + i];
    M[2][i] = this->Planes[p3 * 4 + i];
  }
}
void svtkPlanesIntersection::planesRHS(int p1, int p2, int p3, double r[3]) const
{
  r[0] = -(this->Planes[p1 * 4 + 3]);
  r[1] = -(this->Planes[p2 * 4 + 3]);
  r[2] = -(this->Planes[p3 * 4 + 3]);
}
int svtkPlanesIntersection::outsideRegion(double testv[3])
{
  int i;
  int outside = 0;
  int nplanes = this->GetNumberOfPlanes();

  for (i = 0; i < nplanes; i++)
  {
    int row = i * 4;

    double fx = svtkPlanesIntersection::EvaluatePlaneEquation(testv, this->Planes + row);

    if (fx > SVTK_SMALL_DOUBLE)
    {
      outside = 1;
      break;
    }
  }
  return outside;
}
int svtkPlanesIntersection::Invert3x3(double M[3][3])
{
  int i, j;
  double temp[3][3];

  double det = svtkMath::Determinant3x3(M);

  if ((det > -SVTK_SMALL_DOUBLE) && (det < SVTK_SMALL_DOUBLE))
  {
    return -1;
  }
  svtkMath::Invert3x3(M, temp);

  for (i = 0; i < 3; i++)
  {
    for (j = 0; j < 3; j++)
    {
      M[i][j] = temp[i][j];
    }
  }

  return 0;
}

// Region / box intersection tests *******************************

int svtkPlanesIntersection::IntersectsBoundingBox(svtkPoints* R)
{
  double BoxBounds[6], RegionBounds[6];

  R->GetBounds(BoxBounds);

  this->RegionPts->GetBounds(RegionBounds);

  if ((BoxBounds[1] < RegionBounds[0]) || (BoxBounds[0] > RegionBounds[1]) ||
    (BoxBounds[3] < RegionBounds[2]) || (BoxBounds[2] > RegionBounds[3]) ||
    (BoxBounds[5] < RegionBounds[4]) || (BoxBounds[4] > RegionBounds[5]))
  {
    return 0;
  }
  return 1;
}
int svtkPlanesIntersection::EnclosesBoundingBox(svtkPoints* R)
{
  double BoxBounds[6], RegionBounds[6];

  R->GetBounds(BoxBounds);

  this->RegionPts->GetBounds(RegionBounds);

  if ((BoxBounds[0] > RegionBounds[0]) || (BoxBounds[1] < RegionBounds[1]) ||
    (BoxBounds[2] > RegionBounds[2]) || (BoxBounds[3] < RegionBounds[3]) ||
    (BoxBounds[4] > RegionBounds[4]) || (BoxBounds[5] < RegionBounds[5]))
  {
    return 0;
  }

  return 1;
}
int svtkPlanesIntersection::EvaluateFacePlane(int plane, svtkPoints* R)
{
  int i;
  double n[3], bounds[6];
  double withN[3], oppositeN[3];

  R->GetBounds(bounds);

  this->Normals->GetTuple(plane, n);

  // Find vertex of R most in direction of normal, and find
  //  oppposite vertex

  for (i = 0; i < 3; i++)
  {
    if (n[i] < 0)
    {
      withN[i] = bounds[i * 2];
      oppositeN[i] = bounds[i * 2 + 1];
    }
    else
    {
      withN[i] = bounds[i * 2 + 1];
      oppositeN[i] = bounds[i * 2];
    }
  }

  // Determine whether R is in negative half plane ("inside" frustum),
  //    positive half plane, or whether it straddles the plane.
  //    The normal points in direction of positive half plane.

  double* p = this->Planes + (plane * 4);

  double negVal =

    svtkPlanesIntersection::EvaluatePlaneEquation(oppositeN, p);

  if (negVal > 0)
  {
    return Outside;
  }

  double posVal =

    svtkPlanesIntersection::EvaluatePlaneEquation(withN, p);

  if (posVal < 0)
  {
    return Inside;
  }

  else
    return Straddle;
}
int svtkPlanesIntersection::IntersectsProjection(svtkPoints* R, int dir)
{
  int intersects = 0;

  switch (dir)
  {
    case Xdim:

      intersects = this->RegionPts->RectangleIntersectionX(R);
      break;

    case Ydim:

      intersects = this->RegionPts->RectangleIntersectionY(R);
      break;

    case Zdim:

      intersects = this->RegionPts->RectangleIntersectionZ(R);
      break;
  }

  return intersects;
}

void svtkPlanesIntersection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Planes: " << this->Planes << endl;
  os << indent << "RegionPts: " << this->RegionPts << endl;

  int i, npts;

  if (this->Points)
  {
    npts = this->Points->GetNumberOfPoints();

    for (i = 0; i < npts; i++)
    {
      double* pt = this->Points->GetPoint(i);
      double* n = this->Normals->GetTuple(i);

      os << indent << "Origin " << pt[0] << " " << pt[1] << " " << pt[2] << " ";

      os << indent << "Normal " << n[0] << " " << n[1] << " " << n[2] << endl;
    }
  }

  if (this->RegionPts)
  {
    npts = this->RegionPts->GetNumberOfPoints();

    for (i = 0; i < npts; i++)
    {
      double* pt = this->RegionPts->GetPoint(i);

      os << indent << "Vertex " << pt[0] << " " << pt[1] << " " << pt[2] << endl;
    }
  }
}
