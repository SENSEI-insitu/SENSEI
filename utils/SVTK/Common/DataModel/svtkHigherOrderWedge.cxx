/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderWedge.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHigherOrderWedge.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkHigherOrderCurve.h"
#include "svtkHigherOrderInterpolation.h"
#include "svtkHigherOrderQuadrilateral.h"
#include "svtkHigherOrderTriangle.h"
#include "svtkIdList.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"
#include "svtkWedge.h"

// SVTK_21_POINT_WEDGE is defined (or not) in svtkHigherOrderInterpolation.h
#ifdef SVTK_21_POINT_WEDGE
static double svtkHigherOrderWedge21ParametricCoords[21 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5,
  0.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5,
  1 / 3., 1 / 3., 0.0, 1 / 3., 1 / 3., 1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1 / 3.,
  1 / 3., 0.5 };
// Traversal order of subcells in 1st k-layer above:
static constexpr svtkIdType svtkHigherOrderWedge21EdgePoints[] = { 0, 6, 1, 7, 2, 8, 0 };
// Index of face-center point in 1st k-layer above:
static constexpr svtkIdType svtkHigherOrderWedge21InteriorPt = 15;
// Subcell connectivity:
static constexpr svtkIdType svtkHigherOrderWedge21ApproxCorners[12][6] = {
  { 0, 6, 15, 12, 17, 20 },
  { 6, 1, 15, 17, 13, 20 },
  { 1, 7, 15, 13, 18, 20 },
  { 7, 2, 15, 18, 14, 20 },
  { 2, 8, 15, 14, 19, 20 },
  { 8, 0, 15, 19, 12, 20 },

  { 12, 17, 20, 3, 9, 16 },
  { 17, 13, 20, 9, 4, 16 },
  { 13, 18, 20, 4, 10, 16 },
  { 18, 14, 20, 10, 5, 16 },
  { 14, 19, 20, 5, 11, 16 },
  { 19, 12, 20, 11, 3, 16 },
};
static constexpr svtkIdType svtkHigherOrderWedge21TriFace[2][7] = { { 0, 2, 1, 8, 7, 6, 15 },
  { 3, 4, 5, 9, 10, 11, 16 } };
static constexpr svtkIdType svtkHigherOrderWedge21QuadFace[3][9] = {
  { 0, 1, 4, 3, 6, 13, 9, 12, 17 },
  { 1, 2, 5, 4, 7, 14, 10, 13, 18 },
  { 2, 0, 3, 5, 8, 12, 11, 14, 19 },
};
static constexpr svtkIdType svtkHigherOrderWedge21Edge[9][3] = { { 0, 1, 6 }, { 1, 2, 7 },
  { 2, 0, 8 }, { 3, 4, 9 }, { 4, 5, 10 }, { 5, 3, 11 }, { 0, 3, 12 }, { 1, 4, 13 }, { 2, 5, 14 } };
#endif

// Return the offset into the array of face-DOFs of triangle barycentric integer coordinates (i,j)
// for the given order. Note that (i,j) are indices into the triangle (order >= i + j), not into the
// subtriangle composed solely of face DOFs. Example:
//    *
//    * *
//    * o *
//    * + @ *
//    * ^ % _ *
//    * * * * * *
//
//    (5, 1, 1) ^ -> 0
//    (5, 2, 1) % -> 1
//    (5, 3, 1) _ -> 2
//    (5, 1, 2) + -> 3
//    (5, 2, 2) @ -> 4
//    (5, 3, 1) o -> 5
//    (o, i, j)   -> i + (o - 2) * (o - 1) / 2  - ((o - j - 1) * (o - j) / 2)
//                -> i + o * (j - 1) - (j * (j + 1)) / 2;
//
//    *
//    * *
//    * o *
//    * + @ *
//    * * * * *
//
//    (4, 1, 1) + -> 0
//    (4, 2, 1) @ -> 1
//    (4, 1, 2) o -> 2
// The triangle above is order 4 (5 points per edge) and
// the "o" has coordinates (i,j) = (1,2). This function will
// return offset = 2 since the face-DOF for this triangle
// are ordered { +, @, o }.
//
static int triangleDOFOffset(int order, int i, int j)
{
  int off = i + order * (j - 1) - (j * (j + 1)) / 2;
  return off;
}

/*\brief Given a \a subId in [0,rsOrder*rsOrder*tOrder], return a wedge (i,j,k)+orientation.
 *
 * If false is returned, the inputs were invalid and the outputs are unaltered.
 * If true is returned, \a ii, \a jj, \a kk, and \a orientation are set.
 * Note that \a ii, \a jj, and \a kk refer to the lower, left, front-most
 * point of a hexahedron to be filled with 2 wedges; when \a orientation
 * is true, use (ii, jj, kk) as the right-angle corner of the wedge.
 * When \a orientation is false, use (ii+1, jj+1, kk) as the right-angle
 * corner of the wedge and reverse the order of the i- and j-axes.
 */
static bool linearWedgeLocationFromSubId(
  int subId, int rsOrder, int tOrder, int& ii, int& jj, int& kk, bool& orientation)
{
  int numWedgesPerLayer = rsOrder * rsOrder;
  kk = subId / numWedgesPerLayer;
  if (subId < 0 || kk > tOrder)
  {
    return false;
  }
  // int triId = numWedgesPerLayer - subId % numWedgesPerLayer - 1;

  int triId = subId % numWedgesPerLayer;

  if (rsOrder == 1)
  {
    ii = jj = 0;
    orientation = true;
  }
  else
  {
    svtkIdType nRightSideUp = rsOrder * (rsOrder + 1) / 2;
    if (triId < nRightSideUp)
    {
      // there are nRightSideUp subtriangles whose orientation is the same as
      // the parent triangle. We traverse them here.
      svtkIdType barycentricIndex[3];
      svtkHigherOrderTriangle::BarycentricIndex(triId, barycentricIndex, rsOrder - 1);
      ii = barycentricIndex[0];
      jj = barycentricIndex[1];
      orientation = true;
    }
    else
    {
      // the remaining subtriangles are inverted with respect to the parent
      // triangle. We traverse them here.
      orientation = false;

      if (rsOrder == 2)
      {
        ii = jj = 0;
      }
      else
      {
        svtkIdType barycentricIndex[3];
        svtkHigherOrderTriangle::BarycentricIndex(
          triId - nRightSideUp, barycentricIndex, rsOrder - 2);
        ii = barycentricIndex[0];
        jj = barycentricIndex[1];
      }
    }
  }

  return true;
}

svtkHigherOrderWedge::svtkHigherOrderWedge()
{
  this->Approx = nullptr;
  this->ApproxPD = nullptr;
  this->ApproxCD = nullptr;
  this->Order[0] = this->Order[1] = this->Order[2] = 1;

  // Deliberately leave this unset. When GetOrder() is called, it will construct
  // the accompanying data arrays used for other calculations.
  this->Order[3] = 0;

  this->Points->SetNumberOfPoints(6);
  this->PointIds->SetNumberOfIds(6);
  for (int i = 0; i < 6; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, -1);
  }
}

svtkHigherOrderWedge::~svtkHigherOrderWedge() = default;

void svtkHigherOrderWedge::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Order: " << this->GetOrder(3) << "\n";
  if (this->PointParametricCoordinates)
  {
    os << indent
       << "PointParametricCoordinates: " << this->PointParametricCoordinates->GetNumberOfPoints()
       << " entries\n";
  }
  os << indent << "Approx: " << this->Approx << "\n";
}

void svtkHigherOrderWedge::SetEdgeIdsAndPoints(int edgeId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  const int* order = this->GetOrder();
#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    if (edgeId < 0 || edgeId >= 9)
    {
      svtkErrorMacro("Asked for invalid edge " << edgeId << " of 21-point wedge");
      return;
    }
    set_number_of_ids_and_points(3);
    for (int ii = 0; ii < 3; ++ii)
    {
      set_ids_and_points(ii, svtkHigherOrderWedge21Edge[edgeId][ii]);
    }
  }
#endif
  int oi = svtkHigherOrderInterpolation::GetVaryingParameterOfWedgeEdge(edgeId);
  svtkVector2i eidx = svtkHigherOrderInterpolation::GetPointIndicesBoundingWedgeEdge(edgeId);
  svtkIdType npts = order[oi >= 0 ? oi : 0] + 1;
  int sn = 0;
  set_number_of_ids_and_points(npts);
  for (int i = 0; i < 2; ++i, ++sn)
  {
    set_ids_and_points(sn, eidx[i]);
  }
  // Now add edge-interior points in axis order:
  int offset = 6;
  if (oi == 2)
  {                                          // Edge is in t-direction.
    offset += 6 * (order[0] - 1);            // Skip edges in r-s plane.
    offset += (edgeId - 6) * (order[2] - 1); // Skip any previous t-axis edges.
  }
  else
  {
    // Edge is in r-s plane. Since we require order[0] == order[1], the offset is simple.
    offset += edgeId * (order[0] - 1);
  }
  for (int jj = 0; jj < order[oi >= 0 ? oi : 0] - 1; ++jj, ++sn)
  {
    set_ids_and_points(sn, offset + jj);
  }
}

void svtkHigherOrderWedge::Initialize() {}

/**\brief Obtain the corner points of the nearest bounding face to \a pcoords.
 *
 * This returns non-zero when \a pcoords is inside the wedge and zero otherwise.
 * In any event, \a pts is populated with the IDs of the corner points (and
 * only the corner points, not the higher-order points) of the nearest face
 * **in parameter space** (not in world coordinates).
 */
int svtkHigherOrderWedge::CellBoundary(
  int svtkNotUsed(subId), const double pcoords[3], svtkIdList* pts)
{
  svtkVector3d pp(pcoords);
  int isInside = pp[0] >= 0 && pp[1] >= 0 && (pp[0] + pp[1] <= 1) && pp[2] >= 0 && pp[2] <= 1;

  // To find the (approximate) closest face, we compute the distance
  // to planes (separatrices) that are equidistant in parameter-space.
  // We do not try to evaluate the exactly closest face in world
  // coordinates as that would be too slow to be useful and
  // too chaotic to be numerically stable.
  const double separatrixNormals[9][3] = {
    { 0.00000, 0.70711, -0.70711 },   // face 0-2
    { -0.40825, -0.40825, -0.81650 }, // face 0-3
    { 0.70711, 0.00000, -0.70711 },   // face 0-4

    { 0.00000, 0.70711, 0.70711 },   // face 1-2
    { -0.40825, -0.40825, 0.81650 }, // face 1-3
    { 0.70711, 0.00000, 0.70711 },   // face 1-4

    { -0.31623, -0.94868, 0.00000 }, // face 2-3
    { 0.94868, 0.31623, 0.00000 },   // face 3-4
    { -0.70711, 0.70711, 0.00000 }   // face 4-2
  };
  const double basepoints[3][3] = {
    { 0.25000, 0.25000, 0.25000 }, // face 0-[234]
    { 0.25000, 0.25000, 0.75000 }, // face 1-[234]
    { 0.25000, 0.25000, 0.50000 }  // face [234]-[342]
  };

  double distanceToSeparatrix[9];
  for (int ii = 0; ii < 9; ++ii)
  {
    distanceToSeparatrix[ii] =
      (pp - svtkVector3d(basepoints[ii / 3])).Dot(svtkVector3d(separatrixNormals[ii]));
  }

  bool lowerhalf = pp[2] < 0.5;
  int faceNum = -1;
  if (lowerhalf)
  {
    if (distanceToSeparatrix[0] > 0 && distanceToSeparatrix[1] > 0 && distanceToSeparatrix[2] > 0)
    { // Face 0 (lower triangle) is closest;
      faceNum = 0;
    }
  }
  else
  {
    if (distanceToSeparatrix[3] > 0 && distanceToSeparatrix[4] > 0 && distanceToSeparatrix[5] > 0)
    { // Face 1 (upper triangle) is closest;
      faceNum = 1;
    }
  }
  if (faceNum < 0)
  {
    if (distanceToSeparatrix[8] <= 0 && distanceToSeparatrix[6] >= 0)
    { // Face 2 (i-normal) is closest;
      faceNum = 2;
    }
    else if (distanceToSeparatrix[6] <= 0 && distanceToSeparatrix[7] >= 0)
    { // Face 3 (ij-normal) is closest;
      faceNum = 3;
    }
    else // distanceToSeparatrix[7] <= 0 && distanceToSeparatrix[8] >= 0 must hold
    {    // Face 4 (j-normal) is closest
      faceNum = 4;
    }
  }
  const int* facePts = svtkHigherOrderInterpolation::GetPointIndicesBoundingWedgeFace(faceNum);
  int np = facePts[3] < 0 ? 3 : 4;
  pts->SetNumberOfIds(np);
  for (int ii = 0; ii < np; ++ii)
  {
    pts->SetId(ii, this->PointIds->GetId(facePts[ii]));
  }
  return isInside;
}

int svtkHigherOrderWedge::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& minDist2, double weights[])
{
  int result = 0;

  int dummySubId;
  double linearWeights[8];
  double tmpDist2;
  svtkVector3d params;
  svtkVector3d tmpClosestPt;

  minDist2 = SVTK_DOUBLE_MAX;
  svtkIdType nwedge = this->GetNumberOfApproximatingWedges();
  for (int subCell = 0; subCell < nwedge; ++subCell)
  {
    svtkWedge* approx = this->GetApproximateWedge(subCell, nullptr, nullptr);
    int stat = approx->EvaluatePosition(
      x, tmpClosestPt.GetData(), dummySubId, params.GetData(), tmpDist2, linearWeights);
    if (stat != -1 && tmpDist2 < minDist2)
    {
      result = stat;
      subId = subCell;
      minDist2 = tmpDist2;
      for (int ii = 0; ii < 3; ++ii)
      {
        pcoords[ii] = params[ii]; // We will translate the winning parameter values later.
        if (closestPoint)
        {
          closestPoint[ii] = tmpClosestPt[ii];
        }
      }
    }
  }

  if (result != -1)
  {
    // std::cout << "EvaluatePosition([" << x[0] << " " << x[1] << " " << x[2] << "]) => "
    //  << "subId " << subId << " pc " << pcoords[0] << " " << pcoords[1] << " " << pcoords[2];

    /*
    std::cout << "  " << x[0] << " " << x[1] << " " << x[2] << "  "
      << subId << "    " << pcoords[0] << " " << pcoords[1] << " " << pcoords[2];
      */
    this->TransformApproxToCellParams(subId, pcoords);
    if (closestPoint)
    {
      this->EvaluateLocation(dummySubId, pcoords, closestPoint, weights);
      /*
      std::cout
        << pcoords[0] << " " << pcoords[1] << " " << pcoords[2] << "  "
        << closestPoint[0] << " " << closestPoint[1] << " " << closestPoint[2] << "\n";
        */
    }
    else
    {
      this->InterpolateFunctions(pcoords, weights);
      // std::cout << pcoords[0] << " " << pcoords[1] << " " << pcoords[2] << "\n";
    }
  }

  return result;
}

void svtkHigherOrderWedge::EvaluateLocation(
  int& subId, const double pcoords[3], double x[3], double* weights)
{
  subId = 0; // TODO: Should this be -1?
  this->InterpolateFunctions(pcoords, weights);

  double p[3];
  x[0] = x[1] = x[2] = 0.;
  svtkIdType nPoints = this->GetPoints()->GetNumberOfPoints();
  for (svtkIdType idx = 0; idx < nPoints; ++idx)
  {
    this->Points->GetPoint(idx, p);
    for (svtkIdType jdx = 0; jdx < 3; ++jdx)
    {
      x[jdx] += p[jdx] * weights[idx];
    }
  }
}

void svtkHigherOrderWedge::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  // std::cout << "Contour " << cellId << " with " << inPd->GetNumberOfTuples() << " tuples\n";
  this->PrepareApproxData(
    inPd, inCd, cellId, cellScalars); // writes to this->{CellScalars, ApproxPD, ApproxCD}
  svtkIdType nwedge = this->GetNumberOfApproximatingWedges();
  for (int i = 0; i < nwedge; ++i)
  {
    svtkWedge* approx =
      this->GetApproximateWedge(i, this->CellScalars.GetPointer(), this->Scalars.GetPointer());
    approx->Contour(value, this->Scalars.GetPointer(), locator, verts, lines, polys, this->ApproxPD,
      outPd, this->ApproxCD, cellId, outCd);
  }
}

void svtkHigherOrderWedge::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  this->PrepareApproxData(
    inPd, inCd, cellId, cellScalars); // writes to this->{CellScalars, ApproxPD, ApproxCD}
  svtkIdType nwedge = this->GetNumberOfApproximatingWedges();
  for (int i = 0; i < nwedge; ++i)
  {
    svtkWedge* approx =
      this->GetApproximateWedge(i, this->CellScalars.GetPointer(), this->Scalars.GetPointer());
    approx->Clip(value, this->Scalars.GetPointer(), locator, polys, this->ApproxPD, outPd,
      this->ApproxCD, cellId, outCd, insideOut);
  }
}

int svtkHigherOrderWedge::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  double tFirst = SVTK_DOUBLE_MAX;
  bool intersection = false;
  svtkVector3d tmpX;
  svtkVector3d tmpP;
  int tmpId;
  this->GetOrder(); // Ensure Order is up to date.
  for (int ff = 0; ff < this->GetNumberOfFaces(); ++ff)
  {
    svtkCell* bdy = this->GetFace(ff);
    if (bdy->IntersectWithLine(p1, p2, tol, t, tmpX.GetData(), tmpP.GetData(), tmpId))
    {
      intersection = true;
      if (t < tFirst)
      {
        tFirst = t;
        for (int ii = 0; ii < 3; ++ii)
        {
          x[ii] = tmpX[ii];
          pcoords[ii] = tmpP[ii]; // Translate this after we're sure it's the closest hit.
          subId = ff;
        }
      }
    }
  }
  if (intersection)
  {
    this->TransformFaceToCellParams(subId, pcoords);
  }
  return intersection ? 1 : 0;
}

int svtkHigherOrderWedge::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  // Note that the node numbering between the svtkWedge and svtkHigherOrderWedge is different:
  //
  //    svtkWedge                svtkHigherOrderWedge
  //  4 +-------+ 5               5 +-------+ 4
  //    |\     /|                   |\     /|
  //    | \   / |                   | \   / |
  //    |  \ /  |                   |  \ /  |
  //    | 3 +   |                   | 3 +   |
  //    |   |   |                   |   |   |
  //  1 +...|...+ 2               2 +...|...+ 1
  //     \  |  ,                     \  |  ,
  //      \ | ,                       \ | ,
  //       \|,                         \|,
  //      0 +                         0 +
  //
  // For this reason, in order to not get tetrahedra with negative Jacobian,
  // the nodes 2 and 3 of each tetra are swapped.

  ptIds->Reset();
  pts->Reset();

  svtkIdType nwedge = this->GetNumberOfApproximatingWedges();
  for (int i = 0; i < nwedge; ++i)
  {
    svtkWedge* approx = this->GetApproximateWedge(i);
    if (approx->Triangulate(1, this->TmpIds.GetPointer(), this->TmpPts.GetPointer()))
    {
      // Sigh. Triangulate methods all reset their points/ids
      // so we must copy them to our output.
      svtkIdType np = this->TmpPts->GetNumberOfPoints();
      svtkIdType ii = 0;
      while (ii < np)
      {
        pts->InsertNextPoint(this->TmpPts->GetPoint(ii));
        pts->InsertNextPoint(this->TmpPts->GetPoint(ii + 1));
        pts->InsertNextPoint(this->TmpPts->GetPoint(ii + 3));
        pts->InsertNextPoint(this->TmpPts->GetPoint(ii + 2));

        ptIds->InsertNextId(this->TmpIds->GetId(ii));
        ptIds->InsertNextId(this->TmpIds->GetId(ii + 1));
        ptIds->InsertNextId(this->TmpIds->GetId(ii + 3));
        ptIds->InsertNextId(this->TmpIds->GetId(ii + 2));

        ii += 4;
      }
    }
  }
  return 1;
}

void svtkHigherOrderWedge::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  this->getInterp()->WedgeEvaluateDerivative(
    this->Order, pcoords, this->GetPoints(), values, dim, derivs);
}

void svtkHigherOrderWedge::SetParametricCoords()
{
  const int* order = this->GetOrder();
#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    return;
  }
#endif
  if (!this->PointParametricCoordinates)
  {
    this->PointParametricCoordinates = svtkSmartPointer<svtkPoints>::New();
    this->PointParametricCoordinates->SetDataTypeToDouble();
  }

  // Ensure Order is up-to-date and check that current point size matches:
  if (static_cast<int>(this->PointParametricCoordinates->GetNumberOfPoints()) != order[3])
  {
    this->PointParametricCoordinates->Initialize();
    svtkHigherOrderInterpolation::AppendWedgeCollocationPoints(
      this->PointParametricCoordinates, this->Order);
  }
}

double* svtkHigherOrderWedge::GetParametricCoords()
{
  const int* order = this->GetOrder();
#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    return svtkHigherOrderWedge21ParametricCoords;
  }
#endif
  this->SetParametricCoords();

  return svtkDoubleArray::SafeDownCast(this->PointParametricCoordinates->GetData())->GetPointer(0);
}

double svtkHigherOrderWedge::GetParametricDistance(const double pcoords[3])
{
  double pDist, pDistMax = 0.0;

  for (int ii = 0; ii < 3; ++ii)
  {
    pDist = (pcoords[ii] < 0. ? -pcoords[ii] : (pcoords[ii] > 1. ? pcoords[ii] - 1. : 0.));
    if (pDist > pDistMax)
    {
      pDistMax = pDist;
    }
  }

  return pDistMax;
}

/// A convenience method; see the overloaded variant for more information.
bool svtkHigherOrderWedge::SubCellCoordinatesFromId(svtkVector3i& ijk, int subId)
{
  return this->SubCellCoordinatesFromId(ijk[0], ijk[1], ijk[2], subId);
}

/**\brief Given an integer specifying an approximating linear wedge, compute its IJK
 * coordinate-position in this cell.
 *
 * The \a subId specifies the lower-, left-, front-most vertex of the approximating wedge.
 * This sets the ijk coordinates of that point.
 *
 * For serendipity (21-node) wedges, the returned (i,j,k) coordinate specifies the first
 * node along the first edge of the approximating linear wedge.
 *
 * You must have called this->GetOrder() **before** invoking this method so that the order will be
 * up to date.
 */
bool svtkHigherOrderWedge::SubCellCoordinatesFromId(int& i, int& j, int& k, int subId)
{
  if (subId < 0)
  {
    return false;
  }

#ifdef SVTK_21_POINT_WEDGE
  static constexpr svtkIdType serendipitySubCell[6][2] = { { 0, 0 }, { 1, 0 }, { 2, 0 }, { 1, 1 },
    { 0, 2 }, { 0, 1 } };
  if (this->Order[3] == 21)
  {
    if (subId < 12)
    {
      int m = subId % 6;
      i = serendipitySubCell[m][0];
      j = serendipitySubCell[m][1];
      k = subId / 6;
      return true;
    }
    return false;
  }
#endif

  int layerSize = this->Order[0] * this->Order[1];
  i = subId % this->Order[0];
  j = (subId / this->Order[0]) % this->Order[1];
  k = subId / layerSize;
  return true; // TODO: detect more invalid subId values
}

/**\brief Given (i,j,k) coordinates within the HigherOrder wedge, return an offset into the local
 * connectivity (PointIds) array.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 *
 * This call is invalid for serendipity (21-node) wedge elements.
 */
int svtkHigherOrderWedge::PointIndexFromIJK(int i, int j, int k)
{
  return svtkHigherOrderWedge::PointIndexFromIJK(i, j, k, this->Order);
}

/**\brief Given (i,j,k) coordinates within the HigherOrder wedge, return an offset into the local
 * connectivity (PointIds) array.
 *
 * The \a order parameter must be a pointer to an array of 3 integer values
 * specifying the order along each axis.
 * For wedges, it is assumed that order[0] == order[1] (i.e., the triangular faces have
 * the same order for each direction).
 * The third value specifies the order of the vertical axis of the quadrilateral faces.
 *
 * This call is invalid for serendipity (21-node) wedge elements.
 */
int svtkHigherOrderWedge::PointIndexFromIJK(int i, int j, int k, const int* order)
{
  int rsOrder = order[0];
  int rm1 = rsOrder - 1;
  int tOrder = order[2];
  int tm1 = tOrder - 1;
  bool ibdy = (i == 0);
  bool jbdy = (j == 0);
  bool ijbdy = (i + j == rsOrder);
  bool kbdy = (k == 0 || k == tOrder);
  // How many boundaries do we lie on at once?
  int nbdy = (ibdy ? 1 : 0) + (jbdy ? 1 : 0) + (ijbdy ? 1 : 0) + (kbdy ? 1 : 0);

  // Return an invalid index given invalid coordinates
  if (i < 0 || i > rsOrder || j < 0 || j > rsOrder || i + j > rsOrder || k < 0 || k > tOrder ||
    order[3] == 21)
  {
    return -1;
  }

  if (nbdy == 3) // Vertex DOF
  {              // ijk is a corner node. Return the proper index (somewhere in [0,5]):
    return (ibdy && jbdy ? 0 : (jbdy && ijbdy ? 1 : 2)) + (k ? 3 : 0);
  }

  int offset = 6;
  if (nbdy == 2) // Edge DOF
  {
    if (!kbdy)
    { // Must be on a vertical edge and 2 of {ibdy, jbdy, ijbdy} are true
      offset += rm1 * 6;
      return offset + (k - 1) + ((ibdy && jbdy) ? 0 : (jbdy && ijbdy ? 1 : 2)) * tm1;
    }
    else
    { // Must be on a horizontal edge and kbdy plus 1 of {ibdy, jbdy, ijbdy} is true
      // Skip past first 3 edges if we are on the top (k = tOrder) face:
      offset += (k == tOrder ? 3 * rm1 : 0);
      if (jbdy)
      {
        return offset + i - 1;
      }
      offset += rm1; // Skip the i-axis edge
      if (ijbdy)
      {
        return offset + j - 1;
      }
      offset += rm1; // Skip the ij-axis edge
      // if (ibdy)
      return offset + (rsOrder - j - 1);
    }
  }

  offset += 6 * rm1 + 3 * tm1; // Skip all the edges

  // Number of points on a triangular face (but not on edge/corner):
  int ntfdof = (rm1 - 1) * rm1 / 2;
  int nqfdof = rm1 * tm1;
  if (nbdy == 1) // Face DOF
  {
    if (kbdy)
    { // We are on a triangular face.
      if (k > 0)
      {
        offset += ntfdof;
      }
      return offset + triangleDOFOffset(rsOrder, i, j);
    }
    // Not a k-normal face, so skip them:
    offset += 2 * ntfdof;

    // Face is quadrilateral rsOrder - 1 x tOrder - 1
    // First face is i-normal, then ij-normal, then j-normal
    if (jbdy) // On i-normal face
    {
      return offset + (i - 1) + rm1 * (k - 1);
    }
    offset += nqfdof; // Skip i-normal face
    if (ijbdy)        // on ij-normal face
    {
      return offset + (j - 1) + rm1 * (k - 1);
    }
    offset += nqfdof; // Skip ij-normal face
    return offset + (rsOrder - j - 1) + rm1 * (k - 1);
  }

  // Skip all face DOF
  offset += 2 * ntfdof + 3 * nqfdof;

  // nbdy == 0: Body DOF
  return offset + triangleDOFOffset(rsOrder, i, j) + ntfdof * (k - 1);
  /*
    (i - 1) + (order[0] - 1) * (
      (j - 1) + (order[1] - 1) * (
        (k - 1)));
        */
}

/**\brief Given the index, \a subCell, of a linear approximating-hex, translate pcoords from that
 * hex into this nonlinear hex.
 *
 * You must call this->GetOrder() **before** invoking this method as it assumes
 * the order is up to date.
 */
bool svtkHigherOrderWedge::TransformApproxToCellParams(int subCell, double* pcoords)
{
  const int* order = this->Order;
  int rsOrder = order[0];
  int tOrder = order[2];
  svtkVector3i ijk;
  bool orientation;
#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    int triIdx = subCell % 6;
    svtkVector3d triPt0 = svtkVector3d(
      &svtkHigherOrderWedge21ParametricCoords[3 * svtkHigherOrderWedge21EdgePoints[triIdx]]);
    svtkVector3d triPt1 = svtkVector3d(
      &svtkHigherOrderWedge21ParametricCoords[3 * svtkHigherOrderWedge21EdgePoints[triIdx + 1]]);
    svtkVector3d triPt2 =
      svtkVector3d(&svtkHigherOrderWedge21ParametricCoords[3 * svtkHigherOrderWedge21InteriorPt]);
    svtkVector3d rst(pcoords);
    svtkVector3d rDir = triPt1 - triPt0;
    svtkVector3d sDir = triPt2 - triPt0;
    pcoords[0] = triPt0[0] + rst[0] * rDir[0] + rst[1] * sDir[0];
    pcoords[1] = triPt0[1] + rst[0] * rDir[1] + rst[1] * sDir[1];
    pcoords[2] = ((subCell / 6) ? 0.0 : 0.5) + 0.5 * rst[2];
    return true;
  }
#endif
  if (!linearWedgeLocationFromSubId(subCell, rsOrder, tOrder, ijk[0], ijk[1], ijk[2], orientation))
  {
    return false;
  }

  if (orientation)
  { // positive orientation
    for (int pp = 0; pp < 2; ++pp)
    {
      pcoords[pp] = (pcoords[pp] + ijk[pp]) / order[pp];
    }
  }
  else
  { // negative orientation: wedge origin is at i+1,j+1 and axes point backwards toward i+0,j+0.
    for (int pp = 0; pp < 2; ++pp)
    {
      pcoords[pp] = (ijk[pp] + 1 - pcoords[pp]) / order[pp];
    }
  }

  // k-axis is always positively oriented from k+0 to k+1:
  pcoords[2] = (pcoords[2] + ijk[2]) / tOrder;

  return true;
}

/**\brief Given the index, \a subCell, of a linear approximating-wedge, translate pcoords from that
 * wedge into this nonlinear wedge.
 *
 * You must call this->GetOrder() **before** invoking this method as it assumes
 * the order is up to date.
 */
bool svtkHigherOrderWedge::TransformFaceToCellParams(int bdyFace, double* pcoords)
{
  svtkVector3d tmp(pcoords);
  switch (bdyFace)
  {
      // Triangular faces
    case 0:
      pcoords[0] = tmp[1];
      pcoords[1] = tmp[0];
      pcoords[2] = 0.0;
      return true;
    case 1:
      // First 2 coordinates are unchanged.
      pcoords[2] = 1.0;
      return true;

      // Quadrilateral faces
    case 2:
      pcoords[0] = tmp[0];
      pcoords[1] = 0.0;
      pcoords[2] = tmp[1];
      return true;
    case 3:
      pcoords[0] = 1.0 - tmp[0];
      pcoords[1] = tmp[0];
      pcoords[2] = tmp[1];
      return true;
    case 4:
      pcoords[0] = 0.0;
      pcoords[1] = tmp[0];
      pcoords[2] = tmp[1];
      return true;
    default:
    {
      svtkWarningMacro("Invalid face " << bdyFace << " (expected value in [0,5]).");
    }
  }
  return false;
}

/**\brief Return the number of linear wedges we use to approximate this nonlinear wedge.
 *
 * Note that \a order must be a pointer to an array of **four** integers.
 * The first 3 values specify the order along the r, s, and t parametric axes
 * of the wedge, respectively.
 * The first 2 values must be identical.
 *
 * The final (fourth) value must be the number of points in the wedge's connectivity;
 * it is used to handle the special case of 21-point wedges constructed from 7-point
 * triangles (a serendipity element).
 */
int svtkHigherOrderWedge::GetNumberOfApproximatingWedges(const int* order)
{
  if (!order)
  {
    return 0;
  }
  if (order[1] != order[0])
  {
    svtkGenericWarningMacro("Wedge elements must have same order in "
                           "first 2 dimensions, but had orders "
      << order[0] << " and " << order[1] << " instead.");
  }
#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    return 12;
  }
#endif
  return order[0] * order[0] * order[2];
}

/// Return a linear wedge used to approximate a region of the nonlinear wedge.
svtkWedge* svtkHigherOrderWedge::GetApprox()
{
  if (!this->Approx)
  {
    this->Approx = svtkSmartPointer<svtkWedge>::New();
    this->ApproxPD = svtkSmartPointer<svtkPointData>::New();
    this->ApproxCD = svtkSmartPointer<svtkCellData>::New();
  }
  return this->Approx.GetPointer();
}

/**\brief Prepare point data for use by linear approximating-elements.
 *
 * This copies the point data for the current cell into a new point-data
 * object so that the point ids and scalar ids can match.
 */
void svtkHigherOrderWedge::PrepareApproxData(
  svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars)
{
  this->GetApprox(); // Ensure this->Approx{PD,CD} are non-NULL.
  // this->GetOrder(); // Ensure the order has been updated to match this element.
  this->SetOrderFromCellData(cd, this->Points->GetNumberOfPoints(), cellId);
  svtkIdType npts = this->Order[3];
  svtkIdType nele = this->Order[0] * this->Order[1] * this->Order[2];
  this->ApproxPD->Initialize();
  this->ApproxCD->Initialize();
  this->ApproxPD->CopyAllOn();
  this->ApproxCD->CopyAllOn();
  this->ApproxPD->CopyAllocate(pd, npts);
  this->ApproxCD->CopyAllocate(cd, nele);

  this->CellScalars->SetNumberOfTuples(npts);
  for (int pp = 0; pp < npts; ++pp)
  {
    this->ApproxPD->CopyData(pd, this->PointIds->GetId(pp), pp);
    this->CellScalars->SetValue(pp, cellScalars->GetTuple1(pp));
  }
  for (int ee = 0; ee < nele; ++ee)
  {
    this->ApproxCD->CopyData(cd, cellId, ee);
  }
}

/**\brief Populate the linear wedge returned by GetApprox() with point-data from one wedge-like
 * interval of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkWedge* svtkHigherOrderWedge::GetApproximateWedge(
  int subId, svtkDataArray* scalarsIn, svtkDataArray* scalarsOut)
{
  svtkWedge* approx = this->GetApprox();
  bool doScalars = (scalarsIn && scalarsOut);
  if (doScalars)
  {
    scalarsOut->SetNumberOfTuples(6);
  }
  int i, j, k;
  bool orientation;
  const int* order = this->GetOrder();

#ifdef SVTK_21_POINT_WEDGE
  if (order[3] == 21)
  {
    if (subId < 0 || subId >= 12)
    {
      svtkWarningMacro("Bad subId " << subId << " for 21-point wedge.");
      return nullptr;
    }
    for (int ic = 0; ic < 6; ++ic)
    {
      const svtkIdType corner = svtkHigherOrderWedge21ApproxCorners[subId][ic];
      svtkVector3d cp;
      this->Points->GetPoint(corner, cp.GetData());
      // std::cout << "    corner " << ic << " @ " << corner << ": " << cp << "\n";
      // aconn[ic] = this->PointIds->GetId(corner);
      // aconn[ic] = corner;
      approx->PointIds->SetId(ic, doScalars ? corner : this->PointIds->GetId(corner));
      approx->Points->SetPoint(ic, cp.GetData()); // this->Points->GetPoint(corner));
      if (doScalars)
      {
        // std::cout << "    corner " << ic << " @ " << corner << ": " <<
        // scalarsIn->GetTuple(corner)[0] << "\n";
        scalarsOut->SetTuple(ic, scalarsIn->GetTuple(corner));
      }
    }
    return approx;
  }
#endif

  if (!linearWedgeLocationFromSubId(subId, order[0], order[2], i, j, k, orientation))
  {
    svtkWarningMacro(
      "Bad subId " << subId << " for order " << order[0] << " " << order[1] << " " << order[2]);
    return nullptr;
  }

  // Get the point coordinates (and optionally scalars) for each of the 6 corners
  // in the approximating wedge spanning half of (i, i+1) x (j, j+1) x (k, k+1):
  // svtkIdType aconn[8]; // = {0, 1, 2, 3, 4, 5, 6, 7};
  // std::cout << "Wedgeproximate " << subId << "\n";
  const int deltas[2][3][2] = {
    { { 0, 0 }, { 1, 0 }, { 0, 1 } }, // positive orientation: r, s axes increase as i, j increase
    { { 1, 1 }, { 0, 1 }, { 1, 0 } }  // negative orientation: r, s axes decrease as i, j increase
  };
  for (int ic = 0; ic < 6; ++ic)
  {
    const svtkIdType corner = this->PointIndexFromIJK(i + deltas[orientation ? 0 : 1][ic % 3][0],
      j + deltas[orientation ? 0 : 1][ic % 3][1], k + ((ic / 3) ? 1 : 0));
    svtkVector3d cp;

    if (corner == -1)
    {
      svtkWarningMacro("Could not determine point index for IJK = ("
        << i + deltas[orientation ? 0 : 1][ic % 3][0] << " "
        << j + deltas[orientation ? 0 : 1][ic % 3][1] << " " << k + ((ic / 3) ? 1 : 0) << ")");
      return nullptr;
    }

    this->Points->GetPoint(corner, cp.GetData());
    // std::cout << "    corner " << ic << " @ " << corner << ": " << cp << "\n";
    // aconn[ic] = this->PointIds->GetId(corner);
    // aconn[ic] = corner;
    approx->PointIds->SetId(ic, doScalars ? corner : this->PointIds->GetId(corner));
    approx->Points->SetPoint(ic, cp.GetData()); // this->Points->GetPoint(corner));
    if (doScalars)
    {
      // std::cout << "    corner " << ic << " @ " << corner << ": " <<
      // scalarsIn->GetTuple(corner)[0] << "\n";
      scalarsOut->SetTuple(ic, scalarsIn->GetTuple(corner));
    }
  }
  return approx;
}

void svtkHigherOrderWedge::GetTriangularFace(svtkHigherOrderTriangle* result, int faceId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  const int* order = this->GetOrder();
  int iAxis;
  int kk;
  if (faceId == 0)
  {
    iAxis = 1;
    kk = 0;
  }
  else if (faceId == 1)
  {
    iAxis = 0;
    kk = order[2];
  }
  else
  {
    iAxis = 1;
    kk = 0;
    svtkErrorMacro("faceId for GetTriangularFace should be 0 or 1 " << faceId);
  }

#ifdef SVTK_21_POINT_WEDGE
  const int nptsActual = order[3];
#endif
  const int rsOrder = order[0];

#ifdef SVTK_21_POINT_WEDGE
  if (nptsActual == 21)
  {
    set_number_of_ids_and_points(7);
    result->Initialize();
    for (int ii = 0; ii < 7; ++ii)
    {
      svtkIdType srcId = svtkHigherOrderWedge21TriFace[kk == 0 ? 0 : 1][ii];
      set_ids_and_points(ii, srcId);
    }
    return;
  }
#endif
  svtkIdType npts = (rsOrder + 1) * (rsOrder + 2) / 2;
  set_number_of_ids_and_points(npts);
  result->Initialize();
  svtkIdType bary[3];
  for (int jj = 0; jj <= rsOrder; ++jj)
  {
    for (int ii = 0; ii <= (rsOrder - jj); ++ii)
    {
      svtkIdType srcId =
        iAxis == 0 ? this->PointIndexFromIJK(ii, jj, kk) : this->PointIndexFromIJK(jj, ii, kk);
      bary[0] = ii;
      bary[1] = jj;
      bary[2] = rsOrder - ii - jj;
      svtkIdType dstId = result->Index(bary, rsOrder);
      set_ids_and_points(dstId, srcId);

      /*
      svtkVector3d vpt;
      this->Points->GetPoint(srcId, vpt.GetData());
      std::cout << "  Pt " << ii << " " << jj << "  " << vpt << " src " << srcId << " dst " << dstId
      << "\n";
      */
    }
  }
}

void svtkHigherOrderWedge::GetQuadrilateralFace(svtkHigherOrderQuadrilateral* result, int faceId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  const int* order = this->GetOrder();
  int di;
  int dj;
  if (faceId == 2)
  {
    di = +1;
    dj = 0;
  }
  else if (faceId == 3)
  {
    di = -1;
    dj = +1;
  }
  else if (faceId == 4)
  {
    di = 0;
    dj = -1;
  }
  else
  {
    di = +1;
    dj = 0;
    svtkErrorMacro("faceId for GetTriangularFace should be 2, 3, 4 " << faceId);
  }

#ifdef SVTK_21_POINT_WEDGE
  const int nptsActual = order[3];
  if (nptsActual == 21)
  {
    set_number_of_ids_and_points(9);
    result->Initialize();
    int quadFace = (di == -dj ? 1 : (dj == 0 ? 0 : 2));
    for (int ii = 0; ii < 9; ++ii)
    {
      svtkIdType srcId = svtkHigherOrderWedge21QuadFace[quadFace][ii];
      set_ids_and_points(ii, srcId);
    }
    result->SetOrder(2, 2);
    return;
  }
#endif
  const int rsOrder = order[0];
  const int tOrder = order[2];

  svtkIdType npts = (rsOrder + 1) * (tOrder + 1);
  set_number_of_ids_and_points(npts);
  result->Initialize();
  result->SetOrder(rsOrder, tOrder);

  for (int kk = 0; kk <= tOrder; ++kk)
  {
    int si = (di >= 0 ? 0 : rsOrder);
    int sj = (dj >= 0 ? 0 : rsOrder);
    for (int ii = 0; ii <= rsOrder; ++ii, si += di, sj += dj)
    {
      int srcId = this->PointIndexFromIJK(si, sj, kk);
      int dstId = result->PointIndexFromIJK(ii, kk, 0);
      set_ids_and_points(dstId, srcId);
      /*
      svtkVector3d vpt;
      this->Points->GetPoint(srcId, vpt.GetData());
      std::cout << "  Pt " << ii << " " << kk << "  " << vpt << " src " << srcId << " dst " << dstId
      << "\n";
      */
    }
  }
}

/**\brief Set the degree  of the cell, given a svtkDataSet and cellId
 */
void svtkHigherOrderWedge::SetOrderFromCellData(
  svtkCellData* cell_data, const svtkIdType numPts, const svtkIdType cell_id)
{
  if (cell_data->SetActiveAttribute(
        "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
  {
    double degs[3];
    svtkDataArray* v = cell_data->GetHigherOrderDegrees();
    v->GetTuple(cell_id, degs);
    this->SetOrder(degs[0], degs[1], degs[2], numPts);
  }
  else
  {
    this->SetUniformOrderFromNumPoints(numPts);
  }
}

void svtkHigherOrderWedge::SetUniformOrderFromNumPoints(const svtkIdType numPts)
{
  const double n = static_cast<double>(numPts);
  static const double third(1. / 3.);
  static const double ninth(1. / 9.);
  static const double twentyseventh(1. / 27.);
  const double term =
    std::cbrt(third * sqrt(third) * sqrt((27.0 * n - 2.0) * n) + n - twentyseventh);
  int deg = static_cast<int>(round(term + ninth / term - 4 * third));

#ifdef SVTK_21_POINT_WEDGE
  if (numPts == 21)
  {
    deg = 2;
  }
#endif

  this->SetOrder(deg, deg, deg, numPts);
}

void svtkHigherOrderWedge::SetOrder(const int s, const int t, const int u, const svtkIdType numPts)
{
  if (s != t)
    svtkErrorMacro("For wedges, the first two degrees should be equals.");

  if (this->PointParametricCoordinates && (Order[0] != s || Order[2] != u))
    this->PointParametricCoordinates->Reset();
  Order[0] = s;
  Order[1] = s;
  Order[2] = u;

#ifdef SVTK_21_POINT_WEDGE
  if (numPts == 21)
  {
    Order[3] = numPts;
    if ((s != 2) || (u != 2))
      svtkErrorMacro("For Wedge 21, the degrees should be quadratic.");
  }
  else
  {
    Order[3] = (s + 1) * (s + 2) / 2 * (u + 1);
    if (Order[3] != numPts)
      svtkErrorMacro("The degrees are not correctly set in the input file.");
  }
#else
  Order[3] = (s + 1) * (s + 2) / 2 * (u + 1);
  if (Order[3] != numPts)
    svtkErrorMacro("The degrees are not correctly set in the input file.");
#endif
}

const int* svtkHigherOrderWedge::GetOrder()
{
  //   The interpolation routines can handle different order along each axis
  //   The connectivity array contains three additional entries at the end which specify the Order
  //   in s, t, and u The unstructure grid calls SetOrder with those three additional entries
  svtkIdType numPts = this->Points->GetNumberOfPoints();
  if (this->Order[3] != numPts)
  {
    if (numPts == 6)
      this->SetUniformOrderFromNumPoints(numPts);
    else
      svtkErrorMacro("The degrees might be direction dependents, and should be set before GetOrder "
                    "is called. numPts is "
        << numPts << " and Order[3] " << Order[3]);
  }
  return this->Order;
}
