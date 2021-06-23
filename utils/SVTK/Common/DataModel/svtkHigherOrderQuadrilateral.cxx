/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderQuadrilateral.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHigherOrderQuadrilateral.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkHigherOrderCurve.h"
#include "svtkHigherOrderInterpolation.h"
#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkQuad.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

svtkHigherOrderQuadrilateral::svtkHigherOrderQuadrilateral()
{
  this->Approx = nullptr;
  this->Order[0] = this->Order[1] = 1;
  // Deliberately leave this unset. When GetOrder() is called, it will construct
  // the accompanying data arrays used for other calculations.
  this->Order[2] = 0;

  this->Points->SetNumberOfPoints(4);
  this->PointIds->SetNumberOfIds(4);
  for (svtkIdType i = 0; i < 4; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, -1);
  }
}

svtkHigherOrderQuadrilateral::~svtkHigherOrderQuadrilateral() = default;

void svtkHigherOrderQuadrilateral::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Order: " << this->GetOrder(0) << "\n";
  if (this->PointParametricCoordinates)
  {
    os << indent
       << "PointParametricCoordinates: " << this->PointParametricCoordinates->GetNumberOfPoints()
       << " entries\n";
  }
  os << indent << "Approx: " << this->Approx << "\n";
}

void svtkHigherOrderQuadrilateral::SetEdgeIdsAndPoints(int edgeId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  const int* order = this->GetOrder();
  // Note in calls below: quad has same edges as first 4 of hex
  int oi = svtkHigherOrderInterpolation::GetVaryingParameterOfHexEdge(edgeId);
  svtkVector2i eidx = svtkHigherOrderInterpolation::GetPointIndicesBoundingHexEdge(edgeId);
  svtkIdType npts = order[oi] + 1;
  int sn = 0;

  set_number_of_ids_and_points(npts);
  for (int i = 0; i < 2; ++i, ++sn)
  {
    set_ids_and_points(sn, eidx[i]);
  }
  // Now add edge-interior points in axis order:
  int offset = 4;
  for (int ee = 0; ee < edgeId; ++ee)
  {
    offset += order[ee % 2 == 0 ? 0 : 1] - 1;
  }
  for (int jj = 0; jj < order[oi] - 1; ++jj, ++sn)
  {
    set_ids_and_points(sn, offset + jj);
  }
}

void svtkHigherOrderQuadrilateral::Initialize() {}

int svtkHigherOrderQuadrilateral::CellBoundary(
  int svtkNotUsed(subId), const double pcoords[3], svtkIdList* pts)
{
  double t1 = pcoords[0] - pcoords[1];
  double t2 = 1.0 - pcoords[0] - pcoords[1];

  pts->SetNumberOfIds(2);

  // compare against two lines in parametric space that divide element
  // into four pieces.
  if (t1 >= 0.0 && t2 >= 0.0)
  {
    pts->SetId(0, this->PointIds->GetId(0));
    pts->SetId(1, this->PointIds->GetId(1));
  }

  else if (t1 >= 0.0 && t2 < 0.0)
  {
    pts->SetId(0, this->PointIds->GetId(1));
    pts->SetId(1, this->PointIds->GetId(2));
  }

  else if (t1 < 0.0 && t2 < 0.0)
  {
    pts->SetId(0, this->PointIds->GetId(2));
    pts->SetId(1, this->PointIds->GetId(3));
  }

  else //( t1 < 0.0 && t2 >= 0.0 )
  {
    pts->SetId(0, this->PointIds->GetId(3));
    pts->SetId(1, this->PointIds->GetId(0));
  }

  if (pcoords[0] < 0.0 || pcoords[0] > 1.0 || pcoords[1] < 0.0 || pcoords[1] > 1.0)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

int svtkHigherOrderQuadrilateral::EvaluatePosition(const double x[3], double closestPoint[3],
  int& subId, double pcoords[3], double& minDist2, double weights[])
{
  int result = 0;

  int dummySubId;
  double linearWeights[4];
  double tmpDist2;
  svtkVector3d params;
  svtkVector3d tmpClosestPt;

  minDist2 = SVTK_DOUBLE_MAX;
  svtkIdType nquad = svtkHigherOrderInterpolation::NumberOfIntervals<2>(this->GetOrder());
  for (int subCell = 0; subCell < nquad; ++subCell)
  {
    svtkQuad* approx = this->GetApproximateQuad(subCell, nullptr, nullptr);
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
    this->TransformApproxToCellParams(subId, pcoords);
    if (closestPoint)
    {
      this->EvaluateLocation(dummySubId, pcoords, closestPoint, weights);
    }
    else
    {
      this->InterpolateFunctions(pcoords, weights);
    }
  }

  return result;
}

void svtkHigherOrderQuadrilateral::EvaluateLocation(
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

void svtkHigherOrderQuadrilateral::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  this->PrepareApproxData(
    inPd, inCd, cellId, cellScalars); // writes to this->{CellScalars, ApproxPD, ApproxCD}
  svtkIdType nquad = svtkHigherOrderInterpolation::NumberOfIntervals<2>(this->GetOrder());
  for (int i = 0; i < nquad; ++i)
  {
    svtkQuad* approx =
      this->GetApproximateQuad(i, this->CellScalars.GetPointer(), this->Scalars.GetPointer());
    approx->Contour(value, this->Scalars.GetPointer(), locator, verts, lines, polys, this->ApproxPD,
      outPd, this->ApproxCD, cellId, outCd);
  }
}

void svtkHigherOrderQuadrilateral::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  this->PrepareApproxData(
    inPd, inCd, cellId, cellScalars); // writes to this->{CellScalars, ApproxPD, ApproxCD}
  svtkIdType nquad = svtkHigherOrderInterpolation::NumberOfIntervals<2>(this->GetOrder());
  for (int i = 0; i < nquad; ++i)
  {
    svtkQuad* approx =
      this->GetApproximateQuad(i, this->CellScalars.GetPointer(), this->Scalars.GetPointer());
    approx->Clip(value, this->Scalars.GetPointer(), locator, polys, this->ApproxPD, outPd,
      this->ApproxCD, cellId, outCd, insideOut);
  }
}

int svtkHigherOrderQuadrilateral::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  svtkIdType nquad = svtkHigherOrderInterpolation::NumberOfIntervals<2>(this->GetOrder());
  double tFirst = SVTK_DOUBLE_MAX;
  bool intersection = false;
  svtkVector3d tmpX;
  svtkVector3d tmpP;
  int tmpId;
  for (int i = 0; i < nquad; ++i)
  {
    svtkQuad* approx = this->GetApproximateQuad(i);
    if (approx->IntersectWithLine(p1, p2, tol, t, tmpX.GetData(), tmpP.GetData(), tmpId))
    {
      // Record the point closest to p1 in the direction of p2 unless there is no other
      // intersection, in which case we will report a point "before" p1 (further from p2 than p1).
      if (!intersection || (t >= 0 && (t < tFirst || tFirst < 0)))
      {
        tFirst = t;
        subId = i;
        for (int ii = 0; ii < 3; ++ii)
        {
          x[ii] = tmpX[ii];
          pcoords[ii] = tmpP[ii]; // Translate this after we're sure it's the closest hit.
        }
      }
      intersection = true;
    }
  }
  if (intersection)
  {
    intersection &= this->TransformApproxToCellParams(subId, pcoords);
    t = tFirst;
  }
  return intersection ? 1 : 0;
}

int svtkHigherOrderQuadrilateral::Triangulate(
  int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  ptIds->Reset();
  pts->Reset();

  svtkIdType nquad = svtkHigherOrderInterpolation::NumberOfIntervals<2>(this->GetOrder());
  for (int i = 0; i < nquad; ++i)
  {
    svtkQuad* approx = this->GetApproximateQuad(i);
    if (approx->Triangulate(1, this->TmpIds.GetPointer(), this->TmpPts.GetPointer()))
    {
      // Sigh. Triangulate methods all reset their points/ids
      // so we must copy them to our output.
      svtkIdType np = this->TmpPts->GetNumberOfPoints();
      svtkIdType ni = this->TmpIds->GetNumberOfIds();
      for (svtkIdType ii = 0; ii < np; ++ii)
      {
        pts->InsertNextPoint(this->TmpPts->GetPoint(ii));
      }
      for (svtkIdType ii = 0; ii < ni; ++ii)
      {
        ptIds->InsertNextId(this->TmpIds->GetId(ii));
      }
    }
  }
  return 1;
}

void svtkHigherOrderQuadrilateral::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  svtkIdType numberOfPoints = this->Points->GetNumberOfPoints();

  double sum[2], p[3];
  std::vector<double> functionDerivs(2 * numberOfPoints);
  double *J[3], J0[3], J1[3], J2[3];
  double *JI[3], JI0[3], JI1[3], JI2[3];

  this->InterpolateDerivs(pcoords, functionDerivs.data());

  // Compute transposed Jacobian and inverse Jacobian
  J[0] = J0;
  J[1] = J1;
  J[2] = J2;
  JI[0] = JI0;
  JI[1] = JI1;
  JI[2] = JI2;
  for (int k = 0; k < 3; k++)
  {
    J0[k] = J1[k] = 0.0;
  }

  for (int i = 0; i < numberOfPoints; i++)
  {
    this->Points->GetPoint(i, p);
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        J[j][k] += p[k] * functionDerivs[j + 2 * i];
      }
    }
  }

  // Compute third row vector in transposed Jacobian and normalize it, so that Jacobian determinant
  // stays the same.
  svtkMath::Cross(J0, J1, J2);
  if (svtkMath::Normalize(J2) == 0.0 || !svtkMath::InvertMatrix(J, JI, 3)) // degenerate
  {
    for (int j = 0; j < dim; j++)
    {
      for (int i = 0; i < 3; i++)
      {
        derivs[j * dim + i] = 0.0;
      }
    }
    return;
  }

  // Loop over "dim" derivative values. For each set of values,
  // compute derivatives
  // in local system and then transform into modelling system.
  // First compute derivatives in local x'-y' coordinate system
  for (int j = 0; j < dim; j++)
  {
    sum[0] = sum[1] = 0.0;
    for (int i = 0; i < numberOfPoints; i++) // loop over interp. function derivatives
    {
      sum[0] += functionDerivs[2 * i] * values[dim * i + j];
      sum[1] += functionDerivs[2 * i + 1] * values[dim * i + j];
    }
    //    dBydx = sum[0]*JI[0][0] + sum[1]*JI[0][1];
    //    dBydy = sum[0]*JI[1][0] + sum[1]*JI[1][1];

    // Transform into global system (dot product with global axes)
    derivs[3 * j] = sum[0] * JI[0][0] + sum[1] * JI[0][1];
    derivs[3 * j + 1] = sum[0] * JI[1][0] + sum[1] * JI[1][1];
    derivs[3 * j + 2] = sum[0] * JI[2][0] + sum[1] * JI[2][1];
  }
}

void svtkHigherOrderQuadrilateral::SetParametricCoords()
{
  if (!this->PointParametricCoordinates)
  {
    this->PointParametricCoordinates = svtkSmartPointer<svtkPoints>::New();
    this->PointParametricCoordinates->SetDataTypeToDouble();
  }

  // Ensure Order is up-to-date and check that current point size matches:
  if (static_cast<int>(this->PointParametricCoordinates->GetNumberOfPoints()) != this->GetOrder(2))
  {
    this->PointParametricCoordinates->Initialize();
    svtkHigherOrderInterpolation::AppendQuadrilateralCollocationPoints(
      this->PointParametricCoordinates, this->Order);
  }
}

double* svtkHigherOrderQuadrilateral::GetParametricCoords()
{
  this->SetParametricCoords();

  return svtkDoubleArray::SafeDownCast(this->PointParametricCoordinates->GetData())->GetPointer(0);
}

double svtkHigherOrderQuadrilateral::GetParametricDistance(const double pcoords[3])
{
  double pDist, pDistMax = 0.0;

  for (int ii = 0; ii < 2; ++ii)
  {
    pDist = (pcoords[ii] < 0. ? -pcoords[ii] : (pcoords[ii] > 1. ? pcoords[ii] - 1. : 0.));
    if (pDist > pDistMax)
    {
      pDistMax = pDist;
    }
  }

  // The quadrilateral's 3rd parametric coordinate should always be 0:
  if (pcoords[2] != 0.0 && (pDist = std::abs(pcoords[2])) > pDistMax)
  {
    pDistMax = pDist;
  }

  return pDistMax;
}

/// Return a linear quadrilateral used to approximate a region of the nonlinear quadrilateral.
svtkQuad* svtkHigherOrderQuadrilateral::GetApprox()
{
  if (!this->Approx)
  {
    this->Approx = svtkSmartPointer<svtkQuad>::New();
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
void svtkHigherOrderQuadrilateral::PrepareApproxData(
  svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars)
{
  this->GetApprox(); // Ensure this->Approx{PD,CD} are non-NULL.
  // this->GetOrder(); // Ensure the order has been updated to match this element.
  this->SetOrderFromCellData(cd, this->Points->GetNumberOfPoints(), cellId);
  svtkIdType npts = this->Order[2];
  svtkIdType nele = this->Order[0] * this->Order[1];
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

/// A convenience method; see the overloaded variant for more information.
bool svtkHigherOrderQuadrilateral::SubCellCoordinatesFromId(svtkVector3i& ijk, int subId)
{
  return this->SubCellCoordinatesFromId(ijk[0], ijk[1], ijk[2], subId);
}

/**\brief Given an integer specifying an approximating linear quad, compute its IJK
 * coordinate-position in this cell.
 *
 * The \a subId specifies the lower-, left-, front-most vertex of the approximating quad.
 * This sets the ijk coordinates of that point.
 *
 * You must have called this->GetOrder() **before** invoking this method so that the order will be
 * up to date.
 */
bool svtkHigherOrderQuadrilateral::SubCellCoordinatesFromId(int& i, int& j, int& k, int subId)
{
  if (subId < 0)
  {
    return false;
  }

  i = subId % this->Order[0];
  j = (subId / this->Order[0]) % this->Order[1];
  k = 0;
  return i + this->Order[0] * j == subId ? true : false;
}

/**\brief A convenience function to get a connectivity offset from a control-point tuple.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
int svtkHigherOrderQuadrilateral::PointIndexFromIJK(int i, int j, int svtkNotUsed(k))
{
  return svtkHigherOrderQuadrilateral::PointIndexFromIJK(i, j, this->Order);
}

/**\brief Given (i,j,k) coordinates within the HigherOrder quad, return an offset into the local
 * connectivity (PointIds) array.
 *
 * The \a order parameter must point to the start of an array of 2 integers.
 */
int svtkHigherOrderQuadrilateral::PointIndexFromIJK(int i, int j, const int* order)
{
  bool ibdy = (i == 0 || i == order[0]);
  bool jbdy = (j == 0 || j == order[1]);
  // How many boundaries do we lie on at once?
  int nbdy = (ibdy ? 1 : 0) + (jbdy ? 1 : 0);

  if (nbdy == 2) // Vertex DOF
  {              // ijk is a corner node. Return the proper index (somewhere in [0,7]):
    return (i ? (j ? 2 : 1) : (j ? 3 : 0));
  }

  int offset = 4;
  if (nbdy == 1) // Edge DOF
  {
    if (!ibdy)
    { // On i axis
      return (i - 1) + (j ? order[0] - 1 + order[1] - 1 : 0) + offset;
    }
    if (!jbdy)
    { // On j axis
      return (j - 1) + (i ? order[0] - 1 : 2 * (order[0] - 1) + order[1] - 1) + offset;
    }
  }

  offset += 2 * (order[0] - 1 + order[1] - 1);
  // nbdy == 0: Face DOF
  return offset + (i - 1) + (order[0] - 1) * ((j - 1));
}

/**\brief Given the index, \a subCell, of a linear approximating-quad, translate pcoords from that
 * quad into this nonlinear quad.
 *
 * You must call this->GetOrder() **before** invoking this method as it assumes
 * the order is up to date.
 */
bool svtkHigherOrderQuadrilateral::TransformApproxToCellParams(int subCell, double* pcoords)
{
  svtkVector3i ijk;
  if (!this->SubCellCoordinatesFromId(ijk, subCell))
  {
    return false;
  }
  for (int pp = 0; pp < 2; ++pp)
  {
    pcoords[pp] = (pcoords[pp] + ijk[pp]) / this->Order[pp];
  }
  pcoords[2] = 0.;
  return true;
}

/**\brief Set the degree  of the cell, given a svtkDataSet and cellId
 */
void svtkHigherOrderQuadrilateral::SetOrderFromCellData(
  svtkCellData* cell_data, const svtkIdType numPts, const svtkIdType cell_id)
{
  if (cell_data->SetActiveAttribute(
        "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
  {
    double degs[3];
    svtkDataArray* v = cell_data->GetHigherOrderDegrees();
    v->GetTuple(cell_id, degs);
    this->SetOrder(degs[0], degs[1]);
    if (this->Order[2] != numPts)
      svtkErrorMacro("The degrees are not correctly set in the input file.");
  }
  else
  {
    this->SetUniformOrderFromNumPoints(numPts);
  }
}

void svtkHigherOrderQuadrilateral::SetUniformOrderFromNumPoints(const svtkIdType numPts)
{
  int deg = static_cast<int>(round(std::sqrt(static_cast<int>(numPts)))) - 1;
  this->SetOrder(deg, deg);
  if (static_cast<int>(numPts) != this->Order[2])
    svtkErrorMacro("The degrees are direction dependents, and should be set in the input file.");
}

void svtkHigherOrderQuadrilateral::SetOrder(const int s, const int t)
{
  if (this->PointParametricCoordinates && (Order[0] != s || Order[1] != t))
    this->PointParametricCoordinates->Reset();
  Order[0] = s;
  Order[1] = t;
  Order[2] = (s + 1) * (t + 1);
}

const int* svtkHigherOrderQuadrilateral::GetOrder()
{
  //   The interpolation routines can handle different order along each axis
  //   The connectivity array contains three additional entries at the end which specify the Order
  //   in s, t, and u The unstructure grid calls SetOrder with those three additional entries
  svtkIdType numPts = this->Points->GetNumberOfPoints();
  if (this->Order[2] != numPts)
  {
    if (numPts == 4)
      this->SetUniformOrderFromNumPoints(numPts);
    else
      svtkErrorMacro("The degrees might be direction dependents, and should be set before GetOrder "
                    "is called. numPts is "
        << numPts << " and Order[2] " << Order[2]);
  }
  return this->Order;
}
