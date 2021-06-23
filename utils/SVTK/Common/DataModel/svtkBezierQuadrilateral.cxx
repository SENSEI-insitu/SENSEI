/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierQuadrilateral.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBezierQuadrilateral.h"

#include "svtkBezierCurve.h"
#include "svtkBezierInterpolation.h"
#include "svtkCellData.h"
#include "svtkDataSet.h"
#include "svtkDoubleArray.h"
#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkQuad.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

svtkStandardNewMacro(svtkBezierQuadrilateral);

svtkBezierQuadrilateral::svtkBezierQuadrilateral()
  : svtkHigherOrderQuadrilateral()
{
}

svtkBezierQuadrilateral::~svtkBezierQuadrilateral() = default;

void svtkBezierQuadrilateral::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkBezierQuadrilateral::GetEdge(int edgeId)
{
  svtkBezierCurve* result = EdgeCell;

  if (this->GetRationalWeights()->GetNumberOfTuples() > 0)
  {
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
      result->GetRationalWeights()->SetNumberOfTuples(npts);
    };
    const auto set_ids_and_points = [&](
                                      const svtkIdType& edge_id, const svtkIdType& face_id) -> void {
      result->Points->SetPoint(edge_id, this->Points->GetPoint(face_id));
      result->PointIds->SetId(edge_id, this->PointIds->GetId(face_id));
      result->GetRationalWeights()->SetValue(
        edge_id, this->GetRationalWeights()->GetValue(face_id));
    };
    this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  }
  else
  {
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
      result->GetRationalWeights()->Reset();
    };
    const auto set_ids_and_points = [&](
                                      const svtkIdType& edge_id, const svtkIdType& face_id) -> void {
      result->Points->SetPoint(edge_id, this->Points->GetPoint(face_id));
      result->PointIds->SetId(edge_id, this->PointIds->GetId(face_id));
    };
    this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  }

  return result;
}

/**\brief EvaluateLocation Given a point_id. This is required by Bezier because the interior points
 * are non-interpolatory .
 */
void svtkBezierQuadrilateral::EvaluateLocationProjectedNode(
  int& subId, const svtkIdType point_id, double x[3], double* weights)
{
  this->svtkHigherOrderQuadrilateral::SetParametricCoords();
  double pcoords[3];
  this->PointParametricCoordinates->GetPoint(this->PointIds->FindIdLocation(point_id), pcoords);
  this->svtkHigherOrderQuadrilateral::EvaluateLocation(subId, pcoords, x, weights);
}

/**\brief Populate the linear quadrilateral returned by GetApprox() with point-data from one
 * voxel-like interval of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkQuad* svtkBezierQuadrilateral::GetApproximateQuad(
  int subId, svtkDataArray* scalarsIn, svtkDataArray* scalarsOut)
{
  svtkQuad* approx = this->GetApprox();
  bool doScalars = (scalarsIn && scalarsOut);
  if (doScalars)
  {
    scalarsOut->SetNumberOfTuples(4);
  }
  int i, j, k;
  if (!this->SubCellCoordinatesFromId(i, j, k, subId))
  {
    svtkErrorMacro("Invalid subId " << subId);
    return nullptr;
  }
  // Get the point ids (and optionally scalars) for each of the 4 corners
  // in the approximating quadrilateral spanned by (i, i+1) x (j, j+1):
  for (svtkIdType ic = 0; ic < 4; ++ic)
  {
    const svtkIdType corner =
      this->PointIndexFromIJK(i + ((((ic + 1) / 2) % 2) ? 1 : 0), j + (((ic / 2) % 2) ? 1 : 0), 0);
    svtkVector3d cp;

    // Only the first four corners are interpolatory, we need to project the value of the other
    // nodes
    if (corner < 4)
    {
      this->Points->GetPoint(corner, cp.GetData());
    }
    else
    {
      this->SetParametricCoords();
      double pcoords[3];
      this->PointParametricCoordinates->GetPoint(corner, pcoords);
      int subIdtps;
      std::vector<double> weights(this->Points->GetNumberOfPoints());
      this->svtkHigherOrderQuadrilateral::EvaluateLocation(
        subIdtps, pcoords, cp.GetData(), weights.data());
    }
    approx->Points->SetPoint(ic, cp.GetData());
    approx->PointIds->SetId(ic, doScalars ? corner : this->PointIds->GetId(corner));
    if (doScalars)
    {
      scalarsOut->SetTuple(ic, scalarsIn->GetTuple(corner));
    }
  }
  return approx;
}

void svtkBezierQuadrilateral::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkBezierInterpolation::Tensor2ShapeFunctions(this->GetOrder(), pcoords, weights);

  // If the unit cell has rational weigths: weights_i = weights_i * rationalWeights / sum( weights_i
  // * rationalWeights )
  const bool has_rational_weights = RationalWeights->GetNumberOfTuples() > 0;
  if (has_rational_weights)
  {
    svtkIdType nPoints = this->GetPoints()->GetNumberOfPoints();
    double w = 0;
    for (svtkIdType idx = 0; idx < nPoints; ++idx)
    {
      weights[idx] *= RationalWeights->GetTuple1(idx);
      w += weights[idx];
    }
    const double one_over_rational_weight = 1. / w;
    for (svtkIdType idx = 0; idx < nPoints; ++idx)
      weights[idx] *= one_over_rational_weight;
  }
}

void svtkBezierQuadrilateral::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkBezierInterpolation::Tensor2ShapeDerivatives(this->GetOrder(), pcoords, derivs);
}

/**\brief Set the rational weight of the cell, given a svtkDataSet
 */
void svtkBezierQuadrilateral::SetRationalWeightsFromPointData(
  svtkPointData* point_data, const svtkIdType numPts)
{
  if (point_data->SetActiveAttribute(
        "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
  {
    svtkDataArray* v = point_data->GetRationalWeights();
    this->GetRationalWeights()->SetNumberOfTuples(numPts);
    for (svtkIdType i = 0; i < numPts; i++)
    {
      this->GetRationalWeights()->SetValue(i, v->GetTuple1(this->PointIds->GetId(i)));
    }
  }
  else
    this->GetRationalWeights()->Reset();
}

svtkDoubleArray* svtkBezierQuadrilateral::GetRationalWeights()
{
  return RationalWeights.Get();
}
svtkHigherOrderCurve* svtkBezierQuadrilateral::getEdgeCell()
{
  return EdgeCell;
}
