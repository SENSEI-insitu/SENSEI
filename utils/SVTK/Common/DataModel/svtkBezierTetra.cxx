/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierTetra.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBezierTetra.h"
#include "svtkBezierInterpolation.h"

#include "svtkBezierCurve.h"
#include "svtkBezierTriangle.h"
#include "svtkDataSet.h"
#include "svtkDoubleArray.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTetra.h"

svtkStandardNewMacro(svtkBezierTetra);
//----------------------------------------------------------------------------
svtkBezierTetra::svtkBezierTetra()
  : svtkHigherOrderTetra()
{
}

//----------------------------------------------------------------------------
svtkBezierTetra::~svtkBezierTetra() = default;

void svtkBezierTetra::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkBezierTetra::GetEdge(int edgeId)
{
  svtkBezierCurve* result = EdgeCell;
  if (this->GetRationalWeights()->GetNumberOfTuples() > 0)
  {
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
      result->GetRationalWeights()->SetNumberOfTuples(npts);
    };
    const auto set_ids_and_points = [&](const svtkIdType& edge_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(edge_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(edge_id, this->PointIds->GetId(vol_id));
      result->GetRationalWeights()->SetValue(edge_id, this->GetRationalWeights()->GetValue(vol_id));
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
    const auto set_ids_and_points = [&](const svtkIdType& edge_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(edge_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(edge_id, this->PointIds->GetId(vol_id));
    };
    this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  }

  return result;
}

svtkCell* svtkBezierTetra::GetFace(int faceId)
{
  svtkBezierTriangle* result = FaceCell;
  if (this->GetRationalWeights()->GetNumberOfTuples() > 0)
  {
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
      result->GetRationalWeights()->SetNumberOfTuples(npts);
    };
    const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
      result->GetRationalWeights()->SetValue(face_id, this->GetRationalWeights()->GetValue(vol_id));
    };
    this->SetFaceIdsAndPoints(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
  }
  else
  {
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
      result->GetRationalWeights()->Reset();
    };
    const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
    };
    this->SetFaceIdsAndPoints(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
  }

  return result;
}

/**\brief EvaluateLocation Given a point_id. This is required by Bezier because the interior points
 * are non-interpolatory .
 */
void svtkBezierTetra::EvaluateLocationProjectedNode(
  int& subId, const svtkIdType point_id, double x[3], double* weights)
{
  this->svtkHigherOrderTetra::SetParametricCoords();
  double pcoords[3];
  this->PointParametricCoordinates->GetPoint(this->PointIds->FindIdLocation(point_id), pcoords);
  this->svtkHigherOrderTetra::EvaluateLocation(subId, pcoords, x, weights);
}

/**\brief Set the rational weight of the cell, given a svtkDataSet
 */
void svtkBezierTetra::SetRationalWeightsFromPointData(
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

//----------------------------------------------------------------------------
void svtkBezierTetra::InterpolateFunctions(const double pcoords[3], double* weights)
{
  const int dim = 3;
  const int deg = GetOrder();
  const svtkIdType nPoints = this->GetPoints()->GetNumberOfPoints();
  std::vector<double> coeffs(nPoints, 0.0);

  svtkBezierInterpolation::deCasteljauSimplex(dim, deg, pcoords, &coeffs[0]);
  for (svtkIdType i = 0; i < nPoints; ++i)
  {
    svtkVector3i bv = svtkBezierInterpolation::unflattenSimplex(dim, deg, i);
    svtkIdType lbv[4] = { bv[0], bv[1], bv[2], deg - bv[0] - bv[1] - bv[2] };
    weights[Index(lbv, deg)] = coeffs[i];
  }
  // If the unit cell has rational weigths: weights_i = weights_i * rationalWeights / sum( weights_i
  // * rationalWeights )
  const bool has_rational_weights = RationalWeights->GetNumberOfTuples() > 0;
  if (has_rational_weights)
  {
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

//----------------------------------------------------------------------------
void svtkBezierTetra::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  const int dim = 3;
  const int deg = GetOrder();
  const svtkIdType nPoints = this->GetPoints()->GetNumberOfPoints();

  std::vector<double> coeffs(nPoints, 0.0);
  svtkBezierInterpolation::deCasteljauSimplexDeriv(dim, deg, pcoords, &coeffs[0]);
  for (svtkIdType i = 0; i < nPoints; ++i)
  {
    svtkVector3i bv = svtkBezierInterpolation::unflattenSimplex(dim, deg, i);
    svtkIdType lbv[4] = { bv[0], bv[1], bv[2], deg - bv[0] - bv[1] - bv[2] };
    for (int j = 0; j < dim; ++j)
    {
      derivs[j * nPoints + Index(lbv, deg)] = coeffs[j * nPoints + i];
    }
  }
}

svtkDoubleArray* svtkBezierTetra::GetRationalWeights()
{
  return RationalWeights.Get();
}
svtkHigherOrderCurve* svtkBezierTetra::getEdgeCell()
{
  return EdgeCell;
}
svtkHigherOrderTriangle* svtkBezierTetra::getFaceCell()
{
  return FaceCell;
}
