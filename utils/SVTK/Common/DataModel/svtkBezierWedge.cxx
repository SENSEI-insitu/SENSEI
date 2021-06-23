/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierWedge.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBezierWedge.h"

#include "svtkBezierCurve.h"
#include "svtkBezierInterpolation.h"
#include "svtkBezierQuadrilateral.h"
#include "svtkBezierTriangle.h"
#include "svtkCellData.h"
#include "svtkDoubleArray.h"
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

svtkStandardNewMacro(svtkBezierWedge);

svtkBezierWedge::svtkBezierWedge()
  : svtkHigherOrderWedge()
{
}

svtkBezierWedge::~svtkBezierWedge() = default;

void svtkBezierWedge::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkBezierWedge::GetEdge(int edgeId)
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

svtkCell* svtkBezierWedge::GetFace(int faceId)
{

  std::function<void(const svtkIdType&)> set_number_of_ids_and_points;
  std::function<void(const svtkIdType&, const svtkIdType&)> set_ids_and_points;

  if (faceId < 2)
  {
    svtkBezierTriangle* result = BdyTri;
    if (this->GetRationalWeights()->GetNumberOfTuples() > 0)
    {
      set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
        result->Points->SetNumberOfPoints(npts);
        result->PointIds->SetNumberOfIds(npts);
        result->GetRationalWeights()->SetNumberOfTuples(npts);
      };
      set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
        result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
        result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
        result->GetRationalWeights()->SetValue(
          faceId, this->GetRationalWeights()->GetValue(vol_id));
      };
    }
    else
    {
      set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
        result->Points->SetNumberOfPoints(npts);
        result->PointIds->SetNumberOfIds(npts);
        result->GetRationalWeights()->Reset();
      };
      set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
        result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
        result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
      };
    }
    this->GetTriangularFace(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
    return result;
  }
  else
  {
    svtkBezierQuadrilateral* result = BdyQuad;
    if (this->GetRationalWeights()->GetNumberOfTuples() > 0)
    {
      set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
        result->Points->SetNumberOfPoints(npts);
        result->PointIds->SetNumberOfIds(npts);
        result->GetRationalWeights()->SetNumberOfTuples(npts);
      };
      set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
        result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
        result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
        result->GetRationalWeights()->SetValue(
          faceId, this->GetRationalWeights()->GetValue(vol_id));
      };
    }
    else
    {
      set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
        result->Points->SetNumberOfPoints(npts);
        result->PointIds->SetNumberOfIds(npts);
        result->GetRationalWeights()->Reset();
      };
      set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
        result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
        result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
      };
    }
    this->GetQuadrilateralFace(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
    return result;
  }
}

/**\brief EvaluateLocation Given a point_id. This is required by Bezier because the interior points
 * are non-interpolatory .
 */
void svtkBezierWedge::EvaluateLocationProjectedNode(
  int& subId, const svtkIdType point_id, double x[3], double* weights)
{
  this->svtkHigherOrderWedge::SetParametricCoords();
  double pcoords[3];
  this->PointParametricCoordinates->GetPoint(this->PointIds->FindIdLocation(point_id), pcoords);
  this->svtkHigherOrderWedge::EvaluateLocation(subId, pcoords, x, weights);
}

void svtkBezierWedge::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkBezierInterpolation::WedgeShapeFunctions(
    this->GetOrder(), this->GetOrder()[3], pcoords, weights);

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

void svtkBezierWedge::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkBezierInterpolation::WedgeShapeDerivatives(
    this->GetOrder(), this->GetOrder()[3], pcoords, derivs);
}

/**\brief Set the rational weight of the cell, given a svtkDataSet
 */
void svtkBezierWedge::SetRationalWeightsFromPointData(
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

svtkDoubleArray* svtkBezierWedge::GetRationalWeights()
{
  return RationalWeights.Get();
}
svtkHigherOrderQuadrilateral* svtkBezierWedge::getBdyQuad()
{
  return BdyQuad;
};
svtkHigherOrderTriangle* svtkBezierWedge::getBdyTri()
{
  return BdyTri;
};
svtkHigherOrderCurve* svtkBezierWedge::getEdgeCell()
{
  return EdgeCell;
}
svtkHigherOrderInterpolation* svtkBezierWedge::getInterp()
{
  return Interp;
};
