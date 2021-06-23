/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierCurve.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBezierCurve.h"

#include "svtkBezierInterpolation.h"
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

svtkStandardNewMacro(svtkBezierCurve);

svtkBezierCurve::svtkBezierCurve()
  : svtkHigherOrderCurve()
{
}

svtkBezierCurve::~svtkBezierCurve() = default;

void svtkBezierCurve::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

/**\brief EvaluateLocation Given a point_id. This is required by Bezier because the interior points
 * are non-interpolatory .
 */
void svtkBezierCurve::EvaluateLocationProjectedNode(
  int& subId, const svtkIdType point_id, double x[3], double* weights)
{
  this->svtkHigherOrderCurve::SetParametricCoords();
  double pcoords[3];
  this->PointParametricCoordinates->GetPoint(this->PointIds->FindIdLocation(point_id), pcoords);
  this->svtkHigherOrderCurve::EvaluateLocation(subId, pcoords, x, weights);
}

/**\brief Set the rational weight of the cell, given a svtkDataSet
 */
void svtkBezierCurve::SetRationalWeightsFromPointData(
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

/**\brief Populate the linear segment returned by GetApprox() with point-data from one voxel-like
 * intervals of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkLine* svtkBezierCurve::GetApproximateLine(
  int subId, svtkDataArray* scalarsIn, svtkDataArray* scalarsOut)
{
  svtkLine* approx = this->GetApprox();
  bool doScalars = (scalarsIn && scalarsOut);
  if (doScalars)
  {
    scalarsOut->SetNumberOfTuples(2);
  }
  int i;
  if (!this->SubCellCoordinatesFromId(i, subId))
  {
    svtkErrorMacro("Invalid subId " << subId);
    return nullptr;
  }
  // Get the point ids (and optionally scalars) for each of the 2 corners
  // in the approximating line spanned by (i, i+1):
  for (svtkIdType ic = 0; ic < 2; ++ic)
  {
    const svtkIdType corner = this->PointIndexFromIJK(i + ic, 0, 0);
    svtkVector3d cp;
    // Only the first four corners are interpolatory, we need to project the value of the other
    // nodes
    if (corner < 2)
    {
      this->Points->GetPoint(corner, cp.GetData());
    }
    else
    {
      this->SetParametricCoords();
      double pcoords[3];
      this->PointParametricCoordinates->GetPoint(corner, pcoords);
      int subIdtps;
      const int numtripts = (this->Order[0] + 1);
      std::vector<double> weights(numtripts);
      this->svtkHigherOrderCurve::EvaluateLocation(subIdtps, pcoords, cp.GetData(), weights.data());
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

void svtkBezierCurve::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkBezierInterpolation::Tensor1ShapeFunctions(this->GetOrder(), pcoords, weights);

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

void svtkBezierCurve::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkBezierInterpolation::Tensor1ShapeDerivatives(this->GetOrder(), pcoords, derivs);
}
svtkDoubleArray* svtkBezierCurve::GetRationalWeights()
{
  return RationalWeights.Get();
}
