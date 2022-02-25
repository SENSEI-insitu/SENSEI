/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeQuadrilateral.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLagrangeQuadrilateral.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkIdList.h"
#include "svtkLagrangeCurve.h"
#include "svtkLagrangeInterpolation.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkQuad.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

svtkStandardNewMacro(svtkLagrangeQuadrilateral);

svtkLagrangeQuadrilateral::svtkLagrangeQuadrilateral()
  : svtkHigherOrderQuadrilateral()
{
}

svtkLagrangeQuadrilateral::~svtkLagrangeQuadrilateral() = default;

void svtkLagrangeQuadrilateral::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkLagrangeQuadrilateral::GetEdge(int edgeId)
{
  svtkLagrangeCurve* result = EdgeCell;
  const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
    result->Points->SetNumberOfPoints(npts);
    result->PointIds->SetNumberOfIds(npts);
  };
  const auto set_ids_and_points = [&](const svtkIdType& edge_id, const svtkIdType& face_id) -> void {
    result->Points->SetPoint(edge_id, this->Points->GetPoint(face_id));
    result->PointIds->SetId(edge_id, this->PointIds->GetId(face_id));
  };

  this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  return result;
}

/**\brief Populate the linear quadrilateral returned by GetApprox() with point-data from one
 * voxel-like interval of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkQuad* svtkLagrangeQuadrilateral::GetApproximateQuad(
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
    this->Points->GetPoint(corner, cp.GetData());
    approx->Points->SetPoint(ic, cp.GetData());
    approx->PointIds->SetId(ic, doScalars ? corner : this->PointIds->GetId(corner));
    if (doScalars)
    {
      scalarsOut->SetTuple(ic, scalarsIn->GetTuple(corner));
    }
  }
  return approx;
}

void svtkLagrangeQuadrilateral::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkLagrangeInterpolation::Tensor2ShapeFunctions(this->GetOrder(), pcoords, weights);
}

void svtkLagrangeQuadrilateral::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkLagrangeInterpolation::Tensor2ShapeDerivatives(this->GetOrder(), pcoords, derivs);
}

svtkHigherOrderCurve* svtkLagrangeQuadrilateral::getEdgeCell()
{
  return EdgeCell;
}
