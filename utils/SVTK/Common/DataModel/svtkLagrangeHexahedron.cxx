/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeHexahedron.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLagrangeHexahedron.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkHexahedron.h"
#include "svtkIdList.h"
#include "svtkLagrangeCurve.h"
#include "svtkLagrangeInterpolation.h"
#include "svtkLagrangeQuadrilateral.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

svtkStandardNewMacro(svtkLagrangeHexahedron);

svtkLagrangeHexahedron::svtkLagrangeHexahedron()
  : svtkHigherOrderHexahedron()
{
}

svtkLagrangeHexahedron::~svtkLagrangeHexahedron() = default;

void svtkLagrangeHexahedron::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkLagrangeHexahedron::GetEdge(int edgeId)
{
  svtkLagrangeCurve* result = EdgeCell;
  const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
    result->Points->SetNumberOfPoints(npts);
    result->PointIds->SetNumberOfIds(npts);
  };
  const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
    result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
    result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
  };

  this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  return result;
}

svtkCell* svtkLagrangeHexahedron::GetFace(int faceId)
{
  svtkLagrangeQuadrilateral* result = FaceCell;

  const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
    result->Points->SetNumberOfPoints(npts);
    result->PointIds->SetNumberOfIds(npts);
  };
  const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
    result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
    result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
  };

  this->SetFaceIdsAndPoints(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
  return result;
}

/**\brief Populate the linear hex returned by GetApprox() with point-data from one voxel-like
 * intervals of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkHexahedron* svtkLagrangeHexahedron::GetApproximateHex(
  int subId, svtkDataArray* scalarsIn, svtkDataArray* scalarsOut)
{
  svtkHexahedron* approx = this->GetApprox();
  bool doScalars = (scalarsIn && scalarsOut);
  if (doScalars)
  {
    scalarsOut->SetNumberOfTuples(8);
  }
  int i, j, k;
  if (!this->SubCellCoordinatesFromId(i, j, k, subId))
  {
    svtkErrorMacro("Invalid subId " << subId);
    return nullptr;
  }
  // Get the point coordinates (and optionally scalars) for each of the 8 corners
  // in the approximating hexahedron spanned by (i, i+1) x (j, j+1) x (k, k+1):
  for (svtkIdType ic = 0; ic < 8; ++ic)
  {
    const svtkIdType corner = this->PointIndexFromIJK(
      i + ((((ic + 1) / 2) % 2) ? 1 : 0), j + (((ic / 2) % 2) ? 1 : 0), k + ((ic / 4) ? 1 : 0));
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

void svtkLagrangeHexahedron::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkLagrangeInterpolation::Tensor3ShapeFunctions(this->GetOrder(), pcoords, weights);
}

void svtkLagrangeHexahedron::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkLagrangeInterpolation::Tensor3ShapeDerivatives(this->GetOrder(), pcoords, derivs);
}
svtkHigherOrderCurve* svtkLagrangeHexahedron::getEdgeCell()
{
  return EdgeCell;
}
svtkHigherOrderQuadrilateral* svtkLagrangeHexahedron::getFaceCell()
{
  return FaceCell;
}
svtkHigherOrderInterpolation* svtkLagrangeHexahedron::getInterp()
{
  return Interp;
};
