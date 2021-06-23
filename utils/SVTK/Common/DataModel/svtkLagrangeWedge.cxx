/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeWedge.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLagrangeWedge.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkIdList.h"
#include "svtkLagrangeCurve.h"
#include "svtkLagrangeInterpolation.h"
#include "svtkLagrangeQuadrilateral.h"
#include "svtkLagrangeTriangle.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"
#include "svtkWedge.h"

svtkStandardNewMacro(svtkLagrangeWedge);

svtkLagrangeWedge::svtkLagrangeWedge()
  : svtkHigherOrderWedge()
{
}

svtkLagrangeWedge::~svtkLagrangeWedge() = default;

void svtkLagrangeWedge::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkCell* svtkLagrangeWedge::GetEdge(int edgeId)
{
  svtkLagrangeCurve* result = EdgeCell;
  const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
    result->Points->SetNumberOfPoints(npts);
    result->PointIds->SetNumberOfIds(npts);
  };
  const auto set_ids_and_points = [&](const svtkIdType& edge_id, const svtkIdType& vol_id) -> void {
    result->Points->SetPoint(edge_id, this->Points->GetPoint(vol_id));
    result->PointIds->SetId(edge_id, this->PointIds->GetId(vol_id));
  };
  this->SetEdgeIdsAndPoints(edgeId, set_number_of_ids_and_points, set_ids_and_points);
  return result;
}

svtkCell* svtkLagrangeWedge::GetFace(int faceId)
{
  // If faceId = 0 or 1, triangular face, else if 2, 3, or 4, quad face.
  if (faceId < 2)
  {
    svtkLagrangeTriangle* result = BdyTri;
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
    };
    const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
    };
    this->GetTriangularFace(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
    return result;
  }
  else
  {
    svtkLagrangeQuadrilateral* result = BdyQuad;
    const auto set_number_of_ids_and_points = [&](const svtkIdType& npts) -> void {
      result->Points->SetNumberOfPoints(npts);
      result->PointIds->SetNumberOfIds(npts);
    };
    const auto set_ids_and_points = [&](const svtkIdType& face_id, const svtkIdType& vol_id) -> void {
      result->Points->SetPoint(face_id, this->Points->GetPoint(vol_id));
      result->PointIds->SetId(face_id, this->PointIds->GetId(vol_id));
    };
    this->GetQuadrilateralFace(result, faceId, set_number_of_ids_and_points, set_ids_and_points);
    return result;
  }
}

void svtkLagrangeWedge::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkLagrangeInterpolation::WedgeShapeFunctions(
    this->GetOrder(), this->GetOrder()[3], pcoords, weights);
}

void svtkLagrangeWedge::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkLagrangeInterpolation::WedgeShapeDerivatives(
    this->GetOrder(), this->GetOrder()[3], pcoords, derivs);
}

svtkHigherOrderQuadrilateral* svtkLagrangeWedge::getBdyQuad()
{
  return BdyQuad;
};
svtkHigherOrderTriangle* svtkLagrangeWedge::getBdyTri()
{
  return BdyTri;
};
svtkHigherOrderCurve* svtkLagrangeWedge::getEdgeCell()
{
  return EdgeCell;
}
svtkHigherOrderInterpolation* svtkLagrangeWedge::getInterp()
{
  return Interp;
};
