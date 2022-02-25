/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeCurve.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLagrangeCurve.h"

#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkIdList.h"
#include "svtkLagrangeInterpolation.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTriangle.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

svtkStandardNewMacro(svtkLagrangeCurve);
svtkLagrangeCurve::svtkLagrangeCurve()
  : svtkHigherOrderCurve()
{
}

svtkLagrangeCurve::~svtkLagrangeCurve() = default;

void svtkLagrangeCurve::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

/**\brief Populate the linear segment returned by GetApprox() with point-data from one voxel-like
 * intervals of this cell.
 *
 * Ensure that you have called GetOrder() before calling this method
 * so that this->Order is up to date. This method does no checking
 * before using it to map connectivity-array offsets.
 */
svtkLine* svtkLagrangeCurve::GetApproximateLine(
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

void svtkLagrangeCurve::InterpolateFunctions(const double pcoords[3], double* weights)
{
  svtkLagrangeInterpolation::Tensor1ShapeFunctions(this->GetOrder(), pcoords, weights);
}

void svtkLagrangeCurve::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  svtkLagrangeInterpolation::Tensor1ShapeDerivatives(this->GetOrder(), pcoords, derivs);
}
