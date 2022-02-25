/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeInterpolation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

  =========================================================================*/
#include "svtkLagrangeInterpolation.h"

#include "svtkDoubleArray.h"
#include "svtkLagrangeTriangle.h"
#include "svtkLagrangeWedge.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include <array>
#include <vector>

svtkStandardNewMacro(svtkLagrangeInterpolation);

svtkLagrangeInterpolation::svtkLagrangeInterpolation()
  : svtkHigherOrderInterpolation()
{
}

svtkLagrangeInterpolation::~svtkLagrangeInterpolation() = default;

void svtkLagrangeInterpolation::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

/// Evaluate 1-D shape functions for the given \a order at the given \a pcoord (in [0,1]).
void svtkLagrangeInterpolation::EvaluateShapeFunctions(
  const int order, const double pcoord, double* shape)
{
  int j, k;
  double v = order * pcoord;
  for (j = 0; j <= order; ++j)
  {
    shape[j] = 1.;
    for (k = 0; k <= order; ++k)
    {
      if (j != k)
      { // FIXME: (j - k) could be a register decremented inside the k loop:
        // and even better: normalization 1./(j - k) could be pre-computed and stored
        // somehow for each order that is actually used, removing division operations.
        shape[j] *= (v - k) / (j - k);
      }
    }
  }
}

/// Evaluate 1-D shape functions and their derivatives for the given \a order at the given \a pcoord
/// (in [0,1]).
void svtkLagrangeInterpolation::EvaluateShapeAndGradient(
  int order, double pcoord, double* shape, double* derivs)
{
  int j, k, q;
  double dtmp;
  double v = order * pcoord;
  for (j = 0; j <= order; ++j)
  {
    // std::cout << "ShapeDeriv j = " << j << "  v = " << v << "\n";
    shape[j] = 1.;
    derivs[j] = 0.;
    for (k = 0; k <= order; ++k)
    {
      if (j != k)
      { // FIXME: (j - k) could be a register decremented inside the k loop:
        // and even better: normalization could be pre-computed and stored
        // somehow for each order that is actually used.
        shape[j] *= (v - k) / (j - k);

        // Now compute the derivative of shape[j]; we use the differentiation
        // rule d/dx(a * b) = a * d/dx(b) + b * d/dx(a) instead of faster
        // methods because it keeps the truncation error low(er):
        dtmp = 1.;
        // std::cout << "           k = " << k;
        for (q = 0; q <= order; ++q)
        {
          if (q == j)
          {
            continue;
          }
          dtmp *= (q == k ? 1. : (v - q)) / (j - q);
          // std::cout << "  q " << q << " dtmp *= " << ((q == k ? 1. : (v - q)) / (j - q));
        }
        derivs[j] += order * dtmp;
      }
    }
  }
}

int svtkLagrangeInterpolation::Tensor1ShapeFunctions(
  const int order[1], const double* pcoords, double* shape)
{
  return svtkHigherOrderInterpolation::Tensor1ShapeFunctions(
    order, pcoords, shape, svtkLagrangeInterpolation::EvaluateShapeFunctions);
}

int svtkLagrangeInterpolation::Tensor1ShapeDerivatives(
  const int order[1], const double* pcoords, double* derivs)
{
  return svtkHigherOrderInterpolation::Tensor1ShapeDerivatives(
    order, pcoords, derivs, svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}

/// Quadrilateral shape function computation
int svtkLagrangeInterpolation::Tensor2ShapeFunctions(
  const int order[2], const double pcoords[3], double* shape)
{
  return svtkHigherOrderInterpolation::Tensor2ShapeFunctions(
    order, pcoords, shape, svtkLagrangeInterpolation::EvaluateShapeFunctions);
}

// Quadrilateral shape-function derivatives
int svtkLagrangeInterpolation::Tensor2ShapeDerivatives(
  const int order[2], const double pcoords[3], double* derivs)
{
  return svtkHigherOrderInterpolation::Tensor2ShapeDerivatives(
    order, pcoords, derivs, svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}

/// Hexahedral shape function computation
int svtkLagrangeInterpolation::Tensor3ShapeFunctions(
  const int order[3], const double pcoords[3], double* shape)
{
  return svtkHigherOrderInterpolation::Tensor3ShapeFunctions(
    order, pcoords, shape, svtkLagrangeInterpolation::EvaluateShapeFunctions);
}

int svtkLagrangeInterpolation::Tensor3ShapeDerivatives(
  const int order[3], const double pcoords[3], double* derivs)
{
  return svtkHigherOrderInterpolation::Tensor3ShapeDerivatives(
    order, pcoords, derivs, svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}

void svtkLagrangeInterpolation::Tensor3EvaluateDerivative(const int order[3], const double* pcoords,
  svtkPoints* points, const double* fieldVals, int fieldDim, double* fieldDerivs)
{
  this->svtkHigherOrderInterpolation::Tensor3EvaluateDerivative(order, pcoords, points, fieldVals,
    fieldDim, fieldDerivs, svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}

/// Wedge shape function computation
void svtkLagrangeInterpolation::WedgeShapeFunctions(
  const int order[3], const svtkIdType numberOfPoints, const double pcoords[3], double* shape)
{
  static svtkNew<svtkLagrangeTriangle> tri;
  svtkHigherOrderInterpolation::WedgeShapeFunctions(
    order, numberOfPoints, pcoords, shape, *tri, svtkLagrangeInterpolation::EvaluateShapeFunctions);
}

/// Wedge shape-function derivative evaluation
void svtkLagrangeInterpolation::WedgeShapeDerivatives(
  const int order[3], const svtkIdType numberOfPoints, const double pcoords[3], double* derivs)
{
  static svtkNew<svtkLagrangeTriangle> tri;
  svtkHigherOrderInterpolation::WedgeShapeDerivatives(order, numberOfPoints, pcoords, derivs, *tri,
    svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}

void svtkLagrangeInterpolation::WedgeEvaluate(const int order[3], const svtkIdType numberOfPoints,
  const double* pcoords, double* fieldVals, int fieldDim, double* fieldAtPCoords)
{
  static svtkNew<svtkLagrangeTriangle> tri;
  this->svtkHigherOrderInterpolation::WedgeEvaluate(order, numberOfPoints, pcoords, fieldVals,
    fieldDim, fieldAtPCoords, *tri, svtkLagrangeInterpolation::EvaluateShapeFunctions);
}

void svtkLagrangeInterpolation::WedgeEvaluateDerivative(const int order[3], const double* pcoords,
  svtkPoints* points, const double* fieldVals, int fieldDim, double* fieldDerivs)
{
  static svtkNew<svtkLagrangeTriangle> tri;
  this->svtkHigherOrderInterpolation::WedgeEvaluateDerivative(order, pcoords, points, fieldVals,
    fieldDim, fieldDerivs, *tri, svtkLagrangeInterpolation::EvaluateShapeAndGradient);
}
