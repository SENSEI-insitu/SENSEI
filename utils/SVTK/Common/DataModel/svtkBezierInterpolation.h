/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierInterpolation.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkBezierInterpolation
// .SECTION Description
// .SECTION See Also
#ifndef svtkBezierInterpolation_h
#define svtkBezierInterpolation_h

#include "svtkCommonDataModelModule.h" // For export macro.
#include "svtkHigherOrderInterpolation.h"
#include "svtkSmartPointer.h" // For API.
#include "svtkVector.h"       // For flattenSimplex

#include <vector> // For scratch storage.

// Define this to include support for a "complete" (21- vs 18-point) wedge.
#define SVTK_21_POINT_WEDGE true

class svtkPoints;
class svtkVector2i;
class svtkVector3d;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierInterpolation : public svtkHigherOrderInterpolation
{
public:
  static svtkBezierInterpolation* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;
  svtkTypeMacro(svtkBezierInterpolation, svtkHigherOrderInterpolation);

  // see Geometrically Exact and Analysis Suitable Mesh Generation Using Rational Bernstein–Bezier
  // Elements https://scholar.colorado.edu/cgi/viewcontent.cgi?article=1170&context=mcen_gradetds
  // Chapter 3, pg 25. given a dimmension ( 2 triangle, 3 tetrahedron ) and the degree of the
  // simplex flatten a simplicial bezier function's coordinate to an integer
  static int flattenSimplex(const int dim, const int deg, const svtkVector3i coord);

  // given a dimmension ( 2 triangle, 3 tetrahedron ) and the degree of the simplex,
  // unflatten a simplicial bezier function integer to a simplicial coordinate
  static svtkVector3i unflattenSimplex(const int dim, const int deg, const svtkIdType flat);

  // simplicial version of deCasteljau
  static void deCasteljauSimplex(
    const int dim, const int deg, const double* pcoords, double* weights);
  static void deCasteljauSimplexDeriv(
    const int dim, const int deg, const double* pcoords, double* weights);

  static void EvaluateShapeFunctions(int order, double pcoord, double* shape);
  static void EvaluateShapeAndGradient(int order, double pcoord, double* shape, double* grad);

  static int Tensor1ShapeFunctions(const int order[1], const double* pcoords, double* shape);
  static int Tensor1ShapeDerivatives(const int order[1], const double* pcoords, double* derivs);

  static int Tensor2ShapeFunctions(const int order[2], const double* pcoords, double* shape);
  static int Tensor2ShapeDerivatives(const int order[2], const double* pcoords, double* derivs);

  static int Tensor3ShapeFunctions(const int order[3], const double* pcoords, double* shape);
  static int Tensor3ShapeDerivatives(const int order[3], const double* pcoords, double* derivs);

  virtual void Tensor3EvaluateDerivative(const int order[3], const double* pcoords,
    svtkPoints* points, const double* fieldVals, int fieldDim, double* fieldDerivs) override;

  static void WedgeShapeFunctions(
    const int order[3], const svtkIdType numberOfPoints, const double* pcoords, double* shape);
  static void WedgeShapeDerivatives(
    const int order[3], const svtkIdType numberOfPoints, const double* pcoords, double* derivs);

  virtual void WedgeEvaluate(const int order[3], const svtkIdType numberOfPoints,
    const double* pcoords, double* fieldVals, int fieldDim, double* fieldAtPCoords) override;

  virtual void WedgeEvaluateDerivative(const int order[3], const double* pcoords, svtkPoints* points,
    const double* fieldVals, int fieldDim, double* fieldDerivs) override;

protected:
  svtkBezierInterpolation();
  ~svtkBezierInterpolation() override;

private:
  svtkBezierInterpolation(const svtkBezierInterpolation&) = delete;
  void operator=(const svtkBezierInterpolation&) = delete;
};

#endif // svtkBezierInterpolation_h
