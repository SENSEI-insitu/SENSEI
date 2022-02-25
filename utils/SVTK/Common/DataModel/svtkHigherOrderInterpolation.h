/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderInterpolation.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkHigherOrderInterpolation
// .SECTION Description
// .SECTION See Also
#ifndef svtkHigherOrderInterpolation_h
#define svtkHigherOrderInterpolation_h

#include "svtkCommonDataModelModule.h" // For export macro.
#include "svtkObject.h"
#include "svtkSmartPointer.h" // For API.

#include <vector> // For scratch storage.

// Define this to include support for a "complete" (21- vs 18-point) wedge.
#define SVTK_21_POINT_WEDGE true

class svtkPoints;
class svtkVector2i;
class svtkVector3d;
class svtkHigherOrderTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderInterpolation : public svtkObject
{
public:
  // static svtkHigherOrderInterpolation* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;
  svtkTypeMacro(svtkHigherOrderInterpolation, svtkObject);

  static int Tensor1ShapeFunctions(const int order[1], const double* pcoords, double* shape,
    void (*function_evaluate_shape_functions)(int, double, double*));
  static int Tensor1ShapeDerivatives(const int order[1], const double* pcoords, double* derivs,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  static int Tensor2ShapeFunctions(const int order[2], const double* pcoords, double* shape,
    void (*function_evaluate_shape_functions)(int, double, double*));
  static int Tensor2ShapeDerivatives(const int order[2], const double* pcoords, double* derivs,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  static int Tensor3ShapeFunctions(const int order[3], const double* pcoords, double* shape,
    void (*function_evaluate_shape_functions)(int, double, double*));
  static int Tensor3ShapeDerivatives(const int order[3], const double* pcoords, double* derivs,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  virtual void Tensor3EvaluateDerivative(const int order[3], const double* pcoords,
    svtkPoints* points, const double* fieldVals, int fieldDim, double* fieldDerivs) = 0;

  void Tensor3EvaluateDerivative(const int order[3], const double* pcoords, svtkPoints* points,
    const double* fieldVals, int fieldDim, double* fieldDerivs,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  static void WedgeShapeFunctions(const int order[3], const svtkIdType numberOfPoints,
    const double* pcoords, double* shape, svtkHigherOrderTriangle& tri,
    void (*function_evaluate_shape_functions)(int, double, double*));
  static void WedgeShapeDerivatives(const int order[3], const svtkIdType numberOfPoints,
    const double* pcoords, double* derivs, svtkHigherOrderTriangle& tri,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  /**
   * Compute the inverse of the Jacobian and put the values in `inverse`. Returns
   * 1 for success and 0 for failure (i.e. couldn't invert the Jacobian).
   */
  int JacobianInverse(svtkPoints* points, const double* derivs, double** inverse);
  int JacobianInverseWedge(svtkPoints* points, const double* derivs, double** inverse);

  virtual void WedgeEvaluate(const int order[3], const svtkIdType numberOfPoints,
    const double* pcoords, double* fieldVals, int fieldDim, double* fieldAtPCoords) = 0;

  void WedgeEvaluate(const int order[3], const svtkIdType numberOfPoints, const double* pcoords,
    double* fieldVals, int fieldDim, double* fieldAtPCoords, svtkHigherOrderTriangle& tri,
    void (*function_evaluate_shape_functions)(int, double, double*));

  virtual void WedgeEvaluateDerivative(const int order[3], const double* pcoords, svtkPoints* points,
    const double* fieldVals, int fieldDim, double* fieldDerivs) = 0;

  void WedgeEvaluateDerivative(const int order[3], const double* pcoords, svtkPoints* points,
    const double* fieldVals, int fieldDim, double* fieldDerivs, svtkHigherOrderTriangle& tri,
    void (*function_evaluate_shape_and_gradient)(int, double, double*, double*));

  static svtkVector3d GetParametricHexCoordinates(int vertexId);
  static svtkVector2i GetPointIndicesBoundingHexEdge(int edgeId);
  static int GetVaryingParameterOfHexEdge(int edgeId);
  static svtkVector2i GetFixedParametersOfHexEdge(int edgeId);

  static const int* GetPointIndicesBoundingHexFace(int faceId) SVTK_SIZEHINT(4);
  static const int* GetEdgeIndicesBoundingHexFace(int faceId) SVTK_SIZEHINT(4);
  static svtkVector2i GetVaryingParametersOfHexFace(int faceId);
  static int GetFixedParameterOfHexFace(int faceId);

  static svtkVector3d GetParametricWedgeCoordinates(int vertexId);
  static svtkVector2i GetPointIndicesBoundingWedgeEdge(int edgeId);
  static int GetVaryingParameterOfWedgeEdge(int edgeId);
  static svtkVector2i GetFixedParametersOfWedgeEdge(int edgeId);

  static const int* GetPointIndicesBoundingWedgeFace(int faceId) SVTK_SIZEHINT(4);
  static const int* GetEdgeIndicesBoundingWedgeFace(int faceId) SVTK_SIZEHINT(4);
  static svtkVector2i GetVaryingParametersOfWedgeFace(int faceId);
  static int GetFixedParameterOfWedgeFace(int faceId);

  static void AppendCurveCollocationPoints(svtkSmartPointer<svtkPoints>& pts, const int order[1]);
  static void AppendQuadrilateralCollocationPoints(
    svtkSmartPointer<svtkPoints>& pts, const int order[2]);
  static void AppendHexahedronCollocationPoints(
    svtkSmartPointer<svtkPoints>& pts, const int order[3]);
  static void AppendWedgeCollocationPoints(svtkSmartPointer<svtkPoints>& pts, const int order[3]);

  template <int N>
  static int NumberOfIntervals(const int order[N]);

protected:
  svtkHigherOrderInterpolation();
  ~svtkHigherOrderInterpolation() override;

  void PrepareForOrder(const int order[3], const svtkIdType numberOfPoints);

  std::vector<double> ShapeSpace;
  std::vector<double> DerivSpace;

private:
  svtkHigherOrderInterpolation(const svtkHigherOrderInterpolation&) = delete;
  void operator=(const svtkHigherOrderInterpolation&) = delete;
};

template <int N>
int svtkHigherOrderInterpolation::NumberOfIntervals(const int order[N])
{
  int ni = 1;
  for (int n = 0; n < N; ++n)
  {
    ni *= order[n];
  }
  return ni;
}

#endif // svtkHigherOrderInterpolation_h
