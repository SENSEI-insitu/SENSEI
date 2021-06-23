/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeInterpolation.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkLagrangeInterpolation
// .SECTION Description
// .SECTION See Also
#ifndef svtkLagrangeInterpolation_h
#define svtkLagrangeInterpolation_h

#include "svtkCommonDataModelModule.h" // For export macro.
#include "svtkHigherOrderInterpolation.h"
#include "svtkSmartPointer.h" // For API.

#include <vector> // For scratch storage.

// Define this to include support for a "complete" (21- vs 18-point) wedge.
#define SVTK_21_POINT_WEDGE true

class svtkPoints;
class svtkVector2i;
class svtkVector3d;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeInterpolation : public svtkHigherOrderInterpolation
{
public:
  static svtkLagrangeInterpolation* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;
  svtkTypeMacro(svtkLagrangeInterpolation, svtkHigherOrderInterpolation);

  static void EvaluateShapeFunctions(const int order, const double pcoord, double* shape);
  static void EvaluateShapeAndGradient(
    const int order, const double pcoord, double* shape, double* grad);

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
  svtkLagrangeInterpolation();
  ~svtkLagrangeInterpolation() override;

private:
  svtkLagrangeInterpolation(const svtkLagrangeInterpolation&) = delete;
  void operator=(const svtkLagrangeInterpolation&) = delete;
};

#endif // svtkLagrangeInterpolation_h
