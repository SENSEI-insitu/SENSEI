/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSphere.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSphere
 * @brief   implicit function for a sphere
 *
 * svtkSphere computes the implicit function and/or gradient for a sphere.
 * svtkSphere is a concrete implementation of svtkImplicitFunction. Additional
 * methods are available for sphere-related computations, such as computing
 * bounding spheres for a set of points, or set of spheres.
 */

#ifndef svtkSphere_h
#define svtkSphere_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkSphere : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkSphere, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct sphere with center at (0,0,0) and radius=0.5.
   */
  static svtkSphere* New();

  //@{
  /**
   * Evaluate sphere equation ((x-x0)^2 + (y-y0)^2 + (z-z0)^2) - R^2.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate sphere gradient.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Set / get the radius of the sphere. The default is 0.5.
   */
  svtkSetMacro(Radius, double);
  svtkGetMacro(Radius, double);
  //@}

  //@{
  /**
   * Set / get the center of the sphere. The default is (0,0,0).
   */
  svtkSetVector3Macro(Center, double);
  svtkGetVectorMacro(Center, double, 3);
  //@}

  /**
   * Quick evaluation of the sphere equation ((x-x0)^2 + (y-y0)^2 + (z-z0)^2) - R^2.
   */
  static double Evaluate(double center[3], double R, double x[3])
  {
    return (x[0] - center[0]) * (x[0] - center[0]) + (x[1] - center[1]) * (x[1] - center[1]) +
      (x[2] - center[2]) * (x[2] - center[2]) - R * R;
  }

  //@{
  /**
   * Create a bounding sphere from a set of points. The set of points is
   * defined by an array of doubles, in the order of x-y-z (which repeats for
   * each point).  An optional hints array provides a guess for the initial
   * bounding sphere; the two values in the hints array are the two points
   * expected to be the furthest apart. The output sphere consists of a
   * center (x-y-z) and a radius.
   */
  static void ComputeBoundingSphere(
    float* pts, svtkIdType numPts, float sphere[4], svtkIdType hints[2]);
  static void ComputeBoundingSphere(
    double* pts, svtkIdType numPts, double sphere[4], svtkIdType hints[2]);
  //@}

  //@{
  /**
   * Create a bounding sphere from a set of spheres. The set of input spheres
   * is defined by an array of pointers to spheres. Each sphere is defined by
   * the 4-tuple: center(x-y-z)+radius. An optional hints array provides a
   * guess for the initial bounding sphere; the two values in the hints array
   * are the two spheres expected to be the furthest apart. The output sphere
   * consists of a center (x-y-z) and a radius.
   */
  static void ComputeBoundingSphere(
    float** spheres, svtkIdType numSpheres, float sphere[4], svtkIdType hints[2]);
  static void ComputeBoundingSphere(
    double** spheres, svtkIdType numSpheres, double sphere[4], svtkIdType hints[2]);
  //@}

protected:
  svtkSphere();
  ~svtkSphere() override {}

  double Radius;
  double Center[3];

private:
  svtkSphere(const svtkSphere&) = delete;
  void operator=(const svtkSphere&) = delete;
};

#endif
