/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSpheres.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSpheres
 * @brief   implicit function for a set of spheres
 *
 * svtkSpheres computes the implicit function and function gradient for a set
 * of spheres. The spheres are combined via a union operation (i.e., the
 * minimum value from the evaluation of all spheres is taken).
 *
 * The function value is the distance of a point to the closest sphere, with
 * negative values interior to the spheres, positive outside the spheres, and
 * distance=0 on the spheres surface.  The function gradient is the sphere
 * normal at the function value.
 *
 * @sa
 * svtkPlanes svtkImplicitBoolean
 */

#ifndef svtkSpheres_h
#define svtkSpheres_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkSphere;
class svtkPoints;
class svtkDataArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkSpheres : public svtkImplicitFunction
{
public:
  //@{
  /**
   * Standard methods for instantiation, type information, and printing.
   */
  static svtkSpheres* New();
  svtkTypeMacro(svtkSpheres, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Evaluate spheres equations. Return largest value when evaluating all
   * sphere equations.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate spheres gradient. Gradients point towards the outside of the
   * spheres.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Specify a list of points defining sphere centers.
   */
  virtual void SetCenters(svtkPoints*);
  svtkGetObjectMacro(Centers, svtkPoints);
  //@}

  //@{
  /**
   * Specify a list of radii for the spheres. There is a one-to-one
   * correspondence between sphere points and sphere radii.
   */
  void SetRadii(svtkDataArray* radii);
  svtkGetObjectMacro(Radii, svtkDataArray);
  //@}

  /**
   * Return the number of spheres in the set of spheres.
   */
  int GetNumberOfSpheres();

  /**
   * Create and return a pointer to a svtkSphere object at the ith
   * position. Asking for a sphere outside the allowable range returns
   * nullptr.  This method always returns the same object.  Alternatively use
   * GetSphere(int i, svtkSphere *sphere) to update a user supplied sphere.
   */
  svtkSphere* GetSphere(int i);

  /**
   * If i is within the allowable range, mutates the given sphere's
   * Center and Radius to match the svtkSphere object at the ith
   * position. Does nothing if i is outside the allowable range.
   */
  void GetSphere(int i, svtkSphere* sphere);

protected:
  svtkSpheres();
  ~svtkSpheres() override;

  svtkPoints* Centers;
  svtkDataArray* Radii;
  svtkSphere* Sphere;

private:
  svtkSpheres(const svtkSpheres&) = delete;
  void operator=(const svtkSpheres&) = delete;
};

#endif
