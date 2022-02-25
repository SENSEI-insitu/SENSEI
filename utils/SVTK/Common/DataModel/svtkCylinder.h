/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCylinder.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCylinder
 * @brief   implicit function for a cylinder
 *
 * svtkCylinder computes the implicit function and function gradient
 * for a cylinder using F(r)=r^2-Radius^2. svtkCylinder is a concrete
 * implementation of svtkImplicitFunction. By default the Cylinder is
 * centered at the origin and the axis of rotation is along the
 * y-axis. You can redefine the center and axis of rotation by setting
 * the Center and Axis data members. (Note that it is also possible to
 * use the superclass' svtkImplicitFunction transformation matrix if
 * necessary to reposition by using FunctionValue() and
 * FunctionGradient().)
 *
 * @warning
 * The cylinder is infinite in extent. To truncate the cylinder in
 * modeling operations use the svtkImplicitBoolean in combination with
 * clipping planes.
 */

#ifndef svtkCylinder_h
#define svtkCylinder_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkCylinder : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkCylinder, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct cylinder radius of 0.5; centered at origin with axis
   * along y coordinate axis.
   */
  static svtkCylinder* New();

  //@{
  /**
   * Evaluate cylinder equation F(r) = r^2 - Radius^2.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate cylinder function gradient.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Set/Get the cylinder radius.
   */
  svtkSetMacro(Radius, double);
  svtkGetMacro(Radius, double);
  //@}

  //@{
  /**
   * Set/Get the cylinder center.
   */
  svtkSetVector3Macro(Center, double);
  svtkGetVector3Macro(Center, double);
  //@}

  //@{
  /**
   * Set/Get the axis of the cylinder. If the axis is not specified as
   * a unit vector, it will be normalized. If zero-length axis vector
   * is used as input to this method, it will be ignored.
   */
  void SetAxis(double ax, double ay, double az);
  void SetAxis(double a[3]);
  svtkGetVector3Macro(Axis, double);
  //@}

protected:
  svtkCylinder();
  ~svtkCylinder() override {}

  double Radius;
  double Center[3];
  double Axis[3];

private:
  svtkCylinder(const svtkCylinder&) = delete;
  void operator=(const svtkCylinder&) = delete;
};

#endif
