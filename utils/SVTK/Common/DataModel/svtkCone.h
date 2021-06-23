/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCone.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCone
 * @brief   implicit function for a cone
 *
 * svtkCone computes the implicit function and function gradient for a cone.
 * svtkCone is a concrete implementation of svtkImplicitFunction. The cone vertex
 * is located at the origin with axis of rotation coincident with x-axis. (Use
 * the superclass' svtkImplicitFunction transformation matrix if necessary to
 * reposition.) The angle specifies the angle between the axis of rotation
 * and the side of the cone.
 *
 * @warning
 * The cone is infinite in extent. To truncate the cone use the
 * svtkImplicitBoolean in combination with clipping planes.
 */

#ifndef svtkCone_h
#define svtkCone_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkCone : public svtkImplicitFunction
{
public:
  /**
   * Construct cone with angle of 45 degrees.
   */
  static svtkCone* New();

  svtkTypeMacro(svtkCone, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Evaluate cone equation.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate cone normal.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Set/Get the cone angle (expressed in degrees).
   */
  svtkSetClampMacro(Angle, double, 0.0, 89.0);
  svtkGetMacro(Angle, double);
  //@}

protected:
  svtkCone();
  ~svtkCone() override {}

  double Angle;

private:
  svtkCone(const svtkCone&) = delete;
  void operator=(const svtkCone&) = delete;
};

#endif
