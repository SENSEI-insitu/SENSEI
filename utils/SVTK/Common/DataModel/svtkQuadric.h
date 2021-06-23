/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadric.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadric
 * @brief   evaluate implicit quadric function
 *
 * svtkQuadric evaluates the quadric function F(x,y,z) = a0*x^2 + a1*y^2 +
 * a2*z^2 + a3*x*y + a4*y*z + a5*x*z + a6*x + a7*y + a8*z + a9. svtkQuadric is
 * a concrete implementation of svtkImplicitFunction.
 */

#ifndef svtkQuadric_h
#define svtkQuadric_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadric : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkQuadric, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct quadric with all coefficients = 1.
   */
  static svtkQuadric* New();

  //@{
  /**
   * Evaluate quadric equation.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate the gradient to the quadric equation.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Set / get the 10 coefficients of the quadric equation.
   */
  void SetCoefficients(double a[10]);
  void SetCoefficients(double a0, double a1, double a2, double a3, double a4, double a5, double a6,
    double a7, double a8, double a9);
  svtkGetVectorMacro(Coefficients, double, 10);
  //@}

protected:
  svtkQuadric();
  ~svtkQuadric() override {}

  double Coefficients[10];

private:
  svtkQuadric(const svtkQuadric&) = delete;
  void operator=(const svtkQuadric&) = delete;
};

#endif
