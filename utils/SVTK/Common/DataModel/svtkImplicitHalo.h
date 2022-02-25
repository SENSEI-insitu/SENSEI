/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitHalo.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitHalo
 * @brief   implicit function for an halo
 *
 * svtkImplicitHalo evaluates to 1.0 for each position in the sphere of a
 * given center and radius Radius*(1-FadeOut). It evaluates to 0.0 for each
 * position out the sphere of a given Center and radius Radius. It fades out
 * linearly from 1.0 to 0.0 for points in a radius from Radius*(1-FadeOut) to
 * Radius.
 * svtkImplicitHalo is a concrete implementation of svtkImplicitFunction.
 * It is useful as an input to
 * svtkSampleFunction to generate an 2D image of an halo. It is used this way by
 * svtkShadowMapPass.
 * @warning
 * It does not implement the gradient.
 */

#ifndef svtkImplicitHalo_h
#define svtkImplicitHalo_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitHalo : public svtkImplicitFunction
{
public:
  /**
   * Center=(0.0,0.0,0.0), Radius=1.0, FadeOut=0.01
   */
  static svtkImplicitHalo* New();

  svtkTypeMacro(svtkImplicitHalo, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Evaluate the equation.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate normal. Not implemented.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Radius of the sphere.
   */
  svtkSetMacro(Radius, double);
  svtkGetMacro(Radius, double);
  //@}

  //@{
  /**
   * Center of the sphere.
   */
  svtkSetVector3Macro(Center, double);
  svtkGetVector3Macro(Center, double);
  //@}

  //@{
  /**
   * FadeOut ratio. Valid values are between 0.0 and 1.0.
   */
  svtkSetMacro(FadeOut, double);
  svtkGetMacro(FadeOut, double);
  //@}

protected:
  svtkImplicitHalo();
  ~svtkImplicitHalo() override;

  double Radius;
  double Center[3];
  double FadeOut;

private:
  svtkImplicitHalo(const svtkImplicitHalo&) = delete;
  void operator=(const svtkImplicitHalo&) = delete;
};

#endif
