/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSuperquadric.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSuperquadric
 * @brief   implicit function for a Superquadric
 *
 * svtkSuperquadric computes the implicit function and function gradient
 * for a superquadric. svtkSuperquadric is a concrete implementation of
 * svtkImplicitFunction.  The superquadric is centered at Center and axes
 * of rotation is along the y-axis. (Use the superclass'
 * svtkImplicitFunction transformation matrix if necessary to reposition.)
 * Roundness parameters (PhiRoundness and ThetaRoundness) control
 * the shape of the superquadric.  The Toroidal boolean controls whether
 * a toroidal superquadric is produced.  If so, the Thickness parameter
 * controls the thickness of the toroid:  0 is the thinnest allowable
 * toroid, and 1 has a minimum sized hole.  The Scale parameters allow
 * the superquadric to be scaled in x, y, and z (normal vectors are correctly
 * generated in any case).  The Size parameter controls size of the
 * superquadric.
 *
 * This code is based on "Rigid physically based superquadrics", A. H. Barr,
 * in "Graphics Gems III", David Kirk, ed., Academic Press, 1992.
 *
 * @warning
 * The Size and Thickness parameters control coefficients of superquadric
 * generation, and may do not exactly describe the size of the superquadric.
 *
 */

#ifndef svtkSuperquadric_h
#define svtkSuperquadric_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

#define SVTK_MIN_SUPERQUADRIC_THICKNESS 1e-4

class SVTKCOMMONDATAMODEL_EXPORT svtkSuperquadric : public svtkImplicitFunction
{
public:
  /**
   * Construct with superquadric radius of 0.5, toroidal off, center at 0.0,
   * scale (1,1,1), size 0.5, phi roundness 1.0, and theta roundness 0.0.
   */
  static svtkSuperquadric* New();

  svtkTypeMacro(svtkSuperquadric, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // ImplicitFunction interface
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Set the center of the superquadric. Default is 0,0,0.
   */
  svtkSetVector3Macro(Center, double);
  svtkGetVectorMacro(Center, double, 3);
  //@}

  //@{
  /**
   * Set the scale factors of the superquadric. Default is 1,1,1.
   */
  svtkSetVector3Macro(Scale, double);
  svtkGetVectorMacro(Scale, double, 3);
  //@}

  //@{
  /**
   * Set/Get Superquadric ring thickness (toroids only).
   * Changing thickness maintains the outside diameter of the toroid.
   */
  svtkGetMacro(Thickness, double);
  svtkSetClampMacro(Thickness, double, SVTK_MIN_SUPERQUADRIC_THICKNESS, 1.0);
  //@}

  //@{
  /**
   * Set/Get Superquadric north/south roundness.
   * Values range from 0 (rectangular) to 1 (circular) to higher orders.
   */
  svtkGetMacro(PhiRoundness, double);
  void SetPhiRoundness(double e);
  //@}

  //@{
  /**
   * Set/Get Superquadric east/west roundness.
   * Values range from 0 (rectangular) to 1 (circular) to higher orders.
   */
  svtkGetMacro(ThetaRoundness, double);
  void SetThetaRoundness(double e);
  //@}

  //@{
  /**
   * Set/Get Superquadric isotropic size.
   */
  svtkSetMacro(Size, double);
  svtkGetMacro(Size, double);
  //@}

  //@{
  /**
   * Set/Get whether or not the superquadric is toroidal (1) or ellipsoidal (0).
   */
  svtkBooleanMacro(Toroidal, svtkTypeBool);
  svtkGetMacro(Toroidal, svtkTypeBool);
  svtkSetMacro(Toroidal, svtkTypeBool);
  //@}

protected:
  svtkSuperquadric();
  ~svtkSuperquadric() override {}

  svtkTypeBool Toroidal;
  double Thickness;
  double Size;
  double PhiRoundness;
  double ThetaRoundness;
  double Center[3];
  double Scale[3];

private:
  svtkSuperquadric(const svtkSuperquadric&) = delete;
  void operator=(const svtkSuperquadric&) = delete;
};

#endif
