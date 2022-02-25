/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCylindricalTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCylindricalTransform
 * @brief   cylindrical to rectangular coords and back
 *
 * svtkCylindricalTransform will convert (r,theta,z) coordinates to
 * (x,y,z) coordinates and back again.  The angles are given in radians.
 * By default, it converts cylindrical coordinates to rectangular, but
 * GetInverse() returns a transform that will do the opposite.  The
 * equation that is used is x = r*cos(theta), y = r*sin(theta), z = z.
 * @warning
 * This transform is not well behaved along the line x=y=0 (i.e. along
 * the z-axis)
 * @sa
 * svtkSphericalTransform svtkGeneralTransform
 */

#ifndef svtkCylindricalTransform_h
#define svtkCylindricalTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkWarpTransform.h"

class SVTKCOMMONTRANSFORMS_EXPORT svtkCylindricalTransform : public svtkWarpTransform
{
public:
  static svtkCylindricalTransform* New();
  svtkTypeMacro(svtkCylindricalTransform, svtkWarpTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Make another transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  svtkCylindricalTransform();
  ~svtkCylindricalTransform() override;

  /**
   * Copy this transform from another of the same type.
   */
  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  //@{
  /**
   * Internal functions for calculating the transformation.
   */
  void ForwardTransformPoint(const float in[3], float out[3]) override;
  void ForwardTransformPoint(const double in[3], double out[3]) override;
  //@}

  void ForwardTransformDerivative(const float in[3], float out[3], float derivative[3][3]) override;
  void ForwardTransformDerivative(
    const double in[3], double out[3], double derivative[3][3]) override;

  void InverseTransformPoint(const float in[3], float out[3]) override;
  void InverseTransformPoint(const double in[3], double out[3]) override;

  void InverseTransformDerivative(const float in[3], float out[3], float derivative[3][3]) override;
  void InverseTransformDerivative(
    const double in[3], double out[3], double derivative[3][3]) override;

private:
  svtkCylindricalTransform(const svtkCylindricalTransform&) = delete;
  void operator=(const svtkCylindricalTransform&) = delete;
};

#endif
