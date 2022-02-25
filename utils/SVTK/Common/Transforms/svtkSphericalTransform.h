/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSphericalTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSphericalTransform
 * @brief   spherical to rectangular coords and back
 *
 * svtkSphericalTransform will convert (r,phi,theta) coordinates to
 * (x,y,z) coordinates and back again.  The angles are given in radians.
 * By default, it converts spherical coordinates to rectangular, but
 * GetInverse() returns a transform that will do the opposite.  The equation
 * that is used is x = r*sin(phi)*cos(theta), y = r*sin(phi)*sin(theta),
 * z = r*cos(phi).
 * @warning
 * This transform is not well behaved along the line x=y=0 (i.e. along
 * the z-axis)
 * @sa
 * svtkCylindricalTransform svtkGeneralTransform
 */

#ifndef svtkSphericalTransform_h
#define svtkSphericalTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkWarpTransform.h"

class SVTKCOMMONTRANSFORMS_EXPORT svtkSphericalTransform : public svtkWarpTransform
{
public:
  static svtkSphericalTransform* New();
  svtkTypeMacro(svtkSphericalTransform, svtkWarpTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Make another transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  svtkSphericalTransform();
  ~svtkSphericalTransform() override;

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
  svtkSphericalTransform(const svtkSphericalTransform&) = delete;
  void operator=(const svtkSphericalTransform&) = delete;
};

#endif
