/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHomogeneousTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHomogeneousTransform
 * @brief   superclass for homogeneous transformations
 *
 * svtkHomogeneousTransform provides a generic interface for homogeneous
 * transformations, i.e. transformations which can be represented by
 * multiplying a 4x4 matrix with a homogeneous coordinate.
 * @sa
 * svtkPerspectiveTransform svtkLinearTransform svtkIdentityTransform
 */

#ifndef svtkHomogeneousTransform_h
#define svtkHomogeneousTransform_h

#include "svtkAbstractTransform.h"
#include "svtkCommonTransformsModule.h" // For export macro

class svtkMatrix4x4;

class SVTKCOMMONTRANSFORMS_EXPORT svtkHomogeneousTransform : public svtkAbstractTransform
{
public:
  svtkTypeMacro(svtkHomogeneousTransform, svtkAbstractTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Apply the transformation to a series of points, and append the
   * results to outPts.
   */
  void TransformPoints(svtkPoints* inPts, svtkPoints* outPts) override;

  /**
   * Apply the transformation to a combination of points, normals
   * and vectors.
   */
  void TransformPointsNormalsVectors(svtkPoints* inPts, svtkPoints* outPts, svtkDataArray* inNms,
    svtkDataArray* outNms, svtkDataArray* inVrs, svtkDataArray* outVrs, int nOptionalVectors = 0,
    svtkDataArray** inVrsArr = nullptr, svtkDataArray** outVrsArr = nullptr) override;

  /**
   * Get a copy of the internal transformation matrix.  The
   * transform is Updated first, to guarantee that the matrix
   * is valid.
   */
  void GetMatrix(svtkMatrix4x4* m);

  /**
   * Get a pointer to an internal svtkMatrix4x4 that represents
   * the transformation.  An Update() is called on the transform
   * to ensure that the matrix is up-to-date when you get it.
   * You should not store the matrix pointer anywhere because it
   * might become stale.
   */
  svtkMatrix4x4* GetMatrix()
  {
    this->Update();
    return this->Matrix;
  }

  /**
   * Just like GetInverse(), but includes typecast to svtkHomogeneousTransform.
   */
  svtkHomogeneousTransform* GetHomogeneousInverse()
  {
    return static_cast<svtkHomogeneousTransform*>(this->GetInverse());
  }

  //@{
  /**
   * This will calculate the transformation without calling Update.
   * Meant for use only within other SVTK classes.
   */
  void InternalTransformPoint(const float in[3], float out[3]) override;
  void InternalTransformPoint(const double in[3], double out[3]) override;
  //@}

  //@{
  /**
   * This will calculate the transformation as well as its derivative
   * without calling Update.  Meant for use only within other SVTK
   * classes.
   */
  void InternalTransformDerivative(
    const float in[3], float out[3], float derivative[3][3]) override;
  void InternalTransformDerivative(
    const double in[3], double out[3], double derivative[3][3]) override;
  //@}

protected:
  svtkHomogeneousTransform();
  ~svtkHomogeneousTransform() override;

  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  svtkMatrix4x4* Matrix;

private:
  svtkHomogeneousTransform(const svtkHomogeneousTransform&) = delete;
  void operator=(const svtkHomogeneousTransform&) = delete;
};

#endif
