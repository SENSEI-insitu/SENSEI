/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdentityTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIdentityTransform
 * @brief   a transform that doesn't do anything
 *
 * svtkIdentityTransform is a transformation which will simply pass coordinate
 * data unchanged.  All other transform types can also do this, however,
 * the svtkIdentityTransform does so with much greater efficiency.
 * @sa
 * svtkLinearTransform
 */

#ifndef svtkIdentityTransform_h
#define svtkIdentityTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkLinearTransform.h"

class SVTKCOMMONTRANSFORMS_EXPORT svtkIdentityTransform : public svtkLinearTransform
{
public:
  static svtkIdentityTransform* New();

  svtkTypeMacro(svtkIdentityTransform, svtkLinearTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Apply the transformation to a series of points, and append the
   * results to outPts.
   */
  void TransformPoints(svtkPoints* inPts, svtkPoints* outPts) override;

  /**
   * Apply the transformation to a series of normals, and append the
   * results to outNms.
   */
  void TransformNormals(svtkDataArray* inNms, svtkDataArray* outNms) override;

  /**
   * Apply the transformation to a series of vectors, and append the
   * results to outVrs.
   */
  void TransformVectors(svtkDataArray* inVrs, svtkDataArray* outVrs) override;

  /**
   * Apply the transformation to a combination of points, normals
   * and vectors.
   */
  void TransformPointsNormalsVectors(svtkPoints* inPts, svtkPoints* outPts, svtkDataArray* inNms,
    svtkDataArray* outNms, svtkDataArray* inVrs, svtkDataArray* outVrs, int nOptionalVectors = 0,
    svtkDataArray** inVrsArr = nullptr, svtkDataArray** outVrsArr = nullptr) override;

  // Invert the transformation.  This doesn't do anything to the
  // identity transformation.
  void Inverse() override {}

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
   * This will calculate the transformation without calling Update.
   * Meant for use only within other SVTK classes.
   */
  void InternalTransformNormal(const float in[3], float out[3]) override;
  void InternalTransformNormal(const double in[3], double out[3]) override;
  //@}

  //@{
  /**
   * This will calculate the transformation without calling Update.
   * Meant for use only within other SVTK classes.
   */
  void InternalTransformVector(const float in[3], float out[3]) override;
  void InternalTransformVector(const double in[3], double out[3]) override;
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

  /**
   * Make a transform of the same type.  This will actually
   * return the same transform.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  svtkIdentityTransform();
  ~svtkIdentityTransform() override;

  void InternalDeepCopy(svtkAbstractTransform* t) override;

private:
  svtkIdentityTransform(const svtkIdentityTransform&) = delete;
  void operator=(const svtkIdentityTransform&) = delete;
};

#endif
