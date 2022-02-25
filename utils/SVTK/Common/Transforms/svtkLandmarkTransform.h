/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLandmarkTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLandmarkTransform
 * @brief   a linear transform specified by two corresponding point sets
 *
 * A svtkLandmarkTransform is defined by two sets of landmarks, the
 * transform computed gives the best fit mapping one onto the other, in a
 * least squares sense. The indices are taken to correspond, so point 1
 * in the first set will get mapped close to point 1 in the second set,
 * etc. Call SetSourceLandmarks and SetTargetLandmarks to specify the two
 * sets of landmarks, ensure they have the same number of points.
 * @warning
 * Whenever you add, subtract, or set points you must call Modified()
 * on the svtkPoints object, or the transformation might not update.
 * @sa
 * svtkLinearTransform
 */

#ifndef svtkLandmarkTransform_h
#define svtkLandmarkTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkLinearTransform.h"

#define SVTK_LANDMARK_RIGIDBODY 6
#define SVTK_LANDMARK_SIMILARITY 7
#define SVTK_LANDMARK_AFFINE 12

class SVTKCOMMONTRANSFORMS_EXPORT svtkLandmarkTransform : public svtkLinearTransform
{
public:
  static svtkLandmarkTransform* New();

  svtkTypeMacro(svtkLandmarkTransform, svtkLinearTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Specify the source and target landmark sets. The two sets must have
   * the same number of points.  If you add or change points in these objects,
   * you must call Modified() on them or the transformation might not update.
   */
  void SetSourceLandmarks(svtkPoints* points);
  void SetTargetLandmarks(svtkPoints* points);
  svtkGetObjectMacro(SourceLandmarks, svtkPoints);
  svtkGetObjectMacro(TargetLandmarks, svtkPoints);
  //@}

  //@{
  /**
   * Set the number of degrees of freedom to constrain the solution to.
   * Rigidbody (SVTK_LANDMARK_RIGIDBODY): rotation and translation only.
   * Similarity (SVTK_LANDMARK_SIMILARITY): rotation, translation and
   * isotropic scaling.
   * Affine (SVTK_LANDMARK_AFFINE): collinearity is preserved.
   * Ratios of distances along a line are preserved.
   * The default is similarity.
   */
  svtkSetMacro(Mode, int);
  void SetModeToRigidBody() { this->SetMode(SVTK_LANDMARK_RIGIDBODY); }
  void SetModeToSimilarity() { this->SetMode(SVTK_LANDMARK_SIMILARITY); }
  void SetModeToAffine() { this->SetMode(SVTK_LANDMARK_AFFINE); }
  //@}

  //@{
  /**
   * Get the current transformation mode.
   */
  svtkGetMacro(Mode, int);
  const char* GetModeAsString();
  //@}

  /**
   * Invert the transformation.  This is done by switching the
   * source and target landmarks.
   */
  void Inverse() override;

  /**
   * Get the MTime.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Make another transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  svtkLandmarkTransform();
  ~svtkLandmarkTransform() override;

  // Update the matrix from the quaternion.
  void InternalUpdate() override;

  /**
   * This method does no type checking, use DeepCopy instead.
   */
  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  svtkPoints* SourceLandmarks;
  svtkPoints* TargetLandmarks;

  int Mode;

private:
  svtkLandmarkTransform(const svtkLandmarkTransform&) = delete;
  void operator=(const svtkLandmarkTransform&) = delete;
};

inline const char* svtkLandmarkTransform::GetModeAsString()
{
  switch (this->Mode)
  {
    case SVTK_LANDMARK_RIGIDBODY:
      return "RigidBody";
    case SVTK_LANDMARK_SIMILARITY:
      return "Similarity";
    case SVTK_LANDMARK_AFFINE:
      return "Affine";
    default:
      return "Unrecognized";
  }
}

#endif
