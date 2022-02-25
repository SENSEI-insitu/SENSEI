/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMatrixToLinearTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkMatrixToLinearTransform
 * @brief   convert a matrix to a transform
 *
 * This is a very simple class which allows a svtkMatrix4x4 to be used in
 * place of a svtkLinearTransform or svtkAbstractTransform.  For example,
 * if you use it as a proxy between a matrix and svtkTransformPolyDataFilter
 * then any modifications to the matrix will automatically be reflected in
 * the output of the filter.
 * @sa
 * svtkTransform svtkMatrix4x4 svtkMatrixToHomogeneousTransform
 */

#ifndef svtkMatrixToLinearTransform_h
#define svtkMatrixToLinearTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkLinearTransform.h"

class svtkMatrix4x4;

class SVTKCOMMONTRANSFORMS_EXPORT svtkMatrixToLinearTransform : public svtkLinearTransform
{
public:
  static svtkMatrixToLinearTransform* New();
  svtkTypeMacro(svtkMatrixToLinearTransform, svtkLinearTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set the input matrix.  Any modifications to the matrix will be
   * reflected in the transformation.
   */
  virtual void SetInput(svtkMatrix4x4*);
  svtkGetObjectMacro(Input, svtkMatrix4x4);
  //@}

  /**
   * The input matrix is left as-is, but the transformation matrix
   * is inverted.
   */
  void Inverse() override;

  /**
   * Get the MTime: this is the bit of magic that makes everything work.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Make a new transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  svtkMatrixToLinearTransform();
  ~svtkMatrixToLinearTransform() override;

  void InternalUpdate() override;
  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  int InverseFlag;
  svtkMatrix4x4* Input;

private:
  svtkMatrixToLinearTransform(const svtkMatrixToLinearTransform&) = delete;
  void operator=(const svtkMatrixToLinearTransform&) = delete;
};

#endif
