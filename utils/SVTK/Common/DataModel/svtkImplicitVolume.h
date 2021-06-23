/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitVolume.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitVolume
 * @brief   treat a volume as if it were an implicit function
 *
 * svtkImplicitVolume treats a volume (e.g., structured point dataset)
 * as if it were an implicit function. This means it computes a function
 * value and gradient. svtkImplicitVolume is a concrete implementation of
 * svtkImplicitFunction.
 *
 * svtkImplicitDataSet computes the function (at the point x) by performing
 * cell interpolation. That is, it finds the cell containing x, and then
 * uses the cell's interpolation functions to compute an interpolated
 * scalar value at x. (A similar approach is used to find the
 * gradient, if requested.) Points outside of the dataset are assigned
 * the value of the ivar OutValue, and the gradient value OutGradient.
 *
 * @warning
 * The input volume data is only updated when GetMTime() is called.
 * Works for 3D structured points datasets, 0D-2D datasets won't work properly.
 *
 * @sa
 * svtkImplicitFunction svtkImplicitDataSet svtkClipPolyData svtkCutter
 * svtkImplicitWindowFunction
 */

#ifndef svtkImplicitVolume_h
#define svtkImplicitVolume_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkIdList;
class svtkImageData;

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitVolume : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkImplicitVolume, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct an svtkImplicitVolume with no initial volume; the OutValue
   * set to a large negative number; and the OutGradient set to (0,0,1).
   */
  static svtkImplicitVolume* New();

  /**
   * Returns the mtime also considering the volume.  This also calls Update
   * on the volume, and it therefore must be called before the function is
   * evaluated.
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Evaluate the ImplicitVolume. This returns the interpolated scalar value
   * at x[3].
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate ImplicitVolume gradient.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Specify the volume for the implicit function.
   */
  virtual void SetVolume(svtkImageData*);
  svtkGetObjectMacro(Volume, svtkImageData);
  //@}

  //@{
  /**
   * Set the function value to use for points outside of the dataset.
   */
  svtkSetMacro(OutValue, double);
  svtkGetMacro(OutValue, double);
  //@}

  //@{
  /**
   * Set the function gradient to use for points outside of the dataset.
   */
  svtkSetVector3Macro(OutGradient, double);
  svtkGetVector3Macro(OutGradient, double);
  //@}

protected:
  svtkImplicitVolume();
  ~svtkImplicitVolume() override;

  svtkImageData* Volume; // the structured points
  double OutValue;
  double OutGradient[3];
  // to replace a static
  svtkIdList* PointIds;

private:
  svtkImplicitVolume(const svtkImplicitVolume&) = delete;
  void operator=(const svtkImplicitVolume&) = delete;
};

#endif
