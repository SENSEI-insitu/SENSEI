/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitDataSet
 * @brief   treat a dataset as if it were an implicit function
 *
 * svtkImplicitDataSet treats any type of dataset as if it were an
 * implicit function. This means it computes a function value and
 * gradient. svtkImplicitDataSet is a concrete implementation of
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
 * Any type of dataset can be used as an implicit function as long as it
 * has scalar data associated with it.
 *
 * @sa
 * svtkImplicitFunction svtkImplicitVolume svtkClipPolyData svtkCutter
 * svtkImplicitWindowFunction
 */

#ifndef svtkImplicitDataSet_h
#define svtkImplicitDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitDataSet : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkImplicitDataSet, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct an svtkImplicitDataSet with no initial dataset; the OutValue
   * set to a large negative number; and the OutGradient set to (0,0,1).
   */
  static svtkImplicitDataSet* New();

  /**
   * Return the MTime also considering the DataSet dependency.
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Evaluate the implicit function. This returns the interpolated scalar value
   * at x[3].
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate implicit function gradient.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Set / get the dataset used for the implicit function evaluation.
   */
  virtual void SetDataSet(svtkDataSet*);
  svtkGetObjectMacro(DataSet, svtkDataSet);
  //@}

  //@{
  /**
   * Set / get the function value to use for points outside of the dataset.
   */
  svtkSetMacro(OutValue, double);
  svtkGetMacro(OutValue, double);
  //@}

  //@{
  /**
   * Set / get the function gradient to use for points outside of the dataset.
   */
  svtkSetVector3Macro(OutGradient, double);
  svtkGetVector3Macro(OutGradient, double);
  //@}

protected:
  svtkImplicitDataSet();
  ~svtkImplicitDataSet() override;

  void ReportReferences(svtkGarbageCollector*) override;

  svtkDataSet* DataSet;
  double OutValue;
  double OutGradient[3];

  double* Weights; // used to compute interpolation weights
  int Size;        // keeps track of length of weights array

private:
  svtkImplicitDataSet(const svtkImplicitDataSet&) = delete;
  void operator=(const svtkImplicitDataSet&) = delete;
};

#endif
