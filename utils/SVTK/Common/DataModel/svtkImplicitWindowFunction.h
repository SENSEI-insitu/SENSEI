/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitWindowFunction.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitWindowFunction
 * @brief   implicit function maps another implicit function to lie within a specified range
 *
 * svtkImplicitWindowFunction is used to modify the output of another
 * implicit function to lie within a specified "window", or function
 * range. This can be used to add "thickness" to cutting or clipping
 * functions.
 *
 * This class works as follows. First, it evaluates the function value of the
 * user-specified implicit function. Then, based on the window range specified,
 * it maps the function value into the window values specified.
 *
 *
 * @sa
 * svtkImplicitFunction
 */

#ifndef svtkImplicitWindowFunction_h
#define svtkImplicitWindowFunction_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitWindowFunction : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkImplicitWindowFunction, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct object with window range (0,1) and window values (0,1).
   */
  static svtkImplicitWindowFunction* New();

  //@{
  /**
   * Evaluate window function.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate window function gradient. Just return implicit function gradient.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Specify an implicit function to operate on.
   */
  virtual void SetImplicitFunction(svtkImplicitFunction*);
  svtkGetObjectMacro(ImplicitFunction, svtkImplicitFunction);
  //@}

  //@{
  /**
   * Specify the range of function values which are considered to lie within
   * the window. WindowRange[0] is assumed to be less than WindowRange[1].
   */
  svtkSetVector2Macro(WindowRange, double);
  svtkGetVectorMacro(WindowRange, double, 2);
  //@}

  //@{
  /**
   * Specify the range of output values that the window range is mapped
   * into. This is effectively a scaling and shifting of the original
   * function values.
   */
  svtkSetVector2Macro(WindowValues, double);
  svtkGetVectorMacro(WindowValues, double, 2);
  //@}

  /**
   * Override modified time retrieval because of object dependencies.
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Participate in garbage collection.
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

protected:
  svtkImplicitWindowFunction();
  ~svtkImplicitWindowFunction() override;

  void ReportReferences(svtkGarbageCollector*) override;

  svtkImplicitFunction* ImplicitFunction;
  double WindowRange[2];
  double WindowValues[2];

private:
  svtkImplicitWindowFunction(const svtkImplicitWindowFunction&) = delete;
  void operator=(const svtkImplicitWindowFunction&) = delete;
};

#endif
