/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitSum.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitSum
 * @brief   implicit sum of other implicit functions
 *
 * svtkImplicitSum produces a linear combination of other implicit functions.
 * The contribution of each function is weighted by a scalar coefficient.
 * The NormalizeByWeight option normalizes the output so that the
 * scalar weights add up to 1. Note that this function gives accurate
 * sums and gradients only if the input functions are linear.
 */

#ifndef svtkImplicitSum_h
#define svtkImplicitSum_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkDoubleArray;
class svtkImplicitFunctionCollection;

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitSum : public svtkImplicitFunction
{
public:
  static svtkImplicitSum* New();

  svtkTypeMacro(svtkImplicitSum, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Evaluate implicit function using current functions and weights.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate gradient of the weighted sum of functions.  Input functions
   * should be linear.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  /**
   * Override modified time retrieval because of object dependencies.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Add another implicit function to the list of functions, along with a
   * weighting factor.
   */
  void AddFunction(svtkImplicitFunction* in, double weight);

  /**
   * Add another implicit function to the list of functions, weighting it by
   * a factor of 1.
   */
  void AddFunction(svtkImplicitFunction* in) { this->AddFunction(in, 1.0); }

  /**
   * Remove all functions from the list.
   */
  void RemoveAllFunctions();

  /**
   * Set the weight (coefficient) of the given function to be weight.
   */
  void SetFunctionWeight(svtkImplicitFunction* f, double weight);

  //@{
  /**
   * When calculating the function and gradient values of the
   * composite function, setting NormalizeByWeight on will divide the
   * final result by the total weight of the component functions.
   * This process does not otherwise normalize the gradient vector.
   * By default, NormalizeByWeight is off.
   */
  svtkSetMacro(NormalizeByWeight, svtkTypeBool);
  svtkGetMacro(NormalizeByWeight, svtkTypeBool);
  svtkBooleanMacro(NormalizeByWeight, svtkTypeBool);
  //@}

protected:
  svtkImplicitSum();
  ~svtkImplicitSum() override;

  svtkImplicitFunctionCollection* FunctionList;
  svtkDoubleArray* Weights;
  double TotalWeight;

  void CalculateTotalWeight(void);
  svtkTypeBool NormalizeByWeight;

private:
  svtkImplicitSum(const svtkImplicitSum&) = delete;
  void operator=(const svtkImplicitSum&) = delete;
};

#endif
