/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFunctionSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFunctionSet
 * @brief   Abstract interface for sets of functions
 *
 * svtkFunctionSet specifies an abstract interface for set of functions
 * of the form F_i = F_i(x_j) where F (with i=1..m) are the functions
 * and x (with j=1..n) are the independent variables.
 * The only supported operation is the function evaluation at x_j.
 *
 * @sa
 * svtkImplicitDataSet svtkInterpolatedVelocityField
 * svtkInitialValueProblemSolver
 */

#ifndef svtkFunctionSet_h
#define svtkFunctionSet_h

#include "svtkCommonMathModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONMATH_EXPORT svtkFunctionSet : public svtkObject
{
public:
  svtkTypeMacro(svtkFunctionSet, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Evaluate functions at x_j.
   * x and f have to point to valid double arrays of appropriate
   * sizes obtained with GetNumberOfFunctions() and
   * GetNumberOfIndependentVariables.
   * If you inherit this class, make sure to reimplement at least one of the two
   * FunctionValues signatures.
   */
  virtual int FunctionValues(double* x, double* f) { return this->FunctionValues(x, f, nullptr); }
  virtual int FunctionValues(double* x, double* f, void* svtkNotUsed(userData))
  {
    return this->FunctionValues(x, f);
  }

  /**
   * Return the number of functions. Note that this is constant for
   * a given type of set of functions and can not be changed at
   * run time.
   */
  virtual int GetNumberOfFunctions() { return this->NumFuncs; }

  /**
   * Return the number of independent variables. Note that this is
   * constant for a given type of set of functions and can not be changed
   * at run time.
   */
  virtual int GetNumberOfIndependentVariables() { return this->NumIndepVars; }

protected:
  svtkFunctionSet();
  ~svtkFunctionSet() override {}

  int NumFuncs;
  int NumIndepVars;

private:
  svtkFunctionSet(const svtkFunctionSet&) = delete;
  void operator=(const svtkFunctionSet&) = delete;
};

#endif
