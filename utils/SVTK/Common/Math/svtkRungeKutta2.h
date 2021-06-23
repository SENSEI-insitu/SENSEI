/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRungeKutta2.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkRungeKutta2
 * @brief   Integrate an initial value problem using 2nd
 * order Runge-Kutta method.
 *
 *
 * This is a concrete sub-class of svtkInitialValueProblemSolver.
 * It uses a 2nd order Runge-Kutta method to obtain the values of
 * a set of functions at the next time step.
 *
 * @sa
 * svtkInitialValueProblemSolver svtkRungeKutta4 svtkRungeKutta45 svtkFunctionSet
 */

#ifndef svtkRungeKutta2_h
#define svtkRungeKutta2_h

#include "svtkCommonMathModule.h" // For export macro
#include "svtkInitialValueProblemSolver.h"

class SVTKCOMMONMATH_EXPORT svtkRungeKutta2 : public svtkInitialValueProblemSolver
{
public:
  svtkTypeMacro(svtkRungeKutta2, svtkInitialValueProblemSolver);

  /**
   * Construct a svtkRungeKutta2 with no initial FunctionSet.
   */
  static svtkRungeKutta2* New();

  using Superclass::ComputeNextStep;
  //@{
  /**
   * Given initial values, xprev , initial time, t and a requested time
   * interval, delT calculate values of x at t+delT (xnext).
   * delTActual is always equal to delT.
   * Since this class can not provide an estimate for the error error
   * is set to 0.
   * maxStep, minStep and maxError are unused.
   * This method returns an error code representing the nature of
   * the failure:
   * OutOfDomain = 1,
   * NotInitialized = 2,
   * UnexpectedValue = 3
   */
  int ComputeNextStep(double* xprev, double* xnext, double t, double& delT, double maxError,
    double& error, void* userData) override
  {
    double minStep = delT;
    double maxStep = delT;
    double delTActual;
    return this->ComputeNextStep(
      xprev, nullptr, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }
  int ComputeNextStep(double* xprev, double* dxprev, double* xnext, double t, double& delT,
    double maxError, double& error, void* userData) override
  {
    double minStep = delT;
    double maxStep = delT;
    double delTActual;
    return this->ComputeNextStep(
      xprev, dxprev, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }
  int ComputeNextStep(double* xprev, double* xnext, double t, double& delT, double& delTActual,
    double minStep, double maxStep, double maxError, double& error, void* userData) override
  {
    return this->ComputeNextStep(
      xprev, nullptr, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }
  int ComputeNextStep(double* xprev, double* dxprev, double* xnext, double t, double& delT,
    double& delTActual, double minStep, double maxStep, double maxError, double& error,
    void* userData) override;
  //@}

protected:
  svtkRungeKutta2();
  ~svtkRungeKutta2() override;

private:
  svtkRungeKutta2(const svtkRungeKutta2&) = delete;
  void operator=(const svtkRungeKutta2&) = delete;
};

#endif

// SVTK-HeaderTest-Exclude: svtkRungeKutta2.h
