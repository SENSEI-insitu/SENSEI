/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInitialValueProblemSolver.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInitialValueProblemSolver
 * @brief   Integrate a set of ordinary
 * differential equations (initial value problem) in time.
 *
 *
 * Given a svtkFunctionSet which returns dF_i(x_j, t)/dt given x_j and
 * t, svtkInitialValueProblemSolver computes the value of F_i at t+deltat.
 *
 * @warning
 * svtkInitialValueProblemSolver and it's subclasses are not thread-safe.
 * You should create a new integrator for each thread.
 *
 * @sa
 * svtkRungeKutta2 svtkRungeKutta4
 */

#ifndef svtkInitialValueProblemSolver_h
#define svtkInitialValueProblemSolver_h

#include "svtkCommonMathModule.h" // For export macro
#include "svtkObject.h"

class svtkFunctionSet;

class SVTKCOMMONMATH_EXPORT svtkInitialValueProblemSolver : public svtkObject
{
public:
  svtkTypeMacro(svtkInitialValueProblemSolver, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Given initial values, xprev , initial time, t and a requested time
   * interval, delT calculate values of x at t+delTActual (xnext).
   * For certain concrete sub-classes delTActual != delT. This occurs
   * when the solver supports adaptive stepsize control. If this
   * is the case, the solver tries to change to stepsize such that
   * the (estimated) error of the integration is less than maxError.
   * The solver will not set the stepsize smaller than minStep or
   * larger than maxStep.
   * Also note that delT is an in/out argument. Adaptive solvers
   * will modify delT to reflect the best (estimated) size for the next
   * integration step.
   * An estimated value for the error is returned (by reference) in error.
   * Note that only some concrete sub-classes support this. Otherwise,
   * the error is set to 0.
   * This method returns an error code representing the nature of
   * the failure:
   * OutOfDomain = 1,
   * NotInitialized = 2,
   * UnexpectedValue = 3
   */
  virtual int ComputeNextStep(
    double* xprev, double* xnext, double t, double& delT, double maxError, double& error)
  {
    return this->ComputeNextStep(xprev, xnext, t, delT, maxError, error, nullptr);
  }

  virtual int ComputeNextStep(double* xprev, double* xnext, double t, double& delT, double maxError,
    double& error, void* userData)
  {
    double minStep = delT;
    double maxStep = delT;
    double delTActual;
    return this->ComputeNextStep(
      xprev, nullptr, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }

  virtual int ComputeNextStep(double* xprev, double* dxprev, double* xnext, double t, double& delT,
    double maxError, double& error)
  {
    return this->ComputeNextStep(xprev, dxprev, xnext, t, delT, maxError, error, nullptr);
  }

  virtual int ComputeNextStep(double* xprev, double* dxprev, double* xnext, double t, double& delT,
    double maxError, double& error, void* userData)
  {
    double minStep = delT;
    double maxStep = delT;
    double delTActual;
    return this->ComputeNextStep(
      xprev, dxprev, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }
  virtual int ComputeNextStep(double* xprev, double* xnext, double t, double& delT,
    double& delTActual, double minStep, double maxStep, double maxError, double& error)
  {
    return this->ComputeNextStep(
      xprev, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, nullptr);
  }

  virtual int ComputeNextStep(double* xprev, double* xnext, double t, double& delT,
    double& delTActual, double minStep, double maxStep, double maxError, double& error,
    void* userData)
  {
    return this->ComputeNextStep(
      xprev, nullptr, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, userData);
  }

  virtual int ComputeNextStep(double* xprev, double* dxprev, double* xnext, double t, double& delT,
    double& delTActual, double minStep, double maxStep, double maxError, double& error)
  {
    return this->ComputeNextStep(
      xprev, dxprev, xnext, t, delT, delTActual, minStep, maxStep, maxError, error, nullptr);
  }

  virtual int ComputeNextStep(double* svtkNotUsed(xprev), double* svtkNotUsed(dxprev),
    double* svtkNotUsed(xnext), double svtkNotUsed(t), double& svtkNotUsed(delT),
    double& svtkNotUsed(delTActual), double svtkNotUsed(minStep), double svtkNotUsed(maxStep),
    double svtkNotUsed(maxError), double& svtkNotUsed(error), void* svtkNotUsed(userData))
  {
    return 0;
  }
  //@}

  //@{
  /**
   * Set / get the dataset used for the implicit function evaluation.
   */
  virtual void SetFunctionSet(svtkFunctionSet* functionset);
  svtkGetObjectMacro(FunctionSet, svtkFunctionSet);
  //@}

  /**
   * Returns 1 if the solver uses adaptive stepsize control,
   * 0 otherwise
   */
  virtual svtkTypeBool IsAdaptive() { return this->Adaptive; }

  enum ErrorCodes
  {
    OUT_OF_DOMAIN = 1,
    NOT_INITIALIZED = 2,
    UNEXPECTED_VALUE = 3
  };

protected:
  svtkInitialValueProblemSolver();
  ~svtkInitialValueProblemSolver() override;

  virtual void Initialize();

  svtkFunctionSet* FunctionSet;

  double* Vals;
  double* Derivs;
  int Initialized;
  svtkTypeBool Adaptive;

private:
  svtkInitialValueProblemSolver(const svtkInitialValueProblemSolver&) = delete;
  void operator=(const svtkInitialValueProblemSolver&) = delete;
};

#endif
