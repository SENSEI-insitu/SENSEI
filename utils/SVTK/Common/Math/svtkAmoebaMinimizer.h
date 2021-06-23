/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAmoebaMinimizer.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAmoebaMinimizer
 * @brief   nonlinear optimization with a simplex
 *
 * svtkAmoebaMinimizer will modify a set of parameters in order to find
 * the minimum of a specified function.  The method used is commonly
 * known as the amoeba method, it constructs an n-dimensional simplex
 * in parameter space (i.e. a tetrahedron if the number or parameters
 * is 3) and moves the vertices around parameter space until a local
 * minimum is found.  The amoeba method is robust, reasonably efficient,
 * but is not guaranteed to find the global minimum if several local
 * minima exist.
 */

#ifndef svtkAmoebaMinimizer_h
#define svtkAmoebaMinimizer_h

#include "svtkCommonMathModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONMATH_EXPORT svtkAmoebaMinimizer : public svtkObject
{
public:
  static svtkAmoebaMinimizer* New();
  svtkTypeMacro(svtkAmoebaMinimizer, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Specify the function to be minimized.  When this function
   * is called, it must get the parameter values by calling
   * GetParameterValue() for each parameter, and then must
   * call SetFunctionValue() to tell the minimizer what the result
   * of the function evaluation was.  The number of function
   * evaluations used for the minimization can be retrieved
   * using GetFunctionEvaluations().
   */
  void SetFunction(void (*f)(void*), void* arg);

  /**
   * Set a function to call when a void* argument is being discarded.
   */
  void SetFunctionArgDelete(void (*f)(void*));

  //@{
  /**
   * Set the initial value for the specified parameter.  Calling
   * this function for any parameter will reset the Iterations
   * and the FunctionEvaluations counts to zero.  You must also
   * use SetParameterScale() to specify the step size by which the
   * parameter will be modified during the minimization.  It is
   * preferable to specify parameters by name, rather than by
   * number.
   */
  void SetParameterValue(const char* name, double value);
  void SetParameterValue(int i, double value);
  //@}

  //@{
  /**
   * Set the scale to use when modifying a parameter, i.e. the
   * initial amount by which the parameter will be modified
   * during the search for the minimum.  It is preferable to
   * identify scalars by name rather than by number.
   */
  void SetParameterScale(const char* name, double scale);
  double GetParameterScale(const char* name);
  void SetParameterScale(int i, double scale);
  double GetParameterScale(int i) { return this->ParameterScales[i]; }
  //@}

  //@{
  /**
   * Get the value of a parameter at the current stage of the minimization.
   * Call this method within the function that you are minimizing in order
   * to get the current parameter values.  It is preferable to specify
   * parameters by name rather than by index.
   */
  double GetParameterValue(const char* name);
  double GetParameterValue(int i) { return this->ParameterValues[i]; }
  //@}

  /**
   * For completeness, an unchecked method to get the name for particular
   * parameter (the result will be nullptr if no name was set).
   */
  const char* GetParameterName(int i) { return this->ParameterNames[i]; }

  /**
   * Get the number of parameters that have been set.
   */
  int GetNumberOfParameters() { return this->NumberOfParameters; }

  /**
   * Initialize the minimizer.  This will reset the number of parameters to
   * zero so that the minimizer can be reused.
   */
  void Initialize();

  /**
   * Iterate until the minimum is found to within the specified tolerance,
   * or until the MaxIterations has been reached.
   */
  virtual void Minimize();

  /**
   * Perform one iteration of minimization.  Returns zero if the tolerance
   * stopping criterion has been met.
   */
  virtual int Iterate();

  //@{
  /**
   * Get the function value resulting from the minimization.
   */
  svtkSetMacro(FunctionValue, double);
  double GetFunctionValue() { return this->FunctionValue; }
  //@}

  //@{
  /**
   * Set the amoeba contraction ratio.  The default value of 0.5 gives
   * fast convergence, but larger values such as 0.6 or 0.7 provide
   * greater stability.
   */
  svtkSetClampMacro(ContractionRatio, double, 0.5, 1.0);
  svtkGetMacro(ContractionRatio, double);
  //@}

  //@{
  /**
   * Set the amoeba expansion ratio.  The default value is 2.0, which
   * provides rapid expansion.  Values between 1.1 and 2.0 are valid.
   */
  svtkSetClampMacro(ExpansionRatio, double, 1.0, 2.0);
  svtkGetMacro(ExpansionRatio, double);
  //@}

  //@{
  /**
   * Specify the value tolerance to aim for during the minimization.
   */
  svtkSetMacro(Tolerance, double);
  svtkGetMacro(Tolerance, double);
  //@}

  //@{
  /**
   * Specify the parameter tolerance to aim for during the minimization.
   */
  svtkSetMacro(ParameterTolerance, double);
  svtkGetMacro(ParameterTolerance, double);
  //@}

  //@{
  /**
   * Specify the maximum number of iterations to try before giving up.
   */
  svtkSetMacro(MaxIterations, int);
  svtkGetMacro(MaxIterations, int);
  //@}

  //@{
  /**
   * Return the number of iterations that have been performed.  This
   * is not necessarily the same as the number of function evaluations.
   */
  svtkGetMacro(Iterations, int);
  //@}

  //@{
  /**
   * Return the number of times that the function has been evaluated.
   */
  svtkGetMacro(FunctionEvaluations, int);
  //@}

  /**
   * Evaluate the function.  This is usually called internally by the
   * minimization code, but it is provided here as a public method.
   */
  void EvaluateFunction();

protected:
  svtkAmoebaMinimizer();
  ~svtkAmoebaMinimizer() override;

  void (*Function)(void*);
  void (*FunctionArgDelete)(void*);
  void* FunctionArg;

  int NumberOfParameters;
  char** ParameterNames;
  double* ParameterValues;
  double* ParameterScales;
  double FunctionValue;

  double ContractionRatio;
  double ExpansionRatio;

  double Tolerance;
  double ParameterTolerance;
  int MaxIterations;
  int Iterations;
  int FunctionEvaluations;

private:
  // specific to amoeba simplex minimization

  double** AmoebaVertices;
  double* AmoebaValues;
  double* AmoebaSum;
  double AmoebaSize;
  double AmoebaHighValue;
  int AmoebaNStepsNoImprovement;

  void InitializeAmoeba();
  void GetAmoebaParameterValues();
  void TerminateAmoeba();
  double TryAmoeba(double sum[], int high, double fac);
  int PerformAmoeba();
  int CheckParameterTolerance();

  svtkAmoebaMinimizer(const svtkAmoebaMinimizer&) = delete;
  void operator=(const svtkAmoebaMinimizer&) = delete;
};

#endif
