/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitFunction.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitFunction
 * @brief   abstract interface for implicit functions
 *
 * svtkImplicitFunction specifies an abstract interface for implicit
 * functions. Implicit functions are real valued functions defined in 3D
 * space, w = F(x,y,z). Two primitive operations are required: the ability to
 * evaluate the function, and the function gradient at a given point. The
 * implicit function divides space into three regions: on the surface
 * (F(x,y,z)=w), outside of the surface (F(x,y,z)>c), and inside the
 * surface (F(x,y,z)<c). (When c is zero, positive values are outside,
 * negative values are inside, and zero is on the surface. Note also
 * that the function gradient points from inside to outside.)
 *
 * Implicit functions are very powerful. It is possible to represent almost
 * any type of geometry with the level sets w = const, especially if you use
 * boolean combinations of implicit functions (see svtkImplicitBoolean).
 *
 * svtkImplicitFunction provides a mechanism to transform the implicit
 * function(s) via a svtkAbstractTransform.  This capability can be used to
 * translate, orient, scale, or warp implicit functions.  For example,
 * a sphere implicit function can be transformed into an oriented ellipse.
 *
 * @warning
 * The transformation transforms a point into the space of the implicit
 * function (i.e., the model space). Typically we want to transform the
 * implicit model into world coordinates. In this case the inverse of the
 * transformation is required.
 *
 * @sa
 * svtkAbstractTransform svtkSphere svtkCylinder svtkImplicitBoolean svtkPlane
 * svtkPlanes svtkQuadric svtkImplicitVolume svtkSampleFunction svtkCutter
 * svtkClipPolyData
 */

#ifndef svtkImplicitFunction_h
#define svtkImplicitFunction_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDataArray;

class svtkAbstractTransform;

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitFunction : public svtkObject
{
public:
  svtkTypeMacro(svtkImplicitFunction, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Overload standard modified time function. If Transform is modified,
   * then this object is modified as well.
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Evaluate function at position x-y-z and return value. Point x[3] is
   * transformed through transform (if provided).
   */
  virtual void FunctionValue(svtkDataArray* input, svtkDataArray* output);
  double FunctionValue(const double x[3]);
  double FunctionValue(double x, double y, double z)
  {
    double xyz[3] = { x, y, z };
    return this->FunctionValue(xyz);
  }
  //@}

  //@{
  /**
   * Evaluate function gradient at position x-y-z and pass back vector. Point
   * x[3] is transformed through transform (if provided).
   */
  void FunctionGradient(const double x[3], double g[3]);
  double* FunctionGradient(const double x[3]) SVTK_SIZEHINT(3)
  {
    this->FunctionGradient(x, this->ReturnValue);
    return this->ReturnValue;
  }
  double* FunctionGradient(double x, double y, double z) SVTK_SIZEHINT(3)
  {
    double xyz[3] = { x, y, z };
    return this->FunctionGradient(xyz);
  }
  //@}

  //@{
  /**
   * Set/Get a transformation to apply to input points before
   * executing the implicit function.
   */
  virtual void SetTransform(svtkAbstractTransform*);
  virtual void SetTransform(const double elements[16]);
  svtkGetObjectMacro(Transform, svtkAbstractTransform);
  //@}

  //@{
  /**
   * Evaluate function at position x-y-z and return value.  You should
   * generally not call this method directly, you should use
   * FunctionValue() instead.  This method must be implemented by
   * any derived class.
   */
  virtual double EvaluateFunction(double x[3]) = 0;
  virtual void EvaluateFunction(svtkDataArray* input, svtkDataArray* output);
  virtual double EvaluateFunction(double x, double y, double z)
  {
    double xyz[3] = { x, y, z };
    return this->EvaluateFunction(xyz);
  }
  //@}

  /**
   * Evaluate function gradient at position x-y-z and pass back vector.
   * You should generally not call this method directly, you should use
   * FunctionGradient() instead.  This method must be implemented by
   * any derived class.
   */
  virtual void EvaluateGradient(double x[3], double g[3]) = 0;

protected:
  svtkImplicitFunction();
  ~svtkImplicitFunction() override;

  svtkAbstractTransform* Transform;
  double ReturnValue[3];

private:
  svtkImplicitFunction(const svtkImplicitFunction&) = delete;
  void operator=(const svtkImplicitFunction&) = delete;
};

#endif
