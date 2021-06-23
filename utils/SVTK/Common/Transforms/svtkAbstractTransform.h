/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAbstractTransform
 * @brief   superclass for all geometric transformations
 *
 * svtkAbstractTransform is the superclass for all SVTK geometric
 * transformations.  The SVTK transform hierarchy is split into two
 * major branches: warp transformations and homogeneous (including linear)
 * transformations.  The latter can be represented in terms of a 4x4
 * transformation matrix, the former cannot.
 * <p>Transformations can be pipelined through two mechanisms:
 * <p>1) GetInverse() returns the pipelined
 * inverse of a transformation i.e. if you modify the original transform,
 * any transform previously returned by the GetInverse() method will
 * automatically update itself according to the change.
 * <p>2) You can do pipelined concatenation of transformations through
 * the svtkGeneralTransform class, the svtkPerspectiveTransform class,
 * or the svtkTransform class.
 * @sa
 * svtkGeneralTransform svtkWarpTransform svtkHomogeneousTransform
 * svtkLinearTransform svtkIdentityTransform
 * svtkTransformPolyDataFilter svtkTransformFilter svtkImageReslice
 * svtkImplicitFunction
 */

#ifndef svtkAbstractTransform_h
#define svtkAbstractTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkObject.h"

class svtkDataArray;
class svtkMatrix4x4;
class svtkPoints;
class svtkSimpleCriticalSection;

class SVTKCOMMONTRANSFORMS_EXPORT svtkAbstractTransform : public svtkObject
{
public:
  svtkTypeMacro(svtkAbstractTransform, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Apply the transformation to a coordinate.  You can use the same
   * array to store both the input and output point.
   */
  void TransformPoint(const float in[3], float out[3])
  {
    this->Update();
    this->InternalTransformPoint(in, out);
  }

  /**
   * Apply the transformation to a double-precision coordinate.
   * You can use the same array to store both the input and output point.
   */
  void TransformPoint(const double in[3], double out[3])
  {
    this->Update();
    this->InternalTransformPoint(in, out);
  }

  /**
   * Apply the transformation to a double-precision coordinate.
   * Use this if you are programming in Python or Java.
   */
  double* TransformPoint(double x, double y, double z) SVTK_SIZEHINT(3)
  {
    return this->TransformDoublePoint(x, y, z);
  }
  double* TransformPoint(const double point[3]) SVTK_SIZEHINT(3)
  {
    return this->TransformPoint(point[0], point[1], point[2]);
  }

  //@{
  /**
   * Apply the transformation to an (x,y,z) coordinate.
   * Use this if you are programming in Python or Java.
   */
  float* TransformFloatPoint(float x, float y, float z) SVTK_SIZEHINT(3)
  {
    this->InternalFloatPoint[0] = x;
    this->InternalFloatPoint[1] = y;
    this->InternalFloatPoint[2] = z;
    this->TransformPoint(this->InternalFloatPoint, this->InternalFloatPoint);
    return this->InternalFloatPoint;
  }
  float* TransformFloatPoint(const float point[3]) SVTK_SIZEHINT(3)
  {
    return this->TransformFloatPoint(point[0], point[1], point[2]);
  }
  //@}

  //@{
  /**
   * Apply the transformation to a double-precision (x,y,z) coordinate.
   * Use this if you are programming in Python or Java.
   */
  double* TransformDoublePoint(double x, double y, double z) SVTK_SIZEHINT(3)
  {
    this->InternalDoublePoint[0] = x;
    this->InternalDoublePoint[1] = y;
    this->InternalDoublePoint[2] = z;
    this->TransformPoint(this->InternalDoublePoint, this->InternalDoublePoint);
    return this->InternalDoublePoint;
  }
  double* TransformDoublePoint(const double point[3]) SVTK_SIZEHINT(3)
  {
    return this->TransformDoublePoint(point[0], point[1], point[2]);
  }
  //@}

  //@{
  /**
   * Apply the transformation to a normal at the specified vertex.  If the
   * transformation is a svtkLinearTransform, you can use TransformNormal()
   * instead.
   */
  void TransformNormalAtPoint(const float point[3], const float in[3], float out[3]);
  void TransformNormalAtPoint(const double point[3], const double in[3], double out[3]);
  //@}

  double* TransformNormalAtPoint(const double point[3], const double normal[3]) SVTK_SIZEHINT(3)
  {
    this->TransformNormalAtPoint(point, normal, this->InternalDoublePoint);
    return this->InternalDoublePoint;
  }

  //@{
  /**
   * Apply the transformation to a double-precision normal at the specified
   * vertex.  If the transformation is a svtkLinearTransform, you can use
   * TransformDoubleNormal() instead.
   */
  double* TransformDoubleNormalAtPoint(const double point[3], const double normal[3])
    SVTK_SIZEHINT(3)
  {
    this->TransformNormalAtPoint(point, normal, this->InternalDoublePoint);
    return this->InternalDoublePoint;
  }
  //@}

  //@{
  /**
   * Apply the transformation to a single-precision normal at the specified
   * vertex.  If the transformation is a svtkLinearTransform, you can use
   * TransformFloatNormal() instead.
   */
  float* TransformFloatNormalAtPoint(const float point[3], const float normal[3]) SVTK_SIZEHINT(3)
  {
    this->TransformNormalAtPoint(point, normal, this->InternalFloatPoint);
    return this->InternalFloatPoint;
  }
  //@}

  //@{
  /**
   * Apply the transformation to a vector at the specified vertex.  If the
   * transformation is a svtkLinearTransform, you can use TransformVector()
   * instead.
   */
  void TransformVectorAtPoint(const float point[3], const float in[3], float out[3]);
  void TransformVectorAtPoint(const double point[3], const double in[3], double out[3]);
  //@}

  double* TransformVectorAtPoint(const double point[3], const double vector[3]) SVTK_SIZEHINT(3)
  {
    this->TransformVectorAtPoint(point, vector, this->InternalDoublePoint);
    return this->InternalDoublePoint;
  }

  //@{
  /**
   * Apply the transformation to a double-precision vector at the specified
   * vertex.  If the transformation is a svtkLinearTransform, you can use
   * TransformDoubleVector() instead.
   */
  double* TransformDoubleVectorAtPoint(const double point[3], const double vector[3])
    SVTK_SIZEHINT(3)
  {
    this->TransformVectorAtPoint(point, vector, this->InternalDoublePoint);
    return this->InternalDoublePoint;
  }
  //@}

  //@{
  /**
   * Apply the transformation to a single-precision vector at the specified
   * vertex.  If the transformation is a svtkLinearTransform, you can use
   * TransformFloatVector() instead.
   */
  float* TransformFloatVectorAtPoint(const float point[3], const float vector[3]) SVTK_SIZEHINT(3)
  {
    this->TransformVectorAtPoint(point, vector, this->InternalFloatPoint);
    return this->InternalFloatPoint;
  }
  //@}

  /**
   * Apply the transformation to a series of points, and append the
   * results to outPts.
   */
  virtual void TransformPoints(svtkPoints* inPts, svtkPoints* outPts);

  /**
   * Apply the transformation to a combination of points, normals
   * and vectors.
   */
  virtual void TransformPointsNormalsVectors(svtkPoints* inPts, svtkPoints* outPts,
    svtkDataArray* inNms, svtkDataArray* outNms, svtkDataArray* inVrs, svtkDataArray* outVrs,
    int nOptionalVectors = 0, svtkDataArray** inVrsArr = nullptr,
    svtkDataArray** outVrsArr = nullptr);

  /**
   * Get the inverse of this transform.  If you modify this transform,
   * the returned inverse transform will automatically update.  If you
   * want the inverse of a svtkTransform, you might want to use
   * GetLinearInverse() instead which will type cast the result from
   * svtkAbstractTransform to svtkLinearTransform.
   */
  svtkAbstractTransform* GetInverse();

  /**
   * Set a transformation that this transform will be the inverse of.
   * This transform will automatically update to agree with the
   * inverse transform that you set.
   */
  void SetInverse(svtkAbstractTransform* transform);

  /**
   * Invert the transformation.
   */
  virtual void Inverse() = 0;

  /**
   * Copy this transform from another of the same type.
   */
  void DeepCopy(svtkAbstractTransform*);

  /**
   * Update the transform to account for any changes which
   * have been made.  You do not have to call this method
   * yourself, it is called automatically whenever the
   * transform needs an update.
   */
  void Update();

  //@{
  /**
   * This will calculate the transformation without calling Update.
   * Meant for use only within other SVTK classes.
   */
  virtual void InternalTransformPoint(const float in[3], float out[3]) = 0;
  virtual void InternalTransformPoint(const double in[3], double out[3]) = 0;
  //@}

  //@{
  /**
   * This will transform a point and, at the same time, calculate a
   * 3x3 Jacobian matrix that provides the partial derivatives of the
   * transformation at that point.  This method does not call Update.
   * Meant for use only within other SVTK classes.
   */
  virtual void InternalTransformDerivative(
    const float in[3], float out[3], float derivative[3][3]) = 0;
  virtual void InternalTransformDerivative(
    const double in[3], double out[3], double derivative[3][3]) = 0;
  //@}

  /**
   * Make another transform of the same type.
   */
  virtual SVTK_NEWINSTANCE svtkAbstractTransform* MakeTransform() = 0;

  /**
   * Check for self-reference.  Will return true if concatenating
   * with the specified transform, setting it to be our inverse,
   * or setting it to be our input will create a circular reference.
   * CircuitCheck is automatically called by SetInput(), SetInverse(),
   * and Concatenate(svtkXTransform *).  Avoid using this function,
   * it is experimental.
   */
  virtual int CircuitCheck(svtkAbstractTransform* transform);

  /**
   * Override GetMTime necessary because of inverse transforms.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Needs a special UnRegister() implementation to avoid
   * circular references.
   */
  void UnRegister(svtkObjectBase* O) override;

protected:
  svtkAbstractTransform();
  ~svtkAbstractTransform() override;

  /**
   * Perform any subclass-specific Update.
   */
  virtual void InternalUpdate() {}

  /**
   * Perform any subclass-specific DeepCopy.
   */
  virtual void InternalDeepCopy(svtkAbstractTransform*) {}

  float InternalFloatPoint[3];
  double InternalDoublePoint[3];

private:
  // We need to record the time of the last update, and we also need
  // to do mutex locking so updates don't collide.  These are private
  // because Update() is not virtual.
  // If DependsOnInverse is set, then this transform object will
  // check its inverse on every update, and update itself accordingly
  // if necessary.

  svtkTimeStamp UpdateTime;
  svtkSimpleCriticalSection* UpdateMutex;
  svtkSimpleCriticalSection* InverseMutex;
  int DependsOnInverse;

  // MyInverse is a transform which is the inverse of this one.

  svtkAbstractTransform* MyInverse;

  int InUnRegister;

private:
  svtkAbstractTransform(const svtkAbstractTransform&) = delete;
  void operator=(const svtkAbstractTransform&) = delete;
};

//-------------------------------------------------------------------------
// A simple data structure to hold both a transform and its inverse.
// One of ForwardTransform or InverseTransform might be nullptr,
// and must be acquired by calling GetInverse() on the other.
class svtkTransformPair
{
public:
  svtkTransformPair() {}

  svtkAbstractTransform* ForwardTransform;
  svtkAbstractTransform* InverseTransform;

  void SwapForwardInverse()
  {
    svtkAbstractTransform* tmp = this->ForwardTransform;
    this->ForwardTransform = this->InverseTransform;
    this->InverseTransform = tmp;
  }
};

// .NAME svtkTransformConcatenation - store a series of transformations.
// .SECTION Description
// A helper class (not derived from svtkObject) to store a series of
// transformations in a pipelined concatenation.
class SVTKCOMMONTRANSFORMS_EXPORT svtkTransformConcatenation
{
public:
  static svtkTransformConcatenation* New() { return new svtkTransformConcatenation(); }
  void Delete() { delete this; }

  /**
   * add a transform to the list according to Pre/PostMultiply semantics
   */
  void Concatenate(svtkAbstractTransform* transform);

  /**
   * concatenate with a matrix according to Pre/PostMultiply semantics
   */
  void Concatenate(const double elements[16]);

  //@{
  /**
   * set/get the PreMultiply flag
   */
  void SetPreMultiplyFlag(int flag) { this->PreMultiplyFlag = flag; }
  int GetPreMultiplyFlag() { return this->PreMultiplyFlag; }
  //@}

  //@{
  /**
   * the three basic linear transformations
   */
  void Translate(double x, double y, double z);
  void Rotate(double angle, double x, double y, double z);
  void Scale(double x, double y, double z);
  //@}

  /**
   * invert the concatenation
   */
  void Inverse();

  /**
   * get the inverse flag
   */
  int GetInverseFlag() { return this->InverseFlag; }

  /**
   * identity simply clears the transform list
   */
  void Identity();

  // copy the list
  void DeepCopy(svtkTransformConcatenation* transform);

  /**
   * the number of stored transforms
   */
  int GetNumberOfTransforms() { return this->NumberOfTransforms; }

  /**
   * the number of transforms that were pre-concatenated (note that
   * whenever Inverse() is called, the pre-concatenated and
   * post-concatenated transforms are switched)
   */
  int GetNumberOfPreTransforms() { return this->NumberOfPreTransforms; }

  /**
   * the number of transforms that were post-concatenated.
   */
  int GetNumberOfPostTransforms() { return this->NumberOfTransforms - this->NumberOfPreTransforms; }

  /**
   * get one of the transforms
   */
  svtkAbstractTransform* GetTransform(int i);

  /**
   * get maximum MTime of all transforms
   */
  svtkMTimeType GetMaxMTime();

  void PrintSelf(ostream& os, svtkIndent indent);

protected:
  svtkTransformConcatenation();
  ~svtkTransformConcatenation();

  int InverseFlag;
  int PreMultiplyFlag;

  svtkMatrix4x4* PreMatrix;
  svtkMatrix4x4* PostMatrix;
  svtkAbstractTransform* PreMatrixTransform;
  svtkAbstractTransform* PostMatrixTransform;

  int NumberOfTransforms;
  int NumberOfPreTransforms;
  int MaxNumberOfTransforms;
  svtkTransformPair* TransformList;

private:
  svtkTransformConcatenation(const svtkTransformConcatenation&) = delete;
  void operator=(const svtkTransformConcatenation&) = delete;
};

// .NAME svtkTransformConcatenationStack - Store a stack of concatenations.
// .SECTION Description
// A helper class (not derived from svtkObject) to store a stack of
// concatenations.
class SVTKCOMMONTRANSFORMS_EXPORT svtkTransformConcatenationStack
{
public:
  static svtkTransformConcatenationStack* New() { return new svtkTransformConcatenationStack(); }
  void Delete() { delete this; }

  /**
   * pop will pop delete 'concat', then pop the
   * top item on the stack onto 'concat'.
   */
  void Pop(svtkTransformConcatenation** concat);

  /**
   * push will move 'concat' onto the stack, and
   * make 'concat' a copy of its previous self
   */
  void Push(svtkTransformConcatenation** concat);

  void DeepCopy(svtkTransformConcatenationStack* stack);

protected:
  svtkTransformConcatenationStack();
  ~svtkTransformConcatenationStack();

  int StackSize;
  svtkTransformConcatenation** Stack;
  svtkTransformConcatenation** StackBottom;

private:
  svtkTransformConcatenationStack(const svtkTransformConcatenationStack&) = delete;
  void operator=(const svtkTransformConcatenationStack&) = delete;
};

#endif
