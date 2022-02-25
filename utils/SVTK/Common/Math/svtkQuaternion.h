/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuaternion.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuaternion
 * @brief   templated base type for storage of quaternions.
 *
 * This class is a templated data type for storing and manipulating
 * quaternions. The quaternions have the form [w, x, y, z].
 * Given a rotation of angle theta and axis v, the corresponding
 * quaternion is [w, x, y, z] = [cos(theta/2), v*sin(theta/2)]
 *
 * This class implements the Spherical Linear interpolation (SLERP)
 * and the Spherical Spline Quaternion interpolation (SQUAD).
 * It is advised to use the svtkQuaternionInterpolator when dealing
 * with multiple quaternions and or interpolations.
 *
 * @sa
 * svtkQuaternionInterpolator
 */

#ifndef svtkQuaternion_h
#define svtkQuaternion_h

#include "svtkTuple.h"

template <typename T>
class svtkQuaternion : public svtkTuple<T, 4>
{
public:
  /**
   * Default constructor. Creates an identity quaternion.
   */
  svtkQuaternion();

  /**
   * Initialize all of the quaternion's elements with the supplied scalar.
   */
  explicit svtkQuaternion(const T& scalar)
    : svtkTuple<T, 4>(scalar)
  {
  }

  /**
   * Initialize the quaternion's elements with the elements of the supplied array.
   * Note that the supplied pointer must contain at least as many elements as
   * the quaternion, or it will result in access to out of bounds memory.
   */
  explicit svtkQuaternion(const T* init)
    : svtkTuple<T, 4>(init)
  {
  }

  /**
   * Initialize the quaternion element explicitly.
   */
  svtkQuaternion(const T& w, const T& x, const T& y, const T& z);

  /**
   * Get the squared norm of the quaternion.
   */
  T SquaredNorm() const;

  /**
   * Get the norm of the quaternion, i.e. its length.
   */
  T Norm() const;

  /**
   * Set the quaternion to identity in place.
   */
  void ToIdentity();

  /**
   * Return the identity quaternion.
   * Note that the default constructor also creates an identity quaternion.
   */
  static svtkQuaternion<T> Identity();

  /**
   * Normalize the quaternion in place.
   * Return the norm of the quaternion.
   */
  T Normalize();

  /**
   * Return the normalized form of this quaternion.
   */
  svtkQuaternion<T> Normalized() const;

  /**
   * Conjugate the quaternion in place.
   */
  void Conjugate();

  /**
   * Return the conjugate form of this quaternion.
   */
  svtkQuaternion<T> Conjugated() const;

  /**
   * Invert the quaternion in place.
   * This is equivalent to conjugate the quaternion and then divide
   * it by its squared norm.
   */
  void Invert();

  /**
   * Return the inverted form of this quaternion.
   */
  svtkQuaternion<T> Inverse() const;

  /**
   * Convert this quaternion to a unit log quaternion.
   * The unit log quaternion is defined by:
   * [w, x, y, z] =  [0.0, v*theta].
   */
  void ToUnitLog();

  /**
   * Return the unit log version of this quaternion.
   * The unit log quaternion is defined by:
   * [w, x, y, z] =  [0.0, v*theta].
   */
  svtkQuaternion<T> UnitLog() const;

  /**
   * Convert this quaternion to a unit exponential quaternion.
   * The unit exponential quaternion is defined by:
   * [w, x, y, z] =  [cos(theta), v*sin(theta)].
   */
  void ToUnitExp();

  /**
   * Return the unit exponential version of this quaternion.
   * The unit exponential quaternion is defined by:
   * [w, x, y, z] =  [cos(theta), v*sin(theta)].
   */
  svtkQuaternion<T> UnitExp() const;

  /**
   * Normalize a quaternion in place and transform it to
   * so its angle is in degrees and its axis normalized.
   */
  void NormalizeWithAngleInDegrees();

  /**
   * Returns a quaternion normalized and transformed
   * so its angle is in degrees and its axis normalized.
   */
  svtkQuaternion<T> NormalizedWithAngleInDegrees() const;

  //@{
  /**
   * Set/Get the w, x, y and z components of the quaternion.
   */
  void Set(const T& w, const T& x, const T& y, const T& z);
  void Set(T quat[4]);
  void Get(T quat[4]) const;
  //@}

  //@{
  /**
   * Set/Get the w component of the quaternion, i.e. element 0.
   */
  void SetW(const T& w);
  const T& GetW() const;
  //@}

  //@{
  /**
   * Set/Get the x component of the quaternion, i.e. element 1.
   */
  void SetX(const T& x);
  const T& GetX() const;
  //@}

  //@{
  /**
   * Set/Get the y component of the quaternion, i.e. element 2.
   */
  void SetY(const T& y);
  const T& GetY() const;
  //@}

  //@{
  /**
   * Set/Get the y component of the quaternion, i.e. element 3.
   */
  void SetZ(const T& z);
  const T& GetZ() const;
  //@}

  //@{
  /**
   * Set/Get the angle (in radians) and the axis corresponding to
   * the axis-angle rotation of this quaternion.
   */
  T GetRotationAngleAndAxis(T axis[3]) const;
  void SetRotationAngleAndAxis(T angle, T axis[3]);
  void SetRotationAngleAndAxis(const T& angle, const T& x, const T& y, const T& z);
  //@}

  /**
   * Cast the quaternion to the specified type and return the result.
   */
  template <typename CastTo>
  svtkQuaternion<CastTo> Cast() const;

  /**
   * Convert a quaternion to a 3x3 rotation matrix. The quaternion
   * does not have to be normalized beforehand.
   * @sa FromMatrix3x3()
   */
  void ToMatrix3x3(T A[3][3]) const;

  /**
   * Convert a 3x3 matrix into a quaternion.  This will provide the
   * best possible answer even if the matrix is not a pure rotation matrix.
   * The method used is that of B.K.P. Horn.
   * @sa ToMatrix3x3()
   */
  void FromMatrix3x3(const T A[3][3]);

  /**
   * Interpolate quaternions using spherical linear interpolation between
   * this quaternion and q1 to produce the output.
   * The parametric coordinate t belongs to [0,1] and lies between (this,q1).
   * @sa svtkQuaternionInterpolator
   */
  svtkQuaternion<T> Slerp(T t, const svtkQuaternion<T>& q) const;

  /**
   * Interpolates between quaternions, using spherical quadrangle
   * interpolation.
   * @sa svtkQuaternionInterpolator
   */
  svtkQuaternion<T> InnerPoint(const svtkQuaternion<T>& q1, const svtkQuaternion<T>& q2) const;

  /**
   * Performs addition of quaternion of the same basic type.
   */
  svtkQuaternion<T> operator+(const svtkQuaternion<T>& q) const;

  /**
   * Performs subtraction of quaternions of the same basic type.
   */
  svtkQuaternion<T> operator-(const svtkQuaternion<T>& q) const;

  /**
   * Performs multiplication of quaternion of the same basic type.
   */
  svtkQuaternion<T> operator*(const svtkQuaternion<T>& q) const;

  /**
   * Performs multiplication of the quaternions by a scalar value.
   */
  svtkQuaternion<T> operator*(const T& scalar) const;

  /**
   * Performs in place multiplication of the quaternions by a scalar value.
   */
  void operator*=(const T& scalar) const;

  /**
   * Performs division of quaternions of the same type.
   */
  svtkQuaternion<T> operator/(const svtkQuaternion<T>& q) const;

  /**
   * Performs division of the quaternions by a scalar value.
   */
  svtkQuaternion<T> operator/(const T& scalar) const;

  //@{
  /**
   * Performs in place division of the quaternions by a scalar value.
   */
  void operator/=(const T& scalar);
  //@}
};

/**
 * Several macros to define the various operator overloads for the quaternions.
 * These are necessary for the derived classes that are commonly used.
 */
#define svtkQuaternionIdentity(quaternionType, type)                                                \
  quaternionType Identity() const                                                                  \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::Identity().GetData());                              \
  }
#define svtkQuaternionNormalized(quaternionType, type)                                              \
  quaternionType Normalized() const                                                                \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::Normalized().GetData());                            \
  }
#define svtkQuaternionConjugated(quaternionType, type)                                              \
  quaternionType Conjugated() const                                                                \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::Conjugated().GetData());                            \
  }
#define svtkQuaternionInverse(quaternionType, type)                                                 \
  quaternionType Inverse() const                                                                   \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::Inverse().GetData());                               \
  }
#define svtkQuaternionUnitLog(quaternionType, type)                                                 \
  quaternionType UnitLog() const                                                                   \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::UnitLog().GetData());                               \
  }
#define svtkQuaternionUnitExp(quaternionType, type)                                                 \
  quaternionType UnitExp() const                                                                   \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::UnitExp().GetData());                               \
  }
#define svtkQuaternionNormalizedWithAngleInDegrees(quaternionType, type)                            \
  quaternionType NormalizedWithAngleInDegrees() const                                              \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::NormalizedWithAngleInDegrees().GetData());          \
  }
#define svtkQuaternionSlerp(quaternionType, type)                                                   \
  quaternionType Slerp(type t, const quaternionType& q) const                                      \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::Slerp(t, q).GetData());                             \
  }
#define svtkQuaternionInnerPoint(quaternionType, type)                                              \
  quaternionType InnerPoint(const quaternionType& q1, const quaternionType& q2) const              \
  {                                                                                                \
    return quaternionType(svtkQuaternion<type>::InnerPoint(q1, q2).GetData());                      \
  }
#define svtkQuaternionOperatorPlus(quaternionType, type)                                            \
  inline quaternionType operator+(const quaternionType& q) const                                   \
  {                                                                                                \
    return quaternionType(                                                                         \
      (static_cast<svtkQuaternion<type> >(*this) + static_cast<svtkQuaternion<type> >(q))            \
        .GetData());                                                                               \
  }
#define svtkQuaternionOperatorMinus(quaternionType, type)                                           \
  inline quaternionType operator-(const quaternionType& q) const                                   \
  {                                                                                                \
    return quaternionType(                                                                         \
      (static_cast<svtkQuaternion<type> >(*this) - static_cast<svtkQuaternion<type> >(q))            \
        .GetData());                                                                               \
  }
#define svtkQuaternionOperatorMultiply(quaternionType, type)                                        \
  inline quaternionType operator*(const quaternionType& q) const                                   \
  {                                                                                                \
    return quaternionType(                                                                         \
      (static_cast<svtkQuaternion<type> >(*this) * static_cast<svtkQuaternion<type> >(q))            \
        .GetData());                                                                               \
  }
#define svtkQuaternionOperatorMultiplyScalar(quaternionType, type)                                  \
  inline quaternionType operator*(const type& scalar) const                                        \
  {                                                                                                \
    return quaternionType((static_cast<svtkQuaternion<type> >(*this) * scalar).GetData());          \
  }
#define svtkQuaternionOperatorDivide(quaternionType, type)                                          \
  inline quaternionType operator/(const quaternionType& q) const                                   \
  {                                                                                                \
    return quaternionType(                                                                         \
      (static_cast<svtkQuaternion<type> >(*this) / static_cast<svtkQuaternion<type> >(q))            \
        .GetData());                                                                               \
  }
#define svtkQuaternionOperatorDivideScalar(quaternionType, type)                                    \
  inline quaternionType operator/(const type& scalar) const                                        \
  {                                                                                                \
    return quaternionType((static_cast<svtkQuaternion<type> >(*this) / scalar).GetData());          \
  }

#define svtkQuaternionOperatorMacro(quaternionType, type)                                           \
  svtkQuaternionIdentity(quaternionType, type);                                                     \
  svtkQuaternionNormalized(quaternionType, type);                                                   \
  svtkQuaternionConjugated(quaternionType, type);                                                   \
  svtkQuaternionInverse(quaternionType, type);                                                      \
  svtkQuaternionUnitLog(quaternionType, type);                                                      \
  svtkQuaternionUnitExp(quaternionType, type);                                                      \
  svtkQuaternionNormalizedWithAngleInDegrees(quaternionType, type);                                 \
  svtkQuaternionSlerp(quaternionType, type);                                                        \
  svtkQuaternionInnerPoint(quaternionType, type);                                                   \
  svtkQuaternionOperatorPlus(quaternionType, type);                                                 \
  svtkQuaternionOperatorMinus(quaternionType, type);                                                \
  svtkQuaternionOperatorMultiply(quaternionType, type);                                             \
  svtkQuaternionOperatorMultiplyScalar(quaternionType, type);                                       \
  svtkQuaternionOperatorDivide(quaternionType, type);                                               \
  svtkQuaternionOperatorDivideScalar(quaternionType, type)

// .NAME svtkQuaternionf - Float quaternion type.
//
// .SECTION Description
// This class is uses svtkQuaternion with float type data.
// For further description, see the templated class svtkQuaternion.
// @sa svtkQuaterniond svtkQuaternion
class svtkQuaternionf : public svtkQuaternion<float>
{
public:
  svtkQuaternionf() {}
  explicit svtkQuaternionf(float w, float x, float y, float z)
    : svtkQuaternion<float>(w, x, y, z)
  {
  }
  explicit svtkQuaternionf(float scalar)
    : svtkQuaternion<float>(scalar)
  {
  }
  explicit svtkQuaternionf(const float* init)
    : svtkQuaternion<float>(init)
  {
  }
  svtkQuaternionOperatorMacro(svtkQuaternionf, float);
};

// .NAME svtkQuaterniond - Double quaternion type.
//
// .SECTION Description
// This class is uses svtkQuaternion with double type data.
// For further description, seethe templated class svtkQuaternion.
// @sa svtkQuaternionf svtkQuaternion
class svtkQuaterniond : public svtkQuaternion<double>
{
public:
  svtkQuaterniond() {}
  explicit svtkQuaterniond(double w, double x, double y, double z)
    : svtkQuaternion<double>(w, x, y, z)
  {
  }
  explicit svtkQuaterniond(double scalar)
    : svtkQuaternion<double>(scalar)
  {
  }
  explicit svtkQuaterniond(const double* init)
    : svtkQuaternion<double>(init)
  {
  }
  svtkQuaternionOperatorMacro(svtkQuaterniond, double);
};

#include "svtkQuaternion.txx"

#endif // svtkQuaternion_h
// SVTK-HeaderTest-Exclude: svtkQuaternion.h
