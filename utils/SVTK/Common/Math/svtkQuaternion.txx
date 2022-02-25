/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuaternion.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkQuaternion.h"

#ifndef svtkQuaternion_txx
#define svtkQuaternion_txx

#include "svtkMath.h"

#include <cmath>

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T>::svtkQuaternion()
{
  this->ToIdentity();
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T>::svtkQuaternion(const T& w, const T& x, const T& y, const T& z)
{
  this->Data[0] = w;
  this->Data[1] = x;
  this->Data[2] = y;
  this->Data[3] = z;
}

//----------------------------------------------------------------------------
template <typename T>
T svtkQuaternion<T>::SquaredNorm() const
{
  T result = 0.0;
  for (int i = 0; i < 4; ++i)
  {
    result += this->Data[i] * this->Data[i];
  }
  return result;
}

//----------------------------------------------------------------------------
template <typename T>
T svtkQuaternion<T>::Norm() const
{
  return sqrt(this->SquaredNorm());
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::ToIdentity()
{
  this->Set(1.0, 0.0, 0.0, 0.0);
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::Identity()
{
  svtkQuaternion<T> identity(1.0, 0.0, 0.0, 0.0);
  return identity;
}

//----------------------------------------------------------------------------
template <typename T>
T svtkQuaternion<T>::Normalize()
{
  T norm = this->Norm();
  if (norm != 0.0)
  {
    for (int i = 0; i < 4; ++i)
    {
      this->Data[i] /= norm;
    }
  }
  return norm;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::Normalized() const
{
  svtkQuaternion<T> temp(*this);
  temp.Normalize();
  return temp;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::Conjugate()
{
  for (int i = 1; i < 4; ++i)
  {
    this->Data[i] *= -1.0;
  }
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::Conjugated() const
{
  svtkQuaternion<T> ret(*this);
  ret.Conjugate();
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::Invert()
{
  T squareNorm = this->SquaredNorm();
  if (squareNorm != 0.0)
  {
    this->Conjugate();
    for (int i = 0; i < 4; ++i)
    {
      this->Data[i] /= squareNorm;
    }
  }
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::Inverse() const
{
  svtkQuaternion<T> ret(*this);
  ret.Invert();
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
template <typename CastTo>
svtkQuaternion<CastTo> svtkQuaternion<T>::Cast() const
{
  svtkQuaternion<CastTo> result;
  for (int i = 0; i < 4; ++i)
  {
    result[i] = static_cast<CastTo>(this->Data[i]);
  }
  return result;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::Set(const T& w, const T& x, const T& y, const T& z)
{
  this->Data[0] = w;
  this->Data[1] = x;
  this->Data[2] = y;
  this->Data[3] = z;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::Set(T quat[4])
{
  for (int i = 0; i < 4; ++i)
  {
    this->Data[i] = quat[i];
  }
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::Get(T quat[4]) const
{
  for (int i = 0; i < 4; ++i)
  {
    quat[i] = this->Data[i];
  }
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetW(const T& w)
{
  this->Data[0] = w;
}

//----------------------------------------------------------------------------
template <typename T>
const T& svtkQuaternion<T>::GetW() const
{
  return this->Data[0];
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetX(const T& x)
{
  this->Data[1] = x;
}

//----------------------------------------------------------------------------
template <typename T>
const T& svtkQuaternion<T>::GetX() const
{
  return this->Data[1];
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetY(const T& y)
{
  this->Data[2] = y;
}

//----------------------------------------------------------------------------
template <typename T>
const T& svtkQuaternion<T>::GetY() const
{
  return this->Data[2];
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetZ(const T& z)
{
  this->Data[3] = z;
}

//----------------------------------------------------------------------------
template <typename T>
const T& svtkQuaternion<T>::GetZ() const
{
  return this->Data[3];
}

//----------------------------------------------------------------------------
template <typename T>
T svtkQuaternion<T>::GetRotationAngleAndAxis(T axis[3]) const
{
  T w = this->GetW();
  T x = this->GetX();
  T y = this->GetY();
  T z = this->GetZ();
  T f = sqrt(x * x + y * y + z * z);
  if (f != 0.0)
  {
    axis[0] = x / f;
    axis[1] = y / f;
    axis[2] = z / f;
  }
  else
  {
    w = 1.0;
    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 0.0;
  }

  // atan2() provides a more accurate angle result than acos()
  return 2.0 * atan2(f, w);
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetRotationAngleAndAxis(T angle, T axis[3])
{
  this->SetRotationAngleAndAxis(angle, axis[0], axis[1], axis[2]);
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::SetRotationAngleAndAxis(const T& angle, const T& x, const T& y, const T& z)
{
  T axisNorm = x * x + y * y + z * z;
  if (axisNorm != 0.0)
  {
    T f = sin(0.5 * angle);
    this->SetW(cos(0.5 * angle));
    this->SetX((x / axisNorm) * f);
    this->SetY((y / axisNorm) * f);
    this->SetZ((z / axisNorm) * f);
  }
  else
  {
    // set the quaternion for "no rotation"
    this->Set(1.0, 0.0, 0.0, 0.0);
  }
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator+(const svtkQuaternion<T>& q) const
{
  svtkQuaternion<T> ret;
  for (int i = 0; i < 4; ++i)
  {
    ret[i] = this->Data[i] + q[i];
  }
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator-(const svtkQuaternion<T>& q) const
{
  svtkQuaternion<T> ret;
  for (int i = 0; i < 4; ++i)
  {
    ret[i] = this->Data[i] - q[i];
  }
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator*(const svtkQuaternion<T>& q) const
{
  svtkQuaternion<T> ret;
  T ww = this->Data[0] * q[0];
  T wx = this->Data[0] * q[1];
  T wy = this->Data[0] * q[2];
  T wz = this->Data[0] * q[3];

  T xw = this->Data[1] * q[0];
  T xx = this->Data[1] * q[1];
  T xy = this->Data[1] * q[2];
  T xz = this->Data[1] * q[3];

  T yw = this->Data[2] * q[0];
  T yx = this->Data[2] * q[1];
  T yy = this->Data[2] * q[2];
  T yz = this->Data[2] * q[3];

  T zw = this->Data[3] * q[0];
  T zx = this->Data[3] * q[1];
  T zy = this->Data[3] * q[2];
  T zz = this->Data[3] * q[3];

  ret[0] = ww - xx - yy - zz;
  ret[1] = wx + xw + yz - zy;
  ret[2] = wy - xz + yw + zx;
  ret[3] = wz + xy - yx + zw;
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator*(const T& scalar) const
{
  svtkQuaternion<T> ret;
  for (int i = 0; i < 4; ++i)
  {
    ret[i] = this->Data[i] * scalar;
  }
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::operator*=(const T& scalar) const
{
  for (int i = 0; i < 4; ++i)
  {
    this->Data[i] *= scalar;
  }
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator/(const svtkQuaternion<T>& q) const
{
  svtkQuaternion<T> inverseQuaternion = q.Inverse();
  return (*this) * inverseQuaternion;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::operator/(const T& scalar) const
{
  svtkQuaternion<T> ret;
  for (int i = 0; i < 4; ++i)
  {
    ret[i] = this->Data[i] / scalar;
  }
  return ret;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::operator/=(const T& scalar)
{
  for (int i = 0; i < 4; ++i)
  {
    this->Data[i] /= scalar;
  }
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::ToMatrix3x3(T A[3][3]) const
{
  T ww = this->Data[0] * this->Data[0];
  T wx = this->Data[0] * this->Data[1];
  T wy = this->Data[0] * this->Data[2];
  T wz = this->Data[0] * this->Data[3];

  T xx = this->Data[1] * this->Data[1];
  T yy = this->Data[2] * this->Data[2];
  T zz = this->Data[3] * this->Data[3];

  T xy = this->Data[1] * this->Data[2];
  T xz = this->Data[1] * this->Data[3];
  T yz = this->Data[2] * this->Data[3];

  T rr = xx + yy + zz;
  // normalization factor, just in case quaternion was not normalized
  T f;
  if (ww + rr == 0.0) // means the quaternion is (0, 0, 0, 0)
  {
    A[0][0] = 0.0;
    A[1][0] = 0.0;
    A[2][0] = 0.0;
    A[0][1] = 0.0;
    A[1][1] = 0.0;
    A[2][1] = 0.0;
    A[0][2] = 0.0;
    A[1][2] = 0.0;
    A[2][2] = 0.0;
    return;
  }
  f = 1.0 / (ww + rr);

  T s = (ww - rr) * f;
  f *= 2.0;

  A[0][0] = xx * f + s;
  A[1][0] = (xy + wz) * f;
  A[2][0] = (xz - wy) * f;

  A[0][1] = (xy - wz) * f;
  A[1][1] = yy * f + s;
  A[2][1] = (yz + wx) * f;

  A[0][2] = (xz + wy) * f;
  A[1][2] = (yz - wx) * f;
  A[2][2] = zz * f + s;
}

//----------------------------------------------------------------------------
//  The solution is based on
//  Berthold K. P. Horn (1987),
//  "Closed-form solution of absolute orientation using unit quaternions,"
//  Journal of the Optical Society of America A, 4:629-642
template <typename T>
void svtkQuaternion<T>::FromMatrix3x3(const T A[3][3])
{
  T n[4][4];

  // on-diagonal elements
  n[0][0] = A[0][0] + A[1][1] + A[2][2];
  n[1][1] = A[0][0] - A[1][1] - A[2][2];
  n[2][2] = -A[0][0] + A[1][1] - A[2][2];
  n[3][3] = -A[0][0] - A[1][1] + A[2][2];

  // off-diagonal elements
  n[0][1] = n[1][0] = A[2][1] - A[1][2];
  n[0][2] = n[2][0] = A[0][2] - A[2][0];
  n[0][3] = n[3][0] = A[1][0] - A[0][1];

  n[1][2] = n[2][1] = A[1][0] + A[0][1];
  n[1][3] = n[3][1] = A[0][2] + A[2][0];
  n[2][3] = n[3][2] = A[2][1] + A[1][2];

  T eigenvectors[4][4];
  T eigenvalues[4];

  // convert into format that JacobiN can use,
  // then use Jacobi to find eigenvalues and eigenvectors
  T* nTemp[4];
  T* eigenvectorsTemp[4];
  for (int i = 0; i < 4; ++i)
  {
    nTemp[i] = n[i];
    eigenvectorsTemp[i] = eigenvectors[i];
  }
  svtkMath::JacobiN(nTemp, 4, eigenvalues, eigenvectorsTemp);

  // the first eigenvector is the one we want
  for (int i = 0; i < 4; ++i)
  {
    this->Data[i] = eigenvectors[i][0];
  }
}

//----------------------------------------------------------------------------
// This returns the constant angular velocity interpolation of two quaternions
// on the unit quaternion sphere :
// http://web.mit.edu/2.998/www/QuaternionReport1.pdf
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::Slerp(T t, const svtkQuaternion<T>& q1) const
{
  // Canonical scalar product on quaternion
  T cosTheta = this->GetW() * q1.GetW() + this->GetX() * q1.GetX() + this->GetY() * q1.GetY() +
    this->GetZ() * q1.GetZ();

  // To prevent the SLERP interpolation from taking the long path
  // we first check the relative orientation of the two quaternions
  // If the angle is superior to 90 degrees we take the opposite quaternion
  // which is closer and represents the same rotation
  svtkQuaternion<T> qClosest = q1;
  if (cosTheta < 0)
  {
    cosTheta = -cosTheta;
    qClosest = qClosest * -1;
  }

  // To avoid division by zero, perform a linear interpolation (LERP), if our
  // quarternions are nearly in the same direction, otherwise resort
  // to spherical linear interpolation. In the limiting case (for small
  // angles), SLERP is equivalent to LERP.
  T t1, t2;
  if ((1.0 - fabs(cosTheta)) < 1e-6)
  {
    t1 = 1.0 - t;
    t2 = t;
  }
  else
  {
    // Angle (defined by the canonical scalar product for quaternions)
    // between the two quaternions
    const T theta = acos(cosTheta);
    t1 = sin((1.0 - t) * theta) / sin(theta);
    t2 = sin(t * theta) / sin(theta);
  }

  return (*this) * t1 + qClosest * t2;
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::InnerPoint(
  const svtkQuaternion<T>& q1, const svtkQuaternion<T>& q2) const
{
  svtkQuaternion<T> qInv = q1.Inverse();
  svtkQuaternion<T> qL = qInv * q2;
  svtkQuaternion<T> qR = qInv * (*this);

  svtkQuaternion<T> qLLog = qL.UnitLog();
  svtkQuaternion<T> qRLog = qR.UnitLog();
  svtkQuaternion<T> qSum = qLLog + qRLog;
  T w = qSum.GetW();
  qSum /= -4.0;
  qSum.SetW(w);

  svtkQuaternion<T> qExp = qSum.UnitExp();
  return q1 * qExp;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::ToUnitLog()
{
  T axis[3];
  T angle = 0.5 * this->GetRotationAngleAndAxis(axis);

  this->Set(0.0, angle * axis[0], angle * axis[1], angle * axis[2]);
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::UnitLog() const
{
  svtkQuaternion<T> unitLog(*this);
  unitLog.ToUnitLog();
  return unitLog;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::ToUnitExp()
{
  T x = this->GetX();
  T y = this->GetY();
  T z = this->GetZ();
  T angle = sqrt(x * x + y * y + z * z);
  T sinAngle = sin(angle);
  T cosAngle = cos(angle);
  if (angle != 0.0)
  {
    x /= angle;
    y /= angle;
    z /= angle;
  }

  this->Set(cosAngle, sinAngle * x, sinAngle * y, sinAngle * z);
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::UnitExp() const
{
  svtkQuaternion<T> unitExp(*this);
  unitExp.ToUnitExp();
  return unitExp;
}

//----------------------------------------------------------------------------
template <typename T>
void svtkQuaternion<T>::NormalizeWithAngleInDegrees()
{
  this->Normalize();
  this->SetW(svtkMath::DegreesFromRadians(this->GetW()));
}

//----------------------------------------------------------------------------
template <typename T>
svtkQuaternion<T> svtkQuaternion<T>::NormalizedWithAngleInDegrees() const
{
  svtkQuaternion<T> unitSVTK(*this);
  unitSVTK.Normalize();
  unitSVTK.SetW(svtkMath::DegreesFromRadians(unitSVTK.GetW()));
  return unitSVTK;
}

#endif
