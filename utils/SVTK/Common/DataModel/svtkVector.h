/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVector.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkVector
 * @brief   templated base type for storage of vectors.
 *
 *
 * This class is a templated data type for storing and manipulating fixed size
 * vectors, which can be used to represent two and three dimensional points. The
 * memory layout is a contiguous array of the specified type, such that a
 * float[2] can be cast to a svtkVector2f and manipulated. Also a float[6] could
 * be cast and used as a svtkVector2f[3].
 */

#ifndef svtkVector_h
#define svtkVector_h

#include "svtkObject.h" // for legacy macros
#include "svtkTuple.h"

#include <cmath> // For math functions

template <typename T, int Size>
class svtkVector : public svtkTuple<T, Size>
{
public:
  svtkVector() {}

  /**
   * Initialize all of the vector's elements with the supplied scalar.
   */
  explicit svtkVector(const T& scalar)
    : svtkTuple<T, Size>(scalar)
  {
  }

  /**
   * Initialize the vector's elements with the elements of the supplied array.
   * Note that the supplied pointer must contain at least as many elements as
   * the vector, or it will result in access to out of bounds memory.
   */
  explicit svtkVector(const T* init)
    : svtkTuple<T, Size>(init)
  {
  }

  //@{
  /**
   * Get the squared norm of the vector.
   */
  T SquaredNorm() const
  {
    T result = 0;
    for (int i = 0; i < Size; ++i)
    {
      result += this->Data[i] * this->Data[i];
    }
    return result;
  }
  //@}

  /**
   * Get the norm of the vector, i.e. its length.
   */
  double Norm() const { return sqrt(static_cast<double>(this->SquaredNorm())); }

  //@{
  /**
   * Normalize the vector in place.
   * \return The length of the vector.
   */
  double Normalize()
  {
    const double norm(this->Norm());
    if (norm == 0.0)
    {
      return 0.0;
    }
    const double inv(1.0 / norm);
    for (int i = 0; i < Size; ++i)
    {
      this->Data[i] = static_cast<T>(this->Data[i] * inv);
    }
    return norm;
  }
  //@}

  //@{
  /**
   * Return the normalized form of this vector.
   * \return The normalized form of this vector.
   */
  svtkVector<T, Size> Normalized() const
  {
    svtkVector<T, Size> temp(*this);
    temp.Normalize();
    return temp;
  }
  //@}

  //@{
  /**
   * The dot product of this and the supplied vector.
   */
  T Dot(const svtkVector<T, Size>& other) const
  {
    T result(0);
    for (int i = 0; i < Size; ++i)
    {
      result += this->Data[i] * other[i];
    }
    return result;
  }
  //@}

  //@{
  /**
   * Cast the vector to the specified type, returning the result.
   */
  template <typename TR>
  svtkVector<TR, Size> Cast() const
  {
    svtkVector<TR, Size> result;
    for (int i = 0; i < Size; ++i)
    {
      result[i] = static_cast<TR>(this->Data[i]);
    }
    return result;
  }
  //@}
};

// .NAME svtkVector2 - templated base type for storage of 2D vectors.
//
template <typename T>
class svtkVector2 : public svtkVector<T, 2>
{
public:
  svtkVector2() {}

  explicit svtkVector2(const T& scalar)
    : svtkVector<T, 2>(scalar)
  {
  }

  explicit svtkVector2(const T* init)
    : svtkVector<T, 2>(init)
  {
  }

  svtkVector2(const T& x, const T& y)
  {
    this->Data[0] = x;
    this->Data[1] = y;
  }

  //@{
  /**
   * Set the x and y components of the vector.
   */
  void Set(const T& x, const T& y)
  {
    this->Data[0] = x;
    this->Data[1] = y;
  }
  //@}

  /**
   * Set the x component of the vector, i.e. element 0.
   */
  void SetX(const T& x) { this->Data[0] = x; }

  /**
   * Get the x component of the vector, i.e. element 0.
   */
  const T& GetX() const { return this->Data[0]; }

  /**
   * Set the y component of the vector, i.e. element 1.
   */
  void SetY(const T& y) { this->Data[1] = y; }

  /**
   * Get the y component of the vector, i.e. element 1.
   */
  const T& GetY() const { return this->Data[1]; }

  //@{
  /**
   * Lexicographical comparison of two vector.
   */
  bool operator<(const svtkVector2<T>& v) const
  {
    return (this->Data[0] < v.Data[0]) || (this->Data[0] == v.Data[0] && this->Data[1] < v.Data[1]);
  }
  //@}
};

// .NAME svtkVector3 - templated base type for storage of 3D vectors.
//
template <typename T>
class svtkVector3 : public svtkVector<T, 3>
{
public:
  svtkVector3() {}

  explicit svtkVector3(const T& scalar)
    : svtkVector<T, 3>(scalar)
  {
  }

  explicit svtkVector3(const T* init)
    : svtkVector<T, 3>(init)
  {
  }

  svtkVector3(const T& x, const T& y, const T& z)
  {
    this->Data[0] = x;
    this->Data[1] = y;
    this->Data[2] = z;
  }

  //@{
  /**
   * Set the x, y and z components of the vector.
   */
  void Set(const T& x, const T& y, const T& z)
  {
    this->Data[0] = x;
    this->Data[1] = y;
    this->Data[2] = z;
  }
  //@}

  /**
   * Set the x component of the vector, i.e. element 0.
   */
  void SetX(const T& x) { this->Data[0] = x; }

  /**
   * Get the x component of the vector, i.e. element 0.
   */
  const T& GetX() const { return this->Data[0]; }

  /**
   * Set the y component of the vector, i.e. element 1.
   */
  void SetY(const T& y) { this->Data[1] = y; }

  /**
   * Get the y component of the vector, i.e. element 1.
   */
  const T& GetY() const { return this->Data[1]; }

  /**
   * Set the z component of the vector, i.e. element 2.
   */
  void SetZ(const T& z) { this->Data[2] = z; }

  /**
   * Get the z component of the vector, i.e. element 2.
   */
  const T& GetZ() const { return this->Data[2]; }

  //@{
  /**
   * Return the cross product of this X other.
   */
  svtkVector3<T> Cross(const svtkVector3<T>& other) const
  {
    svtkVector3<T> res;
    res[0] = this->Data[1] * other.Data[2] - this->Data[2] * other.Data[1];
    res[1] = this->Data[2] * other.Data[0] - this->Data[0] * other.Data[2];
    res[2] = this->Data[0] * other.Data[1] - this->Data[1] * other.Data[0];
    return res;
  }
  //@}

  //@{
  /**
   * Lexicographical comparison of two vector.
   */
  bool operator<(const svtkVector3<T>& v) const
  {
    return (this->Data[0] < v.Data[0]) ||
      (this->Data[0] == v.Data[0] && this->Data[1] < v.Data[1]) ||
      (this->Data[0] == v.Data[0] && this->Data[1] == v.Data[1] && this->Data[2] < v.Data[2]);
  }
  //@}
};

// .NAME svtkVector4 - templated base type for storage of 4D vectors.
//
template <typename T>
class svtkVector4 : public svtkVector<T, 4>
{
public:
  svtkVector4() {}

  explicit svtkVector4(const T& scalar)
    : svtkVector<T, 4>(scalar)
  {
  }

  explicit svtkVector4(const T* init)
    : svtkVector<T, 4>(init)
  {
  }

  svtkVector4(const T& x, const T& y, const T& z, const T& w)
  {
    this->Data[0] = x;
    this->Data[1] = y;
    this->Data[2] = z;
    this->Data[3] = w;
  }

  //@{
  /**
   * Set the x, y, z and w components of a 3D vector in homogeneous coordinates.
   */
  void Set(const T& x, const T& y, const T& z, const T& w)
  {
    this->Data[0] = x;
    this->Data[1] = y;
    this->Data[2] = z;
    this->Data[3] = w;
  }
  //@}

  /**
   * Set the x component of the vector, i.e. element 0.
   */
  void SetX(const T& x) { this->Data[0] = x; }

  /**
   * Get the x component of the vector, i.e. element 0.
   */
  const T& GetX() const { return this->Data[0]; }

  /**
   * Set the y component of the vector, i.e. element 1.
   */
  void SetY(const T& y) { this->Data[1] = y; }

  /**
   * Get the y component of the vector, i.e. element 1.
   */
  const T& GetY() const { return this->Data[1]; }

  /**
   * Set the z component of the vector, i.e. element 2.
   */
  void SetZ(const T& z) { this->Data[2] = z; }

  /**
   * Get the z component of the vector, i.e. element 2.
   */
  const T& GetZ() const { return this->Data[2]; }

  /**
   * Set the w component of the vector, i.e. element 3.
   */
  void SetW(const T& w) { this->Data[3] = w; }

  /**
   * Get the w component of the vector, i.e. element 3.
   */
  const T& GetW() const { return this->Data[3]; }
  //@}
};

/**
 * Some inline functions for the derived types.
 */
#define svtkVectorNormalized(vectorType, type, size)                                                \
  vectorType Normalized() const                                                                    \
  {                                                                                                \
    return vectorType(svtkVector<type, size>::Normalized().GetData());                              \
  }

#define svtkVectorDerivedMacro(vectorType, type, size)                                              \
  svtkVectorNormalized(vectorType, type, size);                                                     \
  explicit vectorType(type s)                                                                      \
    : Superclass(s)                                                                                \
  {                                                                                                \
  }                                                                                                \
  explicit vectorType(const type* i)                                                               \
    : Superclass(i)                                                                                \
  {                                                                                                \
  }                                                                                                \
  explicit vectorType(const svtkTuple<type, size>& o)                                               \
    : Superclass(o.GetData())                                                                      \
  {                                                                                                \
  }                                                                                                \
  vectorType(const svtkVector<type, size>& o)                                                       \
    : Superclass(o.GetData())                                                                      \
  {                                                                                                \
  }

//@{
/**
 * Some derived classes for the different vectors commonly used.
 */
class svtkVector2i : public svtkVector2<int>
{
public:
  typedef svtkVector2<int> Superclass;
  svtkVector2i() {}
  svtkVector2i(int x, int y)
    : svtkVector2<int>(x, y)
  {
  }
  svtkVectorDerivedMacro(svtkVector2i, int, 2);
};
//@}

class svtkVector2f : public svtkVector2<float>
{
public:
  typedef svtkVector2<float> Superclass;
  svtkVector2f() {}
  svtkVector2f(float x, float y)
    : svtkVector2<float>(x, y)
  {
  }
  svtkVectorDerivedMacro(svtkVector2f, float, 2);
};

class svtkVector2d : public svtkVector2<double>
{
public:
  typedef svtkVector2<double> Superclass;
  svtkVector2d() {}
  svtkVector2d(double x, double y)
    : svtkVector2<double>(x, y)
  {
  }
  svtkVectorDerivedMacro(svtkVector2d, double, 2);
};

#define svtkVector3Cross(vectorType, type)                                                          \
  vectorType Cross(const vectorType& other) const                                                  \
  {                                                                                                \
    return vectorType(svtkVector3<type>::Cross(other).GetData());                                   \
  }

class svtkVector3i : public svtkVector3<int>
{
public:
  typedef svtkVector3<int> Superclass;
  svtkVector3i() {}
  svtkVector3i(int x, int y, int z)
    : svtkVector3<int>(x, y, z)
  {
  }
  svtkVectorDerivedMacro(svtkVector3i, int, 3);
  svtkVector3Cross(svtkVector3i, int);
};

class svtkVector3f : public svtkVector3<float>
{
public:
  typedef svtkVector3<float> Superclass;
  svtkVector3f() {}
  svtkVector3f(float x, float y, float z)
    : svtkVector3<float>(x, y, z)
  {
  }
  svtkVectorDerivedMacro(svtkVector3f, float, 3);
  svtkVector3Cross(svtkVector3f, float);
};

class svtkVector3d : public svtkVector3<double>
{
public:
  typedef svtkVector3<double> Superclass;
  svtkVector3d() {}
  svtkVector3d(double x, double y, double z)
    : svtkVector3<double>(x, y, z)
  {
  }
  svtkVectorDerivedMacro(svtkVector3d, double, 3);
  svtkVector3Cross(svtkVector3d, double);
};

class svtkVector4d : public svtkVector4<double>
{
public:
  using Superclass = svtkVector4<double>;
  svtkVector4d() {}
  svtkVector4d(double x, double y, double z, double w)
    : svtkVector4<double>(x, y, z, w){};
  svtkVectorDerivedMacro(svtkVector4d, double, 4);
};

#endif // svtkVector_h
// SVTK-HeaderTest-Exclude: svtkVector.h
