/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVectorOperators.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkVectorOperators_h
#define svtkVectorOperators_h

// This set of operators enhance the svtkVector classes, allowing various
// operator overloads one might expect.
#include "svtkVector.h"

// Description:
// Unary minus / negation of vector.
template <typename A, int Size>
svtkVector<A, Size> operator-(const svtkVector<A, Size>& v)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = -v[i];
  }
  return ret;
}

// Description:
// Performs addition of vectors of the same basic type.
template <typename A, int Size>
svtkVector<A, Size> operator+(const svtkVector<A, Size>& v1, const svtkVector<A, Size>& v2)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = v1[i] + v2[i];
  }
  return ret;
}

// Description:
// Performs subtraction of vectors of the same basic type.
template <typename A, int Size>
svtkVector<A, Size> operator-(const svtkVector<A, Size>& v1, const svtkVector<A, Size>& v2)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = v1[i] - v2[i];
  }
  return ret;
}

// Description:
// Performs multiplication of vectors of the same basic type.
template <typename A, int Size>
svtkVector<A, Size> operator*(const svtkVector<A, Size>& v1, const svtkVector<A, Size>& v2)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = v1[i] * v2[i];
  }
  return ret;
}

// Description:
// Performs multiplication of vectors by a scalar value.
template <typename A, typename B, int Size>
svtkVector<A, Size> operator*(const svtkVector<A, Size>& v1, const B& scalar)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = v1[i] * scalar;
  }
  return ret;
}

// Description:
// Performs divisiom of vectors of the same type.
template <typename A, int Size>
svtkVector<A, Size> operator/(const svtkVector<A, Size>& v1, const svtkVector<A, Size>& v2)
{
  svtkVector<A, Size> ret;
  for (int i = 0; i < Size; ++i)
  {
    ret[i] = v1[i] / v2[i];
  }
  return ret;
}

// Description:
// Several macros to define the various operator overloads for the vectors.
#define svtkVectorOperatorNegate(vectorType, type, size)                                            \
  inline vectorType operator-(const vectorType& v)                                                 \
  {                                                                                                \
    return vectorType((-static_cast<svtkVector<type, size> >(v)).GetData());                        \
  }
#define svtkVectorOperatorPlus(vectorType, type, size)                                              \
  inline vectorType operator+(const vectorType& v1, const vectorType& v2)                          \
  {                                                                                                \
    return vectorType(                                                                             \
      (static_cast<svtkVector<type, size> >(v1) + static_cast<svtkVector<type, size> >(v2))          \
        .GetData());                                                                               \
  }
#define svtkVectorOperatorMinus(vectorType, type, size)                                             \
  inline vectorType operator-(const vectorType& v1, const vectorType& v2)                          \
  {                                                                                                \
    return vectorType(                                                                             \
      (static_cast<svtkVector<type, size> >(v1) - static_cast<svtkVector<type, size> >(v2))          \
        .GetData());                                                                               \
  }
#define svtkVectorOperatorMultiply(vectorType, type, size)                                          \
  inline vectorType operator*(const vectorType& v1, const vectorType& v2)                          \
  {                                                                                                \
    return vectorType(                                                                             \
      (static_cast<svtkVector<type, size> >(v1) * static_cast<svtkVector<type, size> >(v2))          \
        .GetData());                                                                               \
  }
#define svtkVectorOperatorMultiplyScalar(vectorType, type, size)                                    \
  template <typename B>                                                                            \
  inline vectorType operator*(const vectorType& v1, const B& scalar)                               \
  {                                                                                                \
    return vectorType((static_cast<svtkVector<type, size> >(v1) * scalar).GetData());               \
  }
#define svtkVectorOperatorMultiplyScalarPre(vectorType, type, size)                                 \
  template <typename B>                                                                            \
  inline vectorType operator*(const B& scalar, const vectorType& v1)                               \
  {                                                                                                \
    return vectorType((static_cast<svtkVector<type, size> >(v1) * scalar).GetData());               \
  }
#define svtkVectorOperatorDivide(vectorType, type, size)                                            \
  inline vectorType operator/(const vectorType& v1, const vectorType& v2)                          \
  {                                                                                                \
    return vectorType(                                                                             \
      (static_cast<svtkVector<type, size> >(v1) / static_cast<svtkVector<type, size> >(v2))          \
        .GetData());                                                                               \
  }

#define svtkVectorOperatorMacro(vectorType, type, size)                                             \
  svtkVectorOperatorNegate(vectorType, type, size);                                                 \
  svtkVectorOperatorPlus(vectorType, type, size);                                                   \
  svtkVectorOperatorMinus(vectorType, type, size);                                                  \
  svtkVectorOperatorMultiply(vectorType, type, size);                                               \
  svtkVectorOperatorMultiplyScalar(vectorType, type, size);                                         \
  svtkVectorOperatorMultiplyScalarPre(vectorType, type, size);                                      \
  svtkVectorOperatorDivide(vectorType, type, size)

// Description:
// Overload the operators for the common types.
svtkVectorOperatorMacro(svtkVector2i, int, 2);
svtkVectorOperatorMacro(svtkVector2f, float, 2);
svtkVectorOperatorMacro(svtkVector2d, double, 2);
svtkVectorOperatorMacro(svtkVector3i, int, 3);
svtkVectorOperatorMacro(svtkVector3f, float, 3);
svtkVectorOperatorMacro(svtkVector3d, double, 3);

#endif
// SVTK-HeaderTest-Exclude: svtkVectorOperators.h
