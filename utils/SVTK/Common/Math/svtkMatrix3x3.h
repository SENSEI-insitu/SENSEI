/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMatrix3x3.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMatrix3x3
 * @brief   represent and manipulate 3x3 transformation matrices
 *
 * svtkMatrix3x3 is a class to represent and manipulate 3x3 matrices.
 * Specifically, it is designed to work on 3x3 transformation matrices
 * found in 2D rendering using homogeneous coordinates [x y w].
 *
 * @sa
 * svtkTransform2D
 */

#ifndef svtkMatrix3x3_h
#define svtkMatrix3x3_h

#include "svtkCommonMathModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONMATH_EXPORT svtkMatrix3x3 : public svtkObject
{
  // Some of the methods in here have a corresponding static (class)
  // method taking a pointer to 9 doubles that constitutes a user
  // supplied matrix. This allows C++ clients to allocate double arrays
  // on the stack and manipulate them using svtkMatrix3x3 methods.
  // This is an alternative to allowing svtkMatrix3x3 instances to be
  // created on the stack (which is frowned upon) or doing lots of
  // temporary heap allocation within svtkTransform2D methods,
  // which is inefficient.

public:
  /**
   * Construct a 3x3 identity matrix.
   */
  static svtkMatrix3x3* New();

  svtkTypeMacro(svtkMatrix3x3, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Set the elements of the matrix to the same values as the elements
   * of the source Matrix.
   */
  void DeepCopy(svtkMatrix3x3* source)
  {
    svtkMatrix3x3::DeepCopy(*this->Element, source);
    this->Modified();
  }
  static void DeepCopy(double elements[9], svtkMatrix3x3* source)
  {
    svtkMatrix3x3::DeepCopy(elements, *source->Element);
  }
  static void DeepCopy(double elements[9], const double newElements[9]);

  /**
   * Non-static member function. Assigns *from* elements array
   */
  void DeepCopy(const double elements[9])
  {
    this->DeepCopy(*this->Element, elements);
    this->Modified();
  }

  /**
   * Set all of the elements to zero.
   */
  void Zero()
  {
    svtkMatrix3x3::Zero(*this->Element);
    this->Modified();
  }
  static void Zero(double elements[9]);

  /**
   * Set equal to Identity matrix
   */
  void Identity()
  {
    svtkMatrix3x3::Identity(*this->Element);
    this->Modified();
  }
  static void Identity(double elements[9]);

  /**
   * Matrix Inversion (adapted from Richard Carling in "Graphics Gems,"
   * Academic Press, 1990).
   */
  static void Invert(svtkMatrix3x3* in, svtkMatrix3x3* out)
  {
    svtkMatrix3x3::Invert(*in->Element, *out->Element);
    out->Modified();
  }
  void Invert() { svtkMatrix3x3::Invert(this, this); }
  static void Invert(const double inElements[9], double outElements[9]);

  /**
   * Transpose the matrix and put it into out.
   */
  static void Transpose(svtkMatrix3x3* in, svtkMatrix3x3* out)
  {
    svtkMatrix3x3::Transpose(*in->Element, *out->Element);
    out->Modified();
  }
  void Transpose() { svtkMatrix3x3::Transpose(this, this); }
  static void Transpose(const double inElements[9], double outElements[9]);

  /**
   * Multiply a homogeneous coordinate by this matrix, i.e. out = A*in.
   * The in[3] and out[3] can be the same array.
   */
  void MultiplyPoint(const float in[3], float out[3])
  {
    svtkMatrix3x3::MultiplyPoint(*this->Element, in, out);
  }
  void MultiplyPoint(const double in[3], double out[3])
  {
    svtkMatrix3x3::MultiplyPoint(*this->Element, in, out);
  }

  static void MultiplyPoint(const double elements[9], const float in[3], float out[3]);
  static void MultiplyPoint(const double elements[9], const double in[3], double out[3]);

  /**
   * Multiplies matrices a and b and stores the result in c (c=a*b).
   */
  static void Multiply3x3(svtkMatrix3x3* a, svtkMatrix3x3* b, svtkMatrix3x3* c)
  {
    svtkMatrix3x3::Multiply3x3(*a->Element, *b->Element, *c->Element);
  }
  static void Multiply3x3(const double a[9], const double b[9], double c[9]);

  /**
   * Compute adjoint of the matrix and put it into out.
   */
  void Adjoint(svtkMatrix3x3* in, svtkMatrix3x3* out)
  {
    svtkMatrix3x3::Adjoint(*in->Element, *out->Element);
  }
  static void Adjoint(const double inElements[9], double outElements[9]);

  /**
   * Compute the determinant of the matrix and return it.
   */
  double Determinant() { return svtkMatrix3x3::Determinant(*this->Element); }
  static double Determinant(const double elements[9]);

  /**
   * Sets the element i,j in the matrix.
   */
  void SetElement(int i, int j, double value);

  /**
   * Returns the element i,j from the matrix.
   */
  double GetElement(int i, int j) const { return this->Element[i][j]; }

  // Descption:
  // Returns true if this matrix is equal to the identity matrix.
  bool IsIdentity();

  /**
   * Return a pointer to the first element of the matrix (double[9]).
   */
  double* GetData() { return *this->Element; }

  /**
   * Return a pointer to the first element of the matrix (double[9]).
   */
  const double* GetData() const { return *this->Element; }

protected:
  svtkMatrix3x3();
  ~svtkMatrix3x3() override;

  double Element[3][3]; // The elements of the 3x3 matrix

private:
  svtkMatrix3x3(const svtkMatrix3x3&) = delete;
  void operator=(const svtkMatrix3x3&) = delete;
};

inline void svtkMatrix3x3::SetElement(int i, int j, double value)
{
  if (this->Element[i][j] != value)
  {
    this->Element[i][j] = value;
    this->Modified();
  }
}

inline bool svtkMatrix3x3::IsIdentity()
{
  double* M = *this->Element;
  if (M[0] == 1.0 && M[4] == 1.0 && M[8] == 1.0 && M[1] == 0.0 && M[2] == 0.0 && M[3] == 0.0 &&
    M[5] == 0.0 && M[6] == 0.0 && M[7] == 0.0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

#endif
