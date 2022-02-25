/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkColor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkColor
 * @brief   templated type for storage of colors.
 *
 *
 * This class is a templated data type for storing and manipulating fixed size
 * colors. It derives from the svtkVector templated data structure.
 */

#ifndef svtkColor_h
#define svtkColor_h

#include "svtkObject.h" // for legacy macros
#include "svtkTuple.h"

// .NAME svtkColor3 - templated base type for storage of 3 component colors.
//
template <typename T>
class svtkColor3 : public svtkTuple<T, 3>
{
public:
  svtkColor3() {}

  explicit svtkColor3(const T& scalar)
    : svtkTuple<T, 3>(scalar)
  {
  }

  explicit svtkColor3(const T* init)
    : svtkTuple<T, 3>(init)
  {
  }

  svtkColor3(const T& red, const T& green, const T& blue)
  {
    this->Data[0] = red;
    this->Data[1] = green;
    this->Data[2] = blue;
  }

  //@{
  /**
   * Set the red, green and blue components of the color.
   */
  void Set(const T& red, const T& green, const T& blue)
  {
    this->Data[0] = red;
    this->Data[1] = green;
    this->Data[2] = blue;
  }
  //@}

  /**
   * Set the red component of the color, i.e. element 0.
   */
  void SetRed(const T& red) { this->Data[0] = red; }

  /**
   * Get the red component of the color, i.e. element 0.
   */
  const T& GetRed() const { return this->Data[0]; }

  /**
   * Set the green component of the color, i.e. element 1.
   */
  void SetGreen(const T& green) { this->Data[1] = green; }

  /**
   * Get the green component of the color, i.e. element 1.
   */
  const T& GetGreen() const { return this->Data[1]; }

  /**
   * Set the blue component of the color, i.e. element 2.
   */
  void SetBlue(const T& blue) { this->Data[2] = blue; }

  /**
   * Get the blue component of the color, i.e. element 2.
   */
  const T& GetBlue() const { return this->Data[2]; }
};

// .NAME svtkColor4 - templated base type for storage of 4 component colors.
//
template <typename T>
class svtkColor4 : public svtkTuple<T, 4>
{
public:
  svtkColor4() {}

  explicit svtkColor4(const T& scalar)
    : svtkTuple<T, 4>(scalar)
  {
  }

  explicit svtkColor4(const T* init)
    : svtkTuple<T, 4>(init)
  {
  }

  svtkColor4(const T& red, const T& green, const T& blue, const T& alpha)
  {
    this->Data[0] = red;
    this->Data[1] = green;
    this->Data[2] = blue;
    this->Data[3] = alpha;
  }

  //@{
  /**
   * Set the red, green and blue components of the color.
   */
  void Set(const T& red, const T& green, const T& blue)
  {
    this->Data[0] = red;
    this->Data[1] = green;
    this->Data[2] = blue;
  }
  //@}

  //@{
  /**
   * Set the red, green, blue and alpha components of the color.
   */
  void Set(const T& red, const T& green, const T& blue, const T& alpha)
  {
    this->Data[0] = red;
    this->Data[1] = green;
    this->Data[2] = blue;
    this->Data[3] = alpha;
  }
  //@}

  /**
   * Set the red component of the color, i.e. element 0.
   */
  void SetRed(const T& red) { this->Data[0] = red; }

  /**
   * Get the red component of the color, i.e. element 0.
   */
  const T& GetRed() const { return this->Data[0]; }

  /**
   * Set the green component of the color, i.e. element 1.
   */
  void SetGreen(const T& green) { this->Data[1] = green; }

  /**
   * Get the green component of the color, i.e. element 1.
   */
  const T& GetGreen() const { return this->Data[1]; }

  /**
   * Set the blue component of the color, i.e. element 2.
   */
  void SetBlue(const T& blue) { this->Data[2] = blue; }

  /**
   * Get the blue component of the color, i.e. element 2.
   */
  const T& GetBlue() const { return this->Data[2]; }

  /**
   * Set the alpha component of the color, i.e. element 3.
   */
  void SetAlpha(const T& alpha) { this->Data[3] = alpha; }

  /**
   * Get the alpha component of the color, i.e. element 3.
   */
  const T& GetAlpha() const { return this->Data[3]; }
};

/**
 * Some derived classes for the different colors commonly used.
 */
class svtkColor3ub : public svtkColor3<unsigned char>
{
public:
  svtkColor3ub() {}
  explicit svtkColor3ub(unsigned char scalar)
    : svtkColor3<unsigned char>(scalar)
  {
  }
  explicit svtkColor3ub(const unsigned char* init)
    : svtkColor3<unsigned char>(init)
  {
  }

  //@{
  /**
   * Construct a color from a hexadecimal representation such as 0x0000FF (blue).
   */
  explicit svtkColor3ub(int hexSigned)
  {
    unsigned int hex = static_cast<unsigned int>(hexSigned);
    this->Data[2] = hex & 0xff;
    hex >>= 8;
    this->Data[1] = hex & 0xff;
    hex >>= 8;
    this->Data[0] = hex & 0xff;
  }
  //@}

  svtkColor3ub(unsigned char r, unsigned char g, unsigned char b)
    : svtkColor3<unsigned char>(r, g, b)
  {
  }
};

class svtkColor3f : public svtkColor3<float>
{
public:
  svtkColor3f() {}
  explicit svtkColor3f(float scalar)
    : svtkColor3<float>(scalar)
  {
  }
  explicit svtkColor3f(const float* init)
    : svtkColor3<float>(init)
  {
  }
  svtkColor3f(float r, float g, float b)
    : svtkColor3<float>(r, g, b)
  {
  }
};

class svtkColor3d : public svtkColor3<double>
{
public:
  svtkColor3d() {}
  explicit svtkColor3d(double scalar)
    : svtkColor3<double>(scalar)
  {
  }
  explicit svtkColor3d(const double* init)
    : svtkColor3<double>(init)
  {
  }
  svtkColor3d(double r, double g, double b)
    : svtkColor3<double>(r, g, b)
  {
  }
};

class svtkColor4ub : public svtkColor4<unsigned char>
{
public:
  svtkColor4ub() {}
  explicit svtkColor4ub(unsigned char scalar)
    : svtkColor4<unsigned char>(scalar)
  {
  }
  explicit svtkColor4ub(const unsigned char* init)
    : svtkColor4<unsigned char>(init)
  {
  }

  //@{
  /**
   * Construct a color from a hexadecimal representation such as 0x0000FFAA
   * (opaque blue).
   */
  explicit svtkColor4ub(int hexSigned)
  {
    unsigned int hex = static_cast<unsigned int>(hexSigned);
    this->Data[3] = hex & 0xff;
    hex >>= 8;
    this->Data[2] = hex & 0xff;
    hex >>= 8;
    this->Data[1] = hex & 0xff;
    hex >>= 8;
    this->Data[0] = hex & 0xff;
  }
  //@}

  svtkColor4ub(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255)
    : svtkColor4<unsigned char>(r, g, b, a)
  {
  }
  svtkColor4ub(const svtkColor3ub& c)
    : svtkColor4<unsigned char>(c[0], c[1], c[2], 255)
  {
  }
};

class svtkColor4f : public svtkColor4<float>
{
public:
  svtkColor4f() {}
  explicit svtkColor4f(float scalar)
    : svtkColor4<float>(scalar)
  {
  }
  explicit svtkColor4f(const float* init)
    : svtkColor4<float>(init)
  {
  }
  svtkColor4f(float r, float g, float b, float a = 1.0)
    : svtkColor4<float>(r, g, b, a)
  {
  }
};

class svtkColor4d : public svtkColor4<double>
{
public:
  svtkColor4d() {}
  explicit svtkColor4d(double scalar)
    : svtkColor4<double>(scalar)
  {
  }
  explicit svtkColor4d(const double* init)
    : svtkColor4<double>(init)
  {
  }
  svtkColor4d(double r, double g, double b, double a = 1.0)
    : svtkColor4<double>(r, g, b, a)
  {
  }
};

#endif // svtkColor_h
// SVTK-HeaderTest-Exclude: svtkColor.h
