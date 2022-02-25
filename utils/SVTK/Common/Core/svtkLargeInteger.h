/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLargeInteger.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLargeInteger
 * @brief   class for arbitrarily large ints
 */

#ifndef svtkLargeInteger_h
#define svtkLargeInteger_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT SVTK_WRAPEXCLUDE svtkLargeInteger
{
public:
  svtkLargeInteger(void);
  svtkLargeInteger(long n);
  svtkLargeInteger(unsigned long n);
  svtkLargeInteger(int n);
  svtkLargeInteger(unsigned int n);
  svtkLargeInteger(const svtkLargeInteger& n);
  svtkLargeInteger(long long n);
  svtkLargeInteger(unsigned long long n);

  ~svtkLargeInteger(void);

  char CastToChar(void) const;
  short CastToShort(void) const;
  int CastToInt(void) const;
  long CastToLong(void) const;
  unsigned long CastToUnsignedLong(void) const;

  int IsEven(void) const;
  int IsOdd(void) const;
  int GetLength(void) const;        // in bits
  int GetBit(unsigned int p) const; // p'th bit (from zero)
  int IsZero() const;               // is zero
  int GetSign(void) const;          // is negative

  void Truncate(unsigned int n); // reduce to lower n bits
  void Complement(void);         // * -1

  bool operator==(const svtkLargeInteger& n) const;
  bool operator!=(const svtkLargeInteger& n) const;
  bool operator<(const svtkLargeInteger& n) const;
  bool operator<=(const svtkLargeInteger& n) const;
  bool operator>(const svtkLargeInteger& n) const;
  bool operator>=(const svtkLargeInteger& n) const;

  svtkLargeInteger& operator=(const svtkLargeInteger& n);
  svtkLargeInteger& operator+=(const svtkLargeInteger& n);
  svtkLargeInteger& operator-=(const svtkLargeInteger& n);
  svtkLargeInteger& operator<<=(int n);
  svtkLargeInteger& operator>>=(int n);
  svtkLargeInteger& operator++(void);
  svtkLargeInteger& operator--(void);
  svtkLargeInteger operator++(int);
  svtkLargeInteger operator--(int);
  svtkLargeInteger& operator*=(const svtkLargeInteger& n);
  svtkLargeInteger& operator/=(const svtkLargeInteger& n);
  svtkLargeInteger& operator%=(const svtkLargeInteger& n);
  // no change of sign for following operators
  svtkLargeInteger& operator&=(const svtkLargeInteger& n);
  svtkLargeInteger& operator|=(const svtkLargeInteger& n);
  svtkLargeInteger& operator^=(const svtkLargeInteger& n);

  svtkLargeInteger operator+(const svtkLargeInteger& n) const;
  svtkLargeInteger operator-(const svtkLargeInteger& n) const;
  svtkLargeInteger operator*(const svtkLargeInteger& n) const;
  svtkLargeInteger operator/(const svtkLargeInteger& n) const;
  svtkLargeInteger operator%(const svtkLargeInteger& n) const;
  // no change of sign for following operators
  svtkLargeInteger operator&(const svtkLargeInteger& n) const;
  svtkLargeInteger operator|(const svtkLargeInteger& n) const;
  svtkLargeInteger operator^(const svtkLargeInteger& n) const;
  svtkLargeInteger operator<<(int n) const;
  svtkLargeInteger operator>>(int n) const;

  friend ostream& operator<<(ostream& s, const svtkLargeInteger& n);
  friend istream& operator>>(istream& s, svtkLargeInteger& n);

private:
  char* Number;
  int Negative;
  unsigned int Sig;
  unsigned int Max;

  // unsigned operators
  bool IsSmaller(const svtkLargeInteger& n) const; // unsigned
  bool IsGreater(const svtkLargeInteger& n) const; // unsigned
  void Expand(unsigned int n);                    // ensure n'th bit exits
  void Contract();                                // remove leading 0s
  void Plus(const svtkLargeInteger& n);            // unsigned
  void Minus(const svtkLargeInteger& n);           // unsigned
};

#endif

// SVTK-HeaderTest-Exclude: svtkLargeInteger.h
