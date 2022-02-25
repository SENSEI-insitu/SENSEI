#ifndef svtkVariantInlineOperators_h
#define svtkVariantInlineOperators_h

#include <climits>

namespace svtk
{

// ----------------------------------------------------------------------

// First we have several helper functions that will determine what
// type we're actually dealing with.  With any luck the compiler will
// inline these so they have very little overhead.

inline bool IsSigned64Bit(int VariantType)
{
  return ((VariantType == SVTK_LONG_LONG) || (VariantType == SVTK_TYPE_INT64));
}

inline bool IsSigned(int VariantType)
{
#if (CHAR_MIN == SCHAR_MIN && CHAR_MAX == SCHAR_MAX)
  // the char type is signed on this compiler
  return ((VariantType == SVTK_CHAR) || (VariantType == SVTK_SIGNED_CHAR) ||
    (VariantType == SVTK_SHORT) || (VariantType == SVTK_INT) || (VariantType == SVTK_LONG) ||
    (VariantType == SVTK_ID_TYPE) || IsSigned64Bit(VariantType));
#else
  // char is unsigned
  return ((VariantType == SVTK_SIGNED_CHAR) || (VariantType == SVTK_SHORT) ||
    (VariantType == SVTK_INT) || (VariantType == SVTK_LONG) || (VariantType == SVTK_ID_TYPE) ||
    IsSigned64Bit(VariantType));
#endif
}

// ----------------------------------------------------------------------

inline bool IsFloatingPoint(int VariantType)
{
  return ((VariantType == SVTK_FLOAT) || (VariantType == SVTK_DOUBLE));
}

// ----------------------------------------------------------------------

inline bool CompareSignedUnsignedEqual(
  const svtkVariant& SignedVariant, const svtkVariant& UnsignedVariant)
{
  // If the signed value is less than zero then they cannot possibly
  // be equal.
  svtkTypeInt64 A = SignedVariant.ToTypeInt64();
  return (A >= 0) && (A == UnsignedVariant.ToTypeInt64());
}

// ----------------------------------------------------------------------

inline bool CompareSignedUnsignedLessThan(
  const svtkVariant& SignedVariant, const svtkVariant& UnsignedVariant)
{
  svtkTypeInt64 A = SignedVariant.ToTypeInt64();
  return ((A < 0) || (static_cast<svtkTypeUInt64>(A) < UnsignedVariant.ToTypeUInt64()));
}

// ----------------------------------------------------------------------

inline bool CompareUnsignedSignedLessThan(
  const svtkVariant& UnsignedVariant, const svtkVariant& SignedVariant)
{
  svtkTypeInt64 B = SignedVariant.ToTypeInt64();
  return ((B > 0) && (UnsignedVariant.ToTypeUInt64() < static_cast<svtkTypeUInt64>(B)));
}

// ----------------------------------------------------------------------

inline bool CompareSignedLessThan(const svtkVariant& A, const svtkVariant& B)
{
  return (A.ToTypeInt64() < B.ToTypeInt64());
}

// ----------------------------------------------------------------------

inline bool CompareUnsignedLessThan(const svtkVariant& A, const svtkVariant& B)
{
  return (A.ToTypeUInt64() < B.ToTypeUInt64());
}
}

// ----------------------------------------------------------------------

inline bool svtkVariant::operator==(const svtkVariant& other) const
{
  // First test: nullptr values are always equal to one another and
  // unequal to anything else.
  if (!(this->Valid && other.Valid))
  {
    return (!(this->Valid || other.Valid));
  }

  // Second test: SVTK objects can only be compared with other SVTK
  // objects.
  if ((this->Type == SVTK_OBJECT) || (other.Type == SVTK_OBJECT))
  {
    return ((this->Type == SVTK_OBJECT) && (other.Type == SVTK_OBJECT) &&
      (this->Data.SVTKObject == other.Data.SVTKObject));
  }

  // Third test: the STRING type dominates all else.  If either item
  // is a string then they must both be compared as strings.
  if ((this->Type == SVTK_STRING) || (other.Type == SVTK_STRING))
  {
    return (this->ToString() == other.ToString());
  }

  // Fourth test: the Unicode STRING type dominates all else.  If either item
  // is a unicode string then they must both be compared as strings.
  if ((this->Type == SVTK_UNICODE_STRING) || (other.Type == SVTK_UNICODE_STRING))
  {
    return (this->ToUnicodeString() == other.ToUnicodeString());
  }

  // Fifth: floating point dominates integer types.
  // Demote to the lowest-floating-point precision for the comparison.
  // This effectively makes the lower-precision number an interval
  // corresponding to the range of double values that get rounded to
  // that float. Otherwise, comparisons of numbers that cannot fit in
  // the smaller mantissa exactly will never be equal to their
  // corresponding higher-precision representations.
  if (this->Type == SVTK_FLOAT || other.Type == SVTK_FLOAT)
  {
    return this->ToFloat() == other.ToFloat();
  }
  else if (this->Type == SVTK_DOUBLE || other.Type == SVTK_DOUBLE)
  {
    return (this->ToDouble() == other.ToDouble());
  }

  // Sixth: we must be comparing integers.

  // 6A: catch signed/unsigned comparison.  If the signed object is
  // less than zero then they cannot be equal.
  bool thisSigned = svtk::IsSigned(this->Type);
  bool otherSigned = svtk::IsSigned(other.Type);

  if (thisSigned ^ otherSigned)
  {
    if (thisSigned)
    {
      return svtk::CompareSignedUnsignedEqual(*this, other);
    }
    else
    {
      return svtk::CompareSignedUnsignedEqual(other, *this);
    }
  }
  else // 6B: both are signed or both are unsigned.  In either event
       // all we have to do is check whether the bit patterns are
       // equal.
  {
    return (this->ToTypeInt64() == other.ToTypeInt64());
  }
}

// ----------------------------------------------------------------------

inline bool svtkVariant::operator<(const svtkVariant& other) const
{
  // First test: a nullptr value is less than anything except another
  // nullptr value.  unequal to anything else.
  if (!(this->Valid && other.Valid))
  {
    return ((!this->Valid) && (other.Valid));
  }

  // Second test: SVTK objects can only be compared with other SVTK
  // objects.
  if ((this->Type == SVTK_OBJECT) || (other.Type == SVTK_OBJECT))
  {
    return ((this->Type == SVTK_OBJECT) && (other.Type == SVTK_OBJECT) &&
      (this->Data.SVTKObject < other.Data.SVTKObject));
  }

  // Third test: the STRING type dominates all else.  If either item
  // is a string then they must both be compared as strings.
  if ((this->Type == SVTK_STRING) || (other.Type == SVTK_STRING))
  {
    return (this->ToString() < other.ToString());
  }

  // Fourth test: the Unicode STRING type dominates all else.  If either item
  // is a unicode string then they must both be compared as strings.
  if ((this->Type == SVTK_UNICODE_STRING) || (other.Type == SVTK_UNICODE_STRING))
  {
    return (this->ToUnicodeString() < other.ToUnicodeString());
  }

  // Fourth: floating point dominates integer types.
  // Demote to the lowest-floating-point precision for the comparison.
  // This effectively makes the lower-precision number an interval
  // corresponding to the range of double values that get rounded to
  // that float. Otherwise, comparisons of numbers that cannot fit in
  // the smaller mantissa exactly will never be equal to their
  // corresponding higher-precision representations.
  if (this->Type == SVTK_FLOAT || other.Type == SVTK_FLOAT)
  {
    return this->ToFloat() < other.ToFloat();
  }
  else if (this->Type == SVTK_DOUBLE || other.Type == SVTK_DOUBLE)
  {
    return (this->ToDouble() < other.ToDouble());
  }

  // Fifth: we must be comparing integers.

  // 5A: catch signed/unsigned comparison.  If the signed object is
  // less than zero then they cannot be equal.
  bool thisSigned = svtk::IsSigned(this->Type);
  bool otherSigned = svtk::IsSigned(other.Type);

  if (thisSigned ^ otherSigned)
  {
    if (thisSigned)
    {
      return svtk::CompareSignedUnsignedLessThan(*this, other);
    }
    else
    {
      return svtk::CompareUnsignedSignedLessThan(*this, other);
    }
  }
  else if (thisSigned)
  {
    return svtk::CompareSignedLessThan(*this, other);
  }
  else
  {
    return svtk::CompareUnsignedLessThan(*this, other);
  }
}

// ----------------------------------------------------------------------

// Below this point are operators defined in terms of other operators.
// Again, this may sacrifice some speed, but reduces the chance of
// inconsistent behavior.

// ----------------------------------------------------------------------

inline bool svtkVariant::operator!=(const svtkVariant& other) const
{
  return !(this->operator==(other));
}

inline bool svtkVariant::operator>(const svtkVariant& other) const
{
  return (!(this->operator==(other) || this->operator<(other)));
}

inline bool svtkVariant::operator<=(const svtkVariant& other) const
{
  return (this->operator==(other) || this->operator<(other));
}

inline bool svtkVariant::operator>=(const svtkVariant& other) const
{
  return (!this->operator<(other));
}


#endif
// SVTK-HeaderTest-Exclude: svtkVariantInlineOperators.h
