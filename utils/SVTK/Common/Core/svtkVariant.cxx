/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariant.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/

#include "svtkVariant.h"

#include "svtkAbstractArray.h"
#include "svtkArrayIteratorIncludes.h"
#include "svtkDataArray.h"
#include "svtkMath.h"
#include "svtkObjectBase.h"
#include "svtkSetGet.h"
#include "svtkStdString.h"
#include "svtkStringArray.h"
#include "svtkType.h"
#include "svtkUnicodeString.h"
#include "svtkVariantArray.h"

#include "svtksys/SystemTools.hxx"
#include <locale> // C++ locale
#include <sstream>

//----------------------------------------------------------------------------

// Implementation of svtkVariant's
// fast-but-potentially-counterintuitive < operation
bool svtkVariantStrictWeakOrder::operator()(const svtkVariant& s1, const svtkVariant& s2) const
{
  // First sort on type if they are different
  if (s1.Type != s2.Type)
  {
    return s1.Type < s2.Type;
  }

  // Next check for nulls
  if (!(s1.Valid && s2.Valid))
  {
    if (!(s1.Valid || s2.Valid))
    {
      return false; // nulls are equal to one another
    }
    else if (!s1.Valid)
    {
      return true; // null is less than any valid value
    }
    else
    {
      return false;
    }
  }

  switch (s1.Type)
  {
    case SVTK_STRING:
      return (*(s1.Data.String) < *(s2.Data.String));

    case SVTK_UNICODE_STRING:
      return (*(s1.Data.UnicodeString) < *(s2.Data.UnicodeString));

    case SVTK_OBJECT:
      return (s1.Data.SVTKObject < s2.Data.SVTKObject);

    case SVTK_CHAR:
      return (s1.Data.Char < s2.Data.Char);

    case SVTK_SIGNED_CHAR:
      return (s1.Data.SignedChar < s2.Data.SignedChar);

    case SVTK_UNSIGNED_CHAR:
      return (s1.Data.UnsignedChar < s2.Data.UnsignedChar);

    case SVTK_SHORT:
      return (s1.Data.Short < s2.Data.Short);

    case SVTK_UNSIGNED_SHORT:
      return (s1.Data.UnsignedShort < s2.Data.UnsignedShort);

    case SVTK_INT:
      return (s1.Data.Int < s2.Data.Int);

    case SVTK_UNSIGNED_INT:
      return (s1.Data.UnsignedInt < s2.Data.UnsignedInt);

    case SVTK_LONG:
      return (s1.Data.Long < s2.Data.Long);

    case SVTK_UNSIGNED_LONG:
      return (s1.Data.UnsignedLong < s2.Data.UnsignedLong);

    case SVTK_LONG_LONG:
      return (s1.Data.LongLong < s2.Data.LongLong);

    case SVTK_UNSIGNED_LONG_LONG:
      return (s1.Data.UnsignedLongLong < s2.Data.UnsignedLongLong);

    case SVTK_FLOAT:
      return (s1.Data.Float < s2.Data.Float);

    case SVTK_DOUBLE:
      return (s1.Data.Double < s2.Data.Double);

    default:
      cerr << "ERROR: Unhandled type " << s1.Type << " in svtkVariantStrictWeakOrder\n";
      return false;
  }
}

// ----------------------------------------------------------------------

bool svtkVariantStrictEquality::operator()(const svtkVariant& s1, const svtkVariant& s2) const
{
  // First sort on type if they are different
  if (s1.Type != s2.Type)
  {
    cerr << "Types differ: " << s1.Type << " and " << s2.Type << "\n";
    return false;
  }

  // Next check for nulls
  if (!(s1.Valid && s2.Valid))
  {
    cerr << "Validity may differ: " << s1.Valid << " and " << s2.Valid << "\n";
    return (s1.Valid == s2.Valid);
  }

  // At this point we know that both variants contain a valid value.
  switch (s1.Type)
  {
    case SVTK_STRING:
    {
      if (*(s1.Data.String) != *(s2.Data.String))
      {
        cerr << "Strings differ: '" << *(s1.Data.String) << "' and '" << *(s2.Data.String) << "'\n";
      }
      return (*(s1.Data.String) == *(s2.Data.String));
    };

    case SVTK_UNICODE_STRING:
      return (*(s1.Data.UnicodeString) == *(s2.Data.UnicodeString));

    case SVTK_OBJECT:
      return (s1.Data.SVTKObject == s2.Data.SVTKObject);

    case SVTK_CHAR:
      return (s1.Data.Char == s2.Data.Char);

    case SVTK_SIGNED_CHAR:
      return (s1.Data.SignedChar == s2.Data.SignedChar);

    case SVTK_UNSIGNED_CHAR:
      return (s1.Data.UnsignedChar == s2.Data.UnsignedChar);

    case SVTK_SHORT:
      return (s1.Data.Short == s2.Data.Short);

    case SVTK_UNSIGNED_SHORT:
      return (s1.Data.UnsignedShort == s2.Data.UnsignedShort);

    case SVTK_INT:
      return (s1.Data.Int == s2.Data.Int);

    case SVTK_UNSIGNED_INT:
      return (s1.Data.UnsignedInt == s2.Data.UnsignedInt);

    case SVTK_LONG:
      return (s1.Data.Long == s2.Data.Long);

    case SVTK_UNSIGNED_LONG:
      return (s1.Data.UnsignedLong == s2.Data.UnsignedLong);

    case SVTK_LONG_LONG:
      return (s1.Data.LongLong == s2.Data.LongLong);

    case SVTK_UNSIGNED_LONG_LONG:
      return (s1.Data.UnsignedLongLong == s2.Data.UnsignedLongLong);

    case SVTK_FLOAT:
      return (s1.Data.Float == s2.Data.Float);

    case SVTK_DOUBLE:
      return (s1.Data.Double == s2.Data.Double);

    default:
      cerr << "ERROR: Unhandled type " << s1.Type << " in svtkVariantStrictEquality\n";
      return false;
  }
}

// ----------------------------------------------------------------------

bool svtkVariantLessThan::operator()(const svtkVariant& v1, const svtkVariant& v2) const
{
  return v1.operator<(v2);
}

// ----------------------------------------------------------------------

bool svtkVariantEqual::operator()(const svtkVariant& v1, const svtkVariant& v2) const
{
  return v1.operator==(v2);
}

//----------------------------------------------------------------------------
svtkVariant::svtkVariant()
{
  this->Valid = 0;
  this->Type = 0;
}

svtkVariant::svtkVariant(const svtkVariant& other)
{
  this->Valid = other.Valid;
  this->Type = other.Type;
  this->Data = other.Data;
  if (this->Valid)
  {
    switch (other.Type)
    {
      case SVTK_STRING:
        this->Data.String = new svtkStdString(*other.Data.String);
        break;
      case SVTK_UNICODE_STRING:
        this->Data.UnicodeString = new svtkUnicodeString(*other.Data.UnicodeString);
        break;
      case SVTK_OBJECT:
        this->Data.SVTKObject->Register(nullptr);
        break;
    }
  }
}

svtkVariant::svtkVariant(const svtkVariant& s2, unsigned int type)
{
  bool valid = false;

  if (s2.Valid)
  {
    switch (type)
    {
      case SVTK_STRING:
        this->Data.String = new svtkStdString(s2.ToString());
        valid = true;
        break;

      case SVTK_UNICODE_STRING:
        this->Data.UnicodeString = new svtkUnicodeString(s2.ToUnicodeString());
        valid = true;
        break;

      case SVTK_OBJECT:
        this->Data.SVTKObject = s2.ToSVTKObject();
        if (this->Data.SVTKObject)
        {
          this->Data.SVTKObject->Register(nullptr);
          valid = true;
        }
        break;

      case SVTK_CHAR:
        this->Data.Char = s2.ToChar(&valid);
        break;

      case SVTK_SIGNED_CHAR:
        this->Data.SignedChar = s2.ToSignedChar(&valid);
        break;

      case SVTK_UNSIGNED_CHAR:
        this->Data.UnsignedChar = s2.ToUnsignedChar(&valid);
        break;

      case SVTK_SHORT:
        this->Data.Short = s2.ToShort(&valid);
        break;

      case SVTK_UNSIGNED_SHORT:
        this->Data.UnsignedShort = s2.ToUnsignedShort(&valid);
        break;

      case SVTK_INT:
        this->Data.Int = s2.ToInt(&valid);
        break;

      case SVTK_UNSIGNED_INT:
        this->Data.UnsignedInt = s2.ToUnsignedInt(&valid);
        break;

      case SVTK_LONG:
        this->Data.Long = s2.ToLong(&valid);
        break;

      case SVTK_UNSIGNED_LONG:
        this->Data.UnsignedLong = s2.ToUnsignedLong(&valid);
        break;

      case SVTK_LONG_LONG:
        this->Data.LongLong = s2.ToLongLong(&valid);
        break;

      case SVTK_UNSIGNED_LONG_LONG:
        this->Data.UnsignedLongLong = s2.ToUnsignedLongLong(&valid);
        break;

      case SVTK_FLOAT:
        this->Data.Float = s2.ToFloat(&valid);
        break;

      case SVTK_DOUBLE:
        this->Data.Double = s2.ToDouble(&valid);
        break;
    }
  }

  this->Type = (valid ? type : 0);
  this->Valid = valid;
}

svtkVariant& svtkVariant::operator=(const svtkVariant& other)
{
  // Short circuit if assigning to self:
  if (this == &other)
  {
    return *this;
  }

  // First delete current variant item.
  if (this->Valid)
  {
    switch (this->Type)
    {
      case SVTK_STRING:
        delete this->Data.String;
        break;
      case SVTK_UNICODE_STRING:
        delete this->Data.UnicodeString;
        break;
      case SVTK_OBJECT:
        this->Data.SVTKObject->Delete();
        break;
    }
  }

  // Then set the appropriate value.
  this->Valid = other.Valid;
  this->Type = other.Type;
  this->Data = other.Data;
  if (this->Valid)
  {
    switch (other.Type)
    {
      case SVTK_STRING:
        this->Data.String = new svtkStdString(*other.Data.String);
        break;
      case SVTK_UNICODE_STRING:
        this->Data.UnicodeString = new svtkUnicodeString(*other.Data.UnicodeString);
        break;
      case SVTK_OBJECT:
        this->Data.SVTKObject->Register(nullptr);
        break;
    }
  }
  return *this;
}

svtkVariant::~svtkVariant()
{
  if (this->Valid)
  {
    switch (this->Type)
    {
      case SVTK_STRING:
        delete this->Data.String;
        break;
      case SVTK_UNICODE_STRING:
        delete this->Data.UnicodeString;
        break;
      case SVTK_OBJECT:
        this->Data.SVTKObject->Delete();
        break;
    }
  }
}

svtkVariant::svtkVariant(bool value)
{
  this->Data.Char = value;
  this->Valid = 1;
  this->Type = SVTK_CHAR;
}

svtkVariant::svtkVariant(char value)
{
  this->Data.Char = value;
  this->Valid = 1;
  this->Type = SVTK_CHAR;
}

svtkVariant::svtkVariant(unsigned char value)
{
  this->Data.UnsignedChar = value;
  this->Valid = 1;
  this->Type = SVTK_UNSIGNED_CHAR;
}

svtkVariant::svtkVariant(signed char value)
{
  this->Data.SignedChar = value;
  this->Valid = 1;
  this->Type = SVTK_SIGNED_CHAR;
}

svtkVariant::svtkVariant(short value)
{
  this->Data.Short = value;
  this->Valid = 1;
  this->Type = SVTK_SHORT;
}

svtkVariant::svtkVariant(unsigned short value)
{
  this->Data.UnsignedShort = value;
  this->Valid = 1;
  this->Type = SVTK_UNSIGNED_SHORT;
}

svtkVariant::svtkVariant(int value)
{
  this->Data.Int = value;
  this->Valid = 1;
  this->Type = SVTK_INT;
}

svtkVariant::svtkVariant(unsigned int value)
{
  this->Data.UnsignedInt = value;
  this->Valid = 1;
  this->Type = SVTK_UNSIGNED_INT;
}

svtkVariant::svtkVariant(long value)
{
  this->Data.Long = value;
  this->Valid = 1;
  this->Type = SVTK_LONG;
}

svtkVariant::svtkVariant(unsigned long value)
{
  this->Data.UnsignedLong = value;
  this->Valid = 1;
  this->Type = SVTK_UNSIGNED_LONG;
}

svtkVariant::svtkVariant(long long value)
{
  this->Data.LongLong = value;
  this->Valid = 1;
  this->Type = SVTK_LONG_LONG;
}

svtkVariant::svtkVariant(unsigned long long value)
{
  this->Data.UnsignedLongLong = value;
  this->Valid = 1;
  this->Type = SVTK_UNSIGNED_LONG_LONG;
}

svtkVariant::svtkVariant(float value)
{
  this->Data.Float = value;
  this->Valid = 1;
  this->Type = SVTK_FLOAT;
}

svtkVariant::svtkVariant(double value)
{
  this->Data.Double = value;
  this->Valid = 1;
  this->Type = SVTK_DOUBLE;
}

svtkVariant::svtkVariant(const char* value)
{
  this->Valid = 0;
  this->Type = 0;
  if (value)
  {
    this->Data.String = new svtkStdString(value);
    this->Valid = 1;
    this->Type = SVTK_STRING;
  }
}

svtkVariant::svtkVariant(svtkStdString value)
{
  this->Data.String = new svtkStdString(value);
  this->Valid = 1;
  this->Type = SVTK_STRING;
}

svtkVariant::svtkVariant(const svtkUnicodeString& value)
{
  this->Data.UnicodeString = new svtkUnicodeString(value);
  this->Valid = 1;
  this->Type = SVTK_UNICODE_STRING;
}

svtkVariant::svtkVariant(svtkObjectBase* value)
{
  this->Valid = 0;
  this->Type = 0;
  if (value)
  {
    value->Register(nullptr);
    this->Data.SVTKObject = value;
    this->Valid = 1;
    this->Type = SVTK_OBJECT;
  }
}

bool svtkVariant::IsValid() const
{
  return this->Valid != 0;
}

bool svtkVariant::IsString() const
{
  return this->Type == SVTK_STRING;
}

bool svtkVariant::IsUnicodeString() const
{
  return this->Type == SVTK_UNICODE_STRING;
}

bool svtkVariant::IsNumeric() const
{
  return this->IsFloat() || this->IsDouble() || this->IsChar() || this->IsUnsignedChar() ||
    this->IsSignedChar() || this->IsShort() || this->IsUnsignedShort() || this->IsInt() ||
    this->IsUnsignedInt() || this->IsLong() || this->IsUnsignedLong() || this->IsLongLong() ||
    this->IsUnsignedLongLong();
}

bool svtkVariant::IsFloat() const
{
  return this->Type == SVTK_FLOAT;
}

bool svtkVariant::IsDouble() const
{
  return this->Type == SVTK_DOUBLE;
}

bool svtkVariant::IsChar() const
{
  return this->Type == SVTK_CHAR;
}

bool svtkVariant::IsUnsignedChar() const
{
  return this->Type == SVTK_UNSIGNED_CHAR;
}

bool svtkVariant::IsSignedChar() const
{
  return this->Type == SVTK_SIGNED_CHAR;
}

bool svtkVariant::IsShort() const
{
  return this->Type == SVTK_SHORT;
}

bool svtkVariant::IsUnsignedShort() const
{
  return this->Type == SVTK_UNSIGNED_SHORT;
}

bool svtkVariant::IsInt() const
{
  return this->Type == SVTK_INT;
}

bool svtkVariant::IsUnsignedInt() const
{
  return this->Type == SVTK_UNSIGNED_INT;
}

bool svtkVariant::IsLong() const
{
  return this->Type == SVTK_LONG;
}

bool svtkVariant::IsUnsignedLong() const
{
  return this->Type == SVTK_UNSIGNED_LONG;
}

bool svtkVariant::Is__Int64() const
{
  return false;
}

bool svtkVariant::IsUnsigned__Int64() const
{
  return false;
}

bool svtkVariant::IsLongLong() const
{
  return this->Type == SVTK_LONG_LONG;
}

bool svtkVariant::IsUnsignedLongLong() const
{
  return this->Type == SVTK_UNSIGNED_LONG_LONG;
}

bool svtkVariant::IsSVTKObject() const
{
  return this->Type == SVTK_OBJECT;
}

bool svtkVariant::IsArray() const
{
  return this->Type == SVTK_OBJECT && this->Valid && this->Data.SVTKObject->IsA("svtkAbstractArray");
}

unsigned int svtkVariant::GetType() const
{
  return this->Type;
}

const char* svtkVariant::GetTypeAsString() const
{
  if (this->Type == SVTK_OBJECT && this->Valid)
  {
    return this->Data.SVTKObject->GetClassName();
  }
  return svtkImageScalarTypeNameMacro(this->Type);
}

template <typename iterT>
svtkStdString svtkVariantArrayToString(iterT* it)
{
  svtkIdType maxInd = it->GetNumberOfValues();
  std::ostringstream ostr;
  for (svtkIdType i = 0; i < maxInd; i++)
  {
    if (i > 0)
    {
      ostr << " ";
    }
    ostr << it->GetValue(i);
  }
  return ostr.str();
}

svtkStdString svtkVariant::ToString() const
{
  if (!this->IsValid())
  {
    return svtkStdString();
  }
  if (this->IsString())
  {
    return svtkStdString(*(this->Data.String));
  }
  if (this->IsUnicodeString())
  {
    return svtkUnicodeString(*(this->Data.UnicodeString)).utf8_str();
  }
  if (this->IsFloat())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.Float;
    return svtkStdString(ostr.str());
  }
  if (this->IsDouble())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.Double;
    return svtkStdString(ostr.str());
  }
  if (this->IsChar())
  {
    std::ostringstream ostr;
    ostr << this->Data.Char;
    return svtkStdString(ostr.str());
  }
  if (this->IsUnsignedChar())
  {
    std::ostringstream ostr;
    ostr << static_cast<unsigned int>(this->Data.UnsignedChar);
    return svtkStdString(ostr.str());
  }
  if (this->IsSignedChar())
  {
    std::ostringstream ostr;
    ostr << this->Data.SignedChar;
    return svtkStdString(ostr.str());
  }
  if (this->IsShort())
  {
    std::ostringstream ostr;
    ostr << this->Data.Short;
    return svtkStdString(ostr.str());
  }
  if (this->IsUnsignedShort())
  {
    std::ostringstream ostr;
    ostr << this->Data.UnsignedShort;
    return svtkStdString(ostr.str());
  }
  if (this->IsInt())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.Int;
    return svtkStdString(ostr.str());
  }
  if (this->IsUnsignedInt())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.UnsignedInt;
    return svtkStdString(ostr.str());
  }
  if (this->IsLong())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.Long;
    return svtkStdString(ostr.str());
  }
  if (this->IsUnsignedLong())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.UnsignedLong;
    return svtkStdString(ostr.str());
  }
  if (this->IsLongLong())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.LongLong;
    return svtkStdString(ostr.str());
  }
  if (this->IsUnsignedLongLong())
  {
    std::ostringstream ostr;
    ostr.imbue(std::locale::classic());
    ostr << this->Data.UnsignedLongLong;
    return svtkStdString(ostr.str());
  }
  if (this->IsArray())
  {
    svtkAbstractArray* arr = svtkAbstractArray::SafeDownCast(this->Data.SVTKObject);
    svtkArrayIterator* iter = arr->NewIterator();
    svtkStdString str;
    switch (arr->GetDataType())
    {
      svtkArrayIteratorTemplateMacro(str = svtkVariantArrayToString(static_cast<SVTK_TT*>(iter)));
    }
    iter->Delete();
    return str;
  }
  svtkGenericWarningMacro(<< "Cannot convert unknown type (" << this->GetTypeAsString()
                         << ") to a string.");
  return svtkStdString();
}

svtkUnicodeString svtkVariant::ToUnicodeString() const
{
  if (!this->IsValid())
  {
    return svtkUnicodeString();
  }
  if (this->IsString())
  {
    return svtkUnicodeString::from_utf8(*this->Data.String);
  }
  if (this->IsUnicodeString())
  {
    return *this->Data.UnicodeString;
  }

  return svtkUnicodeString::from_utf8(this->ToString());
}

svtkObjectBase* svtkVariant::ToSVTKObject() const
{
  if (this->IsSVTKObject())
  {
    return this->Data.SVTKObject;
  }
  return nullptr;
}

svtkAbstractArray* svtkVariant::ToArray() const
{
  if (this->IsArray())
  {
    return svtkAbstractArray::SafeDownCast(this->Data.SVTKObject);
  }
  return nullptr;
}

// Used internally by svtkVariantStringToNumeric to find non-finite numbers.
// Most numerics do not support non-finite numbers, hence the default simply
// fails.  Overload for doubles and floats detect non-finite numbers they
// support
template <typename T>
T svtkVariantStringToNonFiniteNumeric(svtkStdString svtkNotUsed(str), bool* valid)
{
  if (valid)
    *valid = 0;
  return 0;
}

template <>
double svtkVariantStringToNonFiniteNumeric<double>(svtkStdString str, bool* valid)
{
  if (svtksys::SystemTools::Strucmp(str.c_str(), "nan") == 0)
  {
    if (valid)
      *valid = true;
    return svtkMath::Nan();
  }
  if ((svtksys::SystemTools::Strucmp(str.c_str(), "infinity") == 0) ||
    (svtksys::SystemTools::Strucmp(str.c_str(), "inf") == 0))
  {
    if (valid)
      *valid = true;
    return svtkMath::Inf();
  }
  if ((svtksys::SystemTools::Strucmp(str.c_str(), "-infinity") == 0) ||
    (svtksys::SystemTools::Strucmp(str.c_str(), "-inf") == 0))
  {
    if (valid)
      *valid = true;
    return svtkMath::NegInf();
  }
  if (valid)
    *valid = false;
  return svtkMath::Nan();
}

template <>
float svtkVariantStringToNonFiniteNumeric<float>(svtkStdString str, bool* valid)
{
  return static_cast<float>(svtkVariantStringToNonFiniteNumeric<double>(str, valid));
}

template <typename T>
T svtkVariantStringToNumeric(svtkStdString str, bool* valid, T* svtkNotUsed(ignored) = nullptr)
{
  std::istringstream vstr(str);
  T data = 0;
  vstr >> data;
  if (!vstr.eof())
  {
    // take in white space so that it can reach eof.
    vstr >> std::ws;
  }
  bool v = (!vstr.fail() && vstr.eof());
  if (valid)
    *valid = v;
  if (!v)
  {
    data = svtkVariantStringToNonFiniteNumeric<T>(str, valid);
  }
  return data;
}

//----------------------------------------------------------------------------
// Definition of ToNumeric

#include "svtkVariantToNumeric.cxx"

//----------------------------------------------------------------------------
// Explicitly instantiate the ToNumeric member template to make sure
// the symbols are exported from this object file.
// This explicit instantiation exists to resolve SVTK issue #5791.

#if !defined(SVTK_VARIANT_NO_INSTANTIATE)

#define svtkVariantToNumericInstantiateMacro(x) template x svtkVariant::ToNumeric<x>(bool*, x*) const

svtkVariantToNumericInstantiateMacro(char);
svtkVariantToNumericInstantiateMacro(float);
svtkVariantToNumericInstantiateMacro(double);
svtkVariantToNumericInstantiateMacro(unsigned char);
svtkVariantToNumericInstantiateMacro(signed char);
svtkVariantToNumericInstantiateMacro(short);
svtkVariantToNumericInstantiateMacro(unsigned short);
svtkVariantToNumericInstantiateMacro(int);
svtkVariantToNumericInstantiateMacro(unsigned int);
svtkVariantToNumericInstantiateMacro(long);
svtkVariantToNumericInstantiateMacro(unsigned long);
svtkVariantToNumericInstantiateMacro(long long);
svtkVariantToNumericInstantiateMacro(unsigned long long);

#endif

//----------------------------------------------------------------------------
// Callers causing implicit instantiations of ToNumeric

float svtkVariant::ToFloat(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<float*>(nullptr));
}

double svtkVariant::ToDouble(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<double*>(nullptr));
}

char svtkVariant::ToChar(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<char*>(nullptr));
}

unsigned char svtkVariant::ToUnsignedChar(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<unsigned char*>(nullptr));
}

signed char svtkVariant::ToSignedChar(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<signed char*>(nullptr));
}

short svtkVariant::ToShort(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<short*>(nullptr));
}

unsigned short svtkVariant::ToUnsignedShort(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<unsigned short*>(nullptr));
}

int svtkVariant::ToInt(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<int*>(nullptr));
}

unsigned int svtkVariant::ToUnsignedInt(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<unsigned int*>(nullptr));
}

long svtkVariant::ToLong(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<long*>(nullptr));
}

unsigned long svtkVariant::ToUnsignedLong(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<unsigned long*>(nullptr));
}

long long svtkVariant::ToLongLong(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<long long*>(nullptr));
}

unsigned long long svtkVariant::ToUnsignedLongLong(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<unsigned long long*>(nullptr));
}

svtkTypeInt64 svtkVariant::ToTypeInt64(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<svtkTypeInt64*>(nullptr));
}

svtkTypeUInt64 svtkVariant::ToTypeUInt64(bool* valid) const
{
  return this->ToNumeric(valid, static_cast<svtkTypeUInt64*>(nullptr));
}

bool svtkVariant::IsEqual(const svtkVariant& other) const
{
  return this->operator==(other);
}

ostream& operator<<(ostream& os, const svtkVariant& val)
{
  if (!val.Valid)
  {
    os << "(invalid)";
    return os;
  }
  switch (val.Type)
  {
    case SVTK_STRING:
      if (val.Data.String)
      {
        os << "\"" << val.Data.String->c_str() << "\"";
      }
      else
      {
        os << "\"\"";
      }
      break;
    case SVTK_UNICODE_STRING:
      if (val.Data.UnicodeString)
      {
        os << "\"" << val.Data.UnicodeString->utf8_str() << "\"";
      }
      else
      {
        os << "\"\"";
      }
      break;
    case SVTK_FLOAT:
      os << val.Data.Float;
      break;
    case SVTK_DOUBLE:
      os << val.Data.Double;
      break;
    case SVTK_CHAR:
      os << val.Data.Char;
      break;
    case SVTK_UNSIGNED_CHAR:
      os << val.Data.UnsignedChar;
      break;
    case SVTK_SIGNED_CHAR:
      os << val.Data.SignedChar;
      break;
    case SVTK_SHORT:
      os << val.Data.Short;
      break;
    case SVTK_UNSIGNED_SHORT:
      os << val.Data.UnsignedShort;
      break;
    case SVTK_INT:
      os << val.Data.Int;
      break;
    case SVTK_UNSIGNED_INT:
      os << val.Data.UnsignedInt;
      break;
    case SVTK_LONG:
      os << val.Data.Long;
      break;
    case SVTK_UNSIGNED_LONG:
      os << val.Data.UnsignedLong;
      break;
    case SVTK_LONG_LONG:
      os << val.Data.LongLong;
      break;
    case SVTK_UNSIGNED_LONG_LONG:
      os << val.Data.UnsignedLongLong;
      break;
    case SVTK_OBJECT:
      if (val.Data.SVTKObject)
      {
        os << "(" << val.Data.SVTKObject->GetClassName() << ")" << hex << val.Data.SVTKObject << dec;
      }
      else
      {
        os << "(svtkObjectBase)0x0";
      }
      break;
  }
  return os;
}
