/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOStreamWrapper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSystemIncludes.h" // Cannot include svtkOStreamWrapper.h directly.

#include "svtkIOStream.h"
#include "svtkIndent.h"
#include "svtkLargeInteger.h"
#include "svtkObjectBase.h"
#include "svtkSmartPointerBase.h"
#include "svtkStdString.h"

#include <string>

#define SVTKOSTREAM_OPERATOR(type)                                                                  \
  svtkOStreamWrapper& svtkOStreamWrapper::operator<<(type a)                                         \
  {                                                                                                \
    this->ostr << a;                                                                               \
    return *this;                                                                                  \
  }

#define SVTKOSTREAM_OPERATOR_FUNC(arg)                                                              \
  svtkOStreamWrapper& svtkOStreamWrapper::operator<<(arg)                                            \
  {                                                                                                \
    this->ostr << a;                                                                               \
    return *this;                                                                                  \
  }

//----------------------------------------------------------------------------
svtkOStreamWrapper::svtkOStreamWrapper(ostream& os)
  : ostr(os)
{
}

//----------------------------------------------------------------------------
svtkOStreamWrapper::svtkOStreamWrapper(svtkOStreamWrapper& r)
  : ostr(r.ostr)
{
}

//----------------------------------------------------------------------------
svtkOStreamWrapper::~svtkOStreamWrapper() = default;

//----------------------------------------------------------------------------
svtkOStreamWrapper& svtkOStreamWrapper::operator<<(const EndlType&)
{
  this->ostr << endl;
  return *this;
}

//----------------------------------------------------------------------------
SVTKOSTREAM_OPERATOR(const svtkIndent&);
SVTKOSTREAM_OPERATOR(svtkObjectBase&);
SVTKOSTREAM_OPERATOR(const svtkLargeInteger&);
SVTKOSTREAM_OPERATOR(const svtkSmartPointerBase&);
SVTKOSTREAM_OPERATOR(const svtkStdString&);
SVTKOSTREAM_OPERATOR(const char*);
SVTKOSTREAM_OPERATOR(void*);
SVTKOSTREAM_OPERATOR(char);
SVTKOSTREAM_OPERATOR(short);
SVTKOSTREAM_OPERATOR(int);
SVTKOSTREAM_OPERATOR(long);
SVTKOSTREAM_OPERATOR(long long);
SVTKOSTREAM_OPERATOR(unsigned char);
SVTKOSTREAM_OPERATOR(unsigned short);
SVTKOSTREAM_OPERATOR(unsigned int);
SVTKOSTREAM_OPERATOR(unsigned long);
SVTKOSTREAM_OPERATOR(unsigned long long);
SVTKOSTREAM_OPERATOR(float);
SVTKOSTREAM_OPERATOR(double);
SVTKOSTREAM_OPERATOR(bool);
SVTKOSTREAM_OPERATOR_FUNC(void (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(void* (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(int (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(int* (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(float* (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(const char* (*a)(void*));
SVTKOSTREAM_OPERATOR_FUNC(void (*a)(void*, int*));

//----------------------------------------------------------------------------
svtkOStreamWrapper& svtkOStreamWrapper::operator<<(std_string const& s)
{
  this->ostr << reinterpret_cast<std::string const&>(s);
  return *this;
}

//----------------------------------------------------------------------------
#if defined(__IBMCPP__)
svtkOStreamWrapper& svtkOStreamWrapper::WriteInternal(const char* a)
{
  this->ostr << a;
  return *this;
}
svtkOStreamWrapper& svtkOStreamWrapper::WriteInternal(void* a)
{
  this->ostr << a;
  return *this;
}
#endif

//----------------------------------------------------------------------------
svtkOStreamWrapper& svtkOStreamWrapper::write(const char* str, unsigned long size)
{
  this->ostr.write(str, size);
  return *this;
}

//----------------------------------------------------------------------------
ostream& svtkOStreamWrapper::GetOStream()
{
  return this->ostr;
}

//----------------------------------------------------------------------------
svtkOStreamWrapper::operator ostream&()
{
  return this->ostr;
}

//----------------------------------------------------------------------------
svtkOStreamWrapper::operator int()
{
  return this->ostr ? 1 : 0;
}

//----------------------------------------------------------------------------
void svtkOStreamWrapper::flush()
{
  this->ostr.flush();
}
