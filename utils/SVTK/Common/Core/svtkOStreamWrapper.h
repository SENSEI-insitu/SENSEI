/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOStreamWrapper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOStreamWrapper
 * @brief   Wrapper for C++ ostream.  Internal SVTK use only.
 *
 * Provides a wrapper around the C++ ostream so that SVTK source files
 * need not include the full C++ streams library.  This is intended to
 * prevent cluttering of the translation unit and speed up
 * compilation.  Experimentation has revealed between 10% and 60% less
 * time for compilation depending on the platform.  This wrapper is
 * used by the macros in svtkSetGet.h.
 */

#ifndef svtkOStreamWrapper_h
#define svtkOStreamWrapper_h

#include "svtkCommonCoreModule.h"

#ifndef SVTK_SYSTEM_INCLUDES_INSIDE
Do_not_include_svtkOStreamWrapper_directly_svtkSystemIncludes_includes_it;
#endif

class svtkIndent;
class svtkObjectBase;
class svtkLargeInteger;
class svtkSmartPointerBase;
class svtkStdString;

class SVTKCOMMONCORE_EXPORT SVTK_WRAPEXCLUDE svtkOStreamWrapper
{
  class std_string;

public:
  //@{
  /**
   * Construct class to reference a real ostream.  All methods and
   * operators will be forwarded.
   */
  svtkOStreamWrapper(ostream& os);
  svtkOStreamWrapper(svtkOStreamWrapper& r);
  //@}

  virtual ~svtkOStreamWrapper();

  /**
   * Type for a fake endl.
   */
  struct EndlType
  {
  };

  //@{
  /**
   * Forward this output operator to the real ostream.
   */
  svtkOStreamWrapper& operator<<(const EndlType&);
  svtkOStreamWrapper& operator<<(const svtkIndent&);
  svtkOStreamWrapper& operator<<(svtkObjectBase&);
  svtkOStreamWrapper& operator<<(const svtkLargeInteger&);
  svtkOStreamWrapper& operator<<(const svtkSmartPointerBase&);
  svtkOStreamWrapper& operator<<(const svtkStdString&);
  svtkOStreamWrapper& operator<<(const char*);
  svtkOStreamWrapper& operator<<(void*);
  svtkOStreamWrapper& operator<<(char);
  svtkOStreamWrapper& operator<<(short);
  svtkOStreamWrapper& operator<<(int);
  svtkOStreamWrapper& operator<<(long);
  svtkOStreamWrapper& operator<<(long long);
  svtkOStreamWrapper& operator<<(unsigned char);
  svtkOStreamWrapper& operator<<(unsigned short);
  svtkOStreamWrapper& operator<<(unsigned int);
  svtkOStreamWrapper& operator<<(unsigned long);
  svtkOStreamWrapper& operator<<(unsigned long long);
  svtkOStreamWrapper& operator<<(float);
  svtkOStreamWrapper& operator<<(double);
  svtkOStreamWrapper& operator<<(bool);
  //@}

  // Work-around for IBM Visual Age bug in overload resolution.
#if defined(__IBMCPP__)
  svtkOStreamWrapper& WriteInternal(const char*);
  svtkOStreamWrapper& WriteInternal(void*);
  template <typename T>
  svtkOStreamWrapper& operator<<(T* p)
  {
    return this->WriteInternal(p);
  }
#endif

  svtkOStreamWrapper& operator<<(void (*)(void*));
  svtkOStreamWrapper& operator<<(void* (*)(void*));
  svtkOStreamWrapper& operator<<(int (*)(void*));
  svtkOStreamWrapper& operator<<(int* (*)(void*));
  svtkOStreamWrapper& operator<<(float* (*)(void*));
  svtkOStreamWrapper& operator<<(const char* (*)(void*));
  svtkOStreamWrapper& operator<<(void (*)(void*, int*));

  // Accept std::string without a declaration.
  template <template <typename, typename, typename> class S>
  svtkOStreamWrapper& operator<<(const S<char, std::char_traits<char>, std::allocator<char> >& s)
  {
    return *this << reinterpret_cast<std_string const&>(s);
  }

  /**
   * Forward the write method to the real stream.
   */
  svtkOStreamWrapper& write(const char*, unsigned long);

  /**
   * Get a reference to the real ostream.
   */
  ostream& GetOStream();

  /**
   * Allow conversion to the real ostream type.  This allows an
   * instance of svtkOStreamWrapper to look like ostream when passing to a
   * function argument.
   */
  operator ostream&();

  /**
   * Forward conversion to bool to the real ostream.
   */
  operator int();

  /**
   * Forward the flush method to the real ostream.
   */
  void flush();

  //@{
  /**
   * Implementation detail to allow macros to provide an endl that may
   * or may not be used.
   */
  static void UseEndl(const EndlType&) {}
  //@}
protected:
  // Reference to the real ostream.
  ostream& ostr;

private:
  svtkOStreamWrapper& operator=(const svtkOStreamWrapper& r) = delete;
  svtkOStreamWrapper& operator<<(std_string const&);
};

#endif
// SVTK-HeaderTest-Exclude: svtkOStreamWrapper.h
