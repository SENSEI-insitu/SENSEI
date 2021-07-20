/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSetGet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   SetGet
 *
 * The SetGet macros are used to interface to instance variables
 * in a standard fashion. This includes properly treating modified time
 * and printing out debug information.
 *
 * Macros are available for built-in types; for character strings;
 * vector arrays of built-in types size 2,3,4; for setting objects; and
 * debug, warning, and error printout information.
 */

#ifndef svtkSetGet_h
#define svtkSetGet_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"
#include <math.h>
#include <type_traits> // for std::underlying type.
#include <typeinfo>

//----------------------------------------------------------------------------
// Check for unsupported old compilers.
#if defined(_MSC_VER) && _MSC_VER < 1900
#error SVTK requires MSVC++ 14.0 aka Visual Studio 2015 or newer
#endif

#if !defined(SWIG)
#if !defined(__clang__) && defined(__GNUC__) &&                                                    \
  (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
#error SVTK requires GCC 4.8 or newer
#endif
#endif

// Convert a macro representing a value to a string.
//
// Example: svtkQuoteMacro(__LINE__) will expand to "1234" whereas
// svtkInternalQuoteMacro(__LINE__) will expand to "__LINE__"
#define svtkInternalQuoteMacro(x) #x
#define svtkQuoteMacro(x) svtkInternalQuoteMacro(x)

// clang-format off
// A macro to get the name of a type
#define svtkImageScalarTypeNameMacro(type)                                                          \
  (((type) == SVTK_VOID) ? "void" :                                                                 \
  (((type) == SVTK_BIT) ? "bit" :                                                                   \
  (((type) == SVTK_CHAR) ? "char" :                                                                 \
  (((type) == SVTK_SIGNED_CHAR) ? "signed char" :                                                   \
  (((type) == SVTK_UNSIGNED_CHAR) ? "unsigned char" :                                               \
  (((type) == SVTK_SHORT) ? "short" :                                                               \
  (((type) == SVTK_UNSIGNED_SHORT) ? "unsigned short" :                                             \
  (((type) == SVTK_INT) ? "int" :                                                                   \
  (((type) == SVTK_UNSIGNED_INT) ? "unsigned int" :                                                 \
  (((type) == SVTK_LONG) ? "long" :                                                                 \
  (((type) == SVTK_UNSIGNED_LONG) ? "unsigned long" :                                               \
  (((type) == SVTK_LONG_LONG) ? "long long" :                                                       \
  (((type) == SVTK_UNSIGNED_LONG_LONG) ? "unsigned long long" :                                     \
  (((type) == 18 /*SVTK___INT64*/) ? "__int64" :                                                    \
  (((type) == 19 /*SVTK_UNSIGNED___INT64*/) ? "unsigned __int64" :                                  \
  (((type) == SVTK_FLOAT) ? "float" :                                                               \
  (((type) == SVTK_DOUBLE) ? "double" :                                                             \
  (((type) == SVTK_ID_TYPE) ? "idtype" :                                                            \
  (((type) == SVTK_STRING) ? "string" :                                                             \
  (((type) == SVTK_UNICODE_STRING) ? "unicode string" :                                             \
  (((type) == SVTK_VARIANT) ? "variant" :                                                           \
  (((type) == SVTK_OBJECT) ? "object" :                                                             \
  "Undefined"))))))))))))))))))))))
// clang-format on

/* Various compiler-specific performance hints. */
#if defined(SWIG)
#define SVTK_ALWAYS_INLINE inline
#define SVTK_ALWAYS_OPTIMIZE_START
#define SVTK_ALWAYS_OPTIMIZE_END
#else

#if defined(SVTK_COMPILER_GCC) //------------------------------------------------

#define SVTK_ALWAYS_INLINE __attribute__((always_inline)) inline
#define SVTK_ALWAYS_OPTIMIZE_START _Pragma("GCC push_options") _Pragma("GCC optimize (\"O3\")")
#define SVTK_ALWAYS_OPTIMIZE_END _Pragma("GCC pop_options")

#elif defined(SVTK_COMPILER_CLANG) //--------------------------------------------

#define SVTK_ALWAYS_INLINE __attribute__((always_inline)) inline
// Clang doesn't seem to support temporarily increasing optimization level,
// only decreasing it.
#define SVTK_ALWAYS_OPTIMIZE_START
#define SVTK_ALWAYS_OPTIMIZE_END

#elif defined(SVTK_COMPILER_ICC) //----------------------------------------------

#define SVTK_ALWAYS_INLINE __attribute((always_inline)) inline
// ICC doesn't seem to support temporarily increasing optimization level,
// only decreasing it.
#define SVTK_ALWAYS_OPTIMIZE_START
#define SVTK_ALWAYS_OPTIMIZE_END

#elif defined(SVTK_COMPILER_MSVC) //---------------------------------------------

#define SVTK_ALWAYS_INLINE __forceinline
#define SVTK_ALWAYS_OPTIMIZE_START _Pragma("optimize(\"tgs\", on)")
// optimize("", on) resets to command line settings
#define SVTK_ALWAYS_OPTIMIZE_END _Pragma("optimize(\"\", on)")

#else //------------------------------------------------------------------------

#define SVTK_ALWAYS_INLINE inline
#define SVTK_ALWAYS_OPTIMIZE_START
#define SVTK_ALWAYS_OPTIMIZE_END

#endif
#endif

//
// Set built-in type.  Creates member Set"name"() (e.g., SetVisibility());
//
#define svtkSetMacro(name, type)                                                                    \
  virtual void Set##name(type _arg)                                                                \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " #name " to " << _arg);  \
    if (this->name != _arg)                                                                        \
    {                                                                                              \
      this->name = _arg;                                                                           \
      this->Modified();                                                                            \
    }                                                                                              \
  }

//
// Get built-in type.  Creates member Get"name"() (e.g., GetVisibility());
//
#define svtkGetMacro(name, type)                                                                    \
  virtual type Get##name()                                                                         \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " of "       \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }

//
// Set character string.  Creates member Set"name"()
// (e.g., SetFilename(char *));
//
#define svtkSetStringMacro(name)                                                                    \
  virtual void Set##name(const char* _arg) svtkSetStringBodyMacro(name, _arg)

// This macro defines a body of set string macro. It can be used either in
// the header file using svtkSetStringMacro or in the implementation.
#define svtkSetStringBodyMacro(name, _arg)                                                          \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to "         \
                  << (_arg ? _arg : "(null)"));                                                    \
    if (this->name == nullptr && _arg == nullptr)                                                  \
    {                                                                                              \
      return;                                                                                      \
    }                                                                                              \
    if (this->name && _arg && (!strcmp(this->name, _arg)))                                         \
    {                                                                                              \
      return;                                                                                      \
    }                                                                                              \
    delete[] this->name;                                                                           \
    if (_arg)                                                                                      \
    {                                                                                              \
      size_t n = strlen(_arg) + 1;                                                                 \
      char* cp1 = new char[n];                                                                     \
      const char* cp2 = (_arg);                                                                    \
      this->name = cp1;                                                                            \
      do                                                                                           \
      {                                                                                            \
        *cp1++ = *cp2++;                                                                           \
      } while (--n);                                                                               \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
      this->name = nullptr;                                                                        \
    }                                                                                              \
    this->Modified();                                                                              \
  }

//
// Get character string.  Creates member Get"name"()
// (e.g., char *GetFilename());
//
#define svtkGetStringMacro(name)                                                                    \
  virtual char* Get##name()                                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " of "       \
                  << (this->name ? this->name : "(null)"));                                        \
    return this->name;                                                                             \
  }

//
// Set built-in type where value is constrained between min/max limits.
// Create member Set"name"() (eg., SetRadius()). #defines are
// convenience for clamping open-ended values.
// The Get"name"MinValue() and Get"name"MaxValue() members return the
// min and max limits.
//
#define svtkSetClampMacro(name, type, min, max)                                                     \
  virtual void Set##name(type _arg)                                                                \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to "         \
                  << _arg);                                                                        \
    if (this->name != (_arg < min ? min : (_arg > max ? max : _arg)))                              \
    {                                                                                              \
      this->name = (_arg < min ? min : (_arg > max ? max : _arg));                                 \
      this->Modified();                                                                            \
    }                                                                                              \
  }                                                                                                \
  virtual type Get##name##MinValue() { return min; }                                               \
  virtual type Get##name##MaxValue() { return max; }

//
// This macro defines a body of set object macro. It can be used either in
// the header file svtkSetObjectMacro or in the implementation one
// svtkSetObjectMacro. It sets the pointer to object; uses svtkObject
// reference counting methodology. Creates method
// Set"name"() (e.g., SetPoints()).
//
#define svtkSetObjectBodyMacro(name, type, args)                                                    \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to "         \
                  << args);                                                                        \
    if (this->name != args)                                                                        \
    {                                                                                              \
      type* tempSGMacroVar = this->name;                                                           \
      this->name = args;                                                                           \
      if (this->name != nullptr)                                                                   \
      {                                                                                            \
        this->name->Register(this);                                                                \
      }                                                                                            \
      if (tempSGMacroVar != nullptr)                                                               \
      {                                                                                            \
        tempSGMacroVar->UnRegister(this);                                                          \
      }                                                                                            \
      this->Modified();                                                                            \
    }                                                                                              \
  }

//
// This macro defines a body of set object macro with
// a smart pointer class member.
//
#define svtkSetSmartPointerBodyMacro(name, type, args)                                              \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to "         \
                  << args);                                                                        \
    if (this->name != args)                                                                        \
    {                                                                                              \
      this->name = args;                                                                           \
      this->Modified();                                                                            \
    }                                                                                              \
  }

//
// Set pointer to object; uses svtkObject reference counting methodology.
// Creates method Set"name"() (e.g., SetPoints()). This macro should
// be used in the header file.
//
#define svtkSetObjectMacro(name, type)                                                              \
  virtual void Set##name(type* _arg) { svtkSetObjectBodyMacro(name, type, _arg); }

//
// Set pointer to a smart pointer class member.
// Creates method Set"name"() (e.g., SetPoints()). This macro should
// be used in the header file.
//
#define svtkSetSmartPointerMacro(name, type)                                                        \
  virtual void Set##name(type* _arg) { svtkSetSmartPointerBodyMacro(name, type, _arg); }

//
// Set pointer to object; uses svtkObject reference counting methodology.
// Creates method Set"name"() (e.g., SetPoints()). This macro should
// be used in the implementation file. You will also have to write
// prototype in the header file. The prototype should look like this:
// virtual void Set"name"("type" *);
//
// Please use svtkCxxSetObjectMacro not svtkSetObjectImplementationMacro.
// The first one is just for people who already used it.
#define svtkSetObjectImplementationMacro(class, name, type) svtkCxxSetObjectMacro(class, name, type)

#define svtkCxxSetObjectMacro(class, name, type)                                                    \
  void class ::Set##name(type* _arg) { svtkSetObjectBodyMacro(name, type, _arg); }

//
// Set pointer to smart pointer.
// This macro is used to define the implementation.
//
#define svtkCxxSetSmartPointerMacro(class, name, type)                                              \
  void class ::Set##name(type* _arg) { svtkSetSmartPointerBodyMacro(name, type, _arg); }

//
// Get pointer to object wrapped in svtkNew.  Creates member Get"name"
// (e.g., GetPoints()).  This macro should be used in the header file.
//
#define svtkGetNewMacro(name, type)                                                                 \
  virtual type* Get##name()                                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " #name " address "     \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }

//
// Get pointer to object.  Creates member Get"name" (e.g., GetPoints()).
// This macro should be used in the header file.
//
#define svtkGetObjectMacro(name, type)                                                              \
  virtual type* Get##name()                                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " #name " address "     \
                  << static_cast<type*>(this->name));                                              \
    return this->name;                                                                             \
  }

//
// Get pointer to object in a smart pointer class member.
// This is only an alias and is similar to svtkGetObjectMacro.
//
#define svtkGetSmartPointerMacro(name, type) svtkGetObjectMacro(name, type)

//
// Create members "name"On() and "name"Off() (e.g., DebugOn() DebugOff()).
// Set method must be defined to use this macro.
//
#define svtkBooleanMacro(name, type)                                                                \
  virtual void name##On() { this->Set##name(static_cast<type>(1)); }                               \
  virtual void name##Off() { this->Set##name(static_cast<type>(0)); }

//
// Following set macros for vectors define two members for each macro.  The
// first
// allows setting of individual components (e.g, SetColor(float,float,float)),
// the second allows setting from an array (e.g., SetColor(float* rgb[3])).
// The macros vary in the size of the vector they deal with.
//
#define svtkSetVector2Macro(name, type)                                                             \
  virtual void Set##name(type _arg1, type _arg2)                                                   \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to ("        \
                  << _arg1 << "," << _arg2 << ")");                                                \
    if ((this->name[0] != _arg1) || (this->name[1] != _arg2))                                      \
    {                                                                                              \
      this->name[0] = _arg1;                                                                       \
      this->name[1] = _arg2;                                                                       \
      this->Modified();                                                                            \
    }                                                                                              \
  }                                                                                                \
  void Set##name(const type _arg[2]) { this->Set##name(_arg[0], _arg[1]); }

#define svtkGetVector2Macro(name, type)                                                             \
  virtual type* Get##name() SVTK_SIZEHINT(2)                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer "  \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type& _arg1, type& _arg2)                                                 \
  {                                                                                                \
    _arg1 = this->name[0];                                                                         \
    _arg2 = this->name[1];                                                                         \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = ("       \
                  << _arg1 << "," << _arg2 << ")");                                                \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type _arg[2]) { this->Get##name(_arg[0], _arg[1]); }

#define svtkSetVector3Macro(name, type)                                                             \
  virtual void Set##name(type _arg1, type _arg2, type _arg3)                                       \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to ("        \
                  << _arg1 << "," << _arg2 << "," << _arg3 << ")");                                \
    if ((this->name[0] != _arg1) || (this->name[1] != _arg2) || (this->name[2] != _arg3))          \
    {                                                                                              \
      this->name[0] = _arg1;                                                                       \
      this->name[1] = _arg2;                                                                       \
      this->name[2] = _arg3;                                                                       \
      this->Modified();                                                                            \
    }                                                                                              \
  }                                                                                                \
  virtual void Set##name(const type _arg[3]) { this->Set##name(_arg[0], _arg[1], _arg[2]); }

#define svtkGetVector3Macro(name, type)                                                             \
  virtual type* Get##name() SVTK_SIZEHINT(3)                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer "  \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type& _arg1, type& _arg2, type& _arg3)                                    \
  {                                                                                                \
    _arg1 = this->name[0];                                                                         \
    _arg2 = this->name[1];                                                                         \
    _arg3 = this->name[2];                                                                         \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = ("       \
                  << _arg1 << "," << _arg2 << "," << _arg3 << ")");                                \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type _arg[3]) { this->Get##name(_arg[0], _arg[1], _arg[2]); }

#define svtkSetVector4Macro(name, type)                                                             \
  virtual void Set##name(type _arg1, type _arg2, type _arg3, type _arg4)                           \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to ("        \
                  << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << ")");                \
    if ((this->name[0] != _arg1) || (this->name[1] != _arg2) || (this->name[2] != _arg3) ||        \
      (this->name[3] != _arg4))                                                                    \
    {                                                                                              \
      this->name[0] = _arg1;                                                                       \
      this->name[1] = _arg2;                                                                       \
      this->name[2] = _arg3;                                                                       \
      this->name[3] = _arg4;                                                                       \
      this->Modified();                                                                            \
    }                                                                                              \
  }                                                                                                \
  virtual void Set##name(const type _arg[4])                                                       \
  {                                                                                                \
    this->Set##name(_arg[0], _arg[1], _arg[2], _arg[3]);                                           \
  }

#define svtkGetVector4Macro(name, type)                                                             \
  virtual type* Get##name() SVTK_SIZEHINT(4)                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer "  \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type& _arg1, type& _arg2, type& _arg3, type& _arg4)                       \
  {                                                                                                \
    _arg1 = this->name[0];                                                                         \
    _arg2 = this->name[1];                                                                         \
    _arg3 = this->name[2];                                                                         \
    _arg4 = this->name[3];                                                                         \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = ("       \
                  << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << ")");                \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type _arg[4]) { this->Get##name(_arg[0], _arg[1], _arg[2], _arg[3]); }

#define svtkSetVector6Macro(name, type)                                                             \
  virtual void Set##name(type _arg1, type _arg2, type _arg3, type _arg4, type _arg5, type _arg6)   \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to ("        \
                  << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << "," << _arg5 << ","  \
                  << _arg6 << ")");                                                                \
    if ((this->name[0] != _arg1) || (this->name[1] != _arg2) || (this->name[2] != _arg3) ||        \
      (this->name[3] != _arg4) || (this->name[4] != _arg5) || (this->name[5] != _arg6))            \
    {                                                                                              \
      this->name[0] = _arg1;                                                                       \
      this->name[1] = _arg2;                                                                       \
      this->name[2] = _arg3;                                                                       \
      this->name[3] = _arg4;                                                                       \
      this->name[4] = _arg5;                                                                       \
      this->name[5] = _arg6;                                                                       \
      this->Modified();                                                                            \
    }                                                                                              \
  }                                                                                                \
  virtual void Set##name(const type _arg[6])                                                       \
  {                                                                                                \
    this->Set##name(_arg[0], _arg[1], _arg[2], _arg[3], _arg[4], _arg[5]);                         \
  }

#define svtkGetVector6Macro(name, type)                                                             \
  virtual type* Get##name() SVTK_SIZEHINT(6)                                                        \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer "  \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(                                                                          \
    type& _arg1, type& _arg2, type& _arg3, type& _arg4, type& _arg5, type& _arg6)                  \
  {                                                                                                \
    _arg1 = this->name[0];                                                                         \
    _arg2 = this->name[1];                                                                         \
    _arg3 = this->name[2];                                                                         \
    _arg4 = this->name[3];                                                                         \
    _arg5 = this->name[4];                                                                         \
    _arg6 = this->name[5];                                                                         \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = ("       \
                  << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << "," << _arg5 << ","  \
                  << _arg6 << ")");                                                                \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type _arg[6])                                                             \
  {                                                                                                \
    this->Get##name(_arg[0], _arg[1], _arg[2], _arg[3], _arg[4], _arg[5]);                         \
  }

//
// General set vector macro creates a single method that copies specified
// number of values into object.
// Examples: void SetColor(c,3)
//
#define svtkSetVectorMacro(name, type, count)                                                       \
  virtual void Set##name(const type data[])                                                        \
  {                                                                                                \
    int i;                                                                                         \
    for (i = 0; i < count; i++)                                                                    \
    {                                                                                              \
      if (data[i] != this->name[i])                                                                \
      {                                                                                            \
        break;                                                                                     \
      }                                                                                            \
    }                                                                                              \
    if (i < count)                                                                                 \
    {                                                                                              \
      for (i = 0; i < count; i++)                                                                  \
      {                                                                                            \
        this->name[i] = data[i];                                                                   \
      }                                                                                            \
      this->Modified();                                                                            \
    }                                                                                              \
  }

//
// Get vector macro defines two methods. One returns pointer to type
// (i.e., array of type). This is for efficiency. The second copies data
// into user provided array. This is more object-oriented.
// Examples: float *GetColor() and void GetColor(float c[count]).
//
#define svtkGetVectorMacro(name, type, count)                                                       \
  virtual type* Get##name() SVTK_SIZEHINT(count)                                                    \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer "  \
                  << this->name);                                                                  \
    return this->name;                                                                             \
  }                                                                                                \
  SVTK_WRAPEXCLUDE                                                                                  \
  virtual void Get##name(type data[count])                                                         \
  {                                                                                                \
    for (int i = 0; i < count; i++)                                                                \
    {                                                                                              \
      data[i] = this->name[i];                                                                     \
    }                                                                                              \
  }

// Use a global function which actually calls:
//  svtkOutputWindow::GetInstance()->DisplayText();
// This is to avoid svtkObject #include of svtkOutputWindow
// while svtkOutputWindow #includes svtkObject

extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayText(const char*);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayErrorText(const char*);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayWarningText(const char*);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayGenericWarningText(const char*);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayDebugText(const char*);

// overloads that allow providing information about the filename and lineno
// generating the message.
class svtkObject;
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayErrorText(
  const char*, int, const char*, svtkObject* sourceObj);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayWarningText(
  const char*, int, const char*, svtkObject* sourceObj);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayGenericWarningText(
  const char*, int, const char*);
extern SVTKCOMMONCORE_EXPORT void svtkOutputWindowDisplayDebugText(
  const char*, int, const char*, svtkObject* sourceObj);

//
// This macro is used for any output that may not be in an instance method
// svtkGenericWarningMacro(<< "this is debug info" << this->SomeVariable);
//
#define svtkGenericWarningMacro(x)                                                                  \
  do                                                                                               \
  {                                                                                                \
    if (svtkObject::GetGlobalWarningDisplay())                                                      \
    {                                                                                              \
      svtkOStreamWrapper::EndlType endl;                                                            \
      svtkOStreamWrapper::UseEndl(endl);                                                            \
      svtkOStrStreamWrapper svtkmsg;                                                                 \
      svtkmsg << "" x;                                                                              \
      svtkOutputWindowDisplayGenericWarningText(__FILE__, __LINE__, svtkmsg.str());                  \
      svtkmsg.rdbuf()->freeze(0);                                                                   \
    }                                                                                              \
  } while (false)

//
// This macro is used for debug statements in instance methods
// svtkDebugMacro(<< "this is debug info" << this->SomeVariable);
//
#define svtkDebugMacro(x) svtkDebugWithObjectMacro(this, x)

//
// This macro is used to print out warning messages.
// svtkWarningMacro(<< "Warning message" << variable);
//
#define svtkWarningMacro(x) svtkWarningWithObjectMacro(this, x)

//
// This macro is used to print out errors
// svtkErrorMacro(<< "Error message" << variable);
//
#define svtkErrorMacro(x) svtkErrorWithObjectMacro(this, x)

//
// This macro is used to print out errors
// svtkErrorWithObjectMacro(self, << "Error message" << variable);
// self can be null
// Using two casts here so that nvcc compiler can handle const this
// pointer properly
//
#define svtkErrorWithObjectMacro(self, x)                                                           \
  do                                                                                               \
  {                                                                                                \
    if (svtkObject::GetGlobalWarningDisplay())                                                      \
    {                                                                                              \
      svtkOStreamWrapper::EndlType endl;                                                            \
      svtkOStreamWrapper::UseEndl(endl);                                                            \
      svtkOStrStreamWrapper svtkmsg;                                                                 \
      svtkObject* _object = const_cast<svtkObject*>(static_cast<const svtkObject*>(self));            \
      if (_object)                                                                                 \
      {                                                                                            \
        svtkmsg << _object->GetClassName() << " (" << _object << "): ";                             \
      }                                                                                            \
      else                                                                                         \
      {                                                                                            \
        svtkmsg << "(nullptr): ";                                                                   \
      }                                                                                            \
      svtkmsg << "" x;                                                                              \
      svtkOutputWindowDisplayErrorText(__FILE__, __LINE__, svtkmsg.str(), _object);                  \
      svtkmsg.rdbuf()->freeze(0);                                                                   \
      svtkObject::BreakOnError();                                                                   \
    }                                                                                              \
  } while (false)

//
// This macro is used to print out warnings
// svtkWarningWithObjectMacro(self, "Warning message" << variable);
// self can be null
// Using two casts here so that nvcc compiler can handle const this
// pointer properly
//
#define svtkWarningWithObjectMacro(self, x)                                                         \
  do                                                                                               \
  {                                                                                                \
    if (svtkObject::GetGlobalWarningDisplay())                                                      \
    {                                                                                              \
      svtkOStreamWrapper::EndlType endl;                                                            \
      svtkOStreamWrapper::UseEndl(endl);                                                            \
      svtkOStrStreamWrapper svtkmsg;                                                                 \
      svtkObject* _object = const_cast<svtkObject*>(static_cast<const svtkObject*>(self));            \
      if (_object)                                                                                 \
      {                                                                                            \
        svtkmsg << _object->GetClassName() << " (" << _object << "): ";                             \
      }                                                                                            \
      else                                                                                         \
      {                                                                                            \
        svtkmsg << "(nullptr): ";                                                                   \
      }                                                                                            \
      svtkmsg << "" x;                                                                              \
      svtkOutputWindowDisplayWarningText(__FILE__, __LINE__, svtkmsg.str(), _object);                \
      svtkmsg.rdbuf()->freeze(0);                                                                   \
    }                                                                                              \
  } while (false)

/**
 * This macro is used to print out debug message
 * svtkDebugWithObjectMacro(self, "Warning message" << variable);
 * self can be null
 * Using two casts here so that nvcc compiler can handle const this
 * pointer properly
 */
#ifdef NDEBUG
#define svtkDebugWithObjectMacro(self, x)                                                           \
  do                                                                                               \
  {                                                                                                \
  } while (false)
#else
#define svtkDebugWithObjectMacro(self, x)                                                           \
  do                                                                                               \
  {                                                                                                \
    svtkObject* _object = const_cast<svtkObject*>(static_cast<const svtkObject*>(self));              \
    if ((!_object || _object->GetDebug()) && svtkObject::GetGlobalWarningDisplay())                 \
    {                                                                                              \
      svtkOStreamWrapper::EndlType endl;                                                            \
      svtkOStreamWrapper::UseEndl(endl);                                                            \
      svtkOStrStreamWrapper svtkmsg;                                                                 \
      if (_object)                                                                                 \
      {                                                                                            \
        svtkmsg << _object->GetClassName() << " (" << _object << "): ";                             \
      }                                                                                            \
      else                                                                                         \
      {                                                                                            \
        svtkmsg << "(nullptr): ";                                                                   \
      }                                                                                            \
      svtkmsg << "" x;                                                                              \
      svtkOutputWindowDisplayDebugText(__FILE__, __LINE__, svtkmsg.str(), _object);                  \
      svtkmsg.rdbuf()->freeze(0);                                                                   \
    }                                                                                              \
  } while (false)
#endif

//
// This macro is used to quiet compiler warnings about unused parameters
// to methods. Only use it when the parameter really shouldn't be used.
// Don't use it as a way to shut up the compiler while you take your
// sweet time getting around to implementing the method.
//
#define svtkNotUsed(x)

//
// This macro is used for functions which may not be used in a translation unit
// due to different paths taken based on template types. Please give a reason
// why the function may be considered unused (within a translation unit). For
// example, a template specialization might not be used in compiles of sources
// which use different template types.
//
#ifdef __GNUC__
#define svtkMaybeUnused(reason) __attribute__((unused))
#else
#define svtkMaybeUnused(reason)
#endif

#define svtkWorldCoordinateMacro(name)                                                              \
  virtual svtkCoordinate* Get##name##Coordinate()                                                   \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this                                          \
                  << "): returning " #name "Coordinate address " << this->name##Coordinate);       \
    return this->name##Coordinate;                                                                 \
  }                                                                                                \
  virtual void Set##name(double x[3]) { this->Set##name(x[0], x[1], x[2]); }                       \
  virtual void Set##name(double x, double y, double z)                                             \
  {                                                                                                \
    this->name##Coordinate->SetValue(x, y, z);                                                     \
  }                                                                                                \
  virtual double* Get##name() SVTK_SIZEHINT(3) { return this->name##Coordinate->GetValue(); }

#define svtkViewportCoordinateMacro(name)                                                           \
  virtual svtkCoordinate* Get##name##Coordinate()                                                   \
  {                                                                                                \
    svtkDebugMacro(<< this->GetClassName() << " (" << this                                          \
                  << "): returning " #name "Coordinate address " << this->name##Coordinate);       \
    return this->name##Coordinate;                                                                 \
  }                                                                                                \
  virtual void Set##name(double x[2]) { this->Set##name(x[0], x[1]); }                             \
  virtual void Set##name(double x, double y) { this->name##Coordinate->SetValue(x, y); }           \
  virtual double* Get##name() SVTK_SIZEHINT(2) { return this->name##Coordinate->GetValue(); }

// Allows definition of svtkObject API such that NewInstance may return a
// superclass of thisClass.
#define svtkAbstractTypeMacroWithNewInstanceType(                                                   \
  thisClass, superclass, instanceType, thisClassName)                                              \
protected:                                                                                         \
  const char* GetClassNameInternal() const override { return thisClassName; }                      \
                                                                                                   \
public:                                                                                            \
  typedef superclass Superclass;                                                                   \
  static svtkTypeBool IsTypeOf(const char* type)                                                    \
  {                                                                                                \
    if (!strcmp(thisClassName, type))                                                              \
    {                                                                                              \
      return 1;                                                                                    \
    }                                                                                              \
    return superclass::IsTypeOf(type);                                                             \
  }                                                                                                \
  svtkTypeBool IsA(const char* type) override { return this->thisClass::IsTypeOf(type); }           \
  static thisClass* SafeDownCast(svtkObjectBase* o)                                                 \
  {                                                                                                \
    if (o && o->IsA(thisClassName))                                                                \
    {                                                                                              \
      return static_cast<thisClass*>(o);                                                           \
    }                                                                                              \
    return nullptr;                                                                                \
  }                                                                                                \
  /*SVTK_NEWINSTANCE*/ instanceType* NewInstance() const                                                \
  {                                                                                                \
    return instanceType::SafeDownCast(this->NewInstanceInternal());                                \
  }                                                                                                \
  static svtkIdType GetNumberOfGenerationsFromBaseType(const char* type)                            \
  {                                                                                                \
    if (!strcmp(thisClassName, type))                                                              \
    {                                                                                              \
      return 0;                                                                                    \
    }                                                                                              \
    return 1 + superclass::GetNumberOfGenerationsFromBaseType(type);                               \
  }                                                                                                \
  svtkIdType GetNumberOfGenerationsFromBase(const char* type) override                              \
  {                                                                                                \
    return this->thisClass::GetNumberOfGenerationsFromBaseType(type);                              \
  }

// Same as svtkTypeMacro, but adapted for cases where thisClass is abstract.
#define svtkAbstractTypeMacro(thisClass, superclass)                                                \
  svtkAbstractTypeMacroWithNewInstanceType(thisClass, superclass, thisClass, #thisClass);           \
                                                                                                   \
public:

// Macro used to determine whether a class is the same class or
// a subclass of the named class.
#define svtkTypeMacro(thisClass, superclass)                                                        \
  svtkAbstractTypeMacro(thisClass, superclass);                                                     \
                                                                                                   \
protected:                                                                                         \
  svtkObjectBase* NewInstanceInternal() const override { return thisClass::New(); }                 \
                                                                                                   \
public:

// Macro to use when you are a direct child class of svtkObjectBase, instead
// of svtkTypeMacro. This is required to properly specify NewInstanceInternal
// as a virtual method.
// It is used to determine whether a class is the same class or a subclass
// of the named class.

#define svtkBaseTypeMacro(thisClass, superclass)                                                    \
  svtkAbstractTypeMacro(thisClass, superclass);                                                     \
                                                                                                   \
protected:                                                                                         \
  virtual svtkObjectBase* NewInstanceInternal() const { return thisClass::New(); }                  \
                                                                                                   \
public:

// Version of svtkAbstractTypeMacro for when thisClass is templated.
// For templates, we use the compiler generated typeid(...).name() identifier
// to distinguish classes. Otherwise, the template parameter names would appear
// in the class name, rather than the actual parameters. The resulting name may
// not be human readable on some platforms, but it will at least be unique. On
// GCC 4.9.2 release builds, this ends up being the same performance-wise as
// returning a string literal as the name() string is resolved at compile time.
//
// If either class has multiple template parameters, the commas will interfere
// with the macro call. In this case, create a typedef to the multi-parameter
// template class and pass that into the macro instead.
#define svtkAbstractTemplateTypeMacro(thisClass, superclass)                                        \
  svtkAbstractTypeMacroWithNewInstanceType(                                                         \
    thisClass, superclass, thisClass, typeid(thisClass).name());                                   \
                                                                                                   \
public:

// Version of svtkTypeMacro for when thisClass is templated.
// See svtkAbstractTemplateTypeMacro for more info.
#define svtkTemplateTypeMacro(thisClass, superclass)                                                \
  svtkAbstractTemplateTypeMacro(thisClass, superclass);                                             \
                                                                                                   \
protected:                                                                                         \
  svtkObjectBase* NewInstanceInternal() const override { return thisClass::New(); }                 \
                                                                                                   \
public:

// NOTE: This is no longer the prefer method for dispatching an array to a
// worker template. See svtkArrayDispatch for the new approach.
//
// The svtkTemplateMacro is used to centralize the set of types
// supported by Execute methods.  It also avoids duplication of long
// switch statement case lists.
//
// This version of the macro allows the template to take any number of
// arguments.  Example usage:
// switch(array->GetDataType())
//   {
//   svtkTemplateMacro(myFunc(static_cast<SVTK_TT*>(data), arg2));
//   }
#define svtkTemplateMacroCase(typeN, type, call)                                                    \
  case typeN:                                                                                      \
  {                                                                                                \
    typedef type SVTK_TT;                                                                           \
    call;                                                                                          \
  }                                                                                                \
  break
#define svtkTemplateMacro(call)                                                                     \
  svtkTemplateMacroCase(SVTK_DOUBLE, double, call);                                                  \
  svtkTemplateMacroCase(SVTK_FLOAT, float, call);                                                    \
  svtkTemplateMacroCase(SVTK_LONG_LONG, long long, call);                                            \
  svtkTemplateMacroCase(SVTK_UNSIGNED_LONG_LONG, unsigned long long, call);                          \
  svtkTemplateMacroCase(SVTK_ID_TYPE, svtkIdType, call);                                              \
  svtkTemplateMacroCase(SVTK_LONG, long, call);                                                      \
  svtkTemplateMacroCase(SVTK_UNSIGNED_LONG, unsigned long, call);                                    \
  svtkTemplateMacroCase(SVTK_INT, int, call);                                                        \
  svtkTemplateMacroCase(SVTK_UNSIGNED_INT, unsigned int, call);                                      \
  svtkTemplateMacroCase(SVTK_SHORT, short, call);                                                    \
  svtkTemplateMacroCase(SVTK_UNSIGNED_SHORT, unsigned short, call);                                  \
  svtkTemplateMacroCase(SVTK_CHAR, char, call);                                                      \
  svtkTemplateMacroCase(SVTK_SIGNED_CHAR, signed char, call);                                        \
  svtkTemplateMacroCase(SVTK_UNSIGNED_CHAR, unsigned char, call)

// This is same as Template macro with additional case for SVTK_STRING.
#define svtkExtendedTemplateMacro(call)                                                             \
  svtkTemplateMacro(call);                                                                          \
  svtkTemplateMacroCase(SVTK_STRING, svtkStdString, call)

// The svtkTemplate2Macro is used to dispatch like svtkTemplateMacro but
// over two template arguments instead of one.
//
// Example usage:
// switch(svtkTemplate2PackMacro(array1->GetDataType(),
//                              array2->GetDataType()))
//   {
//   svtkTemplateMacro(myFunc(static_cast<SVTK_T1*>(data1),
//                           static_cast<SVTK_T2*>(data2),
//                           otherArg));
//   }
#define svtkTemplate2Macro(call)                                                                    \
  svtkTemplate2MacroCase1(SVTK_DOUBLE, double, call);                                                \
  svtkTemplate2MacroCase1(SVTK_FLOAT, float, call);                                                  \
  svtkTemplate2MacroCase1(SVTK_LONG_LONG, long long, call);                                          \
  svtkTemplate2MacroCase1(SVTK_UNSIGNED_LONG_LONG, unsigned long long, call);                        \
  svtkTemplate2MacroCase1(SVTK_ID_TYPE, svtkIdType, call);                                            \
  svtkTemplate2MacroCase1(SVTK_LONG, long, call);                                                    \
  svtkTemplate2MacroCase1(SVTK_UNSIGNED_LONG, unsigned long, call);                                  \
  svtkTemplate2MacroCase1(SVTK_INT, int, call);                                                      \
  svtkTemplate2MacroCase1(SVTK_UNSIGNED_INT, unsigned int, call);                                    \
  svtkTemplate2MacroCase1(SVTK_SHORT, short, call);                                                  \
  svtkTemplate2MacroCase1(SVTK_UNSIGNED_SHORT, unsigned short, call);                                \
  svtkTemplate2MacroCase1(SVTK_CHAR, char, call);                                                    \
  svtkTemplate2MacroCase1(SVTK_SIGNED_CHAR, signed char, call);                                      \
  svtkTemplate2MacroCase1(SVTK_UNSIGNED_CHAR, unsigned char, call)
#define svtkTemplate2MacroCase1(type1N, type1, call)                                                \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_DOUBLE, double, call);                                 \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_FLOAT, float, call);                                   \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_LONG_LONG, long long, call);                           \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_UNSIGNED_LONG_LONG, unsigned long long, call);         \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_ID_TYPE, svtkIdType, call);                             \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_LONG, long, call);                                     \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_UNSIGNED_LONG, unsigned long, call);                   \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_INT, int, call);                                       \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_UNSIGNED_INT, unsigned int, call);                     \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_SHORT, short, call);                                   \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_UNSIGNED_SHORT, unsigned short, call);                 \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_CHAR, char, call);                                     \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_SIGNED_CHAR, signed char, call);                       \
  svtkTemplate2MacroCase2(type1N, type1, SVTK_UNSIGNED_CHAR, unsigned char, call)
#define svtkTemplate2MacroCase2(type1N, type1, type2N, type2, call)                                 \
  case svtkTemplate2PackMacro(type1N, type2N):                                                      \
  {                                                                                                \
    typedef type1 SVTK_T1;                                                                          \
    typedef type2 SVTK_T2;                                                                          \
    call;                                                                                          \
  };                                                                                               \
  break
#define svtkTemplate2PackMacro(type1N, type2N) ((((type1N)&0xFF) << 8) | ((type2N)&0xFF))

// The svtkArrayIteratorTemplateMacro is used to centralize the set of types
// supported by Execute methods.  It also avoids duplication of long
// switch statement case lists.
//
// This version of the macro allows the template to take any number of
// arguments.
//
// Note that in this macro SVTK_TT is defined to be the type of the iterator
// for the given type of array. One must include the
// svtkArrayIteratorIncludes.h header file to provide for extending of this macro
// by addition of new iterators.
//
// Example usage:
// svtkArrayIter* iter = array->NewIterator();
// switch(array->GetDataType())
//   {
//   svtkArrayIteratorTemplateMacro(myFunc(static_cast<SVTK_TT*>(iter), arg2));
//   }
// iter->Delete();
//
#define svtkArrayIteratorTemplateMacroCase(typeN, type, call)                                       \
  svtkTemplateMacroCase(typeN, svtkArrayIteratorTemplate<type>, call)
#define svtkArrayIteratorTemplateMacro(call)                                                        \
  svtkArrayIteratorTemplateMacroCase(SVTK_DOUBLE, double, call);                                     \
  svtkArrayIteratorTemplateMacroCase(SVTK_FLOAT, float, call);                                       \
  svtkArrayIteratorTemplateMacroCase(SVTK_LONG_LONG, long long, call);                               \
  svtkArrayIteratorTemplateMacroCase(SVTK_UNSIGNED_LONG_LONG, unsigned long long, call);             \
  svtkArrayIteratorTemplateMacroCase(SVTK_ID_TYPE, svtkIdType, call);                                 \
  svtkArrayIteratorTemplateMacroCase(SVTK_LONG, long, call);                                         \
  svtkArrayIteratorTemplateMacroCase(SVTK_UNSIGNED_LONG, unsigned long, call);                       \
  svtkArrayIteratorTemplateMacroCase(SVTK_INT, int, call);                                           \
  svtkArrayIteratorTemplateMacroCase(SVTK_UNSIGNED_INT, unsigned int, call);                         \
  svtkArrayIteratorTemplateMacroCase(SVTK_SHORT, short, call);                                       \
  svtkArrayIteratorTemplateMacroCase(SVTK_UNSIGNED_SHORT, unsigned short, call);                     \
  svtkArrayIteratorTemplateMacroCase(SVTK_CHAR, char, call);                                         \
  svtkArrayIteratorTemplateMacroCase(SVTK_SIGNED_CHAR, signed char, call);                           \
  svtkArrayIteratorTemplateMacroCase(SVTK_UNSIGNED_CHAR, unsigned char, call);                       \
  svtkArrayIteratorTemplateMacroCase(SVTK_STRING, svtkStdString, call);                               \
  svtkTemplateMacroCase(SVTK_BIT, svtkBitArrayIterator, call)

//----------------------------------------------------------------------------
// Setup legacy code policy.

// Define SVTK_LEGACY macro to mark legacy methods where they are
// declared in their class.  Example usage:
//
//   // @deprecated Replaced by MyOtherMethod() as of SVTK 5.0.
//   SVTK_LEGACY(void MyMethod());
#if defined(SVTK_LEGACY_REMOVE)
// Remove legacy methods completely.  Put a bogus declaration in
// place to avoid stray semicolons because this is an error for some
// compilers.  Using a class forward declaration allows any number
// of repeats in any context without generating unique names.

#define SVTK_LEGACY(method) SVTK_LEGACY__0(method, __LINE__)
#define SVTK_LEGACY__0(method, line) SVTK_LEGACY__1(method, line)
#define SVTK_LEGACY__1(method, line) class svtkLegacyMethodRemoved##line

#elif defined(SVTK_LEGACY_SILENT) || defined(SVTK_WRAPPING_CXX) || defined(SWIG)
// Provide legacy methods with no warnings.
#define SVTK_LEGACY(method) method
#else
// Setup compile-time warnings for uses of deprecated methods if
// possible on this compiler.
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define SVTK_LEGACY(method) method __attribute__((deprecated))
#elif defined(_MSC_VER)
#define SVTK_LEGACY(method) __declspec(deprecated) method
#else
#define SVTK_LEGACY(method) method
#endif
#endif

// Macros to create runtime deprecation warning messages in function
// bodies.  Example usage:
//
//   #if !defined(SVTK_LEGACY_REMOVE)
//   void svtkMyClass::MyOldMethod()
//   {
//     SVTK_LEGACY_BODY(svtkMyClass::MyOldMethod, "SVTK 5.0");
//   }
//   #endif
//
//   #if !defined(SVTK_LEGACY_REMOVE)
//   void svtkMyClass::MyMethod()
//   {
//     SVTK_LEGACY_REPLACED_BODY(svtkMyClass::MyMethod, "SVTK 5.0",
//                              svtkMyClass::MyOtherMethod);
//   }
//   #endif
#if defined(SVTK_LEGACY_REMOVE) || defined(SVTK_LEGACY_SILENT)
#define SVTK_LEGACY_BODY(method, version)
#define SVTK_LEGACY_REPLACED_BODY(method, version, replace)
#else
#define SVTK_LEGACY_BODY(method, version)                                                           \
  svtkGenericWarningMacro(                                                                          \
    #method " was deprecated for " version " and will be removed in a future version.")
#define SVTK_LEGACY_REPLACED_BODY(method, version, replace)                                         \
  svtkGenericWarningMacro(                                                                          \
    #method " was deprecated for " version                                                         \
            " and will be removed in a future version.  Use " #replace " instead.")
#endif

//----------------------------------------------------------------------------
// Deprecation attribute.

#if !defined(SVTK_DEPRECATED) && !defined(SVTK_WRAPPING_CXX)
#if __cplusplus >= 201402L && defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define SVTK_DEPRECATED [[deprecated]]
#endif
#elif defined(_MSC_VER)
#define SVTK_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define SVTK_DEPRECATED __attribute__((deprecated))
#endif
#endif

#ifndef SVTK_DEPRECATED
#define SVTK_DEPRECATED
#endif

//----------------------------------------------------------------------------
// format string checking.

#if !defined(SVTK_FORMAT_PRINTF)
#if defined(__GNUC__)
#define SVTK_FORMAT_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define SVTK_FORMAT_PRINTF(a, b)
#endif
#endif

// Qualifiers used for function arguments and return types indicating that the
// class is wrapped externally.
#define SVTK_WRAP_EXTERN

//----------------------------------------------------------------------------
// Switch case fall-through policy.

// Use "SVTK_FALLTHROUGH;" to annotate deliberate fall-through in switches,
// use it analogously to "break;".  The trailing semi-colon is required.
#if !defined(SVTK_FALLTHROUGH) && defined(__has_cpp_attribute)
#if __cplusplus >= 201703L && __has_cpp_attribute(fallthrough)
#define SVTK_FALLTHROUGH [[fallthrough]]
#elif __cplusplus >= 201103L && __has_cpp_attribute(gnu::fallthrough)
#define SVTK_FALLTHROUGH [[gnu::fallthrough]]
#elif __cplusplus >= 201103L && __has_cpp_attribute(clang::fallthrough)
#define SVTK_FALLTHROUGH [[clang::fallthrough]]
#endif
#endif

#ifndef SVTK_FALLTHROUGH
#define SVTK_FALLTHROUGH ((void)0)
#endif

//----------------------------------------------------------------------------
// Macro to generate bitflag operators for C++11 scoped enums.

#define SVTK_GENERATE_BITFLAG_OPS(EnumType)                                                         \
  inline EnumType operator|(EnumType f1, EnumType f2)                                              \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return static_cast<EnumType>(static_cast<T>(f1) | static_cast<T>(f2));                         \
  }                                                                                                \
  inline EnumType operator&(EnumType f1, EnumType f2)                                              \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return static_cast<EnumType>(static_cast<T>(f1) & static_cast<T>(f2));                         \
  }                                                                                                \
  inline EnumType operator^(EnumType f1, EnumType f2)                                              \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return static_cast<EnumType>(static_cast<T>(f1) ^ static_cast<T>(f2));                         \
  }                                                                                                \
  inline EnumType operator~(EnumType f1)                                                           \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return static_cast<EnumType>(~static_cast<T>(f1));                                             \
  }                                                                                                \
  inline EnumType& operator|=(EnumType& f1, EnumType f2)                                           \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return f1 = static_cast<EnumType>(static_cast<T>(f1) | static_cast<T>(f2));                    \
  }                                                                                                \
  inline EnumType& operator&=(EnumType& f1, EnumType f2)                                           \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return f1 = static_cast<EnumType>(static_cast<T>(f1) & static_cast<T>(f2));                    \
  }                                                                                                \
  inline EnumType& operator^=(EnumType& f1, EnumType f2)                                           \
  {                                                                                                \
    using T = typename std::underlying_type<EnumType>::type;                                       \
    return f1 = static_cast<EnumType>(static_cast<T>(f1) ^ static_cast<T>(f2));                    \
  }

#endif
// SVTK-HeaderTest-Exclude: svtkSetGet.h
