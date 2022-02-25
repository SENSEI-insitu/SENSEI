/*=========================================================================

  Program:   ParaView
  Module:    svtkTypeTraits.h

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTypeTraits
 * @brief   Template defining traits of native types used by SVTK.
 *
 * svtkTypeTraits provides information about SVTK's supported scalar types
 * that is useful for templates.
 */

#ifndef svtkTypeTraits_h
#define svtkTypeTraits_h

#include "svtkSystemIncludes.h"

// Forward-declare template.  There is no primary template.
template <class T>
struct svtkTypeTraits;

// Define a macro to simplify trait definitions.
#define SVTK_TYPE_TRAITS(type, macro, isSigned, name, print, format)                                \
  template <>                                                                                      \
  struct svtkTypeTraits<type>                                                                       \
  {                                                                                                \
    /* The type itself.  */                                                                        \
    typedef type ValueType;                                                                        \
                                                                                                   \
    /* the value defined for this type in svtkType */                                               \
    enum                                                                                           \
    {                                                                                              \
      SVTK_TYPE_ID = SVTK_##macro                                                                    \
    };                                                                                             \
    static int SVTKTypeID() { return SVTK_##macro; }                                                 \
                                                                                                   \
    /* The smallest possible value represented by the type.  */                                    \
    static type Min() { return SVTK_##macro##_MIN; }                                                \
                                                                                                   \
    /* The largest possible value represented by the type.  */                                     \
    static type Max() { return SVTK_##macro##_MAX; }                                                \
                                                                                                   \
    /* Whether the type is signed.  */                                                             \
    static int IsSigned() { return isSigned; }                                                     \
                                                                                                   \
    /* An "alias" type that is the same size and signedness.  */                                   \
    typedef svtkType##name SizedType;                                                               \
                                                                                                   \
    /* A name for the type indicating its size and signedness.  */                                 \
    static const char* SizedName() { return #name; }                                               \
                                                                                                   \
    /* The common C++ name for the type (e.g. float, unsigned int, etc).*/                         \
    static const char* Name() { return #type; }                                                    \
                                                                                                   \
    /* A type to use for printing or parsing values in strings.  */                                \
    typedef print PrintType;                                                                       \
                                                                                                   \
    /* A format for parsing values from strings.  Use with PrintType.  */                          \
    static const char* ParseFormat() { return format; }                                            \
  }

// Define traits for floating-point types.
#define SVTK_TYPE_NAME_FLOAT float
#define SVTK_TYPE_NAME_DOUBLE double
#define SVTK_TYPE_SIZED_FLOAT FLOAT32
#define SVTK_TYPE_SIZED_DOUBLE FLOAT64
SVTK_TYPE_TRAITS(float, FLOAT, 1, Float32, float, "%f");
SVTK_TYPE_TRAITS(double, DOUBLE, 1, Float64, double, "%lf");

// Define traits for char types.
// Note the print type is short because not all platforms support formatting integers with char.
#define SVTK_TYPE_NAME_CHAR char
#if SVTK_TYPE_CHAR_IS_SIGNED
#define SVTK_TYPE_SIZED_CHAR INT8
SVTK_TYPE_TRAITS(char, CHAR, 1, Int8, short, "%hd");
#else
#define SVTK_TYPE_SIZED_CHAR UINT8
SVTK_TYPE_TRAITS(char, CHAR, 0, UInt8, unsigned short, "%hu");
#endif
#define SVTK_TYPE_NAME_SIGNED_CHAR signed char
#define SVTK_TYPE_NAME_UNSIGNED_CHAR unsigned char
#define SVTK_TYPE_SIZED_SIGNED_CHAR INT8
#define SVTK_TYPE_SIZED_UNSIGNED_CHAR UINT8
SVTK_TYPE_TRAITS(signed char, SIGNED_CHAR, 1, Int8, short, "%hd");
SVTK_TYPE_TRAITS(unsigned char, UNSIGNED_CHAR, 0, UInt8, unsigned short, "%hu");

// Define traits for short types.
#define SVTK_TYPE_NAME_SHORT short
#define SVTK_TYPE_NAME_UNSIGNED_SHORT unsigned short
#define SVTK_TYPE_SIZED_SHORT INT16
#define SVTK_TYPE_SIZED_UNSIGNED_SHORT UINT16
SVTK_TYPE_TRAITS(short, SHORT, 1, Int16, short, "%hd");
SVTK_TYPE_TRAITS(unsigned short, UNSIGNED_SHORT, 0, UInt16, unsigned short, "%hu");

// Define traits for int types.
#define SVTK_TYPE_NAME_INT int
#define SVTK_TYPE_NAME_UNSIGNED_INT unsigned int
#define SVTK_TYPE_SIZED_INT INT32
#define SVTK_TYPE_SIZED_UNSIGNED_INT UINT32
SVTK_TYPE_TRAITS(int, INT, 1, Int32, int, "%d");
SVTK_TYPE_TRAITS(unsigned int, UNSIGNED_INT, 0, UInt32, unsigned int, "%u");

// Define traits for long types.
#define SVTK_TYPE_NAME_LONG long
#define SVTK_TYPE_NAME_UNSIGNED_LONG unsigned long
#if SVTK_SIZEOF_LONG == 4
#define SVTK_TYPE_SIZED_LONG INT32
#define SVTK_TYPE_SIZED_UNSIGNED_LONG UINT32
SVTK_TYPE_TRAITS(long, LONG, 1, Int32, long, "%ld");
SVTK_TYPE_TRAITS(unsigned long, UNSIGNED_LONG, 0, UInt32, unsigned long, "%lu");
#elif SVTK_SIZEOF_LONG == 8
#define SVTK_TYPE_SIZED_LONG INT64
#define SVTK_TYPE_SIZED_UNSIGNED_LONG UINT64
SVTK_TYPE_TRAITS(long, LONG, 1, Int64, long, "%ld");
SVTK_TYPE_TRAITS(unsigned long, UNSIGNED_LONG, 0, UInt64, unsigned long, "%lu");
#else
#error "Type long is not 4 or 8 bytes in size."
#endif

// Define traits for long long types if they are enabled.
#define SVTK_TYPE_NAME_LONG_LONG long long
#define SVTK_TYPE_NAME_UNSIGNED_LONG_LONG unsigned long long
#if SVTK_SIZEOF_LONG_LONG == 8
#define SVTK_TYPE_SIZED_LONG_LONG INT64
#define SVTK_TYPE_SIZED_UNSIGNED_LONG_LONG UINT64
#define SVTK_TYPE_LONG_LONG_FORMAT "%ll"
SVTK_TYPE_TRAITS(long long, LONG_LONG, 1, Int64, long long, SVTK_TYPE_LONG_LONG_FORMAT "d");
SVTK_TYPE_TRAITS(unsigned long long, UNSIGNED_LONG_LONG, 0, UInt64, unsigned long long,
  SVTK_TYPE_LONG_LONG_FORMAT "u");
#undef SVTK_TYPE_LONG_LONG_FORMAT
#else
#error "Type long long is not 8 bytes in size."
#endif

// Define traits for svtkIdType.  The template specialization is
// already defined for the corresponding native type.
#define SVTK_TYPE_NAME_ID_TYPE svtkIdType
#if defined(SVTK_USE_64BIT_IDS)
#define SVTK_TYPE_SIZED_ID_TYPE INT64
#else
#define SVTK_TYPE_SIZED_ID_TYPE INT32
#endif

#undef SVTK_TYPE_TRAITS

#endif
// SVTK-HeaderTest-Exclude: svtkTypeTraits.h
