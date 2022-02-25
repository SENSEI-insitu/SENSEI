/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkType.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkType_h
#define svtkType_h

#include "svtkConfigure.h"
#include "svtk_kwiml.h"

#define SVTK_SIZEOF_CHAR KWIML_ABI_SIZEOF_CHAR
#define SVTK_SIZEOF_SHORT KWIML_ABI_SIZEOF_SHORT
#define SVTK_SIZEOF_INT KWIML_ABI_SIZEOF_INT
#define SVTK_SIZEOF_LONG KWIML_ABI_SIZEOF_LONG
#define SVTK_SIZEOF_LONG_LONG KWIML_ABI_SIZEOF_LONG_LONG
#define SVTK_SIZEOF_FLOAT KWIML_ABI_SIZEOF_FLOAT
#define SVTK_SIZEOF_DOUBLE KWIML_ABI_SIZEOF_DOUBLE
#define SVTK_SIZEOF_VOID_P KWIML_ABI_SIZEOF_DATA_PTR

/* Whether type "char" is signed (it may be signed or unsigned).  */
#if defined(KWIML_ABI_CHAR_IS_SIGNED)
#define SVTK_TYPE_CHAR_IS_SIGNED 1
#else
#define SVTK_TYPE_CHAR_IS_SIGNED 0
#endif

/*--------------------------------------------------------------------------*/
/* Define a unique integer identifier for each native scalar type.  */

/* These types are returned by GetDataType to indicate pixel type.  */
#define SVTK_VOID 0
#define SVTK_BIT 1
#define SVTK_CHAR 2
#define SVTK_SIGNED_CHAR 15
#define SVTK_UNSIGNED_CHAR 3
#define SVTK_SHORT 4
#define SVTK_UNSIGNED_SHORT 5
#define SVTK_INT 6
#define SVTK_UNSIGNED_INT 7
#define SVTK_LONG 8
#define SVTK_UNSIGNED_LONG 9
#define SVTK_FLOAT 10
#define SVTK_DOUBLE 11
#define SVTK_ID_TYPE 12

/* These types are not currently supported by GetDataType, but are for
   completeness.  */
#define SVTK_STRING 13
#define SVTK_OPAQUE 14

#define SVTK_LONG_LONG 16
#define SVTK_UNSIGNED_LONG_LONG 17

#if !defined(SVTK_LEGACY_REMOVE)

/* Legacy.  This type is never enabled.  */
#define SVTK___INT64 18

/* Legacy.  This type is never enabled.  */
#define SVTK_UNSIGNED___INT64 19

#endif

/* These types are required by svtkVariant and svtkVariantArray */
#define SVTK_VARIANT 20
#define SVTK_OBJECT 21

/* Storage for Unicode strings */
#define SVTK_UNICODE_STRING 22

/*--------------------------------------------------------------------------*/
/* Define a unique integer identifier for each svtkDataObject type.          */
/* When adding a new data type here, make sure to update                    */
/* svtkDataObjectTypes as well.                                              */
#define SVTK_POLY_DATA 0
#define SVTK_STRUCTURED_POINTS 1
#define SVTK_STRUCTURED_GRID 2
#define SVTK_RECTILINEAR_GRID 3
#define SVTK_UNSTRUCTURED_GRID 4
#define SVTK_PIECEWISE_FUNCTION 5
#define SVTK_IMAGE_DATA 6
#define SVTK_DATA_OBJECT 7
#define SVTK_DATA_SET 8
#define SVTK_POINT_SET 9
#define SVTK_UNIFORM_GRID 10
#define SVTK_COMPOSITE_DATA_SET 11
#define SVTK_MULTIGROUP_DATA_SET 12
#define SVTK_MULTIBLOCK_DATA_SET 13
#define SVTK_HIERARCHICAL_DATA_SET 14
#define SVTK_HIERARCHICAL_BOX_DATA_SET 15
#define SVTK_GENERIC_DATA_SET 16
#define SVTK_HYPER_OCTREE 17
#define SVTK_TEMPORAL_DATA_SET 18
#define SVTK_TABLE 19
#define SVTK_GRAPH 20
#define SVTK_TREE 21
#define SVTK_SELECTION 22
#define SVTK_DIRECTED_GRAPH 23
#define SVTK_UNDIRECTED_GRAPH 24
#define SVTK_MULTIPIECE_DATA_SET 25
#define SVTK_DIRECTED_ACYCLIC_GRAPH 26
#define SVTK_ARRAY_DATA 27
#define SVTK_REEB_GRAPH 28
#define SVTK_UNIFORM_GRID_AMR 29
#define SVTK_NON_OVERLAPPING_AMR 30
#define SVTK_OVERLAPPING_AMR 31
#define SVTK_HYPER_TREE_GRID 32
#define SVTK_MOLECULE 33
#define SVTK_PISTON_DATA_OBJECT 34
#define SVTK_PATH 35
#define SVTK_UNSTRUCTURED_GRID_BASE 36
#define SVTK_PARTITIONED_DATA_SET 37
#define SVTK_PARTITIONED_DATA_SET_COLLECTION 38
#define SVTK_UNIFORM_HYPER_TREE_GRID 39
#define SVTK_EXPLICIT_STRUCTURED_GRID 40

/*--------------------------------------------------------------------------*/
/* Define a casting macro for use by the constants below.  */
#if defined(__cplusplus)
#define SVTK_TYPE_CAST(T, V) static_cast<T>(V)
#else
#define SVTK_TYPE_CAST(T, V) ((T)(V))
#endif

/*--------------------------------------------------------------------------*/
/* Define min/max constants for each type.  */
#define SVTK_BIT_MIN 0
#define SVTK_BIT_MAX 1
#if SVTK_TYPE_CHAR_IS_SIGNED
#define SVTK_CHAR_MIN SVTK_TYPE_CAST(char, 0x80)
#define SVTK_CHAR_MAX SVTK_TYPE_CAST(char, 0x7f)
#else
#define SVTK_CHAR_MIN SVTK_TYPE_CAST(char, 0u)
#define SVTK_CHAR_MAX SVTK_TYPE_CAST(char, 0xffu)
#endif
#define SVTK_SIGNED_CHAR_MIN SVTK_TYPE_CAST(signed char, 0x80)
#define SVTK_SIGNED_CHAR_MAX SVTK_TYPE_CAST(signed char, 0x7f)
#define SVTK_UNSIGNED_CHAR_MIN SVTK_TYPE_CAST(unsigned char, 0u)
#define SVTK_UNSIGNED_CHAR_MAX SVTK_TYPE_CAST(unsigned char, 0xffu)
#define SVTK_SHORT_MIN SVTK_TYPE_CAST(short, 0x8000)
#define SVTK_SHORT_MAX SVTK_TYPE_CAST(short, 0x7fff)
#define SVTK_UNSIGNED_SHORT_MIN SVTK_TYPE_CAST(unsigned short, 0u)
#define SVTK_UNSIGNED_SHORT_MAX SVTK_TYPE_CAST(unsigned short, 0xffffu)
#define SVTK_INT_MIN SVTK_TYPE_CAST(int, ~(~0u >> 1))
#define SVTK_INT_MAX SVTK_TYPE_CAST(int, ~0u >> 1)
#define SVTK_UNSIGNED_INT_MIN SVTK_TYPE_CAST(unsigned int, 0)
#define SVTK_UNSIGNED_INT_MAX SVTK_TYPE_CAST(unsigned int, ~0u)
#define SVTK_LONG_MIN SVTK_TYPE_CAST(long, ~(~0ul >> 1))
#define SVTK_LONG_MAX SVTK_TYPE_CAST(long, ~0ul >> 1)
#define SVTK_UNSIGNED_LONG_MIN SVTK_TYPE_CAST(unsigned long, 0ul)
#define SVTK_UNSIGNED_LONG_MAX SVTK_TYPE_CAST(unsigned long, ~0ul)
#define SVTK_FLOAT_MIN SVTK_TYPE_CAST(float, -1.0e+38f)
#define SVTK_FLOAT_MAX SVTK_TYPE_CAST(float, 1.0e+38f)
#define SVTK_DOUBLE_MIN SVTK_TYPE_CAST(double, -1.0e+299)
#define SVTK_DOUBLE_MAX SVTK_TYPE_CAST(double, 1.0e+299)
#define SVTK_LONG_LONG_MIN SVTK_TYPE_CAST(long long, ~(~0ull >> 1))
#define SVTK_LONG_LONG_MAX SVTK_TYPE_CAST(long long, ~0ull >> 1)
#define SVTK_UNSIGNED_LONG_LONG_MIN SVTK_TYPE_CAST(unsigned long long, 0ull)
#define SVTK_UNSIGNED_LONG_LONG_MAX SVTK_TYPE_CAST(unsigned long long, ~0ull)

/*--------------------------------------------------------------------------*/
/* Define named types and constants corresponding to specific integer
   and floating-point sizes and signedness.  */

/* Select an 8-bit integer type.  */
#if SVTK_SIZEOF_CHAR == 1
typedef unsigned char svtkTypeUInt8;
typedef signed char svtkTypeInt8;
#define SVTK_TYPE_UINT8 SVTK_UNSIGNED_CHAR
#define SVTK_TYPE_UINT8_MIN SVTK_UNSIGNED_CHAR_MIN
#define SVTK_TYPE_UINT8_MAX SVTK_UNSIGNED_CHAR_MAX
#if SVTK_TYPE_CHAR_IS_SIGNED
#define SVTK_TYPE_INT8 SVTK_CHAR
#define SVTK_TYPE_INT8_MIN SVTK_CHAR_MIN
#define SVTK_TYPE_INT8_MAX SVTK_CHAR_MAX
#else
#define SVTK_TYPE_INT8 SVTK_SIGNED_CHAR
#define SVTK_TYPE_INT8_MIN SVTK_SIGNED_CHAR_MIN
#define SVTK_TYPE_INT8_MAX SVTK_SIGNED_CHAR_MAX
#endif
#else
#error "No native data type can represent an 8-bit integer."
#endif

/* Select a 16-bit integer type.  */
#if SVTK_SIZEOF_SHORT == 2
typedef unsigned short svtkTypeUInt16;
typedef signed short svtkTypeInt16;
#define SVTK_TYPE_UINT16 SVTK_UNSIGNED_SHORT
#define SVTK_TYPE_UINT16_MIN SVTK_UNSIGNED_SHORT_MIN
#define SVTK_TYPE_UINT16_MAX SVTK_UNSIGNED_SHORT_MAX
#define SVTK_TYPE_INT16 SVTK_SHORT
#define SVTK_TYPE_INT16_MIN SVTK_SHORT_MIN
#define SVTK_TYPE_INT16_MAX SVTK_SHORT_MAX
#elif SVTK_SIZEOF_INT == 2
typedef unsigned int svtkTypeUInt16;
typedef signed int svtkTypeInt16;
#define SVTK_TYPE_UINT16 SVTK_UNSIGNED_INT
#define SVTK_TYPE_UINT16_MIN SVTK_UNSIGNED_INT_MIN
#define SVTK_TYPE_UINT16_MAX SVTK_UNSIGNED_INT_MAX
#define SVTK_TYPE_INT16 SVTK_INT
#define SVTK_TYPE_INT16_MIN SVTK_INT_MIN
#define SVTK_TYPE_INT16_MAX SVTK_INT_MAX
#else
#error "No native data type can represent a 16-bit integer."
#endif

/* Select a 32-bit integer type.  */
#if SVTK_SIZEOF_INT == 4
typedef unsigned int svtkTypeUInt32;
typedef signed int svtkTypeInt32;
#define SVTK_TYPE_UINT32 SVTK_UNSIGNED_INT
#define SVTK_TYPE_UINT32_MIN SVTK_UNSIGNED_INT_MIN
#define SVTK_TYPE_UINT32_MAX SVTK_UNSIGNED_INT_MAX
#define SVTK_TYPE_INT32 SVTK_INT
#define SVTK_TYPE_INT32_MIN SVTK_INT_MIN
#define SVTK_TYPE_INT32_MAX SVTK_INT_MAX
#elif SVTK_SIZEOF_LONG == 4
typedef unsigned long svtkTypeUInt32;
typedef signed long svtkTypeInt32;
#define SVTK_TYPE_UINT32 SVTK_UNSIGNED_LONG
#define SVTK_TYPE_UINT32_MIN SVTK_UNSIGNED_LONG_MIN
#define SVTK_TYPE_UINT32_MAX SVTK_UNSIGNED_LONG_MAX
#define SVTK_TYPE_INT32 SVTK_LONG
#define SVTK_TYPE_INT32_MIN SVTK_LONG_MIN
#define SVTK_TYPE_INT32_MAX SVTK_LONG_MAX
#else
#error "No native data type can represent a 32-bit integer."
#endif

/* Select a 64-bit integer type.  */
#if SVTK_SIZEOF_LONG_LONG == 8
typedef unsigned long long svtkTypeUInt64;
typedef signed long long svtkTypeInt64;
#define SVTK_TYPE_UINT64 SVTK_UNSIGNED_LONG_LONG
#define SVTK_TYPE_UINT64_MIN SVTK_UNSIGNED_LONG_LONG_MIN
#define SVTK_TYPE_UINT64_MAX SVTK_UNSIGNED_LONG_LONG_MAX
#define SVTK_TYPE_INT64 SVTK_LONG_LONG
#define SVTK_TYPE_INT64_MIN SVTK_LONG_LONG_MIN
#define SVTK_TYPE_INT64_MAX SVTK_LONG_LONG_MAX
#elif SVTK_SIZEOF_LONG == 8
typedef unsigned long svtkTypeUInt64;
typedef signed long svtkTypeInt64;
#define SVTK_TYPE_UINT64 SVTK_UNSIGNED_LONG
#define SVTK_TYPE_UINT64_MIN SVTK_UNSIGNED_LONG_MIN
#define SVTK_TYPE_UINT64_MAX SVTK_UNSIGNED_LONG_MAX
#define SVTK_TYPE_INT64 SVTK_LONG
#define SVTK_TYPE_INT64_MIN SVTK_LONG_MIN
#define SVTK_TYPE_INT64_MAX SVTK_LONG_MAX
#else
#error "No native data type can represent a 64-bit integer."
#endif

#if !defined(SVTK_LEGACY_REMOVE)
// Provide this define to facilitate apps that need to support older
// versions that do not have svtkMTimeType
#define SVTK_HAS_MTIME_TYPE
#endif

// If this is a 64-bit platform, or the user has indicated that 64-bit
// timestamps should be used, select an unsigned 64-bit integer type
// for use in MTime values. If possible, use 'unsigned long' as we have
// historically.
#if defined(SVTK_USE_64BIT_TIMESTAMPS) || SVTK_SIZEOF_VOID_P == 8
#if SVTK_SIZEOF_LONG == 8
typedef unsigned long svtkMTimeType;
#define SVTK_MTIME_TYPE_IMPL SVTK_UNSIGNED_LONG
#define SVTK_MTIME_MIN SVTK_UNSIGNED_LONG_MIN
#define SVTK_MTIME_MAX SVTK_UNSIGNED_LONG_MAX
#else
typedef svtkTypeUInt64 svtkMTimeType;
#define SVTK_MTIME_TYPE_IMPL SVTK_TYPE_UINT64
#define SVTK_MTIME_MIN SVTK_TYPE_UINT64_MIN
#define SVTK_MTIME_MAX SVTK_TYPE_UINT64_MAX
#endif
#else
#if SVTK_SIZEOF_LONG == 4
typedef unsigned long svtkMTimeType;
#define SVTK_MTIME_TYPE_IMPL SVTK_UNSIGNED_LONG
#define SVTK_MTIME_MIN SVTK_UNSIGNED_LONG_MIN
#define SVTK_MTIME_MAX SVTK_UNSIGNED_LONG_MAX
#else
typedef svtkTypeUInt32 svtkMTimeType;
#define SVTK_MTIME_TYPE_IMPL SVTK_TYPE_UINT32
#define SVTK_MTIME_MIN SVTK_TYPE_UINT32_MIN
#define SVTK_MTIME_MAX SVTK_TYPE_UINT32_MAX
#endif
#endif

/* Select a 32-bit floating point type.  */
#if SVTK_SIZEOF_FLOAT == 4
typedef float svtkTypeFloat32;
#define SVTK_TYPE_FLOAT32 SVTK_FLOAT
#else
#error "No native data type can represent a 32-bit floating point value."
#endif

/* Select a 64-bit floating point type.  */
#if SVTK_SIZEOF_DOUBLE == 8
typedef double svtkTypeFloat64;
#define SVTK_TYPE_FLOAT64 SVTK_DOUBLE
#else
#error "No native data type can represent a 64-bit floating point value."
#endif

/*--------------------------------------------------------------------------*/
/* Choose an implementation for svtkIdType.  */
#define SVTK_HAS_ID_TYPE
#ifdef SVTK_USE_64BIT_IDS
#if SVTK_SIZEOF_LONG_LONG == 8
typedef long long svtkIdType;
#define SVTK_ID_TYPE_IMPL SVTK_LONG_LONG
#define SVTK_SIZEOF_ID_TYPE SVTK_SIZEOF_LONG_LONG
#define SVTK_ID_MIN SVTK_LONG_LONG_MIN
#define SVTK_ID_MAX SVTK_LONG_LONG_MAX
#define SVTK_ID_TYPE_PRId "lld"
#elif SVTK_SIZEOF_LONG == 8
typedef long svtkIdType;
#define SVTK_ID_TYPE_IMPL SVTK_LONG
#define SVTK_SIZEOF_ID_TYPE SVTK_SIZEOF_LONG
#define SVTK_ID_MIN SVTK_LONG_MIN
#define SVTK_ID_MAX SVTK_LONG_MAX
#define SVTK_ID_TYPE_PRId "ld"
#else
#error "SVTK_USE_64BIT_IDS is ON but no 64-bit integer type is available."
#endif
#else
typedef int svtkIdType;
#define SVTK_ID_TYPE_IMPL SVTK_INT
#define SVTK_SIZEOF_ID_TYPE SVTK_SIZEOF_INT
#define SVTK_ID_MIN SVTK_INT_MIN
#define SVTK_ID_MAX SVTK_INT_MAX
#define SVTK_ID_TYPE_PRId "d"
#endif

#ifndef __cplusplus
// Make sure that when SVTK headers are used by the C compiler we make
// sure to define the bool type. This is possible when using IO features
// like svtkXMLWriterC.h
#include "stdbool.h"
#endif

/*--------------------------------------------------------------------------*/
/* If not already defined, define svtkTypeBool. When SVTK was started, some   */
/* compilers did not yet support the bool type, and so SVTK often used int,  */
/* or more rarely unsigned int, where it should have used bool.             */
/* Eventually svtkTypeBool will switch to real bool.                         */
#ifndef SVTK_TYPE_BOOL_TYPEDEFED
#define SVTK_TYPE_BOOL_TYPEDEFED
#if 1
typedef int svtkTypeBool;
typedef unsigned int svtkTypeUBool;
#else
typedef bool svtkTypeBool;
typedef bool svtkTypeUBool;
#endif
#endif

#if defined(__cplusplus)
/* Description:
 * Returns true if data type tags a and b point to the same data type. This
 * is intended to handle svtkIdType, which does not have the same tag as its
 * underlying data type.
 * @note This method is only available when included from a C++ source file. */
inline svtkTypeBool svtkDataTypesCompare(int a, int b)
{
  return (a == b ||
    ((a == SVTK_ID_TYPE || a == SVTK_ID_TYPE_IMPL) && (b == SVTK_ID_TYPE || b == SVTK_ID_TYPE_IMPL)));
}
#endif

/*--------------------------------------------------------------------------*/
/** A macro to instantiate a template over all numerical types */
#define svtkInstantiateTemplateMacro(decl)                                                          \
  decl<float>;                                                                                     \
  decl<double>;                                                                                    \
  decl<char>;                                                                                      \
  decl<signed char>;                                                                               \
  decl<unsigned char>;                                                                             \
  decl<short>;                                                                                     \
  decl<unsigned short>;                                                                            \
  decl<int>;                                                                                       \
  decl<unsigned int>;                                                                              \
  decl<long>;                                                                                      \
  decl<unsigned long>;                                                                             \
  decl<long long>;                                                                                 \
  decl<unsigned long long>

/** A macro to declare extern templates for all numerical types */
#ifdef SVTK_USE_EXTERN_TEMPLATE
#define svtkExternTemplateMacro(decl) svtkInstantiateTemplateMacro(decl)
#else
#define svtkExternTemplateMacro(decl)
#endif

#endif
// SVTK-HeaderTest-Exclude: svtkType.h
