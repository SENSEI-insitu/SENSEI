/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkABI.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkABI
 * @brief   manage macros for exporting symbols in the binary interface.
 *
 * This header defines the macros for importing and exporting symbols in shared
 * objects (DLLs, etc). All SVTK headers should use these macros to define the
 * kit specific import/export macros. So for the svtkCommon kit this might be,
 *
 * \code
 * #include "svtkABI.h"
 *
 * #if defined(SVTK_BUILD_SHARED_LIBS)
 * # if defined(svtkCommon_EXPORTS)
 * #  define SVTK_COMMON_EXPORT SVTK_ABI_EXPORT
 * # else
 * #  define SVTK_COMMON_EXPORT SVTK_ABI_IMPORT
 * # endif
 * #else
 * # define SVTK_COMMON_EXPORT
 * #endif
 * \endcode
 *
 * See http://gcc.gnu.org/wiki/Visibility for a discussion of the symbol
 * visibility support in GCC. The project must pass extra CFLAGS/CXXFLAGS in
 * order to change the default symbol visibility when using GCC.
 * Currently hidden is not used, but it can be used to explicitly hide
 * symbols from external linkage.
 */

#ifndef svtkABI_h
#define svtkABI_h

#if defined(_WIN32)
#define SVTK_ABI_IMPORT __declspec(dllimport)
#define SVTK_ABI_EXPORT __declspec(dllexport)
#define SVTK_ABI_HIDDEN
#elif __GNUC__ >= 4
#define SVTK_ABI_IMPORT __attribute__((visibility("default")))
#define SVTK_ABI_EXPORT __attribute__((visibility("default")))
#define SVTK_ABI_HIDDEN __attribute__((visibility("hidden")))
#else
#define SVTK_ABI_IMPORT
#define SVTK_ABI_EXPORT
#define SVTK_ABI_HIDDEN
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

#endif // svtkABI_h
// SVTK-HeaderTest-Exclude: svtkABI.h
