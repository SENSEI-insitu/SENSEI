/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWin32Header.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWin32Header
 * @brief   manage Windows system differences
 *
 * The svtkWin32Header captures some system differences between Unix and
 * Windows operating systems.
 */

#ifndef svtkWin32Header_h
#define svtkWin32Header_h

#ifndef SVTK_SYSTEM_INCLUDES_INSIDE
Do_not_include_svtkWin32Header_directly_svtkSystemIncludes_includes_it;
#endif

#include "svtkABI.h"
#include "svtkConfigure.h"

/*
 * This is a support for files on the disk that are larger than 2GB.
 * Since this is the first place that any include should happen, do this here.
 */
#ifdef SVTK_REQUIRE_LARGE_FILE_SUPPORT
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _LARGE_FILES
#define _LARGE_FILES
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#endif

//
// Windows specific stuff------------------------------------------
#if defined(_WIN32)

// define strict header for windows
#ifndef STRICT
#define STRICT
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#endif

// Never include the windows header here when building SVTK itself.
#if defined(SVTK_IN_SVTK)
#undef SVTK_INCLUDE_WINDOWS_H
#endif

#if defined(_WIN32)
// Include the windows header here only if requested by user code.
#if defined(SVTK_INCLUDE_WINDOWS_H)
#include <windows.h>
// Define types from the windows header file.
typedef DWORD svtkWindowsDWORD;
typedef PVOID svtkWindowsPVOID;
typedef LPVOID svtkWindowsLPVOID;
typedef HANDLE svtkWindowsHANDLE;
typedef LPTHREAD_START_ROUTINE svtkWindowsLPTHREAD_START_ROUTINE;
#else
// Define types from the windows header file.
typedef unsigned long svtkWindowsDWORD;
typedef void* svtkWindowsPVOID;
typedef svtkWindowsPVOID svtkWindowsLPVOID;
typedef svtkWindowsPVOID svtkWindowsHANDLE;
typedef svtkWindowsDWORD(__stdcall* svtkWindowsLPTHREAD_START_ROUTINE)(svtkWindowsLPVOID);
#endif
// Enable workaround for windows header name mangling.
// See SVTK/Utilities/Upgrading/README.WindowsMangling.txt for details.
#if !defined(__SVTK_WRAP__) && !defined(__WRAP_GCCXML__)
#define SVTK_WORKAROUND_WINDOWS_MANGLE
#endif

#if defined(_MSC_VER) // Visual studio
#pragma warning(disable : 4311)
#pragma warning(disable : 4312)
#endif

#define svtkGetWindowLong GetWindowLongPtr
#define svtkSetWindowLong SetWindowLongPtr
#define svtkLONG LONG_PTR
#define svtkGWL_WNDPROC GWLP_WNDPROC
#define svtkGWL_HINSTANCE GWLP_HINSTANCE
#define svtkGWL_USERDATA GWLP_USERDATA

#endif

#if defined(_MSC_VER)
// Enable MSVC compiler warning messages that are useful but off by default.
#pragma warning(default : 4263) /* no override, call convention differs */
// Disable MSVC compiler warning messages that often occur in valid code.
#if !defined(SVTK_DISPLAY_WIN32_WARNINGS)
#pragma warning(disable : 4003) /* not enough actual parameters for macro */
#pragma warning(disable : 4097) /* typedef is synonym for class */
#pragma warning(disable : 4127) /* conditional expression is constant */
#pragma warning(disable : 4244) /* possible loss in conversion */
#pragma warning(disable : 4251) /* missing DLL-interface */
#pragma warning(disable : 4305) /* truncation from type1 to type2 */
#pragma warning(disable : 4309) /* truncation of constant value */
#pragma warning(disable : 4514) /* unreferenced inline function */
#pragma warning(disable : 4706) /* assignment in conditional expression */
#pragma warning(disable : 4710) /* function not inlined */
#pragma warning(disable : 4786) /* identifier truncated in debug info */
#endif
#endif

// Now set up the generic SVTK export macro.
#if defined(SVTK_BUILD_SHARED_LIBS)
#define SVTK_EXPORT SVTK_ABI_EXPORT
#else
#define SVTK_EXPORT
#endif

#endif
// SVTK-HeaderTest-Exclude: svtkWin32Header.h
