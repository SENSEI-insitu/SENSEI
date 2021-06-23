/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWindows.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkWindows_h
#define svtkWindows_h

/* This header is useless when not on windows or when the windows
   header has already been included at the top of SVTK.  Block the
   whole thing on this condition.  */
#if defined(_WIN32) && !defined(SVTK_INCLUDE_WINDOWS_H)

/*
Define some macros to shorten the windows header.  Record which ones
we defined here so that we can undefine them later.

See this page for details:
http://msdn.microsoft.com/library/en-us/vccore/html/_core_faster_builds_and_smaller_header_files.asp
*/
#if !defined(SVTK_WINDOWS_FULL)
#if !defined(VC_EXTRALEAN)
#define VC_EXTRALEAN
#define SVTK_WINDOWS_VC_EXTRALEAN
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#define SVTK_WINDOWS_WIN32_LEAN_AND_MEAN
#endif
#if !defined(NOSERVICE)
#define NOSERVICE
#define SVTK_WINDOWS_NOSERVICE
#endif
#if !defined(NOMCX)
#define NOMCX
#define SVTK_WINDOWS_NOMCX
#endif
#if !defined(NOIME)
#define NOIME
#define SVTK_WINDOWS_NOIME
#endif
#if !defined(NONLS)
#define NONLS
#define SVTK_WINDOWS_NONLS
#endif
#endif

/* Include the real windows header. */
#include <windows.h>

/* Undefine any macros we defined to shorten the windows header.
   Leave the SVTK_WINDOWS_* versions defined so that user code can tell
   what parts of the windows header were included.  */
#if !defined(SVTK_WINDOWS_FULL)
#if defined(SVTK_WINDOWS_VC_EXTRALEAN)
#undef VC_EXTRALEAN
#endif
#if defined(SVTK_WINDOWS_WIN32_LEAN_AND_MEAN)
#undef WIN32_LEAN_AND_MEAN
#endif
#if defined(SVTK_WINDOWS_NOSERVICE)
#undef NOSERVICE
#endif
#if defined(SVTK_WINDOWS_NOMCX)
#undef NOMCX
#endif
#if defined(SVTK_WINDOWS_NOIME)
#undef NOIME
#endif
#if defined(SVTK_WINDOWS_NONLS)
#undef NONLS
#endif
#endif

#endif /* defined(_WIN32) && !defined(SVTK_INCLUDE_WINDOWS_H) */

#endif
// SVTK-HeaderTest-Exclude: svtkWindows.h
