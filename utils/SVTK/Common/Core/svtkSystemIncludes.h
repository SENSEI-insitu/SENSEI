/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSystemIncludes.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSystemIncludes
 * @brief   transition SVTK to ANSI C++, centralize
 * inclusion of system files
 *
 * The svtkSystemIncludes centralizes the inclusion of system include
 * files.
 */

#ifndef svtkSystemIncludes_h
#define svtkSystemIncludes_h

/* first include the local configuration for this machine */
#define SVTK_SYSTEM_INCLUDES_INSIDE
#include "svtkWin32Header.h"
#undef SVTK_SYSTEM_INCLUDES_INSIDE

// The language wrapper files do not need the real streams.  They
// define SVTK_STREAMS_FWD_ONLY so that the streams are only
// forward-declared.  This significantly improves compile time on some
// platforms.
#if defined(SVTK_STREAMS_FWD_ONLY)
#include "svtkIOStreamFwd.h" // Forward-declare the C++ streams.
#else
#include "svtkIOStream.h" // Include the real C++ streams.
#endif

// Setup the basic types to be used by SVTK.
#include "svtkType.h"

// Define some macros to provide wrapping hints
#include "svtkWrappingHints.h"

// this should be removed at some point
#define SVTK_USE_EXECUTIVES

#define SVTK_SYSTEM_INCLUDES_INSIDE
#include "svtkOStreamWrapper.h" // Include the ostream wrapper.

#include "svtkOStrStreamWrapper.h" // Include the ostrstream wrapper.
#undef SVTK_SYSTEM_INCLUDES_INSIDE

// Include generic stuff.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// These types define error codes for svtk functions
#define SVTK_OK 1
#define SVTK_ERROR 2

// These types define different text properties
#define SVTK_ARIAL 0
#define SVTK_COURIER 1
#define SVTK_TIMES 2
#define SVTK_UNKNOWN_FONT 3
#define SVTK_FONT_FILE 4

#define SVTK_TEXT_LEFT 0
#define SVTK_TEXT_CENTERED 1
#define SVTK_TEXT_RIGHT 2

#define SVTK_TEXT_BOTTOM 0
#define SVTK_TEXT_TOP 2

#define SVTK_TEXT_GLOBAL_ANTIALIASING_SOME 0
#define SVTK_TEXT_GLOBAL_ANTIALIASING_NONE 1
#define SVTK_TEXT_GLOBAL_ANTIALIASING_ALL 2

#define SVTK_LUMINANCE 1
#define SVTK_LUMINANCE_ALPHA 2
#define SVTK_RGB 3
#define SVTK_RGBA 4

#define SVTK_COLOR_MODE_DEFAULT 0
#define SVTK_COLOR_MODE_MAP_SCALARS 1
#define SVTK_COLOR_MODE_DIRECT_SCALARS 2

// Constants for InterpolationType
#define SVTK_NEAREST_INTERPOLATION 0
#define SVTK_LINEAR_INTERPOLATION 1
#define SVTK_CUBIC_INTERPOLATION 2

// Constants for SlabType
#define SVTK_IMAGE_SLAB_MIN 0
#define SVTK_IMAGE_SLAB_MAX 1
#define SVTK_IMAGE_SLAB_MEAN 2
#define SVTK_IMAGE_SLAB_SUM 3

// For volume rendering
#define SVTK_MAX_VRCOMP 4

// If SVTK_USE_PTHREADS is defined, then the multithreaded
// function is of type void *, and returns nullptr
// Otherwise the type is void which is correct for WIN32
#ifdef SVTK_USE_PTHREADS
#define SVTK_THREAD_RETURN_VALUE nullptr
#define SVTK_THREAD_RETURN_TYPE void*
#endif

#ifdef SVTK_USE_WIN32_THREADS
#define SVTK_THREAD_RETURN_VALUE 0
#define SVTK_THREAD_RETURN_TYPE svtkWindowsDWORD __stdcall
#endif

#if !defined(SVTK_USE_PTHREADS) && !defined(SVTK_USE_WIN32_THREADS)
#define SVTK_THREAD_RETURN_VALUE
#define SVTK_THREAD_RETURN_TYPE void
#endif

// For encoding

#define SVTK_ENCODING_NONE 0 // to specify that no encoding should occur
#define SVTK_ENCODING_US_ASCII 1
#define SVTK_ENCODING_UNICODE 2
#define SVTK_ENCODING_UTF_8 3
#define SVTK_ENCODING_ISO_8859_1 4
#define SVTK_ENCODING_ISO_8859_2 5
#define SVTK_ENCODING_ISO_8859_3 6
#define SVTK_ENCODING_ISO_8859_4 7
#define SVTK_ENCODING_ISO_8859_5 8
#define SVTK_ENCODING_ISO_8859_6 9
#define SVTK_ENCODING_ISO_8859_7 10
#define SVTK_ENCODING_ISO_8859_8 11
#define SVTK_ENCODING_ISO_8859_9 12
#define SVTK_ENCODING_ISO_8859_10 13
#define SVTK_ENCODING_ISO_8859_11 14
#define SVTK_ENCODING_ISO_8859_12 15
#define SVTK_ENCODING_ISO_8859_13 16
#define SVTK_ENCODING_ISO_8859_14 17
#define SVTK_ENCODING_ISO_8859_15 18
#define SVTK_ENCODING_ISO_8859_16 19
#define SVTK_ENCODING_UNKNOWN 20 // leave this one at the end

#endif
// SVTK-HeaderTest-Exclude: svtkSystemIncludes.h
