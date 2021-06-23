/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWrappingHints.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWrappingHints
 * @brief   hint macros for wrappers
 *
 * The macros defined in this file can be used to supply hints for the
 * wrappers.
 */

#ifndef svtkWrappingHints_h
#define svtkWrappingHints_h

#ifdef __SVTK_WRAP__
#define SVTK_WRAP_HINTS_DEFINED
// Exclude a method or class from wrapping
#define SVTK_WRAPEXCLUDE [[svtk::wrapexclude]]
// The return value points to a newly-created SVTK object.
#define SVTK_NEWINSTANCE [[svtk::newinstance]]
// The parameter is a pointer to a zerocopy buffer.
#define SVTK_ZEROCOPY [[svtk::zerocopy]]
// Set preconditions for a function
#define SVTK_EXPECTS(x) [[svtk::expects(x)]]
// Set size hint for parameter or return value
#define SVTK_SIZEHINT(...) [[svtk::sizehint(__VA_ARGS__)]]
#endif

#ifndef SVTK_WRAP_HINTS_DEFINED
#define SVTK_WRAPEXCLUDE
#define SVTK_NEWINSTANCE
#define SVTK_ZEROCOPY
#define SVTK_EXPECTS(x)
#define SVTK_SIZEHINT(...)
#endif

#endif
// SVTK-HeaderTest-Exclude: svtkWrappingHints.h
