/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIOStreamFwd.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIOStreamFwd
 * @brief   Forward-declare C++ iostreams as used by SVTK.
 *
 * This header forward-declares the proper streams.
 */

#ifndef svtkIOStreamFwd_h
#define svtkIOStreamFwd_h

#include "svtkConfigure.h"

#ifdef _MSC_VER
#pragma warning(push, 3)
#endif

// Forward-declare ansi streams.
#include <iosfwd>
using std::filebuf;
using std::fstream;
using std::ios;
using std::iostream;
using std::istream;
using std::ostream;
using std::streambuf;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // svtkIOStreamFwd_h
// SVTK-HeaderTest-Exclude: svtkIOStreamFwd.h
