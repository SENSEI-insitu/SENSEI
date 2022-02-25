/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFloatingPointExceptions.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFloatingPointExceptions
 * @brief   Deal with floating-point exceptions
 *
 * Right now it is really basic and it only provides a function to enable
 * floating point exceptions on some compilers.
 * Note that Borland C++ has floating-point exceptions by default, not
 * Visual studio nor gcc. It is mainly use to optionally enable floating
 * point exceptions in the C++ tests.
 */

#ifndef svtkFloatingPointExceptions_h
#define svtkFloatingPointExceptions_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"   // For SVTKCOMMONCORE_EXPORT

class SVTKCOMMONCORE_EXPORT svtkFloatingPointExceptions
{
public:
  /**
   * Enable floating point exceptions.
   */
  static void Enable();

  /**
   * Disable floating point exceptions.
   */
  static void Disable();

private:
  svtkFloatingPointExceptions() = delete;
  svtkFloatingPointExceptions(const svtkFloatingPointExceptions&) = delete;
  void operator=(const svtkFloatingPointExceptions&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkFloatingPointExceptions.h
