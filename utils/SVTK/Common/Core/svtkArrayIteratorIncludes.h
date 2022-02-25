/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayIteratorIncludes.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkArrayIteratorIncludes
 * @brief   centralize array iterator type includes
 * required when using the svtkArrayIteratorTemplateMacro.
 *
 *
 * A CXX file using svtkArrayIteratorTemplateMacro needs to include the
 * header files for all types of iterators supported by the macro.  As
 * new arrays and new iterators are added, svtkArrayIteratorTemplateMacro
 * will also need to be updated to switch to the additional cases.
 * However, this would imply any code using the macro will start giving
 * compilation errors unless they include the new iterator headers. The
 * svtkArrayIteratorIncludes.h will streamline this issue. Every file
 * using the svtkArrayIteratorTemplateMacro must include this
 * svtkArrayIteratorIncludes.h. As new iterators are added and the
 * svtkArrayIteratorTemplateMacro updated, one needs to update this header
 * file alone.
 */

#ifndef svtkArrayIteratorIncludes_h
#define svtkArrayIteratorIncludes_h

// Iterators.
#include "svtkArrayIteratorTemplate.h"
#include "svtkBitArrayIterator.h"

#endif

// SVTK-HeaderTest-Exclude: svtkArrayIteratorIncludes.h
