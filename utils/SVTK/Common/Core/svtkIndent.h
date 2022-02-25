/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIndent.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkIndent
 * @brief   a simple class to control print indentation
 *
 * svtkIndent is used to control indentation during the chaining print
 * process. This way nested objects can correctly indent themselves.
 */

#ifndef svtkIndent_h
#define svtkIndent_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

class svtkIndent;
SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream& os, const svtkIndent& o);

class SVTKCOMMONCORE_EXPORT svtkIndent
{
public:
  void Delete() { delete this; }
  explicit svtkIndent(int ind = 0) { this->Indent = ind; }
  static svtkIndent* New();

  /**
   * Determine the next indentation level. Keep indenting by two until the
   * max of forty.
   */
  svtkIndent GetNextIndent();

  /**
   * Print out the indentation. Basically output a bunch of spaces.
   */
  friend SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream& os, const svtkIndent& o);

protected:
  int Indent;
};

#endif
// SVTK-HeaderTest-Exclude: svtkIndent.h
