/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIndent.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkIndent.h"
#include "svtkObjectFactory.h"

//------------------------------------------------------------------------------
svtkIndent* svtkIndent::New()
{
  return new svtkIndent; // not a SVTK object, don't use object factory macros
}

#define SVTK_STD_INDENT 2
#define SVTK_NUMBER_OF_BLANKS 40

static const char blanks[SVTK_NUMBER_OF_BLANKS + 1] = "                                        ";

// Determine the next indentation level. Keep indenting by two until the
// max of forty.
svtkIndent svtkIndent::GetNextIndent()
{
  int indent = this->Indent + SVTK_STD_INDENT;
  if (indent > SVTK_NUMBER_OF_BLANKS)
  {
    indent = SVTK_NUMBER_OF_BLANKS;
  }
  return svtkIndent(indent);
}

// Print out the indentation. Basically output a bunch of spaces.
ostream& operator<<(ostream& os, const svtkIndent& ind)
{
  os << blanks + (SVTK_NUMBER_OF_BLANKS - ind.Indent);
  return os;
}
