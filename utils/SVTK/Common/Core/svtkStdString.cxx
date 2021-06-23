/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStdString.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkStdString.h"

//----------------------------------------------------------------------------
ostream& operator<<(ostream& os, const svtkStdString& s)
{
  return os << s.c_str();
}
