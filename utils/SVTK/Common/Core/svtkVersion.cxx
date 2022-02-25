/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVersion.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkVersion.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkVersion);
const char* GetSVTKVersion()
{
  return svtkVersion::GetSVTKVersion();
}
