/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDynamicLoader.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDynamicLoader.h"

#include "svtkDebugLeaks.h"
#include "svtkObjectFactory.h"

//-----------------------------------------------------------------------------
svtkDynamicLoader* svtkDynamicLoader::New()
{
  SVTK_STANDARD_NEW_BODY(svtkDynamicLoader);
}

// ----------------------------------------------------------------------------
svtkLibHandle svtkDynamicLoader::OpenLibrary(const char* libname)
{
  return svtksys::DynamicLoader::OpenLibrary(libname);
}

// ----------------------------------------------------------------------------
svtkLibHandle svtkDynamicLoader::OpenLibrary(const char* libname, int flags)
{
  return svtksys::DynamicLoader::OpenLibrary(libname, flags);
}

// ----------------------------------------------------------------------------
int svtkDynamicLoader::CloseLibrary(svtkLibHandle lib)
{
  return svtksys::DynamicLoader::CloseLibrary(lib);
}

// ----------------------------------------------------------------------------
svtkSymbolPointer svtkDynamicLoader::GetSymbolAddress(svtkLibHandle lib, const char* sym)
{
  return svtksys::DynamicLoader::GetSymbolAddress(lib, sym);
}

// ----------------------------------------------------------------------------
const char* svtkDynamicLoader::LibPrefix()
{
  return svtksys::DynamicLoader::LibPrefix();
}

// ----------------------------------------------------------------------------
const char* svtkDynamicLoader::LibExtension()
{
  return svtksys::DynamicLoader::LibExtension();
}

// ----------------------------------------------------------------------------
const char* svtkDynamicLoader::LastError()
{
  return svtksys::DynamicLoader::LastError();
}
