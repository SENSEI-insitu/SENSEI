/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDynamicLoader.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDynamicLoader
 * @brief   class interface to system dynamic libraries
 *
 * svtkDynamicLoader provides a portable interface to loading dynamic
 * libraries into a process.
 * @sa
 * A more portable and lightweight solution is kwsys::DynamicLoader
 */

#ifndef svtkDynamicLoader_h
#define svtkDynamicLoader_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
#include <svtksys/DynamicLoader.hxx>

typedef svtksys::DynamicLoader::LibraryHandle svtkLibHandle;
typedef svtksys::DynamicLoader::SymbolPointer svtkSymbolPointer;

class SVTKCOMMONCORE_EXPORT svtkDynamicLoader : public svtkObject
{
public:
  static svtkDynamicLoader* New();
  svtkTypeMacro(svtkDynamicLoader, svtkObject);

  /**
   * Load a dynamic library into the current process.
   * The returned svtkLibHandle can be used to access the symbols in the
   * library.
   */
  static svtkLibHandle OpenLibrary(const char*);
  static svtkLibHandle OpenLibrary(const char*, int);

  /**
   * Attempt to detach a dynamic library from the
   * process.  A value of true is returned if it is successful.
   */
  static int CloseLibrary(svtkLibHandle);

  /**
   * Find the address of the symbol in the given library
   */
  static svtkSymbolPointer GetSymbolAddress(svtkLibHandle, const char*);

  /**
   * Return the library prefix for the given architecture
   */
  static const char* LibPrefix();

  /**
   * Return the library extension for the given architecture
   */
  static const char* LibExtension();

  /**
   * Return the last error produced from a calls made on this class.
   */
  static const char* LastError();

protected:
  svtkDynamicLoader() {}
  ~svtkDynamicLoader() override {}

private:
  svtkDynamicLoader(const svtkDynamicLoader&) = delete;
  void operator=(const svtkDynamicLoader&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkDynamicLoader.h
