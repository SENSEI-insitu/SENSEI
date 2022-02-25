/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVersion.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkVersion
 * @brief   Versioning class for svtk
 *
 * Holds methods for defining/determining the current svtk version
 * (major, minor, build).
 *
 * @warning
 * This file will change frequently to update the SVTKSourceVersion which
 * timestamps a particular source release.
 */

#ifndef svtkVersion_h
#define svtkVersion_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
#include "svtkVersionMacros.h" // For version macros

class SVTKCOMMONCORE_EXPORT svtkVersion : public svtkObject
{
public:
  static svtkVersion* New();
  svtkTypeMacro(svtkVersion, svtkObject);

  /**
   * Return the version of svtk this object is a part of.
   * A variety of methods are included. GetSVTKSourceVersion returns a string
   * with an identifier which timestamps a particular source tree.
   */
  static const char* GetSVTKVersion() { return SVTK_VERSION; }
  static int GetSVTKMajorVersion() { return SVTK_MAJOR_VERSION; }
  static int GetSVTKMinorVersion() { return SVTK_MINOR_VERSION; }
  static int GetSVTKBuildVersion() { return SVTK_BUILD_VERSION; }
  static const char* GetSVTKSourceVersion() { return SVTK_SOURCE_VERSION; }

protected:
  svtkVersion() {} // insure constructor/destructor protected
  ~svtkVersion() override {}

private:
  svtkVersion(const svtkVersion&) = delete;
  void operator=(const svtkVersion&) = delete;
};

extern "C"
{
  SVTKCOMMONCORE_EXPORT const char* GetSVTKVersion();
}

#endif

// SVTK-HeaderTest-Exclude: svtkVersion.h
