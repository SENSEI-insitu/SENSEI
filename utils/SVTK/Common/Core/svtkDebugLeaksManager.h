/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDebugLeaksManager.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDebugLeaksManager
 * @brief   Manages the svtkDebugLeaks singleton.
 *
 * svtkDebugLeaksManager should be included in any translation unit
 * that will use svtkDebugLeaks or that implements the singleton
 * pattern.  It makes sure that the svtkDebugLeaks singleton is created
 * before and destroyed after all other singletons in SVTK.
 */

#ifndef svtkDebugLeaksManager_h
#define svtkDebugLeaksManager_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

class SVTKCOMMONCORE_EXPORT svtkDebugLeaksManager
{
public:
  svtkDebugLeaksManager();
  ~svtkDebugLeaksManager();

private:
  svtkDebugLeaksManager(const svtkDebugLeaksManager&);
  svtkDebugLeaksManager& operator=(const svtkDebugLeaksManager&);
};

// This instance will show up in any translation unit that uses
// svtkDebugLeaks or that has a singleton.  It will make sure
// svtkDebugLeaks is initialized before it is used and is the last
// static object destroyed.
static svtkDebugLeaksManager svtkDebugLeaksManagerInstance;

#endif
// SVTK-HeaderTest-Exclude: svtkDebugLeaksManager.h
