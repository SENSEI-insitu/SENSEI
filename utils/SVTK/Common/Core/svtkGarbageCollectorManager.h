/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGarbageCollectorManager.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGarbageCollectorManager
 * @brief   Manages the svtkGarbageCollector singleton.
 *
 * svtkGarbageCollectorManager should be included in any translation unit
 * that will use svtkGarbageCollector or that implements the singleton
 * pattern.  It makes sure that the svtkGarbageCollector singleton is created
 * before and destroyed after it is used.
 */

#ifndef svtkGarbageCollectorManager_h
#define svtkGarbageCollectorManager_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

#include "svtkDebugLeaksManager.h" // DebugLeaks is around longer than
                                  // the garbage collector.

class SVTKCOMMONCORE_EXPORT svtkGarbageCollectorManager
{
public:
  svtkGarbageCollectorManager();
  ~svtkGarbageCollectorManager();

private:
  svtkGarbageCollectorManager(const svtkGarbageCollectorManager&);
  svtkGarbageCollectorManager& operator=(const svtkGarbageCollectorManager&);
};

// This instance will show up in any translation unit that uses
// svtkGarbageCollector or that has a singleton.  It will make sure
// svtkGarbageCollector is initialized before it is used finalized when
// it is done being used.
static svtkGarbageCollectorManager svtkGarbageCollectorManagerInstance;

#endif
// SVTK-HeaderTest-Exclude: svtkGarbageCollectorManager.h
