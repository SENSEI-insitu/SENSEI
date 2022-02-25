/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCommonInformationKeyManager.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCommonInformationKeyManager
 * @brief   Manages key types in svtkCommon.
 *
 * svtkCommonInformationKeyManager is included in the header of any
 * subclass of svtkInformationKey defined in the svtkCommon library.
 * It makes sure that the table of keys is created before and
 * destroyed after it is used.
 */

#ifndef svtkCommonInformationKeyManager_h
#define svtkCommonInformationKeyManager_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

#include "svtkDebugLeaksManager.h" // DebugLeaks exists longer than info keys.

class svtkInformationKey;

class SVTKCOMMONCORE_EXPORT svtkCommonInformationKeyManager
{
public:
  svtkCommonInformationKeyManager();
  ~svtkCommonInformationKeyManager();

  /**
   * Called by constructors of svtkInformationKey subclasses defined in
   * svtkCommon to register themselves with the manager.  The
   * instances will be deleted when svtkCommon is unloaded on
   * program exit.
   */
  static void Register(svtkInformationKey* key);

private:
  // Unimplemented
  svtkCommonInformationKeyManager(const svtkCommonInformationKeyManager&);
  svtkCommonInformationKeyManager& operator=(const svtkCommonInformationKeyManager&);

  static void ClassInitialize();
  static void ClassFinalize();
};

// This instance will show up in any translation unit that uses key
// types defined in svtkCommon or that has a singleton.  It will
// make sure svtkCommonInformationKeyManager's vector of keys is
// initialized before and destroyed after it is used.
static svtkCommonInformationKeyManager svtkCommonInformationKeyManagerInstance;

#endif
// SVTK-HeaderTest-Exclude: svtkCommonInformationKeyManager.h
