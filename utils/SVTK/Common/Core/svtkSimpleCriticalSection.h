/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCriticalSection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSimpleCriticalSection
 * @brief   Critical section locking class
 *
 * svtkCriticalSection allows the locking of variables which are accessed
 * through different threads.  This header file also defines
 * svtkSimpleCriticalSection which is not a subclass of svtkObject.
 * The API is identical to that of svtkMutexLock, and the behavior is
 * identical as well, except on Windows 9x/NT platforms. The only difference
 * on these platforms is that svtkMutexLock is more flexible, in that
 * it works across processes as well as across threads, but also costs
 * more, in that it evokes a 600-cycle x86 ring transition. The
 * svtkCriticalSection provides a higher-performance equivalent (on
 * Windows) but won't work across processes. Since it is unclear how,
 * in svtk, an object at the svtk level can be shared across processes
 * in the first place, one should use svtkCriticalSection unless one has
 * a very good reason to use svtkMutexLock. If higher-performance equivalents
 * for non-Windows platforms (Irix, SunOS, etc) are discovered, they
 * should replace the implementations in this class
 */

#ifndef svtkSimpleCriticalSection_h
#define svtkSimpleCriticalSection_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

#if defined(SVTK_USE_PTHREADS)
#include <pthread.h> // Needed for pthreads implementation of mutex
typedef pthread_mutex_t svtkCritSecType;
#endif

#ifdef SVTK_USE_WIN32_THREADS
#include "svtkWindows.h" // Needed for win32 implementation of mutex
typedef CRITICAL_SECTION svtkCritSecType;
#endif

#ifndef SVTK_USE_PTHREADS
#ifndef SVTK_USE_WIN32_THREADS
typedef int svtkCritSecType;
#endif
#endif

// Critical Section object that is not a svtkObject.
class SVTKCOMMONCORE_EXPORT svtkSimpleCriticalSection
{
public:
  // Default cstor
  svtkSimpleCriticalSection() { this->Init(); }
  // Construct object locked if isLocked is different from 0
  svtkSimpleCriticalSection(int isLocked)
  {
    this->Init();
    if (isLocked)
    {
      this->Lock();
    }
  }
  // Destructor
  virtual ~svtkSimpleCriticalSection();

  void Init();

  /**
   * Lock the svtkCriticalSection
   */
  void Lock();

  /**
   * Unlock the svtkCriticalSection
   */
  void Unlock();

protected:
  svtkCritSecType CritSec;

private:
  svtkSimpleCriticalSection(const svtkSimpleCriticalSection& other) = delete;
  svtkSimpleCriticalSection& operator=(const svtkSimpleCriticalSection& rhs) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkSimpleCriticalSection.h
