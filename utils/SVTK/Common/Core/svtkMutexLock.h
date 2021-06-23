/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMutexLock.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMutexLock
 * @brief   mutual exclusion locking class
 *
 * svtkMutexLock allows the locking of variables which are accessed
 * through different threads.  This header file also defines
 * svtkSimpleMutexLock which is not a subclass of svtkObject.
 */

#ifndef svtkMutexLock_h
#define svtkMutexLock_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#if defined(SVTK_USE_PTHREADS)
#include <pthread.h> // Needed for PTHREAD implementation of mutex
typedef pthread_mutex_t svtkMutexType;
#endif

#ifdef SVTK_USE_WIN32_THREADS
typedef svtkWindowsHANDLE svtkMutexType;
#endif

#ifndef SVTK_USE_PTHREADS
#ifndef SVTK_USE_WIN32_THREADS
typedef int svtkMutexType;
#endif
#endif

// Mutex lock that is not a svtkObject.
class SVTKCOMMONCORE_EXPORT svtkSimpleMutexLock
{
public:
  // left public purposely
  svtkSimpleMutexLock();
  virtual ~svtkSimpleMutexLock();

  static svtkSimpleMutexLock* New();

  void Delete() { delete this; }

  /**
   * Lock the svtkMutexLock
   */
  void Lock(void);

  /**
   * Unlock the svtkMutexLock
   */
  void Unlock(void);

protected:
  friend class svtkSimpleConditionVariable;
  svtkMutexType MutexLock;

private:
  svtkSimpleMutexLock(const svtkSimpleMutexLock& other) = delete;
  svtkSimpleMutexLock& operator=(const svtkSimpleMutexLock& rhs) = delete;
};

class SVTKCOMMONCORE_EXPORT svtkMutexLock : public svtkObject
{
public:
  static svtkMutexLock* New();

  svtkTypeMacro(svtkMutexLock, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Lock the svtkMutexLock
   */
  void Lock(void);

  /**
   * Unlock the svtkMutexLock
   */
  void Unlock(void);

protected:
  friend class svtkConditionVariable; // needs to get at SimpleMutexLock.

  svtkSimpleMutexLock SimpleMutexLock;
  svtkMutexLock() {}

private:
  svtkMutexLock(const svtkMutexLock&) = delete;
  void operator=(const svtkMutexLock&) = delete;
};

inline void svtkMutexLock::Lock(void)
{
  this->SimpleMutexLock.Lock();
}

inline void svtkMutexLock::Unlock(void)
{
  this->SimpleMutexLock.Unlock();
}

#endif
