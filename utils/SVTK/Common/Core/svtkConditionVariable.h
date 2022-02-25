/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkConditionVariable.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkConditionVariable
 * @brief   mutual exclusion locking class
 *
 * svtkConditionVariable allows the locking of variables which are accessed
 * through different threads.  This header file also defines
 * svtkSimpleConditionVariable which is not a subclass of svtkObject.
 *
 * The win32 implementation is based on notes provided by
 * Douglas C. Schmidt and Irfan Pyarali,
 * Department of Computer Science,
 * Washington University, St. Louis, Missouri.
 * http://www.cs.wustl.edu/~schmidt/win32-cv-1.html
 */

#ifndef svtkConditionVariable_h
#define svtkConditionVariable_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include "svtkMutexLock.h" // Need for friend access to svtkSimpleMutexLock

#if defined(SVTK_USE_PTHREADS)
#include <pthread.h> // Need POSIX thread implementation of mutex (even win32 provides mutexes)
typedef pthread_cond_t svtkConditionType;
#endif

// Typically a top level windows application sets _WIN32_WINNT. If it is not set we set it to
// 0x0501 (Windows XP)
#ifdef SVTK_USE_WIN32_THREADS
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // 0x0501 means target Windows XP or later
#endif
#include "svtkWindows.h" // Needed for win32 CRITICAL_SECTION, HANDLE, etc.
#endif

#ifdef SVTK_USE_WIN32_THREADS
#if 1
typedef struct
{
  // Number of threads waiting on condition.
  int WaitingThreadCount;

  // Lock for WaitingThreadCount
  CRITICAL_SECTION WaitingThreadCountCritSec;

  // Semaphore to block threads waiting for the condition to change.
  svtkWindowsHANDLE Semaphore;

  // An event used to wake up thread(s) waiting on the semaphore
  // when pthread_cond_signal or pthread_cond_broadcast is called.
  svtkWindowsHANDLE DoneWaiting;

  // Was pthread_cond_broadcast called?
  size_t WasBroadcast;
} pthread_cond_t;

typedef pthread_cond_t svtkConditionType;
#else  // 0
typedef struct
{
  // Number of threads waiting on condition.
  int WaitingThreadCount;

  // Lock for WaitingThreadCount
  CRITICAL_SECTION WaitingThreadCountCritSec;

  // Number of threads to release when pthread_cond_broadcast()
  // or pthread_cond_signal() is called.
  int ReleaseCount;

  // Used to prevent one thread from decrementing ReleaseCount all
  // by itself instead of letting others respond.
  int NotifyCount;

  // A manual-reset event that's used to block and release waiting threads.
  svtkWindowsHANDLE Event;
} pthread_cond_t;

typedef pthread_cond_t svtkConditionType;
#endif // 0
#endif // SVTK_USE_WIN32_THREADS

#ifndef SVTK_USE_PTHREADS
#ifndef SVTK_USE_WIN32_THREADS
typedef int svtkConditionType;
#endif
#endif

// Condition variable that is not a svtkObject.
class SVTKCOMMONCORE_EXPORT svtkSimpleConditionVariable
{
public:
  svtkSimpleConditionVariable();
  ~svtkSimpleConditionVariable();

  static svtkSimpleConditionVariable* New();

  void Delete() { delete this; }

  /**
   * Wake one thread waiting for the condition to change.
   */
  void Signal();

  /**
   * Wake all threads waiting for the condition to change.
   */
  void Broadcast();

  /**
   * Wait for the condition to change.
   * Upon entry, the mutex must be locked and the lock held by the calling thread.
   * Upon exit, the mutex will be locked and held by the calling thread.
   * Between entry and exit, the mutex will be unlocked and may be held by other threads.

   * @param mutex The mutex that should be locked on entry and will be locked on exit (but not in
   between)
   * @retval Normally, this function returns 0. Should a thread be interrupted by a signal, a
   non-zero value may be returned.
   */
  int Wait(svtkSimpleMutexLock& mutex);

protected:
  svtkConditionType ConditionVariable;

private:
  svtkSimpleConditionVariable(const svtkSimpleConditionVariable& other) = delete;
  svtkSimpleConditionVariable& operator=(const svtkSimpleConditionVariable& rhs) = delete;
};

class SVTKCOMMONCORE_EXPORT svtkConditionVariable : public svtkObject
{
public:
  static svtkConditionVariable* New();
  svtkTypeMacro(svtkConditionVariable, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Wake one thread waiting for the condition to change.
   */
  void Signal();

  /**
   * Wake all threads waiting for the condition to change.
   */
  void Broadcast();

  /**
   * Wait for the condition to change.
   * Upon entry, the mutex must be locked and the lock held by the calling thread.
   * Upon exit, the mutex will be locked and held by the calling thread.
   * Between entry and exit, the mutex will be unlocked and may be held by other threads.

   * @param mutex The mutex that should be locked on entry and will be locked on exit (but not in
   between)
   * @retval Normally, this function returns 0. Should a thread be interrupted by a signal, a
   non-zero value may be returned.
   */
  int Wait(svtkMutexLock* mutex);

protected:
  svtkConditionVariable() {}

  svtkSimpleConditionVariable SimpleConditionVariable;

private:
  svtkConditionVariable(const svtkConditionVariable&) = delete;
  void operator=(const svtkConditionVariable&) = delete;
};

inline void svtkConditionVariable::Signal()
{
  this->SimpleConditionVariable.Signal();
}

inline void svtkConditionVariable::Broadcast()
{
  this->SimpleConditionVariable.Broadcast();
}

inline int svtkConditionVariable::Wait(svtkMutexLock* lock)
{
  return this->SimpleConditionVariable.Wait(lock->SimpleMutexLock);
}

#endif // svtkConditionVariable_h
