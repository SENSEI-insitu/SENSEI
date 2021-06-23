/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkThreadMessager.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkThreadMessager
 * @brief   A class for performing inter-thread messaging
 *
 * svtkThreadMessager is a class that provides support for messaging between
 * threads multithreaded using pthreads or Windows messaging.
 */

#ifndef svtkThreadMessager_h
#define svtkThreadMessager_h

#include "svtkCommonSystemModule.h" // For export macro
#include "svtkObject.h"

#if defined(SVTK_USE_PTHREADS)
#include <pthread.h> // Needed for pthread types
#endif

class SVTKCOMMONSYSTEM_EXPORT svtkThreadMessager : public svtkObject
{
public:
  static svtkThreadMessager* New();

  svtkTypeMacro(svtkThreadMessager, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Wait (block, non-busy) until another thread sends a
   * message.
   */
  void WaitForMessage();

  /**
   * Send a message to all threads who are waiting via
   * WaitForMessage().
   */
  void SendWakeMessage();

  /**
   * pthreads only. If the wait is enabled, the thread who
   * is to call WaitForMessage() will block until a receiver
   * thread is ready to receive.
   */
  void EnableWaitForReceiver();

  /**
   * pthreads only. If the wait is enabled, the thread who
   * is to call WaitForMessage() will block until a receiver
   * thread is ready to receive.
   */
  void DisableWaitForReceiver();

  /**
   * pthreads only.
   * If wait is enable, this will block until one thread is ready
   * to receive a message.
   */
  void WaitForReceiver();

protected:
  svtkThreadMessager();
  ~svtkThreadMessager() override;

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_t Mutex;
  pthread_cond_t PSignal;
#endif

#ifdef SVTK_USE_WIN32_THREADS
  svtkWindowsHANDLE WSignal;
#endif

private:
  svtkThreadMessager(const svtkThreadMessager&) = delete;
  void operator=(const svtkThreadMessager&) = delete;
};

#endif
